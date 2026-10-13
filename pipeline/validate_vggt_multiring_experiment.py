"""Validate multi-ring VGGT evidence with cross-view-only reprojection.

The validator deliberately excludes the target view when scoring it.  This
prevents a dense point map from receiving a high score merely by projecting
back into the same image from which it was predicted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import cKDTree


VIEW_ORDER = ("front", "top", "iso", "lsv", "rsv")


def _records(path: Path) -> dict[str, dict[str, Any]]:
    return {
        row["object_id"]: row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _square(path: Path, size: int, grayscale: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(path)
    height, width = image.shape[:2]
    side = max(height, width)
    shape = (side, side) if grayscale else (side, side, 3)
    canvas = np.zeros(shape, dtype=np.uint8)
    y = (side - height) // 2
    x = (side - width) // 2
    canvas[y : y + height, x : x + width] = image
    interpolation = cv2.INTER_NEAREST if grayscale else cv2.INTER_CUBIC
    return cv2.resize(canvas, (size, size), interpolation=interpolation)


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    predicted = int(prediction.sum())
    truth = int(target.sum())
    return {
        "iou": round(intersection / max(union, 1), 6),
        "precision": round(intersection / max(predicted, 1), 6),
        "recall": round(intersection / max(truth, 1), 6),
    }


def _project(
    points: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    size: int,
) -> np.ndarray:
    camera = points @ extrinsic[:, :3].T + extrinsic[:, 3]
    visible = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-6)
    camera = camera[visible]
    if not len(camera):
        return np.zeros((size, size), dtype=np.uint8)
    pixels = camera @ intrinsic.T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    pixels = np.rint(pixels).astype(np.int32)
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < size)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < size)
    )
    pixels = pixels[inside]
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[pixels[:, 1], pixels[:, 0]] = 255
    return cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)


def _agreement(source: np.ndarray, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    cell = np.full_like(source, 245)
    both = (prediction > 0) & (target > 0)
    pred_only = (prediction > 0) & ~both
    target_only = (target > 0) & ~both
    cell[both] = (70, 170, 70)
    cell[pred_only] = (65, 65, 220)
    cell[target_only] = (220, 130, 60)
    return cv2.addWeighted(source, 0.35, cell, 0.65, 0)


def _label(image: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    header = np.full((50, image.shape[1], 3), 248, dtype=np.uint8)
    cv2.putText(header, title, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (18, 18, 18), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(header, subtitle, (7, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.29, (75, 75, 75), 1, cv2.LINE_AA)
    return np.vstack((header, image))


def _cloud_preview(
    before: np.ndarray,
    after: np.ndarray,
    output: Path,
    object_id: str,
) -> None:
    figure = plt.figure(figsize=(12, 7), dpi=150, facecolor="white")
    for panel, (title, points) in enumerate(
        (("MASK + CONFIDENCE", before), ("MULTI-VIEW SUPPORTED", after)), start=1
    ):
        axis = figure.add_subplot(1, 2, panel, projection="3d")
        if len(points):
            count = min(35_000, len(points))
            indices = np.linspace(0, len(points) - 1, count, dtype=np.int64)
            sample = points[indices]
            bounds = np.percentile(sample, (1, 99), axis=0)
            centre = bounds.mean(axis=0)
            radius = max(float(np.max(bounds[1] - bounds[0])) / 2.0, 1e-6)
            axis.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=0.5, c="#5f879a", linewidths=0)
            axis.set_xlim(centre[0] - radius, centre[0] + radius)
            axis.set_ylim(centre[1] - radius, centre[1] + radius)
            axis.set_zlim(centre[2] - radius, centre[2] + radius)
        axis.view_init(elev=24, azim=-45)
        axis.set_box_aspect((1, 1, 1))
        axis.set_title(title, loc="left", fontsize=10)
        axis.set_axis_off()
    figure.suptitle(f"{object_id} · VGGT cross-view support filter · non-metric", fontsize=13)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def validate_object(
    root: Path,
    object_dir: Path,
    record: dict[str, Any],
    confidence_quantile: float,
    support_radius_fraction: float,
    minimum_supported_views: int,
) -> dict[str, Any]:
    report = json.loads((object_dir / "vggt_report.json").read_text(encoding="utf-8"))
    cameras = np.load(object_dir / "vggt_cameras.npz")
    dense = np.load(object_dir / "vggt_dense_evidence.npz")
    extrinsic = cameras["extrinsic"][0]
    intrinsic = cameras["intrinsic"][0]
    world = dense["world_points"][0]
    confidence = dense["world_points_conf"][0]
    size = int(report["image_size"])
    input_manifest_path = object_dir / "input_manifest.json"
    input_manifest = (
        json.loads(input_manifest_path.read_text(encoding="utf-8"))
        if input_manifest_path.is_file()
        else None
    )

    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    candidates: list[np.ndarray] = []
    colours: list[np.ndarray] = []
    thresholds: list[float] = []
    for index, view in enumerate(VIEW_ORDER):
        if input_manifest:
            source_path = root / input_manifest["views"][view]["image"]
            mask_path = root / input_manifest["views"][view]["mask"]
        else:
            source_path = root / record["views"][view]["image_path"]
            mask_path = root / record["views"][view]["jewelry_mask_path"]
        source = _square(source_path, size)
        target = _square(mask_path, size, grayscale=True) > 0
        finite = np.isfinite(world[index]).all(axis=-1) & np.isfinite(confidence[index])
        values = confidence[index][target & finite]
        threshold = float(np.quantile(values, confidence_quantile))
        valid = target & finite & (confidence[index] >= threshold)
        sources.append(source)
        targets.append(target)
        candidates.append(world[index][valid])
        colours.append(cv2.cvtColor(source, cv2.COLOR_BGR2RGB)[valid])
        thresholds.append(threshold)

    all_candidates = np.concatenate(candidates, axis=0)
    robust_bounds = np.percentile(all_candidates, (1, 99), axis=0)
    robust_diagonal = float(np.linalg.norm(robust_bounds[1] - robust_bounds[0]))
    support_radius = max(robust_diagonal * support_radius_fraction, 1e-7)
    trees = [cKDTree(points) for points in candidates]

    filtered: list[np.ndarray] = []
    filtered_colours: list[np.ndarray] = []
    support_histograms: list[dict[str, int]] = []
    for source_index, points in enumerate(candidates):
        support = np.ones(len(points), dtype=np.uint8)
        for target_index, tree in enumerate(trees):
            if source_index == target_index:
                continue
            distances, _ = tree.query(points, k=1, workers=-1)
            support += distances <= support_radius
        keep = support >= minimum_supported_views
        filtered.append(points[keep])
        filtered_colours.append(colours[source_index][keep])
        unique, counts = np.unique(support, return_counts=True)
        support_histograms.append({str(int(key)): int(value) for key, value in zip(unique, counts)})

    before = np.concatenate(candidates, axis=0)
    after = np.concatenate(filtered, axis=0)
    after_colours = np.concatenate(filtered_colours, axis=0)
    cloud_path = object_dir / f"{record['object_id']}_cross_view_supported.ply"
    trimesh.points.PointCloud(after, colors=after_colours).export(cloud_path)
    cloud_preview = object_dir / f"{record['object_id']}_cross_view_filter_review.png"
    _cloud_preview(before, after, cloud_preview, record["object_id"])

    rows = []
    view_metrics = []
    for target_index, view in enumerate(VIEW_ORDER):
        leave_one_out = np.concatenate(
            [points for index, points in enumerate(filtered) if index != target_index], axis=0
        )
        projected = _project(
            leave_one_out, extrinsic[target_index], intrinsic[target_index], size
        )
        target_u8 = np.uint8(targets[target_index]) * 255
        metrics = _metrics(projected, target_u8)
        view_metrics.append({"view": view, **metrics})
        rows.append(
            np.hstack(
                (
                    _label(sources[target_index], f"{view.upper()} SOURCE"),
                    _label(cv2.cvtColor(target_u8, cv2.COLOR_GRAY2BGR), "PHASE 2 MASK", "machine proposal"),
                    _label(cv2.cvtColor(projected, cv2.COLOR_GRAY2BGR), "LEAVE-ONE-VIEW-OUT", "only the other four views"),
                    _label(
                        _agreement(sources[target_index], projected, target_u8),
                        "REPROJECTION AGREEMENT",
                        f"IoU {metrics['iou']:.3f} / green match",
                    ),
                )
            )
        )
    reprojection_preview = object_dir / f"{record['object_id']}_cross_view_reprojection.png"
    if not cv2.imwrite(str(reprojection_preview), np.vstack(rows)):
        raise RuntimeError(f"Could not write {reprojection_preview}")

    mean_iou = float(np.mean([item["iou"] for item in view_metrics]))
    mean_precision = float(np.mean([item["precision"] for item in view_metrics]))
    mean_recall = float(np.mean([item["recall"] for item in view_metrics]))
    retention = len(after) / max(len(before), 1)
    result = {
        "schema_version": "vggt_cross_view_validation_v1",
        "object_id": record["object_id"],
        "dataset": "STL-1",
        "split": record["split"],
        "image_size": size,
        "input_preprocessing": report.get(
            "input_preprocessing", "prepared_v1_full_frame"
        ),
        "filter": {
            "confidence_quantile": confidence_quantile,
            "confidence_thresholds": [round(value, 6) for value in thresholds],
            "support_radius_fraction_of_robust_diagonal": support_radius_fraction,
            "support_radius": round(support_radius, 8),
            "minimum_supported_views": minimum_supported_views,
            "candidate_points": int(len(before)),
            "supported_points": int(len(after)),
            "retention_fraction": round(retention, 6),
            "support_histograms": support_histograms,
        },
        "leave_one_view_out_reprojection": {
            "views": view_metrics,
            "mean_iou": round(mean_iou, 6),
            "minimum_iou": round(min(item["iou"] for item in view_metrics), 6),
            "mean_partial_surface_precision": round(mean_precision, 6),
            "mean_surface_completion_recall": round(mean_recall, 6),
            "interpretation": {
                "precision": "containment of partial cross-view surface evidence inside the target silhouette",
                "recall": "target-silhouette coverage supplied by the other four visible-surface maps",
                "iou": "combined surface-completion score; not an isolated camera-pose score",
            },
        },
        "checks": {
            "all_five_views_scored": len(view_metrics) == 5,
            "supported_cloud_nonempty": len(after) > 0,
            "filter_reduces_points": 0 < len(after) < len(before),
            "no_self_view_in_reprojection": True,
            "cross_view_exit_gate_passed": mean_iou >= 0.80 and min(item["iou"] for item in view_metrics) >= 0.70,
        },
        "artifacts": {
            "supported_cloud": str(cloud_path.relative_to(root)),
            "filter_preview": str(cloud_preview.relative_to(root)),
            "reprojection_preview": str(reprojection_preview.relative_to(root)),
        },
        "decision": "accepted_for_geometry_refinement" if mean_iou >= 0.80 and min(item["iou"] for item in view_metrics) >= 0.70 else "rejected_as_cross_view_reconstruction_geometry",
        "claims": {
            "metric_scale_validated": False,
            "camera_pose_accuracy_validated": False,
            "cross_view_geometry_validated": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "The target view is excluded from each reprojection score, but the masks remain machine proposals.",
            "Each VGGT point map describes visible surface evidence, so leave-one-view-out recall and IoU also measure missing-surface completion rather than camera accuracy alone.",
            "VGGT coordinates are relative and cannot provide millimetres without scale evidence.",
            "Independently rendered product views can violate photogrammetric scene assumptions.",
        ],
    }
    (object_dir / "cross_view_validation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repo_root.resolve()
    run_root = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir
    run_report = json.loads((run_root / "run_report.json").read_text(encoding="utf-8"))
    records = _records(root / "dataset/prepared_v1/manifest.jsonl")
    results = [
        validate_object(
            root,
            run_root / item["object_id"],
            records[item["object_id"]],
            args.confidence_quantile,
            args.support_radius_fraction,
            args.minimum_supported_views,
        )
        for item in run_report["objects"]
    ]
    summary = {
        "schema_version": "vggt_multiring_cross_view_validation_v1",
        "dataset": "STL-1",
        "object_count": len(results),
        "objects": [
            {
                "object_id": item["object_id"],
                "split": item["split"],
                "mean_leave_one_view_out_iou": item["leave_one_view_out_reprojection"]["mean_iou"],
                "minimum_leave_one_view_out_iou": item["leave_one_view_out_reprojection"]["minimum_iou"],
                "mean_partial_surface_precision": item["leave_one_view_out_reprojection"]["mean_partial_surface_precision"],
                "mean_surface_completion_recall": item["leave_one_view_out_reprojection"]["mean_surface_completion_recall"],
                "retention_fraction": item["filter"]["retention_fraction"],
                "decision": item["decision"],
                "artifacts": item["artifacts"],
            }
            for item in results
        ],
        "accepted_object_count": sum(item["decision"] == "accepted_for_geometry_refinement" for item in results),
        "decision": "research_only",
        "claims": {
            "dataset_generalization_validated": False,
            "cross_view_geometry_validated": False,
            "manufacturing_accuracy_validated": False,
        },
    }
    (run_root / "validation_report.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("dataset/phase_runs/v1/vggt_multiring_224_v1"),
    )
    parser.add_argument("--confidence-quantile", type=float, default=0.50)
    parser.add_argument("--support-radius-fraction", type=float, default=0.025)
    parser.add_argument("--minimum-supported-views", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
