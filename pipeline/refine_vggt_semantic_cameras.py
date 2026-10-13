"""Snap VGGT optical axes to STL-1 view semantics and revalidate fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from pipeline.validate_semantic_camera_priors import EXPECTED_FORWARD
from pipeline.validate_vggt_multiring_experiment import (
    VIEW_ORDER,
    _agreement,
    _cloud_preview,
    _label,
    _metrics,
    _project,
    _records,
    _square,
)


def _camera_centres(extrinsic: np.ndarray) -> np.ndarray:
    return np.stack(
        [-(matrix[:, :3].T @ matrix[:, 3]) for matrix in extrinsic], axis=0
    )


def _semantic_snap(
    extrinsic: np.ndarray,
    object_centre: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    predicted = np.stack(
        [matrix[:, :3].T @ np.asarray((0.0, 0.0, 1.0)) for matrix in extrinsic]
    )
    predicted /= np.linalg.norm(predicted, axis=1, keepdims=True)
    expected = np.stack([EXPECTED_FORWARD[view] for view in VIEW_ORDER])
    gauge, alignment_rmsd = Rotation.align_vectors(predicted, expected)
    desired = gauge.apply(expected)
    original_centres = _camera_centres(extrinsic)
    distances = np.linalg.norm(original_centres - object_centre[None, :], axis=1)
    fallback_distance = max(float(np.median(distances)), 1e-3)

    corrected = np.zeros_like(extrinsic)
    roll_adjustments = []
    for index, (matrix, forward) in enumerate(zip(extrinsic, desired)):
        camera_to_world = matrix[:, :3].T
        right = camera_to_world[:, 0]
        right = right - forward * float(np.dot(right, forward))
        norm = float(np.linalg.norm(right))
        if norm < 1e-6:
            helper = np.asarray((0.0, 0.0, 1.0))
            if abs(float(np.dot(helper, forward))) > 0.9:
                helper = np.asarray((0.0, 1.0, 0.0))
            right = np.cross(forward, helper)
            norm = float(np.linalg.norm(right))
        right /= norm
        down = np.cross(forward, right)
        down /= max(float(np.linalg.norm(down)), 1e-9)
        corrected_camera_to_world = np.column_stack((right, down, forward))
        corrected_rotation = corrected_camera_to_world.T
        distance = float(distances[index]) if distances[index] > 1e-3 else fallback_distance
        corrected_centre = object_centre - forward * distance
        corrected[index, :, :3] = corrected_rotation
        corrected[index, :, 3] = -(corrected_rotation @ corrected_centre)
        roll_adjustments.append(
            float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            np.dot(camera_to_world[:, 0], right), -1.0, 1.0
                        )
                    )
                )
            )
        )
    return corrected, {
        "global_gauge_alignment_rmsd": round(float(alignment_rmsd), 6),
        "camera_distance_median": round(float(np.median(distances)), 6),
        "roll_adjustment_deg": [round(value, 6) for value in roll_adjustments],
        "method": "semantic_optical_axis_snap_preserve_predicted_roll_v1",
    }


def _transform_world_points(
    world: np.ndarray,
    original: np.ndarray,
    corrected: np.ndarray,
) -> np.ndarray:
    transformed = np.empty_like(world)
    for index in range(len(world)):
        original_rotation = original[index, :, :3]
        camera = world[index] @ original_rotation.T + original[index, :, 3]
        corrected_rotation = corrected[index, :, :3]
        transformed[index] = (
            camera - corrected[index, :, 3]
        ) @ corrected_rotation
    return transformed


def refine_object(
    root: Path,
    record: dict[str, Any],
    baseline_dir: Path,
    output_dir: Path,
    confidence_quantile: float,
    support_radius_fraction: float,
    minimum_supported_views: int,
) -> dict[str, Any]:
    baseline_validation = json.loads(
        (baseline_dir / "cross_view_validation.json").read_text(encoding="utf-8")
    )
    run_report = json.loads(
        (baseline_dir / "vggt_report.json").read_text(encoding="utf-8")
    )
    cameras = np.load(baseline_dir / "vggt_cameras.npz")
    dense = np.load(baseline_dir / "vggt_dense_evidence.npz")
    original_extrinsic = cameras["extrinsic"][0]
    intrinsic = cameras["intrinsic"][0]
    original_world = dense["world_points"][0]
    confidence = dense["world_points_conf"][0]
    size = int(run_report["image_size"])

    sources = []
    targets = []
    preliminary = []
    thresholds = []
    for index, view in enumerate(VIEW_ORDER):
        source = _square(root / record["views"][view]["image_path"], size)
        target = _square(
            root / record["views"][view]["jewelry_mask_path"], size, grayscale=True
        ) > 0
        finite = np.isfinite(original_world[index]).all(axis=-1) & np.isfinite(
            confidence[index]
        )
        values = confidence[index][target & finite]
        threshold = float(np.quantile(values, confidence_quantile))
        valid = target & finite & (confidence[index] >= threshold)
        sources.append(source)
        targets.append(target)
        preliminary.append(valid)
        thresholds.append(threshold)

    object_points = np.concatenate(
        [original_world[index][valid] for index, valid in enumerate(preliminary)],
        axis=0,
    )
    object_centre = np.median(object_points, axis=0)
    corrected_extrinsic, snap = _semantic_snap(original_extrinsic, object_centre)
    corrected_world = _transform_world_points(
        original_world, original_extrinsic, corrected_extrinsic
    )
    candidates = [
        corrected_world[index][valid]
        for index, valid in enumerate(preliminary)
    ]
    colours = [
        cv2.cvtColor(sources[index], cv2.COLOR_BGR2RGB)[valid]
        for index, valid in enumerate(preliminary)
    ]
    all_candidates = np.concatenate(candidates, axis=0)
    bounds = np.percentile(all_candidates, (1, 99), axis=0)
    diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
    support_radius = max(diagonal * support_radius_fraction, 1e-7)
    trees = [cKDTree(points) for points in candidates]

    filtered = []
    filtered_colours = []
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

    output_dir.mkdir(parents=True, exist_ok=True)
    corrected_cameras_path = output_dir / "semantic_corrected_cameras.npz"
    np.savez_compressed(
        corrected_cameras_path,
        extrinsic=corrected_extrinsic[None, ...],
        intrinsic=intrinsic[None, ...],
    )
    fused_points = np.concatenate(filtered, axis=0)
    fused_colours = np.concatenate(filtered_colours, axis=0)
    cloud_path = output_dir / f"{record['object_id']}_semantic_corrected_points.ply"
    trimesh.points.PointCloud(fused_points, colors=fused_colours).export(cloud_path)
    cloud_preview = output_dir / f"{record['object_id']}_semantic_corrected_cloud.png"
    _cloud_preview(all_candidates, fused_points, cloud_preview, record["object_id"])

    rows = []
    metrics = []
    for target_index, view in enumerate(VIEW_ORDER):
        leave_one_out = np.concatenate(
            [points for index, points in enumerate(filtered) if index != target_index],
            axis=0,
        )
        projection = _project(
            leave_one_out,
            corrected_extrinsic[target_index],
            intrinsic[target_index],
            size,
        )
        target_u8 = np.uint8(targets[target_index]) * 255
        values = _metrics(projection, target_u8)
        metrics.append({"view": view, **values})
        rows.append(
            np.hstack(
                (
                    _label(sources[target_index], f"{view.upper()} SOURCE"),
                    _label(cv2.cvtColor(target_u8, cv2.COLOR_GRAY2BGR), "PHASE 2 MASK"),
                    _label(
                        cv2.cvtColor(projection, cv2.COLOR_GRAY2BGR),
                        "SEMANTIC-CORRECTED LOO",
                        "other four views only",
                    ),
                    _label(
                        _agreement(sources[target_index], projection, target_u8),
                        "AGREEMENT",
                        f"IoU {values['iou']:.3f} / precision {values['precision']:.3f}",
                    ),
                )
            )
        )
    preview = output_dir / f"{record['object_id']}_semantic_camera_reprojection.png"
    if not cv2.imwrite(str(preview), np.vstack(rows)):
        raise RuntimeError(f"Could not write {preview}")

    baseline_views = baseline_validation["leave_one_view_out_reprojection"]["views"]
    baseline_mean_iou = float(np.mean([item["iou"] for item in baseline_views]))
    baseline_mean_precision = float(
        np.mean([item["precision"] for item in baseline_views])
    )
    corrected_mean_iou = float(np.mean([item["iou"] for item in metrics]))
    corrected_mean_precision = float(np.mean([item["precision"] for item in metrics]))
    result = {
        "schema_version": "vggt_semantic_camera_refinement_v1",
        "object_id": record["object_id"],
        "split": record["split"],
        "snap": snap,
        "filter": {
            "confidence_quantile": confidence_quantile,
            "confidence_thresholds": [round(value, 6) for value in thresholds],
            "support_radius_fraction": support_radius_fraction,
            "support_radius": round(support_radius, 8),
            "minimum_supported_views": minimum_supported_views,
            "candidate_points": int(len(all_candidates)),
            "supported_points": int(len(fused_points)),
        },
        "comparison": {
            "baseline_mean_iou": round(baseline_mean_iou, 6),
            "corrected_mean_iou": round(corrected_mean_iou, 6),
            "iou_delta": round(corrected_mean_iou - baseline_mean_iou, 6),
            "baseline_mean_precision": round(baseline_mean_precision, 6),
            "corrected_mean_precision": round(corrected_mean_precision, 6),
            "precision_delta": round(
                corrected_mean_precision - baseline_mean_precision, 6
            ),
            "corrected_views": metrics,
        },
        "checks": {
            "all_five_views_scored": len(metrics) == 5,
            "corrected_cloud_nonempty": len(fused_points) > 0,
            "iou_improved": corrected_mean_iou > baseline_mean_iou,
            "precision_improved": corrected_mean_precision > baseline_mean_precision,
            "geometry_exit_gate_passed": corrected_mean_iou >= 0.80,
        },
        "decision": (
            "accepted_as_camera_initialization_candidate"
            if corrected_mean_iou > baseline_mean_iou
            and corrected_mean_precision > baseline_mean_precision
            else "semantic_camera_snap_rejected"
        ),
        "artifacts": {
            "corrected_cameras": str(corrected_cameras_path.relative_to(root)),
            "corrected_cloud": str(cloud_path.relative_to(root)),
            "cloud_preview": str(cloud_preview.relative_to(root)),
            "reprojection_preview": str(preview.relative_to(root)),
        },
        "claims": {
            "camera_calibration_validated": False,
            "cross_view_geometry_validated": False,
            "manufacturing_accuracy_validated": False,
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def run(root: Path, baseline_root: Path, output_root: Path) -> dict[str, Any]:
    run_report = json.loads(
        (baseline_root / "run_report.json").read_text(encoding="utf-8")
    )
    records = _records(root / "dataset/prepared_v1/manifest.jsonl")
    results = [
        refine_object(
            root,
            records[item["object_id"]],
            baseline_root / item["object_id"],
            output_root / item["object_id"],
            0.50,
            0.025,
            2,
        )
        for item in run_report["objects"]
    ]
    summary = {
        "schema_version": "vggt_semantic_camera_refinement_multiring_v1",
        "dataset": "STL-1",
        "object_count": len(results),
        "objects": [
            {
                "object_id": item["object_id"],
                "split": item["split"],
                "comparison": item["comparison"],
                "decision": item["decision"],
                "artifacts": item["artifacts"],
            }
            for item in results
        ],
        "cohort": {
            "mean_baseline_iou": round(float(np.mean([item["comparison"]["baseline_mean_iou"] for item in results])), 6),
            "mean_corrected_iou": round(float(np.mean([item["comparison"]["corrected_mean_iou"] for item in results])), 6),
            "mean_iou_delta": round(float(np.mean([item["comparison"]["iou_delta"] for item in results])), 6),
            "mean_precision_delta": round(float(np.mean([item["comparison"]["precision_delta"] for item in results])), 6),
            "accepted_initialization_count": sum(item["decision"] == "accepted_as_camera_initialization_candidate" for item in results),
            "geometry_exit_gate_count": sum(item["checks"]["geometry_exit_gate_passed"] for item in results),
        },
        "decision": "research_only",
        "claims": {
            "camera_calibration_validated": False,
            "cross_view_geometry_validated": False,
            "manufacturing_accuracy_validated": False,
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "report.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("dataset/phase_runs/v1/vggt_multiring_224_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/phase_runs/v1/vggt_semantic_camera_refinement_v1"),
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    baseline = args.baseline_root if args.baseline_root.is_absolute() else root / args.baseline_root
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    print(json.dumps(run(root, baseline, output), indent=2))


if __name__ == "__main__":
    main()
