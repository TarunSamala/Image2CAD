"""Evaluate silhouette visual-hull completion with VGGT camera hypotheses.

The experiment excludes the target silhouette while constructing each hull.
It tests whether intersecting the four remaining silhouettes can complete more
of the held-out view than the retained VGGT visible-surface point maps.  The
result is non-metric silhouette evidence: it cannot recover concavities,
relief, cavities, component identity, or manufacturing geometry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline.validate_vggt_multiring_experiment import (
    VIEW_ORDER,
    _agreement,
    _label,
    _metrics,
    _project,
    _records,
    _square,
)


VALIDATION_OBJECTS = ("ring_004", "ring_012", "ring_020")
EVALUATION_OBJECTS = ("ring_001", "ring_007", "ring_015", "ring_023")
SUPPORT_CANDIDATES = (2, 3, 4)


def _paths(root: Path, object_id: str) -> tuple[Path, Path]:
    if object_id in VALIDATION_OBJECTS:
        run_dir = root / "dataset/phase_runs/v1/vggt_resolution_518_validation_v1"
    elif object_id == "ring_015":
        run_dir = root / "dataset/phase_runs/v1/vggt_resolution_518_test_extension_v1"
    else:
        run_dir = root / "dataset/phase_runs/v1/vggt_resolution_518_v1"
    return run_dir / object_id, run_dir / "validation_report.json"


def _source_candidates(
    world: np.ndarray,
    confidence: np.ndarray,
    masks: list[np.ndarray],
    confidence_quantile: float,
) -> list[np.ndarray]:
    candidates = []
    for index in range(len(VIEW_ORDER)):
        finite = np.isfinite(world[index]).all(axis=-1) & np.isfinite(confidence[index])
        values = confidence[index][masks[index] & finite]
        threshold = float(np.quantile(values, confidence_quantile))
        valid = masks[index] & finite & (confidence[index] >= threshold)
        candidates.append(world[index][valid])
    return candidates


def _voxel_hulls(
    source_indices: list[int],
    candidates: list[np.ndarray],
    masks: list[np.ndarray],
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    resolution: int,
) -> dict[int, np.ndarray]:
    source_points = np.concatenate([candidates[index] for index in source_indices])
    lower, upper = np.percentile(source_points, (1, 99), axis=0)
    axes = [
        np.linspace(lower[axis], upper[axis], resolution, dtype=np.float32)
        for axis in range(3)
    ]
    xy_x, xy_y = np.meshgrid(axes[0], axes[1], indexing="xy")
    slabs: dict[int, list[np.ndarray]] = {value: [] for value in SUPPORT_CANDIDATES}
    for z_value in axes[2]:
        slab = np.column_stack(
            (
                xy_x.ravel(),
                xy_y.ravel(),
                np.full(xy_x.size, z_value, dtype=np.float32),
            )
        )
        support = np.zeros(len(slab), dtype=np.uint8)
        for source_index in source_indices:
            camera = slab @ extrinsic[source_index, :, :3].T + extrinsic[source_index, :, 3]
            valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-6)
            pixels = camera @ intrinsic[source_index].T
            pixels = pixels[:, :2] / pixels[:, 2:3]
            pixels = np.rint(pixels).astype(np.int32)
            valid &= (
                (pixels[:, 0] >= 0)
                & (pixels[:, 0] < masks[source_index].shape[1])
                & (pixels[:, 1] >= 0)
                & (pixels[:, 1] < masks[source_index].shape[0])
            )
            indices = np.flatnonzero(valid)
            support[indices] += masks[source_index][
                pixels[indices, 1], pixels[indices, 0]
            ]
        for minimum_support in SUPPORT_CANDIDATES:
            retained = slab[support >= minimum_support]
            if len(retained):
                slabs[minimum_support].append(retained)
    return {
        support: np.concatenate(parts) if parts else np.empty((0, 3), np.float32)
        for support, parts in slabs.items()
    }


def _evaluate_object(
    root: Path,
    object_id: str,
    record: dict[str, Any],
    resolution: int,
    confidence_quantile: float,
) -> dict[str, Any]:
    object_dir, baseline_report_path = _paths(root, object_id)
    report = json.loads((object_dir / "vggt_report.json").read_text(encoding="utf-8"))
    cameras = np.load(object_dir / "vggt_cameras.npz")
    dense = np.load(object_dir / "vggt_dense_evidence.npz")
    extrinsic = cameras["extrinsic"][0]
    intrinsic = cameras["intrinsic"][0]
    world = dense["world_points"][0]
    confidence = dense["world_points_conf"][0]
    image_size = int(report["image_size"])
    sources = [
        _square(root / record["views"][view]["image_path"], image_size)
        for view in VIEW_ORDER
    ]
    masks = [
        _square(
            root / record["views"][view]["jewelry_mask_path"],
            image_size,
            grayscale=True,
        )
        > 0
        for view in VIEW_ORDER
    ]
    candidates = _source_candidates(world, confidence, masks, confidence_quantile)
    views = []
    projections: dict[int, list[np.ndarray]] = {
        support: [] for support in SUPPORT_CANDIDATES
    }
    for target_index, view in enumerate(VIEW_ORDER):
        source_indices = [index for index in range(len(VIEW_ORDER)) if index != target_index]
        hulls = _voxel_hulls(
            source_indices,
            candidates,
            masks,
            extrinsic,
            intrinsic,
            resolution,
        )
        candidate_metrics = {}
        for support, points in hulls.items():
            projection = _project(
                points,
                extrinsic[target_index],
                intrinsic[target_index],
                image_size,
            )
            projections[support].append(projection)
            candidate_metrics[str(support)] = {
                **_metrics(projection, np.uint8(masks[target_index]) * 255),
                "occupied_voxels": int(len(points)),
            }
        views.append({"view": view, "candidates": candidate_metrics})

    baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    baseline_object = next(
        item for item in baseline_report["objects"] if item["object_id"] == object_id
    )
    return {
        "object_id": object_id,
        "split": record["split"],
        "image_size": image_size,
        "voxel_resolution": resolution,
        "views": views,
        "sources": sources,
        "masks": masks,
        "projections": projections,
        "baseline": {
            "mean_iou": baseline_object["mean_leave_one_view_out_iou"],
            "minimum_iou": baseline_object["minimum_leave_one_view_out_iou"],
            "mean_precision": baseline_object["mean_partial_surface_precision"],
            "mean_recall": baseline_object["mean_surface_completion_recall"],
        },
    }


def _candidate_summary(objects: list[dict[str, Any]], support: int) -> dict[str, float]:
    metrics = [
        view["candidates"][str(support)]
        for item in objects
        for view in item["views"]
    ]
    return {
        "mean_iou": round(float(np.mean([item["iou"] for item in metrics])), 6),
        "minimum_iou": round(float(np.min([item["iou"] for item in metrics])), 6),
        "mean_precision": round(float(np.mean([item["precision"] for item in metrics])), 6),
        "mean_recall": round(float(np.mean([item["recall"] for item in metrics])), 6),
    }


def _object_summary(item: dict[str, Any], support: int) -> dict[str, Any]:
    metrics = [view["candidates"][str(support)] for view in item["views"]]
    mean_iou = float(np.mean([metric["iou"] for metric in metrics]))
    minimum_iou = float(np.min([metric["iou"] for metric in metrics]))
    baseline = item["baseline"]
    return {
        "object_id": item["object_id"],
        "split": item["split"],
        "mean_iou": round(mean_iou, 6),
        "minimum_iou": round(minimum_iou, 6),
        "mean_precision": round(float(np.mean([metric["precision"] for metric in metrics])), 6),
        "mean_recall": round(float(np.mean([metric["recall"] for metric in metrics])), 6),
        "baseline_point_mean_iou": baseline["mean_iou"],
        "mean_iou_delta": round(mean_iou - baseline["mean_iou"], 6),
        "views": [
            {"view": view["view"], **view["candidates"][str(support)]}
            for view in item["views"]
        ],
        "exit_gate_passed": mean_iou >= 0.80 and minimum_iou >= 0.70,
    }


def _preview(
    item: dict[str, Any],
    selected_support: int,
    output: Path,
) -> None:
    rows = []
    for index, view in enumerate(item["views"]):
        target = np.uint8(item["masks"][index]) * 255
        prediction = item["projections"][selected_support][index]
        metrics = view["candidates"][str(selected_support)]
        rows.append(
            np.hstack(
                (
                    _label(item["sources"][index], f"{view['view'].upper()} SOURCE"),
                    _label(cv2.cvtColor(target, cv2.COLOR_GRAY2BGR), "TARGET MASK", "score only"),
                    _label(
                        cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR),
                        "4-VIEW VISUAL HULL",
                        "target silhouette excluded",
                    ),
                    _label(
                        _agreement(item["sources"][index], prediction, target),
                        "AGREEMENT",
                        f"IoU {metrics['iou']:.3f} / green match",
                    ),
                )
            )
        )
    if not cv2.imwrite(str(output), np.vstack(rows)):
        raise RuntimeError(f"Could not write {output}")


def run(
    root: Path,
    output_dir: Path,
    resolution: int = 96,
    confidence_quantile: float = 0.50,
) -> dict[str, Any]:
    records = _records(root / "dataset/prepared_v1/manifest.jsonl")
    object_ids = (*VALIDATION_OBJECTS, *EVALUATION_OBJECTS)
    objects = [
        _evaluate_object(
            root,
            object_id,
            records[object_id],
            resolution,
            confidence_quantile,
        )
        for object_id in object_ids
    ]
    validation = [item for item in objects if item["object_id"] in VALIDATION_OBJECTS]
    candidates = {
        str(support): _candidate_summary(validation, support)
        for support in SUPPORT_CANDIDATES
    }
    selected_support = max(
        SUPPORT_CANDIDATES,
        key=lambda support: (candidates[str(support)]["mean_iou"], support),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for item in objects:
        preview = output_dir / f"{item['object_id']}_visual_hull_completion.png"
        _preview(item, selected_support, preview)
        summary = _object_summary(item, selected_support)
        summary["preview"] = str(preview.relative_to(root))
        summaries.append(summary)

    validation_summaries = [
        item for item in summaries if item["object_id"] in VALIDATION_OBJECTS
    ]
    test_summaries = [item for item in summaries if item["split"] == "test"]
    report = {
        "schema_version": "vggt_silhouette_visual_hull_completion_v1",
        "dataset": "STL-1",
        "authority": "non_metric_machine_silhouette_hypothesis",
        "input_resolution": 518,
        "voxel_resolution": resolution,
        "validation_selection": {
            "object_ids": list(VALIDATION_OBJECTS),
            "support_candidates": candidates,
            "selected_minimum_source_views": selected_support,
            "selection_rule": "highest validation mean IoU; target silhouette excluded from hull construction",
        },
        "validation": _candidate_summary(validation, selected_support),
        "test_evaluation": {
            "object_ids": [item["object_id"] for item in test_summaries],
            "mean_iou": round(float(np.mean([item["mean_iou"] for item in test_summaries])), 6),
            "minimum_iou": round(float(np.min([item["minimum_iou"] for item in test_summaries])), 6),
            "mean_precision": round(float(np.mean([item["mean_precision"] for item in test_summaries])), 6),
            "mean_recall": round(float(np.mean([item["mean_recall"] for item in test_summaries])), 6),
            "mean_iou_delta_over_point_baseline": round(
                float(np.mean([item["mean_iou_delta"] for item in test_summaries])), 6
            ),
        },
        "objects": summaries,
        "checks": {
            "selection_uses_validation_split_only": all(
                item["split"] == "val" for item in validation_summaries
            ),
            "target_silhouette_excluded_from_each_hull": True,
            "all_objects_improve_over_point_baseline": all(
                item["mean_iou_delta"] > 0 for item in summaries
            ),
            "geometry_exit_gate_passed": all(
                item["exit_gate_passed"] for item in summaries
            ),
        },
        "decision": "retained_as_silhouette_completion_prior_not_approved_geometry",
        "claims": {
            "metric_scale_validated": False,
            "surface_detail_reconstructed": False,
            "hidden_concavities_reconstructed": False,
            "component_topology_validated": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "VGGT cameras were inferred jointly from all five images; target pixels can influence the camera hypothesis even though the target silhouette and point map are excluded from hull construction.",
            "A visual hull reproduces outer silhouettes but cannot recover concavities, undercuts, cavities, relief, engraving, stone seats or component boundaries.",
            "STL-1 masks are machine proposals and coordinates remain non-metric.",
            "The test identities were observed during earlier exploratory development and are not a final unbiased external generalization estimate.",
        ],
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/phase_runs/v1/vggt_visual_hull_completion_v1"),
    )
    parser.add_argument("--voxel-resolution", type=int, default=96)
    parser.add_argument("--confidence-quantile", type=float, default=0.50)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    report = run(root, output_dir, args.voxel_resolution, args.confidence_quantile)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "selected_minimum_source_views": report["validation_selection"]["selected_minimum_source_views"],
                "validation": report["validation"],
                "test_evaluation": report["test_evaluation"],
                "checks": report["checks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
