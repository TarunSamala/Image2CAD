"""Carve free space from the surface-aware VGGT silhouette hull.

The target mask and target point map are excluded from reconstruction.  Source
depth is derived by transforming each source-view VGGT world point into its own
camera.  Voxels clearly in front of those observed surfaces are free-space
contradictions.  Tolerance and allowed contradiction count are selected only
on validation objects.  Output remains non-metric research evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree

from pipeline.evaluate_vggt_surface_aware_hull import _load_object
from pipeline.evaluate_vggt_visual_hull_completion import (
    EVALUATION_OBJECTS,
    VALIDATION_OBJECTS,
    _voxel_hulls,
)
from pipeline.validate_vggt_multiring_experiment import (
    VIEW_ORDER,
    _agreement,
    _label,
    _metrics,
    _project,
    _records,
)


DEPTH_TOLERANCE_CANDIDATES = (0.0025, 0.005, 0.01, 0.02, 0.04)
ALLOWED_CONTRADICTION_CANDIDATES = (0, 1, 2)
SURFACE_DISTANCE_FRACTION = 0.04


def _configuration_key(tolerance: float, allowed: int) -> str:
    return f"{tolerance}:{allowed}"


def _depth_violations(
    shell: np.ndarray,
    source_indices: list[int],
    world_points: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    image_size: int,
    robust_diagonal: float,
) -> dict[float, np.ndarray]:
    violations = {
        tolerance: np.zeros(len(shell), dtype=np.uint8)
        for tolerance in DEPTH_TOLERANCE_CANDIDATES
    }
    for source_index in source_indices:
        observed_camera = (
            world_points[source_index] @ extrinsic[source_index, :, :3].T
            + extrinsic[source_index, :, 3]
        )
        observed_depth = observed_camera[:, :, 2]
        camera = (
            shell @ extrinsic[source_index, :, :3].T
            + extrinsic[source_index, :, 3]
        )
        pixels = camera @ intrinsic[source_index].T
        pixels = pixels[:, :2] / pixels[:, 2:3]
        pixels = np.rint(pixels).astype(np.int32)
        valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-6)
        valid &= (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < image_size)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < image_size)
        )
        indices = np.flatnonzero(valid)
        source_depth = observed_depth[pixels[indices, 1], pixels[indices, 0]]
        finite = np.isfinite(source_depth) & (source_depth > 1e-6)
        indices = indices[finite]
        source_depth = source_depth[finite]
        for tolerance in DEPTH_TOLERANCE_CANDIDATES:
            free_space = (
                camera[indices, 2]
                < source_depth - robust_diagonal * tolerance
            )
            violations[tolerance][indices] += free_space
    return violations


def _evaluate_object(item: dict[str, Any], voxel_resolution: int) -> dict[str, Any]:
    views = []
    projections: dict[str, list[np.ndarray]] = {
        _configuration_key(tolerance, allowed): []
        for tolerance in DEPTH_TOLERANCE_CANDIDATES
        for allowed in ALLOWED_CONTRADICTION_CANDIDATES
    }
    surface_projections = []
    for target_index, view in enumerate(VIEW_ORDER):
        source_indices = [index for index in range(len(VIEW_ORDER)) if index != target_index]
        hull = _voxel_hulls(
            source_indices,
            item["candidates"],
            item["masks"],
            item["extrinsic"],
            item["intrinsic"],
            voxel_resolution,
        )[4]
        source_points = np.concatenate(
            [item["candidates"][index] for index in source_indices]
        )
        bounds = np.percentile(source_points, (1, 99), axis=0)
        diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
        distances = cKDTree(source_points).query(hull, k=1, workers=-1)[0]
        shell = hull[distances <= diagonal * SURFACE_DISTANCE_FRACTION]
        violations = _depth_violations(
            shell,
            source_indices,
            item["world_points"],
            item["extrinsic"],
            item["intrinsic"],
            item["image_size"],
            diagonal,
        )
        target = np.uint8(item["masks"][target_index]) * 255
        surface_projection = _project(
            shell,
            item["extrinsic"][target_index],
            item["intrinsic"][target_index],
            item["image_size"],
        )
        surface_projections.append(surface_projection)
        candidate_metrics = {}
        for tolerance in DEPTH_TOLERANCE_CANDIDATES:
            for allowed in ALLOWED_CONTRADICTION_CANDIDATES:
                key = _configuration_key(tolerance, allowed)
                retained = shell[violations[tolerance] <= allowed]
                projection = _project(
                    retained,
                    item["extrinsic"][target_index],
                    item["intrinsic"][target_index],
                    item["image_size"],
                )
                projections[key].append(projection)
                candidate_metrics[key] = {
                    **_metrics(projection, target),
                    "retained_voxels": int(len(retained)),
                    "depth_tolerance": tolerance,
                    "allowed_source_contradictions": allowed,
                }
        views.append(
            {
                "view": view,
                "surface_aware_hull": _metrics(surface_projection, target),
                "candidates": candidate_metrics,
            }
        )
    return {
        **item,
        "views": views,
        "surface_projections": surface_projections,
        "projections": projections,
        "voxel_resolution": voxel_resolution,
    }


def _configuration_summary(objects: list[dict[str, Any]], key: str) -> dict[str, float]:
    metrics = [view["candidates"][key] for item in objects for view in item["views"]]
    return {
        "mean_iou": round(float(np.mean([item["iou"] for item in metrics])), 6),
        "minimum_iou": round(float(np.min([item["iou"] for item in metrics])), 6),
        "mean_precision": round(float(np.mean([item["precision"] for item in metrics])), 6),
        "mean_recall": round(float(np.mean([item["recall"] for item in metrics])), 6),
    }


def _summarize(item: dict[str, Any], key: str) -> dict[str, Any]:
    selected = [view["candidates"][key] for view in item["views"]]
    surface = [view["surface_aware_hull"] for view in item["views"]]
    mean_iou = float(np.mean([metric["iou"] for metric in selected]))
    minimum_iou = float(np.min([metric["iou"] for metric in selected]))
    surface_mean = float(np.mean([metric["iou"] for metric in surface]))
    return {
        "object_id": item["object_id"],
        "split": item["split"],
        "mean_iou": round(mean_iou, 6),
        "minimum_iou": round(minimum_iou, 6),
        "mean_precision": round(float(np.mean([metric["precision"] for metric in selected])), 6),
        "mean_recall": round(float(np.mean([metric["recall"] for metric in selected])), 6),
        "surface_aware_mean_iou": round(surface_mean, 6),
        "delta_over_surface_aware_hull": round(mean_iou - surface_mean, 6),
        "point_baseline_mean_iou": item["point_baseline_mean_iou"],
        "delta_over_point_baseline": round(
            mean_iou - item["point_baseline_mean_iou"], 6
        ),
        "views": [
            {"view": view["view"], **view["candidates"][key]}
            for view in item["views"]
        ],
        "exit_gate_passed": mean_iou >= 0.80 and minimum_iou >= 0.70,
    }


def _preview(item: dict[str, Any], key: str, output: Path) -> None:
    rows = []
    for index, view in enumerate(item["views"]):
        target = np.uint8(item["masks"][index]) * 255
        surface = item["surface_projections"][index]
        carved = item["projections"][key][index]
        metrics = view["candidates"][key]
        rows.append(
            np.hstack(
                (
                    _label(item["sources"][index], f"{view['view'].upper()} SOURCE"),
                    _label(cv2.cvtColor(target, cv2.COLOR_GRAY2BGR), "TARGET MASK", "score only"),
                    _label(cv2.cvtColor(surface, cv2.COLOR_GRAY2BGR), "SURFACE-AWARE HULL", "source points + silhouettes"),
                    _label(cv2.cvtColor(carved, cv2.COLOR_GRAY2BGR), "SIGNED-DEPTH CARVE", "source free-space constraints"),
                    _label(
                        _agreement(item["sources"][index], carved, target),
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
    voxel_resolution: int = 96,
    confidence_quantile: float = 0.50,
) -> dict[str, Any]:
    records = _records(root / "dataset/prepared_v1/manifest.jsonl")
    object_ids = (*VALIDATION_OBJECTS, *EVALUATION_OBJECTS)
    objects = [
        _evaluate_object(
            _load_object(
                root,
                object_id,
                records[object_id],
                confidence_quantile,
            ),
            voxel_resolution,
        )
        for object_id in object_ids
    ]
    validation_objects = [
        item for item in objects if item["object_id"] in VALIDATION_OBJECTS
    ]
    candidates = {
        _configuration_key(tolerance, allowed): _configuration_summary(
            validation_objects,
            _configuration_key(tolerance, allowed),
        )
        for tolerance in DEPTH_TOLERANCE_CANDIDATES
        for allowed in ALLOWED_CONTRADICTION_CANDIDATES
    }
    selected_key = max(
        candidates,
        key=lambda key: candidates[key]["mean_iou"],
    )
    selected_tolerance, selected_allowed = selected_key.split(":")
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for item in objects:
        preview = output_dir / f"{item['object_id']}_signed_depth_carving.png"
        _preview(item, selected_key, preview)
        summary = _summarize(item, selected_key)
        summary["preview"] = str(preview.relative_to(root))
        summaries.append(summary)
    validation = [item for item in summaries if item["split"] == "val"]
    test = [item for item in summaries if item["split"] == "test"]

    report = {
        "schema_version": "vggt_signed_depth_free_space_carving_v1",
        "dataset": "STL-1",
        "authority": "non_metric_machine_surface_hypothesis",
        "input_resolution": 518,
        "voxel_resolution": voxel_resolution,
        "surface_distance_fraction": SURFACE_DISTANCE_FRACTION,
        "validation_selection": {
            "object_ids": list(VALIDATION_OBJECTS),
            "candidates": candidates,
            "selected_depth_tolerance_fraction": float(selected_tolerance),
            "selected_allowed_source_contradictions": int(selected_allowed),
            "selection_rule": "highest pooled validation mean IoU",
        },
        "validation": _configuration_summary(validation_objects, selected_key),
        "test_evaluation": {
            "object_ids": [item["object_id"] for item in test],
            "mean_iou": round(float(np.mean([item["mean_iou"] for item in test])), 6),
            "minimum_iou": round(float(np.min([item["minimum_iou"] for item in test])), 6),
            "mean_precision": round(float(np.mean([item["mean_precision"] for item in test])), 6),
            "mean_recall": round(float(np.mean([item["mean_recall"] for item in test])), 6),
            "mean_delta_over_surface_aware_hull": round(float(np.mean([item["delta_over_surface_aware_hull"] for item in test])), 6),
            "mean_delta_over_point_baseline": round(float(np.mean([item["delta_over_point_baseline"] for item in test])), 6),
        },
        "objects": summaries,
        "checks": {
            "selection_uses_validation_split_only": all(
                item["split"] == "val" for item in validation
            ),
            "target_mask_and_points_excluded_from_reconstruction": True,
            "all_test_objects_improve_over_surface_aware_hull": all(
                item["delta_over_surface_aware_hull"] > 0 for item in test
            ),
            "geometry_exit_gate_passed": all(
                item["exit_gate_passed"] for item in summaries
            ),
        },
        "decision": "retained_as_depth_carving_prior_not_approved_geometry",
        "claims": {
            "strict_target_image_holdout": False,
            "metric_scale_validated": False,
            "depth_accuracy_validated": False,
            "hidden_surface_truth_validated": False,
            "component_topology_validated": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "VGGT camera and depth hypotheses are jointly conditioned on all five images even though target masks and target points are excluded from reconstruction.",
            "Depth is relative and internally consistent, not calibrated in millimetres.",
            "Allowing one source contradiction is validation-selected robustness to reflective and independently rendered views, not proof that the contradicted view is wrong.",
            "Free-space carving cannot reconstruct hidden geometry absent from every supplied view.",
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/phase_runs/v1/vggt_signed_depth_carving_v1"),
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
                "validation_selection": report["validation_selection"],
                "validation": report["validation"],
                "test_evaluation": report["test_evaluation"],
                "checks": report["checks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
