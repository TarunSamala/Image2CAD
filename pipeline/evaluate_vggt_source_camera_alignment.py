"""Refine source-camera image-plane alignment before signed-depth carving.

For each held-out target view, the remaining source cameras are adjusted with a
small image-plane similarity transform.  Each adjustment aligns points from the
other source views to that source view's silhouette.  The target mask and target
point map are excluded from alignment and reconstruction.  Search profile is
selected only on validation objects, then frozen for the test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree

from pipeline.evaluate_vggt_signed_depth_carving import (
    SURFACE_DISTANCE_FRACTION,
    _depth_violations,
)
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


DEPTH_TOLERANCE_FRACTION = 0.0025
ALLOWED_SOURCE_CONTRADICTIONS = 1
ALIGNMENT_PROFILES = {
    "identity": {"scales": (1.0,), "offsets": (0,)},
    "translation_4px": {"scales": (1.0,), "offsets": (-4, 0, 4)},
    "similarity_4px": {"scales": (0.98, 1.0, 1.02), "offsets": (-4, 0, 4)},
    "similarity_8px": {"scales": (0.96, 1.0, 1.04), "offsets": (-8, 0, 8)},
}


def _image_transform(size: int, scale: float, tx: int, ty: int) -> np.ndarray:
    centre = (size - 1) / 2.0
    return np.asarray(
        (
            (scale, 0.0, (1.0 - scale) * centre + tx),
            (0.0, scale, (1.0 - scale) * centre + ty),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )


def _fit_image_transform(
    projection: np.ndarray,
    source_mask: np.ndarray,
    profile: dict[str, tuple[float, ...] | tuple[int, ...]],
) -> tuple[np.ndarray, dict[str, Any]]:
    size = source_mask.shape[0]
    target = np.uint8(source_mask) * 255
    baseline = _metrics(projection, target)
    candidates = []
    for scale in profile["scales"]:
        for tx in profile["offsets"]:
            for ty in profile["offsets"]:
                transform = _image_transform(size, float(scale), int(tx), int(ty))
                warped = cv2.warpAffine(
                    projection,
                    transform[:2],
                    (size, size),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                metric = _metrics(warped, target)
                adjustment = abs(float(np.log(scale))) + (
                    abs(int(tx)) + abs(int(ty))
                ) / size
                candidates.append((metric["iou"], -adjustment, transform, metric, scale, tx, ty))
    _, _, transform, metric, scale, tx, ty = max(
        candidates, key=lambda item: (item[0], item[1])
    )
    return transform, {
        "baseline_iou": baseline["iou"],
        "aligned_iou": metric["iou"],
        "iou_delta": round(metric["iou"] - baseline["iou"], 6),
        "scale": float(scale),
        "translation_x_px": int(tx),
        "translation_y_px": int(ty),
    }


def _refine_source_intrinsics(
    item: dict[str, Any],
    target_index: int,
    profile: dict[str, tuple[float, ...] | tuple[int, ...]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    source_indices = [index for index in range(len(VIEW_ORDER)) if index != target_index]
    corrected = item["intrinsic"].copy()
    records = []
    for source_index in source_indices:
        cross_points = np.concatenate(
            [
                item["candidates"][index]
                for index in source_indices
                if index != source_index
            ]
        )
        projection = _project(
            cross_points,
            item["extrinsic"][source_index],
            item["intrinsic"][source_index],
            item["image_size"],
        )
        transform, result = _fit_image_transform(
            projection, item["masks"][source_index], profile
        )
        corrected[source_index] = transform @ item["intrinsic"][source_index]
        records.append(
            {
                "view": VIEW_ORDER[source_index],
                **result,
            }
        )
    return corrected, records


def _signed_depth_voxels(
    item: dict[str, Any],
    target_index: int,
    intrinsic: np.ndarray,
    voxel_resolution: int,
) -> np.ndarray:
    source_indices = [index for index in range(len(VIEW_ORDER)) if index != target_index]
    hull = _voxel_hulls(
        source_indices,
        item["candidates"],
        item["masks"],
        item["extrinsic"],
        intrinsic,
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
        intrinsic,
        item["image_size"],
        diagonal,
    )[DEPTH_TOLERANCE_FRACTION]
    return shell[violations <= ALLOWED_SOURCE_CONTRADICTIONS]


def _evaluate_object(
    item: dict[str, Any],
    profile_name: str,
    voxel_resolution: int,
) -> dict[str, Any]:
    profile = ALIGNMENT_PROFILES[profile_name]
    views = []
    baseline_projections = []
    aligned_projections = []
    for target_index, view in enumerate(VIEW_ORDER):
        corrected_intrinsic, alignments = _refine_source_intrinsics(
            item, target_index, profile
        )
        baseline_voxels = _signed_depth_voxels(
            item, target_index, item["intrinsic"], voxel_resolution
        )
        aligned_voxels = _signed_depth_voxels(
            item, target_index, corrected_intrinsic, voxel_resolution
        )
        target = np.uint8(item["masks"][target_index]) * 255
        baseline_projection = _project(
            baseline_voxels,
            item["extrinsic"][target_index],
            item["intrinsic"][target_index],
            item["image_size"],
        )
        aligned_projection = _project(
            aligned_voxels,
            item["extrinsic"][target_index],
            item["intrinsic"][target_index],
            item["image_size"],
        )
        baseline_projections.append(baseline_projection)
        aligned_projections.append(aligned_projection)
        views.append(
            {
                "view": view,
                "baseline": _metrics(baseline_projection, target),
                "aligned": _metrics(aligned_projection, target),
                "source_alignment": alignments,
                "baseline_voxels": int(len(baseline_voxels)),
                "aligned_voxels": int(len(aligned_voxels)),
            }
        )
    baseline_mean = float(np.mean([entry["baseline"]["iou"] for entry in views]))
    aligned_mean = float(np.mean([entry["aligned"]["iou"] for entry in views]))
    return {
        **item,
        "profile": profile_name,
        "views": views,
        "baseline_projections": baseline_projections,
        "aligned_projections": aligned_projections,
        "baseline_mean_iou": round(baseline_mean, 6),
        "mean_iou": round(aligned_mean, 6),
        "minimum_iou": round(
            float(np.min([entry["aligned"]["iou"] for entry in views])), 6
        ),
        "mean_precision": round(
            float(np.mean([entry["aligned"]["precision"] for entry in views])), 6
        ),
        "mean_recall": round(
            float(np.mean([entry["aligned"]["recall"] for entry in views])), 6
        ),
        "mean_delta_over_signed_depth": round(aligned_mean - baseline_mean, 6),
        "mean_source_alignment_delta": round(
            float(
                np.mean(
                    [
                        source["iou_delta"]
                        for entry in views
                        for source in entry["source_alignment"]
                    ]
                )
            ),
            6,
        ),
    }


def _summary(items: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "mean_iou": round(float(np.mean([item["mean_iou"] for item in items])), 6),
        "minimum_iou": round(float(np.min([item["minimum_iou"] for item in items])), 6),
        "mean_precision": round(float(np.mean([item["mean_precision"] for item in items])), 6),
        "mean_recall": round(float(np.mean([item["mean_recall"] for item in items])), 6),
        "mean_delta_over_signed_depth": round(
            float(np.mean([item["mean_delta_over_signed_depth"] for item in items])), 6
        ),
        "mean_source_alignment_delta": round(
            float(np.mean([item["mean_source_alignment_delta"] for item in items])), 6
        ),
    }


def _object_summary(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "object_id": item["object_id"],
        "split": item["split"],
        "profile": item["profile"],
        "baseline_mean_iou": item["baseline_mean_iou"],
        "mean_iou": item["mean_iou"],
        "minimum_iou": item["minimum_iou"],
        "mean_precision": item["mean_precision"],
        "mean_recall": item["mean_recall"],
        "mean_delta_over_signed_depth": item["mean_delta_over_signed_depth"],
        "mean_source_alignment_delta": item["mean_source_alignment_delta"],
        "views": item["views"],
        "exit_gate_passed": item["mean_iou"] >= 0.80 and item["minimum_iou"] >= 0.70,
    }


def _preview(item: dict[str, Any], output: Path) -> None:
    rows = []
    for index, entry in enumerate(item["views"]):
        target = np.uint8(item["masks"][index]) * 255
        baseline = item["baseline_projections"][index]
        aligned = item["aligned_projections"][index]
        rows.append(
            np.hstack(
                (
                    _label(item["sources"][index], f"{entry['view'].upper()} SOURCE"),
                    _label(cv2.cvtColor(target, cv2.COLOR_GRAY2BGR), "TARGET MASK", "score only"),
                    _label(cv2.cvtColor(baseline, cv2.COLOR_GRAY2BGR), "SIGNED DEPTH", "original cameras"),
                    _label(cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR), "SOURCE-ALIGNED", item["profile"]),
                    _label(
                        _agreement(item["sources"][index], aligned, target),
                        "AGREEMENT",
                        f"IoU {entry['aligned']['iou']:.3f} / green match",
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
    validation_inputs = {
        object_id: _load_object(
            root, object_id, records[object_id], confidence_quantile
        )
        for object_id in VALIDATION_OBJECTS
    }
    profile_results = {}
    for profile_name in ALIGNMENT_PROFILES:
        evaluated = [
            _evaluate_object(validation_inputs[object_id], profile_name, voxel_resolution)
            for object_id in VALIDATION_OBJECTS
        ]
        profile_results[profile_name] = {
            "summary": _summary(evaluated),
            "objects": evaluated,
        }
    selected_profile = max(
        ALIGNMENT_PROFILES,
        key=lambda name: profile_results[name]["summary"]["mean_iou"],
    )
    selected_validation = profile_results[selected_profile]["objects"]

    evaluation_inputs = {
        object_id: _load_object(
            root, object_id, records[object_id], confidence_quantile
        )
        for object_id in EVALUATION_OBJECTS
    }
    evaluation = [
        _evaluate_object(evaluation_inputs[object_id], selected_profile, voxel_resolution)
        for object_id in EVALUATION_OBJECTS
    ]
    all_items = [*selected_validation, *evaluation]
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for item in all_items:
        preview = output_dir / f"{item['object_id']}_source_camera_alignment.png"
        _preview(item, preview)
        summary = _object_summary(item)
        summary["preview"] = str(preview.relative_to(root))
        summaries.append(summary)
    validation = [item for item in summaries if item["split"] == "val"]
    test = [item for item in summaries if item["split"] == "test"]
    test_summary = _summary(test)
    accepted = (
        selected_profile != "identity"
        and test_summary["mean_delta_over_signed_depth"] > 0
        and all(item["mean_delta_over_signed_depth"] > 0 for item in test)
    )
    report = {
        "schema_version": "vggt_source_camera_image_alignment_v1",
        "dataset": "STL-1",
        "authority": "non_metric_machine_camera_hypothesis",
        "input_resolution": 518,
        "voxel_resolution": voxel_resolution,
        "validation_selection": {
            "object_ids": list(VALIDATION_OBJECTS),
            "profile_candidates": {
                name: result["summary"] for name, result in profile_results.items()
            },
            "selected_profile": selected_profile,
            "selected_profile_definition": ALIGNMENT_PROFILES[selected_profile],
            "selection_rule": "highest pooled validation mean IoU",
        },
        "validation": _summary(selected_validation),
        "test_evaluation": {
            "object_ids": [item["object_id"] for item in test],
            **test_summary,
        },
        "objects": summaries,
        "checks": {
            "selection_uses_validation_split_only": all(item["split"] == "val" for item in validation),
            "target_mask_and_points_excluded_from_alignment_and_reconstruction": True,
            "target_camera_unchanged": True,
            "all_test_objects_improve_over_signed_depth": all(item["mean_delta_over_signed_depth"] > 0 for item in test),
            "geometry_exit_gate_passed": all(item["exit_gate_passed"] for item in summaries),
        },
        "decision": (
            "retained_as_source_camera_alignment_prior_not_calibration"
            if accepted
            else "rejected_no_consistent_heldout_gain"
        ),
        "claims": {
            "strict_target_image_holdout": False,
            "camera_calibration_validated": False,
            "metric_scale_validated": False,
            "hidden_surface_truth_validated": False,
            "component_topology_validated": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "The fitted image-plane similarity transform is a bounded reprojection correction, not physical camera calibration.",
            "Source silhouettes are machine proposals and may contain reflection-sensitive boundary errors.",
            "VGGT cameras and point maps remain jointly conditioned on all five input images.",
            "A source-only correction can improve reprojection without proving hidden geometry or metric scale.",
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
        default=Path("dataset/phase_runs/v1/vggt_source_camera_alignment_v1"),
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
