"""Evaluate source-view normal consistency after signed-depth carving.

World-space normals are derived from local VGGT point-map tangents.  For every
target view, only the other four masks, point maps, confidences and derived
normals may constrain the voxel shell.  Voxels with fewer than two source
normal observations remain uncertain and are retained.  The consistency
threshold is selected only on validation objects; output remains non-metric
research evidence rather than approved geometry or CAD.
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
NORMAL_COSINE_CANDIDATES = (0.0, 0.25, 0.50, 0.65, 0.75, 0.85, 0.95)


def _world_normals(world: np.ndarray) -> np.ndarray:
    """Derive sign-ambiguous world normals from point-map finite differences."""
    horizontal = np.empty_like(world)
    vertical = np.empty_like(world)
    horizontal[:, 1:-1] = world[:, 2:] - world[:, :-2]
    horizontal[:, 0] = world[:, 1] - world[:, 0]
    horizontal[:, -1] = world[:, -1] - world[:, -2]
    vertical[1:-1] = world[2:] - world[:-2]
    vertical[0] = world[1] - world[0]
    vertical[-1] = world[-1] - world[-2]
    normals = np.cross(horizontal, vertical)
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    valid = np.isfinite(normals).all(axis=-1) & (lengths[..., 0] > 1e-9)
    output = np.full_like(normals, np.nan)
    output[valid] = normals[valid] / lengths[valid]
    return output


def _source_normal_samples(
    item: dict[str, Any], confidence_quantile: float
) -> list[tuple[np.ndarray, np.ndarray]]:
    samples = []
    for index in range(len(VIEW_ORDER)):
        world = item["world_points"][index]
        confidence = item["world_points_conf"][index]
        normals = _world_normals(world)
        finite = np.isfinite(world).all(axis=-1) & np.isfinite(normals).all(axis=-1)
        values = confidence[item["masks"][index] & finite]
        threshold = float(np.quantile(values, confidence_quantile))
        valid = (
            item["masks"][index]
            & finite
            & np.isfinite(confidence)
            & (confidence >= threshold)
        )
        samples.append((world[valid], normals[valid]))
    return samples


def _normal_consistency(
    voxels: np.ndarray,
    source_indices: list[int],
    samples: list[tuple[np.ndarray, np.ndarray]],
    maximum_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sign-invariant pairwise normal agreement for each voxel."""
    sampled_normals = []
    supported = []
    for source_index in source_indices:
        points, normals = samples[source_index]
        distances, indices = cKDTree(points).query(voxels, k=1, workers=-1)
        sampled_normals.append(normals[indices])
        supported.append(distances <= maximum_distance)
    sampled = np.stack(sampled_normals, axis=0)
    support = np.stack(supported, axis=0)
    agreement_sum = np.zeros(len(voxels), dtype=np.float32)
    pair_count = np.zeros(len(voxels), dtype=np.uint8)
    for first in range(len(source_indices)):
        for second in range(first + 1, len(source_indices)):
            valid = support[first] & support[second]
            similarity = np.abs(
                np.sum(sampled[first] * sampled[second], axis=1)
            )
            agreement_sum[valid] += np.clip(similarity[valid], 0.0, 1.0)
            pair_count[valid] += 1
    score = np.ones(len(voxels), dtype=np.float32)
    assessed = pair_count > 0
    score[assessed] = agreement_sum[assessed] / pair_count[assessed]
    return score, pair_count, support.sum(axis=0).astype(np.uint8)


def _evaluate_object(
    item: dict[str, Any], voxel_resolution: int, confidence_quantile: float
) -> dict[str, Any]:
    samples = _source_normal_samples(item, confidence_quantile)
    projections = {threshold: [] for threshold in NORMAL_COSINE_CANDIDATES}
    signed_depth_projections = []
    views = []
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
        )[DEPTH_TOLERANCE_FRACTION]
        signed_depth = shell[violations <= ALLOWED_SOURCE_CONTRADICTIONS]
        normal_score, pair_count, support_count = _normal_consistency(
            signed_depth,
            source_indices,
            samples,
            diagonal * SURFACE_DISTANCE_FRACTION,
        )
        target = np.uint8(item["masks"][target_index]) * 255
        baseline_projection = _project(
            signed_depth,
            item["extrinsic"][target_index],
            item["intrinsic"][target_index],
            item["image_size"],
        )
        signed_depth_projections.append(baseline_projection)
        candidate_metrics = {}
        assessed = pair_count > 0
        for threshold in NORMAL_COSINE_CANDIDATES:
            if threshold == 0.0:
                retained = signed_depth
            else:
                retained = signed_depth[(~assessed) | (normal_score >= threshold)]
            projection = _project(
                retained,
                item["extrinsic"][target_index],
                item["intrinsic"][target_index],
                item["image_size"],
            )
            projections[threshold].append(projection)
            candidate_metrics[str(threshold)] = {
                **_metrics(projection, target),
                "retained_voxels": int(len(retained)),
                "removed_voxels": int(len(signed_depth) - len(retained)),
                "assessed_voxel_fraction": round(float(np.mean(assessed)), 6),
                "mean_source_support": round(float(np.mean(support_count)), 6),
                "normal_cosine_threshold": threshold,
            }
        views.append(
            {
                "view": view,
                "signed_depth": _metrics(baseline_projection, target),
                "candidates": candidate_metrics,
            }
        )
    return {
        **item,
        "views": views,
        "signed_depth_projections": signed_depth_projections,
        "projections": projections,
        "voxel_resolution": voxel_resolution,
    }


def _candidate_summary(
    objects: list[dict[str, Any]], threshold: float
) -> dict[str, float]:
    metrics = [
        view["candidates"][str(threshold)]
        for item in objects
        for view in item["views"]
    ]
    return {
        "mean_iou": round(float(np.mean([item["iou"] for item in metrics])), 6),
        "minimum_iou": round(float(np.min([item["iou"] for item in metrics])), 6),
        "mean_precision": round(float(np.mean([item["precision"] for item in metrics])), 6),
        "mean_recall": round(float(np.mean([item["recall"] for item in metrics])), 6),
        "mean_removed_voxels": round(float(np.mean([item["removed_voxels"] for item in metrics])), 3),
        "mean_assessed_voxel_fraction": round(float(np.mean([item["assessed_voxel_fraction"] for item in metrics])), 6),
    }


def _summarize(item: dict[str, Any], threshold: float) -> dict[str, Any]:
    selected = [view["candidates"][str(threshold)] for view in item["views"]]
    baseline = [view["signed_depth"] for view in item["views"]]
    mean_iou = float(np.mean([metric["iou"] for metric in selected]))
    minimum_iou = float(np.min([metric["iou"] for metric in selected]))
    baseline_mean = float(np.mean([metric["iou"] for metric in baseline]))
    return {
        "object_id": item["object_id"],
        "split": item["split"],
        "mean_iou": round(mean_iou, 6),
        "minimum_iou": round(minimum_iou, 6),
        "mean_precision": round(float(np.mean([metric["precision"] for metric in selected])), 6),
        "mean_recall": round(float(np.mean([metric["recall"] for metric in selected])), 6),
        "signed_depth_mean_iou": round(baseline_mean, 6),
        "delta_over_signed_depth": round(mean_iou - baseline_mean, 6),
        "views": [
            {"view": view["view"], **view["candidates"][str(threshold)]}
            for view in item["views"]
        ],
        "exit_gate_passed": mean_iou >= 0.80 and minimum_iou >= 0.70,
    }


def _preview(item: dict[str, Any], threshold: float, output: Path) -> None:
    rows = []
    for index, view in enumerate(item["views"]):
        target = np.uint8(item["masks"][index]) * 255
        baseline = item["signed_depth_projections"][index]
        refined = item["projections"][threshold][index]
        metrics = view["candidates"][str(threshold)]
        rows.append(
            np.hstack(
                (
                    _label(item["sources"][index], f"{view['view'].upper()} SOURCE"),
                    _label(cv2.cvtColor(target, cv2.COLOR_GRAY2BGR), "TARGET MASK", "score only"),
                    _label(cv2.cvtColor(baseline, cv2.COLOR_GRAY2BGR), "SIGNED DEPTH", "source masks + depth"),
                    _label(cv2.cvtColor(refined, cv2.COLOR_GRAY2BGR), "NORMAL CONSISTENCY", f"cosine >= {threshold:.2f}"),
                    _label(
                        _agreement(item["sources"][index], refined, target),
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
            _load_object(root, object_id, records[object_id], confidence_quantile),
            voxel_resolution,
            confidence_quantile,
        )
        for object_id in object_ids
    ]
    validation_objects = [
        item for item in objects if item["object_id"] in VALIDATION_OBJECTS
    ]
    candidates = {
        str(threshold): _candidate_summary(validation_objects, threshold)
        for threshold in NORMAL_COSINE_CANDIDATES
    }
    selected_threshold = max(
        NORMAL_COSINE_CANDIDATES,
        key=lambda value: candidates[str(value)]["mean_iou"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for item in objects:
        preview = output_dir / f"{item['object_id']}_normal_consistency.png"
        _preview(item, selected_threshold, preview)
        summary = _summarize(item, selected_threshold)
        summary["preview"] = str(preview.relative_to(root))
        summaries.append(summary)
    validation = [item for item in summaries if item["split"] == "val"]
    test = [item for item in summaries if item["split"] == "test"]
    test_delta = float(np.mean([item["delta_over_signed_depth"] for item in test]))
    useful = selected_threshold > 0 and test_delta > 0

    report = {
        "schema_version": "vggt_source_normal_consistency_v1",
        "dataset": "STL-1",
        "authority": "non_metric_machine_surface_hypothesis",
        "input_resolution": 518,
        "voxel_resolution": voxel_resolution,
        "upstream": {
            "surface_distance_fraction": SURFACE_DISTANCE_FRACTION,
            "depth_tolerance_fraction": DEPTH_TOLERANCE_FRACTION,
            "allowed_source_contradictions": ALLOWED_SOURCE_CONTRADICTIONS,
        },
        "validation_selection": {
            "object_ids": list(VALIDATION_OBJECTS),
            "candidates": candidates,
            "selected_normal_cosine_threshold": selected_threshold,
            "selection_rule": "highest pooled validation mean IoU",
        },
        "validation": _candidate_summary(validation_objects, selected_threshold),
        "test_evaluation": {
            "object_ids": [item["object_id"] for item in test],
            "mean_iou": round(float(np.mean([item["mean_iou"] for item in test])), 6),
            "minimum_iou": round(float(np.min([item["minimum_iou"] for item in test])), 6),
            "mean_precision": round(float(np.mean([item["mean_precision"] for item in test])), 6),
            "mean_recall": round(float(np.mean([item["mean_recall"] for item in test])), 6),
            "mean_delta_over_signed_depth": round(test_delta, 6),
        },
        "objects": summaries,
        "checks": {
            "selection_uses_validation_split_only": all(item["split"] == "val" for item in validation),
            "target_mask_points_and_normals_excluded_from_reconstruction": True,
            "unassessed_voxels_retained_as_uncertain": True,
            "all_test_objects_improve_over_signed_depth": all(item["delta_over_signed_depth"] > 0 for item in test),
            "geometry_exit_gate_passed": all(item["exit_gate_passed"] for item in summaries),
        },
        "decision": (
            "retained_as_normal_consistency_prior_not_approved_geometry"
            if useful
            else "rejected_no_heldout_gain_over_signed_depth"
        ),
        "claims": {
            "strict_target_image_holdout": False,
            "normal_accuracy_validated": False,
            "metric_scale_validated": False,
            "hidden_surface_truth_validated": False,
            "component_topology_validated": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "Normals are finite-difference hypotheses derived from VGGT world-point maps, not independent normal ground truth.",
            "Normal direction is treated as sign-ambiguous because source-view tangent orientation may differ.",
            "Voxels observed by fewer than two source normal maps remain uncertain and are retained.",
            "VGGT cameras and point maps are jointly conditioned on all five images even though target geometry evidence is excluded.",
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
        default=Path("dataset/phase_runs/v1/vggt_normal_consistency_v1"),
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
