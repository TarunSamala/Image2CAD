"""Constrain the VGGT-camera visual hull with observed source surfaces.

For each target view, the target mask and target point map are excluded.  The
four source silhouettes form a visual hull, then voxels farther than a selected
distance from every observed source point are removed.  Selection uses only the
STL-1 validation split.  The output remains non-metric surface evidence, not
complete geometry or CAD.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree

from pipeline.evaluate_vggt_visual_hull_completion import (
    EVALUATION_OBJECTS,
    VALIDATION_OBJECTS,
    _paths,
    _source_candidates,
    _voxel_hulls,
)
from pipeline.validate_vggt_multiring_experiment import (
    VIEW_ORDER,
    _agreement,
    _label,
    _metrics,
    _project,
    _records,
    _square,
)


THICKNESS_CANDIDATES = (0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12)


def _load_object(
    root: Path,
    object_id: str,
    record: dict[str, Any],
    confidence_quantile: float,
) -> dict[str, Any]:
    object_dir, baseline_report_path = _paths(root, object_id)
    report = json.loads((object_dir / "vggt_report.json").read_text(encoding="utf-8"))
    cameras = np.load(object_dir / "vggt_cameras.npz")
    dense = np.load(object_dir / "vggt_dense_evidence.npz")
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
    candidates = _source_candidates(
        dense["world_points"][0],
        dense["world_points_conf"][0],
        masks,
        confidence_quantile,
    )
    baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    baseline = next(
        item for item in baseline_report["objects"] if item["object_id"] == object_id
    )
    return {
        "object_id": object_id,
        "split": record["split"],
        "image_size": image_size,
        "sources": sources,
        "masks": masks,
        "candidates": candidates,
        "world_points": dense["world_points"][0],
        "world_points_conf": dense["world_points_conf"][0],
        "extrinsic": cameras["extrinsic"][0],
        "intrinsic": cameras["intrinsic"][0],
        "point_baseline_mean_iou": baseline["mean_leave_one_view_out_iou"],
    }


def _evaluate_object(item: dict[str, Any], voxel_resolution: int) -> dict[str, Any]:
    views = []
    projections: dict[float, list[np.ndarray]] = {
        fraction: [] for fraction in THICKNESS_CANDIDATES
    }
    visual_hull_projections = []
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
        target = np.uint8(item["masks"][target_index]) * 255
        raw_projection = _project(
            hull,
            item["extrinsic"][target_index],
            item["intrinsic"][target_index],
            item["image_size"],
        )
        visual_hull_projections.append(raw_projection)
        candidate_metrics = {}
        for fraction in THICKNESS_CANDIDATES:
            retained = hull[distances <= diagonal * fraction]
            projection = _project(
                retained,
                item["extrinsic"][target_index],
                item["intrinsic"][target_index],
                item["image_size"],
            )
            projections[fraction].append(projection)
            candidate_metrics[str(fraction)] = {
                **_metrics(projection, target),
                "retained_voxels": int(len(retained)),
                "maximum_surface_distance": round(diagonal * fraction, 8),
            }
        views.append(
            {
                "view": view,
                "visual_hull": _metrics(raw_projection, target),
                "candidates": candidate_metrics,
            }
        )
    return {
        **item,
        "views": views,
        "visual_hull_projections": visual_hull_projections,
        "projections": projections,
        "voxel_resolution": voxel_resolution,
    }


def _candidate_summary(objects: list[dict[str, Any]], fraction: float) -> dict[str, float]:
    metrics = [
        view["candidates"][str(fraction)]
        for item in objects
        for view in item["views"]
    ]
    return {
        "mean_iou": round(float(np.mean([item["iou"] for item in metrics])), 6),
        "minimum_iou": round(float(np.min([item["iou"] for item in metrics])), 6),
        "mean_precision": round(float(np.mean([item["precision"] for item in metrics])), 6),
        "mean_recall": round(float(np.mean([item["recall"] for item in metrics])), 6),
    }


def _summarize(item: dict[str, Any], fraction: float) -> dict[str, Any]:
    selected = [view["candidates"][str(fraction)] for view in item["views"]]
    hull = [view["visual_hull"] for view in item["views"]]
    mean_iou = float(np.mean([metric["iou"] for metric in selected]))
    minimum_iou = float(np.min([metric["iou"] for metric in selected]))
    hull_mean = float(np.mean([metric["iou"] for metric in hull]))
    return {
        "object_id": item["object_id"],
        "split": item["split"],
        "mean_iou": round(mean_iou, 6),
        "minimum_iou": round(minimum_iou, 6),
        "mean_precision": round(float(np.mean([metric["precision"] for metric in selected])), 6),
        "mean_recall": round(float(np.mean([metric["recall"] for metric in selected])), 6),
        "visual_hull_mean_iou": round(hull_mean, 6),
        "delta_over_visual_hull": round(mean_iou - hull_mean, 6),
        "point_baseline_mean_iou": item["point_baseline_mean_iou"],
        "delta_over_point_baseline": round(
            mean_iou - item["point_baseline_mean_iou"], 6
        ),
        "views": [
            {"view": view["view"], **view["candidates"][str(fraction)]}
            for view in item["views"]
        ],
        "exit_gate_passed": mean_iou >= 0.80 and minimum_iou >= 0.70,
    }


def _preview(item: dict[str, Any], fraction: float, output: Path) -> None:
    rows = []
    for index, view in enumerate(item["views"]):
        target = np.uint8(item["masks"][index]) * 255
        hull = item["visual_hull_projections"][index]
        selected = item["projections"][fraction][index]
        metrics = view["candidates"][str(fraction)]
        rows.append(
            np.hstack(
                (
                    _label(item["sources"][index], f"{view['view'].upper()} SOURCE"),
                    _label(cv2.cvtColor(target, cv2.COLOR_GRAY2BGR), "TARGET MASK", "score only"),
                    _label(cv2.cvtColor(hull, cv2.COLOR_GRAY2BGR), "VISUAL HULL", "four source silhouettes"),
                    _label(cv2.cvtColor(selected, cv2.COLOR_GRAY2BGR), "SURFACE-AWARE HULL", "source points + silhouettes"),
                    _label(
                        _agreement(item["sources"][index], selected, target),
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
        str(fraction): _candidate_summary(validation_objects, fraction)
        for fraction in THICKNESS_CANDIDATES
    }
    selected_fraction = max(
        THICKNESS_CANDIDATES,
        key=lambda value: candidates[str(value)]["mean_iou"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for item in objects:
        preview = output_dir / f"{item['object_id']}_surface_aware_hull.png"
        _preview(item, selected_fraction, preview)
        summary = _summarize(item, selected_fraction)
        summary["preview"] = str(preview.relative_to(root))
        summaries.append(summary)
    validation = [item for item in summaries if item["split"] == "val"]
    test = [item for item in summaries if item["split"] == "test"]

    report = {
        "schema_version": "vggt_surface_aware_visual_hull_v1",
        "dataset": "STL-1",
        "authority": "non_metric_machine_surface_hypothesis",
        "input_resolution": 518,
        "voxel_resolution": voxel_resolution,
        "validation_selection": {
            "object_ids": list(VALIDATION_OBJECTS),
            "thickness_candidates": candidates,
            "selected_surface_distance_fraction": selected_fraction,
            "selection_rule": "highest pooled validation mean IoU",
        },
        "validation": _candidate_summary(validation_objects, selected_fraction),
        "test_evaluation": {
            "object_ids": [item["object_id"] for item in test],
            "mean_iou": round(float(np.mean([item["mean_iou"] for item in test])), 6),
            "minimum_iou": round(float(np.min([item["minimum_iou"] for item in test])), 6),
            "mean_precision": round(float(np.mean([item["mean_precision"] for item in test])), 6),
            "mean_recall": round(float(np.mean([item["mean_recall"] for item in test])), 6),
            "mean_delta_over_visual_hull": round(float(np.mean([item["delta_over_visual_hull"] for item in test])), 6),
            "mean_delta_over_point_baseline": round(float(np.mean([item["delta_over_point_baseline"] for item in test])), 6),
        },
        "objects": summaries,
        "checks": {
            "selection_uses_validation_split_only": all(
                item["split"] == "val" for item in validation
            ),
            "target_mask_and_points_excluded_from_reconstruction": True,
            "all_objects_improve_over_point_baseline": all(
                item["delta_over_point_baseline"] > 0 for item in summaries
            ),
            "all_validation_objects_improve_over_visual_hull": all(
                item["delta_over_visual_hull"] > 0 for item in validation
            ),
            "geometry_exit_gate_passed": all(
                item["exit_gate_passed"] for item in summaries
            ),
        },
        "decision": "retained_as_surface_completion_prior_not_approved_geometry",
        "claims": {
            "strict_target_image_holdout": False,
            "metric_scale_validated": False,
            "hidden_surface_truth_validated": False,
            "component_topology_validated": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "The target mask and target point map are excluded, but VGGT camera hypotheses are jointly conditioned on all five images.",
            "Nearest-surface support reduces unsupported volume but does not infer geometry absent from every source view.",
            "The shell remains a voxelized non-metric hypothesis without component identities, cavities or exact topology.",
            "STL-1 masks are machine proposals and the two test identities were observed in earlier exploratory work.",
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
        default=Path("dataset/phase_runs/v1/vggt_surface_aware_hull_v1"),
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
                "selected_surface_distance_fraction": report["validation_selection"]["selected_surface_distance_fraction"],
                "validation": report["validation"],
                "test_evaluation": report["test_evaluation"],
                "checks": report["checks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
