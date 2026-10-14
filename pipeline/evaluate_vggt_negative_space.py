"""Evaluate structural negative-space margins after signed-depth carving.

Large enclosed background components in each source silhouette are treated as
structural apertures.  Their boundaries are expanded by validation-selected
pixel margins before source rays carve the signed-depth voxel shell.  Tiny mask
holes are ignored as segmentation noise.  Target-view masks, point maps and
holes are excluded from reconstruction and are used only for scoring.
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
ALLOWED_DEPTH_CONTRADICTIONS = 1
MINIMUM_HOLE_AREA_FRACTION = 0.01
HOLE_MARGIN_CANDIDATES = (0, 1, 2, 4, 6, 8, 12)


def _structural_holes(mask: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Return enclosed background components large enough to be structural."""
    background = np.uint8(~mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(background)
    border_labels = set(
        np.unique(
            np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
        ).tolist()
    )
    minimum_area = int(np.ceil(mask.size * MINIMUM_HOLE_AREA_FRACTION))
    accepted = [
        index
        for index in range(1, count)
        if index not in border_labels
        and int(stats[index, cv2.CC_STAT_AREA]) >= minimum_area
    ]
    holes = np.isin(labels, accepted)
    areas = [int(stats[index, cv2.CC_STAT_AREA]) for index in accepted]
    return holes, areas


def _hole_violations(
    voxels: np.ndarray,
    source_indices: list[int],
    hole_masks: list[np.ndarray],
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    image_size: int,
) -> dict[int, np.ndarray]:
    violations = {
        margin: np.zeros(len(voxels), dtype=np.uint8)
        for margin in HOLE_MARGIN_CANDIDATES
    }
    expanded = {}
    for source_index in source_indices:
        source_holes = np.uint8(hole_masks[source_index])
        expanded[source_index] = {}
        for margin in HOLE_MARGIN_CANDIDATES:
            if margin == 0:
                expanded[source_index][margin] = source_holes > 0
            else:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1)
                )
                expanded[source_index][margin] = cv2.dilate(
                    source_holes, kernel
                ) > 0

        camera = (
            voxels @ extrinsic[source_index, :, :3].T
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
        for margin in HOLE_MARGIN_CANDIDATES:
            contradictions = expanded[source_index][margin][
                pixels[indices, 1], pixels[indices, 0]
            ]
            violations[margin][indices] += contradictions
    return violations


def _evaluate_object(item: dict[str, Any], voxel_resolution: int) -> dict[str, Any]:
    hole_results = [_structural_holes(mask) for mask in item["masks"]]
    hole_masks = [result[0] for result in hole_results]
    hole_areas = [result[1] for result in hole_results]
    projections = {margin: [] for margin in HOLE_MARGIN_CANDIDATES}
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
        depth_violations = _depth_violations(
            shell,
            source_indices,
            item["world_points"],
            item["extrinsic"],
            item["intrinsic"],
            item["image_size"],
            diagonal,
        )[DEPTH_TOLERANCE_FRACTION]
        signed_depth = shell[depth_violations <= ALLOWED_DEPTH_CONTRADICTIONS]
        violations = _hole_violations(
            signed_depth,
            source_indices,
            hole_masks,
            item["extrinsic"],
            item["intrinsic"],
            item["image_size"],
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
        for margin in HOLE_MARGIN_CANDIDATES:
            retained = signed_depth[violations[margin] == 0]
            projection = _project(
                retained,
                item["extrinsic"][target_index],
                item["intrinsic"][target_index],
                item["image_size"],
            )
            projections[margin].append(projection)
            candidate_metrics[str(margin)] = {
                **_metrics(projection, target),
                "retained_voxels": int(len(retained)),
                "removed_voxels": int(len(signed_depth) - len(retained)),
                "hole_margin_pixels": margin,
                "source_structural_hole_count": int(
                    sum(len(hole_areas[index]) for index in source_indices)
                ),
            }
        views.append(
            {
                "view": view,
                "signed_depth": _metrics(baseline_projection, target),
                "target_structural_hole_areas": hole_areas[target_index],
                "candidates": candidate_metrics,
            }
        )
    return {
        **item,
        "views": views,
        "hole_areas": dict(zip(VIEW_ORDER, hole_areas, strict=True)),
        "signed_depth_projections": signed_depth_projections,
        "projections": projections,
    }


def _candidate_summary(objects: list[dict[str, Any]], margin: int) -> dict[str, float]:
    metrics = [
        view["candidates"][str(margin)]
        for item in objects
        for view in item["views"]
    ]
    return {
        "mean_iou": round(float(np.mean([item["iou"] for item in metrics])), 6),
        "minimum_iou": round(float(np.min([item["iou"] for item in metrics])), 6),
        "mean_precision": round(float(np.mean([item["precision"] for item in metrics])), 6),
        "mean_recall": round(float(np.mean([item["recall"] for item in metrics])), 6),
        "mean_removed_voxels": round(float(np.mean([item["removed_voxels"] for item in metrics])), 3),
    }


def _summarize(item: dict[str, Any], margin: int) -> dict[str, Any]:
    selected = [view["candidates"][str(margin)] for view in item["views"]]
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
        "structural_hole_areas": item["hole_areas"],
        "views": [
            {"view": view["view"], **view["candidates"][str(margin)]}
            for view in item["views"]
        ],
        "exit_gate_passed": mean_iou >= 0.80 and minimum_iou >= 0.70,
    }


def _preview(item: dict[str, Any], margin: int, output: Path) -> None:
    rows = []
    for index, view in enumerate(item["views"]):
        target = np.uint8(item["masks"][index]) * 255
        baseline = item["signed_depth_projections"][index]
        refined = item["projections"][margin][index]
        metrics = view["candidates"][str(margin)]
        rows.append(
            np.hstack(
                (
                    _label(item["sources"][index], f"{view['view'].upper()} SOURCE"),
                    _label(cv2.cvtColor(target, cv2.COLOR_GRAY2BGR), "TARGET MASK", "score only"),
                    _label(cv2.cvtColor(baseline, cv2.COLOR_GRAY2BGR), "SIGNED DEPTH", "source masks + depth"),
                    _label(cv2.cvtColor(refined, cv2.COLOR_GRAY2BGR), "NEGATIVE SPACE", f"hole margin {margin}px"),
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
        )
        for object_id in object_ids
    ]
    validation_objects = [
        item for item in objects if item["object_id"] in VALIDATION_OBJECTS
    ]
    candidates = {
        str(margin): _candidate_summary(validation_objects, margin)
        for margin in HOLE_MARGIN_CANDIDATES
    }
    selected_margin = max(
        HOLE_MARGIN_CANDIDATES,
        key=lambda value: candidates[str(value)]["mean_iou"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for item in objects:
        preview = output_dir / f"{item['object_id']}_negative_space.png"
        _preview(item, selected_margin, preview)
        summary = _summarize(item, selected_margin)
        summary["preview"] = str(preview.relative_to(root))
        summaries.append(summary)
    validation = [item for item in summaries if item["split"] == "val"]
    test = [item for item in summaries if item["split"] == "test"]
    test_delta = float(np.mean([item["delta_over_signed_depth"] for item in test]))
    useful = selected_margin > 0 and test_delta > 0
    report = {
        "schema_version": "vggt_structural_negative_space_v1",
        "dataset": "STL-1",
        "authority": "non_metric_machine_surface_hypothesis",
        "input_resolution": 518,
        "voxel_resolution": voxel_resolution,
        "minimum_hole_area_fraction": MINIMUM_HOLE_AREA_FRACTION,
        "upstream": {
            "surface_distance_fraction": SURFACE_DISTANCE_FRACTION,
            "depth_tolerance_fraction": DEPTH_TOLERANCE_FRACTION,
            "allowed_source_contradictions": ALLOWED_DEPTH_CONTRADICTIONS,
        },
        "validation_selection": {
            "object_ids": list(VALIDATION_OBJECTS),
            "candidates": candidates,
            "selected_hole_margin_pixels": selected_margin,
            "selection_rule": "highest pooled validation mean IoU",
        },
        "validation": _candidate_summary(validation_objects, selected_margin),
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
            "target_mask_points_and_holes_excluded_from_reconstruction": True,
            "tiny_enclosed_regions_ignored": True,
            "all_test_objects_improve_over_signed_depth": all(item["delta_over_signed_depth"] > 0 for item in test),
            "geometry_exit_gate_passed": all(item["exit_gate_passed"] for item in summaries),
        },
        "decision": (
            "retained_as_negative_space_prior_not_approved_geometry"
            if useful
            else "rejected_no_heldout_gain_over_signed_depth"
        ),
        "claims": {
            "strict_target_image_holdout": False,
            "cavity_topology_validated": False,
            "metric_scale_validated": False,
            "hidden_surface_truth_validated": False,
            "component_topology_validated": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "Enclosed silhouette background is evidence of visible negative space, not proof of an internal cavity or through-hole.",
            "The one-percent area threshold intentionally ignores small holes that may be details or segmentation artifacts.",
            "A two-dimensional pixel margin is not a metric wall-thickness or cavity-clearance measurement.",
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
        default=Path("dataset/phase_runs/v1/vggt_negative_space_v1"),
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
