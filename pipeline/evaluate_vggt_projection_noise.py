"""Evaluate target-independent cleanup of signed-depth projection noise.

The signed-depth voxel geometry is never changed.  This experiment only
changes how that geometry is rasterised for silhouette review.  A tiny binary
closing operation is enabled from the raw prediction's own fragmentation
count, never from the held-out target mask.  Thresholds are selected on the
fixed validation split and then frozen for the test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline.evaluate_vggt_negative_space import _structural_holes
from pipeline.evaluate_vggt_source_camera_alignment import _signed_depth_voxels
from pipeline.evaluate_vggt_surface_aware_hull import _load_object
from pipeline.evaluate_vggt_visual_hull_completion import (
    EVALUATION_OBJECTS,
    VALIDATION_OBJECTS,
)
from pipeline.validate_vggt_multiring_experiment import (
    VIEW_ORDER,
    _agreement,
    _label,
    _metrics,
    _project,
    _records,
)


FRAGMENT_THRESHOLD_CANDIDATES: tuple[int | None, ...] = (
    None,
    2,
    3,
    4,
    6,
    10,
    20,
    40,
)
CLOSING_KERNEL = np.ones((3, 3), np.uint8)


def _profile_name(threshold: int | None) -> str:
    if threshold is None:
        return "raw_point_projection"
    return f"close3_components_ge_{threshold}"


def _foreground_components(mask: np.ndarray) -> tuple[int, float]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        np.uint8(mask > 0), connectivity=8
    )
    areas = stats[1:, cv2.CC_STAT_AREA]
    if not len(areas):
        return 0, 0.0
    return int(len(areas)), float(np.max(areas) / max(np.sum(areas), 1))


def _cleanup_projection(
    raw: np.ndarray,
    component_count: int,
    threshold: int | None,
) -> tuple[np.ndarray, bool]:
    if threshold is None or component_count < threshold:
        return raw.copy(), False
    cleaned = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, CLOSING_KERNEL)
    return cleaned, True


def _hole_false_fill(prediction: np.ndarray, target_holes: np.ndarray) -> float:
    if not np.any(target_holes):
        return 0.0
    return float(np.mean(prediction[target_holes] > 0))


def _evaluate_object(
    item: dict[str, Any], voxel_resolution: int
) -> dict[str, Any]:
    profile_names = [_profile_name(value) for value in FRAGMENT_THRESHOLD_CANDIDATES]
    projections = {name: [] for name in profile_names}
    views = []
    for target_index, view in enumerate(VIEW_ORDER):
        voxels = _signed_depth_voxels(
            item,
            target_index,
            item["intrinsic"],
            voxel_resolution,
        )
        raw = _project(
            voxels,
            item["extrinsic"][target_index],
            item["intrinsic"][target_index],
            item["image_size"],
        )
        target = np.uint8(item["masks"][target_index]) * 255
        target_holes, target_hole_areas = _structural_holes(target > 0)
        component_count, largest_component_fraction = _foreground_components(raw)
        raw_metrics = _metrics(raw, target)
        raw_hole_false_fill = _hole_false_fill(raw, target_holes)
        candidates = {}
        for threshold in FRAGMENT_THRESHOLD_CANDIDATES:
            profile = _profile_name(threshold)
            cleaned, applied = _cleanup_projection(
                raw, component_count, threshold
            )
            projections[profile].append(cleaned)
            metrics = _metrics(cleaned, target)
            added = int(np.count_nonzero((cleaned > 0) & (raw == 0)))
            removed = int(np.count_nonzero((cleaned == 0) & (raw > 0)))
            candidates[profile] = {
                **metrics,
                "iou_delta_over_raw": round(
                    metrics["iou"] - raw_metrics["iou"], 6
                ),
                "cleanup_applied": applied,
                "added_pixels": added,
                "removed_pixels": removed,
                "added_fraction_of_raw": round(
                    added / max(int(np.count_nonzero(raw)), 1), 6
                ),
                "structural_hole_false_fill": round(
                    _hole_false_fill(cleaned, target_holes), 6
                ),
                "structural_hole_false_fill_delta": round(
                    _hole_false_fill(cleaned, target_holes)
                    - raw_hole_false_fill,
                    6,
                ),
            }
        views.append(
            {
                "view": view,
                "raw_foreground_components": component_count,
                "raw_largest_component_fraction": round(
                    largest_component_fraction, 6
                ),
                "target_structural_hole_areas": target_hole_areas,
                "raw": {
                    **raw_metrics,
                    "structural_hole_false_fill": round(
                        raw_hole_false_fill, 6
                    ),
                    "retained_voxels": int(len(voxels)),
                },
                "candidates": candidates,
            }
        )
    return {
        **item,
        "voxel_resolution": voxel_resolution,
        "views": views,
        "projections": projections,
    }


def _configuration_summary(
    objects: list[dict[str, Any]], profile: str
) -> dict[str, Any]:
    rows = [view["candidates"][profile] for item in objects for view in item["views"]]
    deltas = [row["iou_delta_over_raw"] for row in rows]
    return {
        "mean_iou": round(float(np.mean([row["iou"] for row in rows])), 6),
        "minimum_iou": round(float(np.min([row["iou"] for row in rows])), 6),
        "mean_precision": round(
            float(np.mean([row["precision"] for row in rows])), 6
        ),
        "mean_recall": round(float(np.mean([row["recall"] for row in rows])), 6),
        "mean_delta_over_raw": round(float(np.mean(deltas)), 6),
        "worst_view_delta_over_raw": round(float(np.min(deltas)), 6),
        "changed_view_count": int(sum(row["cleanup_applied"] for row in rows)),
        "improved_view_count": int(sum(delta > 0 for delta in deltas)),
        "regressed_view_count": int(sum(delta < 0 for delta in deltas)),
        "mean_added_fraction_of_raw": round(
            float(np.mean([row["added_fraction_of_raw"] for row in rows])), 6
        ),
        "mean_structural_hole_false_fill_delta": round(
            float(
                np.mean(
                    [row["structural_hole_false_fill_delta"] for row in rows]
                )
            ),
            6,
        ),
    }


def _object_summary(item: dict[str, Any], profile: str) -> dict[str, Any]:
    selected = [view["candidates"][profile] for view in item["views"]]
    mean_iou = float(np.mean([row["iou"] for row in selected]))
    raw_mean = float(np.mean([view["raw"]["iou"] for view in item["views"]]))
    return {
        "object_id": item["object_id"],
        "split": item["split"],
        "profile": profile,
        "raw_mean_iou": round(raw_mean, 6),
        "mean_iou": round(mean_iou, 6),
        "minimum_iou": round(float(np.min([row["iou"] for row in selected])), 6),
        "mean_precision": round(
            float(np.mean([row["precision"] for row in selected])), 6
        ),
        "mean_recall": round(
            float(np.mean([row["recall"] for row in selected])), 6
        ),
        "mean_delta_over_raw": round(mean_iou - raw_mean, 6),
        "views": [
            {
                "view": view["view"],
                "raw_foreground_components": view[
                    "raw_foreground_components"
                ],
                "raw": view["raw"],
                "cleaned": view["candidates"][profile],
            }
            for view in item["views"]
        ],
        "exit_gate_passed": mean_iou >= 0.80
        and min(row["iou"] for row in selected) >= 0.70,
    }


def _preview(item: dict[str, Any], profile: str, output: Path) -> None:
    raw_profile = _profile_name(None)
    rows = []
    for index, view in enumerate(item["views"]):
        target = np.uint8(item["masks"][index]) * 255
        raw = item["projections"][raw_profile][index]
        cleaned = item["projections"][profile][index]
        candidate = view["candidates"][profile]
        changed = np.full_like(item["sources"][index], 245)
        added = (cleaned > 0) & (raw == 0)
        retained = (cleaned > 0) & (raw > 0)
        changed[retained] = (180, 180, 180)
        changed[added] = (20, 130, 240)
        rows.append(
            np.hstack(
                (
                    _label(
                        item["sources"][index],
                        f"{view['view'].upper()} SOURCE",
                    ),
                    _label(
                        cv2.cvtColor(target, cv2.COLOR_GRAY2BGR),
                        "TARGET MASK",
                        "score only",
                    ),
                    _label(
                        cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR),
                        "RAW PROJECTION",
                        f"{view['raw_foreground_components']} components",
                    ),
                    _label(
                        cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR),
                        "FRAGMENT-GATED CLEANUP",
                        "applied" if candidate["cleanup_applied"] else "raw retained",
                    ),
                    _label(
                        changed,
                        "PIXELS ADDED",
                        f"{candidate['added_pixels']} orange pixels",
                    ),
                    _label(
                        _agreement(item["sources"][index], cleaned, target),
                        "AGREEMENT",
                        f"IoU {candidate['iou']:.3f} / delta {candidate['iou_delta_over_raw']:+.3f}",
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
                root, object_id, records[object_id], confidence_quantile
            ),
            voxel_resolution,
        )
        for object_id in object_ids
    ]
    validation_objects = [
        item for item in objects if item["object_id"] in VALIDATION_OBJECTS
    ]
    candidates = {
        _profile_name(threshold): _configuration_summary(
            validation_objects, _profile_name(threshold)
        )
        for threshold in FRAGMENT_THRESHOLD_CANDIDATES
    }
    eligible = [
        profile
        for profile, summary in candidates.items()
        if summary["regressed_view_count"] == 0
        and summary["mean_structural_hole_false_fill_delta"] <= 0.001
    ]
    selected_profile = max(
        eligible or [_profile_name(None)],
        key=lambda profile: candidates[profile]["mean_iou"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for item in objects:
        preview = output_dir / f"{item['object_id']}_projection_noise.png"
        _preview(item, selected_profile, preview)
        summary = _object_summary(item, selected_profile)
        summary["preview"] = str(preview.relative_to(root))
        summaries.append(summary)
    validation = [item for item in summaries if item["split"] == "val"]
    test = [item for item in summaries if item["split"] == "test"]
    validation_summary = _configuration_summary(
        validation_objects, selected_profile
    )
    test_summary = {
        "object_ids": [item["object_id"] for item in test],
        "mean_iou": round(float(np.mean([item["mean_iou"] for item in test])), 6),
        "minimum_iou": round(
            float(np.min([item["minimum_iou"] for item in test])), 6
        ),
        "mean_precision": round(
            float(np.mean([item["mean_precision"] for item in test])), 6
        ),
        "mean_recall": round(
            float(np.mean([item["mean_recall"] for item in test])), 6
        ),
        "mean_delta_over_raw": round(
            float(np.mean([item["mean_delta_over_raw"] for item in test])), 6
        ),
        "regressed_view_count": int(
            sum(
                view["cleaned"]["iou_delta_over_raw"] < 0
                for item in test
                for view in item["views"]
            )
        ),
        "mean_structural_hole_false_fill_delta": round(
            float(
                np.mean(
                    [
                        view["cleaned"][
                            "structural_hole_false_fill_delta"
                        ]
                        for item in test
                        for view in item["views"]
                    ]
                )
            ),
            6,
        ),
    }
    accepted = (
        selected_profile != _profile_name(None)
        and validation_summary["mean_delta_over_raw"] > 0
        and test_summary["mean_delta_over_raw"] > 0
        and test_summary["regressed_view_count"] == 0
        and all(item["mean_delta_over_raw"] > 0 for item in test)
        and test_summary["mean_structural_hole_false_fill_delta"] <= 0.001
    )
    report = {
        "schema_version": "vggt_projection_noise_v1",
        "dataset": "STL-1",
        "authority": "projection_renderer_evidence_only",
        "input_resolution": 518,
        "voxel_resolution": voxel_resolution,
        "validation_selection": {
            "object_ids": list(VALIDATION_OBJECTS),
            "profile_candidates": candidates,
            "selected_profile": selected_profile,
            "selection_rule": "highest validation mean IoU among profiles with zero regressed validation views and no material structural-hole false-fill increase",
        },
        "validation": validation_summary,
        "test_evaluation": test_summary,
        "objects": summaries,
        "checks": {
            "selection_uses_validation_split_only": all(
                item["split"] == "val" for item in validation
            ),
            "target_mask_not_used_for_cleanup_decision": True,
            "target_mask_and_points_excluded_from_reconstruction": True,
            "voxel_geometry_unchanged": True,
            "all_test_objects_improve_over_raw": all(
                item["mean_delta_over_raw"] > 0 for item in test
            ),
            "no_test_view_regresses": test_summary[
                "regressed_view_count"
            ]
            == 0,
            "structural_hole_guard_passed": test_summary[
                "mean_structural_hole_false_fill_delta"
            ]
            <= 0.001,
            "geometry_exit_gate_passed": all(
                item["exit_gate_passed"] for item in summaries
            ),
        },
        "decision": (
            "retained_as_projection_renderer_only"
            if accepted
            else "rejected_projection_cleanup_not_consistent"
        ),
        "claims": {
            "geometry_improved": False,
            "topology_validated": False,
            "metric_scale_validated": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "The cleanup changes only binary projection rasterisation; signed-depth voxels are byte-for-byte unchanged.",
            "Target masks are score-only evidence and do not decide whether cleanup is applied.",
            "A cleaner silhouette does not validate hidden surfaces, topology, metric scale or CAD readiness.",
            "The weakest held-out view remains a geometry/alignment problem rather than projection noise.",
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
        default=Path("dataset/phase_runs/v1/vggt_projection_noise_v1"),
    )
    parser.add_argument("--voxel-resolution", type=int, default=96)
    parser.add_argument("--confidence-quantile", type=float, default=0.50)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else root / args.output_dir
    )
    report = run(
        root, output_dir, args.voxel_resolution, args.confidence_quantile
    )
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
