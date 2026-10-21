"""Test fixed multiscale relief patches against Phase 5.17 matched controls.

Training-only descriptor discrimination. Existing reciprocal tracks, target
decoys and split assignments are frozen. No patch is fit to a target image,
and a descriptor win would not constitute 3D surface or CAD validation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline.build_phase5_stl1_relief_evidence import _metric_depth_records
from pipeline.evaluate_phase5_dense_relief_pilot import _preserve_baseline_gate
from pipeline.evaluate_phase5_relief_tracks import (
    _load_view_evidence,
    _metric3d_normal_records,
    _sample,
    evaluate_track_consistency,
)
from pipeline.evaluate_phase5_training_relief_descriptors import (
    MINIMUM_CONTROL_COUNT,
    MINIMUM_POSITIVE_COUNT_FOR_VIEW,
    SELECTION_MAXIMUM_OBJECT_AUC_REGRESSION,
    SELECTION_MINIMUM_IMPROVED_OBJECT_COUNT,
    SELECTION_MINIMUM_MEAN_AUC_DELTA,
    _controls,
    _load_descriptors,
)
from pipeline.evaluate_phase5_training_track_topology import (
    BASELINE_TRACK_ROOT,
    HYBRID_TRACK_ROOT,
    PHASE1_REPORT,
    SOURCE_INDEX,
    SOURCE_VIEW,
    TARGET_VIEWS,
    _load_tracks,
    _relative,
)
from pipeline.extract_vggt_track_evidence import TRAIN_OBJECTS, VIEW_ORDER


DESCRIPTOR_WEIGHTS = {
    "signed_relief_baseline": (1.0,),
    "local_signed_profile": (0.55, 0.25, 0.12, 0.08),
    "radial_relief_profile": (0.30, 0.25, 0.15, 0.20, 0.10),
    "multiscale_shape_fusion": (0.35, 0.25, 0.15, 0.10, 0.10, 0.05),
}
DESCRIPTOR_ORDER = tuple(DESCRIPTOR_WEIGHTS)
SIGMAS_IMAGE_PIXELS = (4.0, 10.0, 20.0)
FROZEN_BASELINE_MEAN = 0.606918
FROZEN_INCLUDED_TRACK_COUNT = 3259


def _weighted_blur(values: np.ndarray, support: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth only supported relief; avoid drawing background into the patch."""
    support_f = np.asarray(support, np.float32)
    numerator = cv2.GaussianBlur(values * support_f, (0, 0), sigma)
    denominator = cv2.GaussianBlur(support_f, (0, 0), sigma)
    result = np.divide(
        numerator, denominator, out=np.zeros_like(numerator),
        where=denominator > 1e-6,
    )
    return np.asarray(result, np.float32)


def _shape_maps(maps: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    signed = np.asarray(maps["signed_relief_baseline"], np.float32)
    support = np.asarray(maps["support"], bool)
    mean4, mean10, mean20 = (
        _weighted_blur(signed, support, sigma) for sigma in SIGMAS_IMAGE_PIXELS
    )
    second4 = _weighted_blur(np.square(signed), support, SIGMAS_IMAGE_PIXELS[0])
    second10 = _weighted_blur(np.square(signed), support, SIGMAS_IMAGE_PIXELS[1])
    std4 = np.sqrt(np.maximum(second4 - np.square(mean4), 0.0))
    std10 = np.sqrt(np.maximum(second10 - np.square(mean10), 0.0))
    high4 = signed - mean4
    band10 = mean4 - mean10
    band20 = mean10 - mean20
    return {
        "signed_relief_baseline": signed[..., None],
        "local_signed_profile": np.stack((signed, high4, band10, band20), axis=-1),
        "radial_relief_profile": np.stack(
            (mean4, mean10, mean20, std4, std10), axis=-1
        ),
        "multiscale_shape_fusion": np.stack(
            (signed, high4, band10, band20, std4, std10), axis=-1
        ),
    }


def _paired_score(
    source: np.ndarray,
    positive: np.ndarray,
    controls: np.ndarray,
    valid_controls: np.ndarray,
    weights: tuple[float, ...],
) -> dict[str, Any]:
    weights_array = np.asarray(weights, np.float32)
    if source.shape != positive.shape or source.shape[-1] != len(weights_array):
        raise ValueError("descriptor shapes or channel weights differ")
    if controls.shape[:2] != valid_controls.shape or controls.shape[-1] != len(weights_array):
        raise ValueError("control shapes or channel weights differ")
    positive_error = np.sum(np.abs(source - positive) * weights_array, axis=-1)
    individual_errors = np.sum(
        np.abs(source[:, None, :] - controls) * weights_array, axis=-1
    )
    control_error = np.sum(
        np.where(valid_controls, individual_errors, 0.0), axis=1
    ) / np.maximum(np.count_nonzero(valid_controls, axis=1), 1)
    ties = np.isclose(positive_error, control_error, atol=1e-6)
    wins = (positive_error < control_error) & ~ties
    paired = np.mean(wins.astype(np.float32) + 0.5 * ties.astype(np.float32))
    return {
        "positive_error": round(float(np.mean(positive_error)), 6),
        "control_error": round(float(np.mean(control_error)), 6),
        "mean_margin": round(float(np.mean(control_error - positive_error)), 6),
        "paired_discrimination_auc": round(float(paired), 6),
        "positive_beats_control_count": int(np.count_nonzero(wins)),
    }


def _evaluate_view(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    accepted: np.ndarray,
    source_maps: dict[str, np.ndarray],
    target_maps: dict[str, np.ndarray],
    source_shape: dict[str, np.ndarray],
    target_shape: dict[str, np.ndarray],
) -> dict[str, Any]:
    indices = np.flatnonzero(accepted)
    if not len(indices):
        return {"accepted_track_count": 0, "control_eligible_track_count": 0,
                "descriptors": {}}
    positive_xy = target_xy[indices]
    displaced, valid = _controls(positive_xy, target_maps)
    eligible = np.count_nonzero(valid, axis=1) >= MINIMUM_CONTROL_COUNT
    source_points = source_xy[indices][eligible]
    positive_points = positive_xy[eligible]
    control_points = displaced[eligible]
    control_valid = valid[eligible]
    descriptor_metrics = {}
    for name, weights in DESCRIPTOR_WEIGHTS.items():
        if not len(source_points):
            descriptor_metrics[name] = {
                "positive_error": None, "control_error": None,
                "mean_margin": None, "paired_discrimination_auc": None,
                "positive_beats_control_count": 0,
            }
            continue
        descriptor_metrics[name] = _paired_score(
            _sample(source_shape[name], source_points),
            _sample(target_shape[name], positive_points),
            _sample(target_shape[name], control_points),
            control_valid, weights,
        )
    return {
        "accepted_track_count": int(len(indices)),
        "control_eligible_track_count": int(len(source_points)),
        "descriptors": descriptor_metrics,
    }


def _render_audit(report: dict[str, Any], output: Path) -> None:
    objects = report["objects"]
    figure, axes = plt.subplots(2, 1, figsize=(17, 10), facecolor="#f4f4f4")
    positions = np.arange(len(objects))
    colors = ("#8a8f95", "#497f9f", "#bb7190", "#589972")
    width = 0.19
    for index, name in enumerate(DESCRIPTOR_ORDER):
        scores = [item["descriptors"][name]["paired_discrimination_auc"] for item in objects]
        axes[0].bar(positions + (index - 1.5) * width, scores, width,
                    color=colors[index], label=name.replace("_", " "))
    axes[0].axhline(0.5, color="#222222", linestyle="--", linewidth=1)
    axes[0].set_ylim(0, 1)
    axes[0].set_xticks(positions, [item["object_id"] for item in objects],
                      rotation=45, ha="right")
    axes[0].set_title("Same accepted tracks and matched target decoys, by training ring", loc="left")
    axes[0].legend(frameon=False, ncol=4, fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)
    means = [report["summary"]["descriptors"][name]["mean_object_auc"]
             for name in DESCRIPTOR_ORDER]
    axes[1].barh(np.arange(len(DESCRIPTOR_ORDER)), means, color=colors)
    axes[1].axvline(means[0], color="#333333", linestyle="--", linewidth=1,
                    label="signed-relief baseline")
    axes[1].axvline(
        means[0] + SELECTION_MINIMUM_MEAN_AUC_DELTA,
        color="#a34a65", linestyle=":", linewidth=2,
        label="minimum training gain",
    )
    axes[1].set_xlim(0, 1)
    axes[1].set_yticks(np.arange(len(DESCRIPTOR_ORDER)),
                      [name.replace("_", " ") for name in DESCRIPTOR_ORDER])
    axes[1].set_title("Equal-weight mean across 18 training rings", loc="left")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(axis="x", alpha=0.2)
    figure.suptitle("PHASE 5.19 - TRAINING LOCAL SHAPE DESCRIPTOR TEST",
                    fontsize=17, fontweight="bold")
    figure.text(0.5, 0.012,
                "Image-space descriptor comparison only - validation surface and CAD gates unchanged",
                ha="center", fontsize=10)
    figure.tight_layout(rect=(0.02, 0.045, 0.98, 0.94))
    figure.savefig(output, dpi=165, bbox_inches="tight")
    plt.close(figure)


def run(root: Path, output: Path) -> dict[str, Any]:
    root, output = root.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    phase1 = json.loads((root / PHASE1_REPORT).read_text(encoding="utf-8"))
    records = {item["object_id"]: item for item in phase1["objects"]}
    metric_depths = _metric_depth_records(root)
    direct_normals = _metric3d_normal_records(root)
    objects = []
    for object_id in TRAIN_OBJECTS:
        record = records[object_id]
        if record["split"] != "train":
            raise RuntimeError(f"{object_id} is outside the training split")
        evidence = _load_view_evidence(root, record, metric_depths, direct_normals)
        maps = _load_descriptors(root, record, metric_depths)
        shapes = {view: _shape_maps(maps[view]) for view in VIEW_ORDER}
        baseline = _load_tracks(root / BASELINE_TRACK_ROOT / object_id / "vggt_track_evidence.npz")
        tracks = _load_tracks(root / HYBRID_TRACK_ROOT / object_id / "vggt_track_evidence.npz")
        prefix_matches = bool(np.allclose(
            tracks["query_points_xy"][:, :384], baseline["query_points_xy"]
        ))
        if not prefix_matches:
            raise RuntimeError(f"{object_id} does not preserve baseline queries")
        evaluated = _preserve_baseline_gate(
            evaluate_track_consistency(tracks, evidence),
            evaluate_track_consistency(baseline, evidence),
        )
        by_view = {}
        for target_index, view in enumerate(VIEW_ORDER):
            if target_index == SOURCE_INDEX:
                continue
            by_view[view] = _evaluate_view(
                tracks["query_points_xy"][SOURCE_INDEX],
                tracks["tracks_xy"][SOURCE_INDEX, target_index],
                evaluated["accepted"][SOURCE_INDEX, target_index],
                maps[SOURCE_VIEW], maps[view],
                shapes[SOURCE_VIEW], shapes[view],
            )
        weights = {
            view: item["control_eligible_track_count"]
            for view, item in by_view.items()
            if item["control_eligible_track_count"] >= MINIMUM_POSITIVE_COUNT_FOR_VIEW
        }
        object_descriptors = {}
        for name in DESCRIPTOR_ORDER:
            weighted = [
                (by_view[view]["descriptors"][name]["paired_discrimination_auc"], weight)
                for view, weight in weights.items()
            ]
            object_descriptors[name] = {
                "paired_discrimination_auc": round(
                    sum(value * weight for value, weight in weighted)
                    / max(sum(weight for _, weight in weighted), 1), 6
                )
            }
        objects.append({
            "object_id": object_id, "split": "train", "target_views": by_view,
            "included_target_views": sorted(weights),
            "included_track_count": sum(weights.values()),
            "descriptors": object_descriptors,
            "baseline_query_prefix_reproduced": prefix_matches,
        })
    included = sum(item["included_track_count"] for item in objects)
    baseline_auc = np.asarray(
        [item["descriptors"][DESCRIPTOR_ORDER[0]]["paired_discrimination_auc"]
         for item in objects], np.float32
    )
    summaries = {}
    for name in DESCRIPTOR_ORDER:
        scores = np.asarray(
            [item["descriptors"][name]["paired_discrimination_auc"] for item in objects],
            np.float32,
        )
        delta = scores - baseline_auc
        summaries[name] = {
            "mean_object_auc": round(float(np.mean(scores)), 6),
            "mean_auc_delta_vs_signed_relief": round(float(np.mean(delta)), 6),
            "improved_object_count": int(np.count_nonzero(delta > 0)),
            "regressed_object_count": int(np.count_nonzero(delta < 0)),
            "maximum_object_auc_regression": round(float(max(-np.min(delta), 0.0)), 6),
        }
    baseline_reproduced = (
        included == FROZEN_INCLUDED_TRACK_COUNT
        and abs(summaries[DESCRIPTOR_ORDER[0]]["mean_object_auc"] - FROZEN_BASELINE_MEAN)
        <= 1e-6
    )
    if not baseline_reproduced:
        raise RuntimeError("Phase 5.17 matched-control baseline did not reproduce")
    candidates = [
        name for name in DESCRIPTOR_ORDER[1:]
        if summaries[name]["mean_auc_delta_vs_signed_relief"]
        >= SELECTION_MINIMUM_MEAN_AUC_DELTA
        and summaries[name]["improved_object_count"]
        >= SELECTION_MINIMUM_IMPROVED_OBJECT_COUNT
        and summaries[name]["maximum_object_auc_regression"]
        <= SELECTION_MAXIMUM_OBJECT_AUC_REGRESSION
    ]
    selected = max(candidates, key=lambda name: summaries[name]["mean_object_auc"]) if candidates else "none"
    audit = output / "training_local_shape_audit.png"
    report = {
        "schema_version": "lalitha_phase5_training_local_shape_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "Phase 5.19 training local-shape descriptor experiment",
        "scope": {"split": "train", "object_ids": list(TRAIN_OBJECTS),
                  "source_view": SOURCE_VIEW, "target_views": list(TARGET_VIEWS)},
        "policy": {
            "fixed_gaussian_scales_image_pixels": list(SIGMAS_IMAGE_PIXELS),
            "descriptor_weights": {name: list(values) for name, values in DESCRIPTOR_WEIGHTS.items()},
            "matched_control_policy": "unchanged Phase 5.17 target offsets, support, confidence and photo-detail matching",
            "minimum_positive_count_for_view": MINIMUM_POSITIVE_COUNT_FOR_VIEW,
            "selection_minimum_mean_auc_delta": SELECTION_MINIMUM_MEAN_AUC_DELTA,
            "selection_minimum_improved_object_count": SELECTION_MINIMUM_IMPROVED_OBJECT_COUNT,
            "selection_maximum_object_auc_regression": SELECTION_MAXIMUM_OBJECT_AUC_REGRESSION,
        },
        "summary": {
            "object_count": len(objects), "included_track_count": included,
            "target_view_eligible_track_count": {
                view: sum(item["target_views"][view]["control_eligible_track_count"]
                          for item in objects) for view in TARGET_VIEWS
            },
            "descriptors": summaries,
            "selected_descriptor": selected,
        },
        "checks": {
            "training_objects_only": all(item["split"] == "train" for item in objects),
            "validation_and_test_objects_not_processed": True,
            "baseline_query_prefix_reproduced": all(
                item["baseline_query_prefix_reproduced"] for item in objects
            ),
            "phase5_17_matched_control_baseline_reproduced": baseline_reproduced,
            "phase5_13_surface_gate_unchanged": True,
        },
        "objects": objects,
        "decision": (
            "freeze_selected_descriptor_for_one_validation_discrimination_test"
            if selected != "none" else "reject_all_local_shape_candidates_on_training"
        ),
        "claims": {
            "training_descriptor_discrimination_measured": True,
            "validation_relief_improved": False,
            "physical_relief_validated": False,
            "surface_promotion_authorized": False,
            "cad_fusion_authorized": False,
        },
        "limitations": [
            "Accepted VGGT tracks are machine hypotheses, not physical correspondences.",
            "The retained field name paired_discrimination_auc denotes a paired win fraction, not a full ROC AUC.",
            "Gaussian patch pooling is rotation-tolerant, not perspective-invariant or reflection-proof.",
            "Only two top-view tracks have eligible controls; this experiment cannot settle top-view matching.",
            "The same training cohort selects among three fixed candidates; validation and test remain untouched.",
            "Discrimination does not measure transferred 3D relief or CAD accuracy.",
        ],
        "artifacts": {"audit": _relative(audit, root)},
    }
    _render_audit(report, audit)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("data/phase5_training_local_shape_v1"))
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    report = run(root, output)
    print(json.dumps({"summary": report["summary"], "decision": report["decision"]}, indent=2))


if __name__ == "__main__":
    main()
