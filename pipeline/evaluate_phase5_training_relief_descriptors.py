"""Compare cross-view relief descriptors against nearby wrong locations.

This Phase 5.17 training experiment keeps the frozen Phase 5.7 reciprocal
correspondences. Each accepted front-to-target observation is compared against
deterministic displaced controls on the *same* target ring. Descriptor
discrimination is an image-evidence test, not a surface or CAD validation.
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

from pipeline.build_phase5_stl1_relief_evidence import (
    _metric_depth_records,
    _strength,
    build_relief_evidence,
)
from pipeline.evaluate_phase5_dense_relief_pilot import _preserve_baseline_gate
from pipeline.evaluate_phase5_relief_tracks import (
    _load_view_evidence,
    _metric3d_normal_records,
    _sample,
    evaluate_track_consistency,
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


DESCRIPTOR_ORDER = (
    "signed_relief_baseline",
    "unsigned_depth_detail",
    "photo_detail",
    "depth_photo_fusion",
)
CONTROL_OFFSETS_XY = np.asarray(
    (
        (16, 0), (-16, 0), (0, 16), (0, -16),
        (24, 0), (-24, 0), (0, 24), (0, -24),
        (17, 17), (-17, -17), (17, -17), (-17, 17),
    ),
    np.float32,
)
MINIMUM_CONTROL_COUNT = 2
MINIMUM_POSITIVE_COUNT_FOR_VIEW = 8
MINIMUM_CONTROL_RELIEF_CONFIDENCE = 0.05
MAXIMUM_CONTROL_PHOTO_DETAIL_DELTA = 0.15
SELECTION_MINIMUM_MEAN_AUC_DELTA = 0.02
SELECTION_MINIMUM_IMPROVED_OBJECT_COUNT = 12
SELECTION_MAXIMUM_OBJECT_AUC_REGRESSION = 0.05


def _descriptor_maps(evidence: dict[str, Any]) -> dict[str, np.ndarray]:
    support = np.asarray(evidence["support"], dtype=bool)
    relief = np.asarray(evidence["relief"], np.float32)
    selected = np.abs(relief[support])
    relief_scale = float(np.percentile(selected, 98.0)) if selected.size else 1.0
    signed = np.clip(relief / max(relief_scale, 1e-8), -1.0, 1.0)
    depth = _strength(evidence["signed_depth_residual"], support)
    photo = np.clip(np.asarray(evidence["photo_detail"], np.float32), 0.0, 1.0)
    fusion = np.sqrt(depth * photo).astype(np.float32)
    return {
        "signed_relief_baseline": signed.astype(np.float32),
        "unsigned_depth_detail": depth.astype(np.float32),
        "photo_detail": photo.astype(np.float32),
        "depth_photo_fusion": fusion,
    }


def _load_descriptors(
    root: Path,
    object_record: dict[str, Any],
    metric_depths: dict[str, tuple[Path, ...]],
) -> dict[str, dict[str, np.ndarray]]:
    object_id = object_record["object_id"]
    result = {}
    for view in VIEW_ORDER:
        with np.load(root / object_record["views"][view]["bundle"]) as bundle:
            evidence = build_relief_evidence(
                depth=bundle["relative_depth_regularized_f32"],
                uncertainty_u16=bundle["relative_depth_uncertainty_u16"],
                gradient=bundle["gradient_magnitude_f32"],
                local_contrast=bundle["local_contrast_f32"],
                jewelry_mask=bundle["mask_jewelry_u8"],
                opening_mask=bundle["mask_negative_space_proposal_u8"],
                metric_depths=tuple(
                    np.load(path)
                    for path in metric_depths.get(f"{object_id}:{view}", ())
                ),
            )
        result[view] = {
            "support": np.asarray(evidence["support"], dtype=bool),
            "confidence": np.asarray(evidence["confidence"], np.float32),
            **_descriptor_maps(evidence),
        }
    return result


def _controls(
    target_xy: np.ndarray,
    target_maps: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Keep displaced controls with comparable target detail and confidence."""
    displaced = target_xy[:, None, :] + CONTROL_OFFSETS_XY[None, :, :]
    inside = (
        (displaced[..., 0] >= 0)
        & (displaced[..., 0] <= 517)
        & (displaced[..., 1] >= 0)
        & (displaced[..., 1] <= 517)
    )
    positive_photo = _sample(target_maps["photo_detail"], target_xy)
    displaced_photo = _sample(target_maps["photo_detail"], displaced)
    valid = (
        inside
        & _sample(target_maps["support"], displaced)
        & (_sample(target_maps["confidence"], displaced)
           >= MINIMUM_CONTROL_RELIEF_CONFIDENCE)
        & (np.abs(displaced_photo - positive_photo[:, None])
           <= MAXIMUM_CONTROL_PHOTO_DETAIL_DELTA)
    )
    return displaced, valid


def _evaluate_view(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    accepted: np.ndarray,
    source_maps: dict[str, np.ndarray],
    target_maps: dict[str, np.ndarray],
) -> dict[str, Any]:
    accepted_indices = np.flatnonzero(accepted)
    if not len(accepted_indices):
        return {
            "accepted_track_count": 0,
            "control_eligible_track_count": 0,
            "descriptors": {},
        }
    positive_xy = target_xy[accepted_indices]
    displaced, control_valid = _controls(
        positive_xy, target_maps
    )
    eligible = np.count_nonzero(control_valid, axis=1) >= MINIMUM_CONTROL_COUNT
    source_points = source_xy[accepted_indices][eligible]
    positive_points = positive_xy[eligible]
    decoy_points = displaced[eligible]
    decoy_valid = control_valid[eligible]
    descriptor_metrics = {}
    for name in DESCRIPTOR_ORDER:
        if not len(source_points):
            descriptor_metrics[name] = {
                "positive_error": None,
                "control_error": None,
                "mean_margin": None,
                "paired_discrimination_auc": None,
                "positive_beats_control_count": 0,
            }
            continue
        source_values = _sample(source_maps[name], source_points)
        positive_values = _sample(target_maps[name], positive_points)
        decoy_values = _sample(target_maps[name], decoy_points)
        positive_error = np.abs(source_values - positive_values)
        decoy_error = np.abs(source_values[:, None] - decoy_values)
        control_error = np.sum(
            np.where(decoy_valid, decoy_error, 0.0), axis=1
        ) / np.maximum(np.count_nonzero(decoy_valid, axis=1), 1)
        ties = np.isclose(positive_error, control_error, atol=1e-6)
        wins = (positive_error < control_error) & ~ties
        auc = np.mean(wins.astype(np.float32) + 0.5 * ties.astype(np.float32))
        descriptor_metrics[name] = {
            "positive_error": round(float(np.mean(positive_error)), 6),
            "control_error": round(float(np.mean(control_error)), 6),
            "mean_margin": round(
                float(np.mean(control_error - positive_error)), 6
            ),
            "paired_discrimination_auc": round(float(auc), 6),
            "positive_beats_control_count": int(np.count_nonzero(wins)),
        }
    return {
        "accepted_track_count": int(len(accepted_indices)),
        "control_eligible_track_count": int(len(source_points)),
        "descriptors": descriptor_metrics,
    }


def _render_audit(report: dict[str, Any], output: Path) -> None:
    objects = report["objects"]
    names = [item["object_id"] for item in objects]
    positions = np.arange(len(objects))
    figure, axes = plt.subplots(2, 1, figsize=(17, 10), facecolor="#f4f4f4")
    width = 0.19
    colors = ("#9da2a8", "#4f82bb", "#b44978", "#3f9b65")
    for index, name in enumerate(DESCRIPTOR_ORDER):
        values = [item["descriptors"][name]["paired_discrimination_auc"] for item in objects]
        axes[0].bar(
            positions + (index - 1.5) * width, values, width,
            label=name.replace("_", " "), color=colors[index],
        )
    axes[0].axhline(0.5, color="#222222", linestyle="--", linewidth=1, label="chance")
    axes[0].set_ylim(0, 1)
    axes[0].set_xticks(positions, names, rotation=45, ha="right")
    axes[0].set_title("Paired positive-versus-displaced discrimination by training ring", loc="left")
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)
    means = [report["summary"]["descriptors"][name]["mean_object_auc"] for name in DESCRIPTOR_ORDER]
    axes[1].barh(
        np.arange(len(DESCRIPTOR_ORDER)), means, color=colors
    )
    axes[1].axvline(0.5, color="#222222", linestyle="--", linewidth=1)
    axes[1].set_yticks(
        np.arange(len(DESCRIPTOR_ORDER)),
        [name.replace("_", " ") for name in DESCRIPTOR_ORDER],
    )
    axes[1].set_xlim(0, 1)
    axes[1].set_title("Equal-weight mean across 18 training rings", loc="left")
    axes[1].grid(axis="x", alpha=0.2)
    figure.suptitle(
        "PHASE 5.17 - TRAINING CROSS-VIEW RELIEF DESCRIPTOR TEST",
        fontsize=17, fontweight="bold",
    )
    figure.text(
        0.5, 0.012,
        "Fixed reciprocal tracks and same-view displaced controls - no surface or CAD claim",
        ha="center", fontsize=10,
    )
    figure.tight_layout(rect=(0.02, 0.045, 0.98, 0.94))
    figure.savefig(output, dpi=165, bbox_inches="tight")
    plt.close(figure)


def run(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    phase1 = json.loads((root / PHASE1_REPORT).read_text(encoding="utf-8"))
    phase1_by_id = {item["object_id"]: item for item in phase1["objects"]}
    metric_depths = _metric_depth_records(root)
    direct_normals = _metric3d_normal_records(root)
    objects = []
    for object_id in TRAIN_OBJECTS:
        record = phase1_by_id[object_id]
        if record["split"] != "train":
            raise RuntimeError(f"{object_id} is not a training ring")
        view_evidence = _load_view_evidence(
            root, record, metric_depths, direct_normals
        )
        descriptors = _load_descriptors(root, record, metric_depths)
        baseline = _load_tracks(
            root / BASELINE_TRACK_ROOT / object_id / "vggt_track_evidence.npz"
        )
        tracks = _load_tracks(
            root / HYBRID_TRACK_ROOT / object_id / "vggt_track_evidence.npz"
        )
        baseline_gate = evaluate_track_consistency(baseline, view_evidence)
        evaluated = _preserve_baseline_gate(
            evaluate_track_consistency(tracks, view_evidence), baseline_gate
        )
        by_view = {}
        for target_index, target_view in enumerate(VIEW_ORDER):
            if target_index == SOURCE_INDEX:
                continue
            by_view[target_view] = _evaluate_view(
                tracks["query_points_xy"][SOURCE_INDEX],
                tracks["tracks_xy"][SOURCE_INDEX, target_index],
                evaluated["accepted"][SOURCE_INDEX, target_index],
                descriptors[SOURCE_VIEW],
                descriptors[target_view],
            )
        view_weights = {
            view: item["control_eligible_track_count"]
            for view, item in by_view.items()
            if item["control_eligible_track_count"] >= MINIMUM_POSITIVE_COUNT_FOR_VIEW
        }
        object_descriptors = {}
        for name in DESCRIPTOR_ORDER:
            weighted = [
                (by_view[view]["descriptors"][name]["paired_discrimination_auc"], weight)
                for view, weight in view_weights.items()
            ]
            object_descriptors[name] = {
                "paired_discrimination_auc": round(
                    sum(value * weight for value, weight in weighted)
                    / max(sum(weight for _, weight in weighted), 1),
                    6,
                ),
            }
        objects.append(
            {
                "object_id": object_id,
                "split": "train",
                "target_views": by_view,
                "included_target_views": sorted(view_weights),
                "included_track_count": sum(view_weights.values()),
                "descriptors": object_descriptors,
                "baseline_query_prefix_reproduced": bool(
                    np.allclose(
                        tracks["query_points_xy"][:, :384],
                        baseline["query_points_xy"],
                    )
                ),
            }
        )
    descriptor_summary = {}
    baseline_auc = np.asarray(
        [item["descriptors"][DESCRIPTOR_ORDER[0]]["paired_discrimination_auc"] for item in objects],
        np.float32,
    )
    for name in DESCRIPTOR_ORDER:
        auc = np.asarray(
            [item["descriptors"][name]["paired_discrimination_auc"] for item in objects],
            np.float32,
        )
        delta = auc - baseline_auc
        descriptor_summary[name] = {
            "mean_object_auc": round(float(np.mean(auc)), 6),
            "mean_auc_delta_vs_signed_relief": round(float(np.mean(delta)), 6),
            "improved_object_count": int(np.count_nonzero(delta > 0)),
            "regressed_object_count": int(np.count_nonzero(delta < 0)),
            "maximum_object_auc_regression": round(float(max(-np.min(delta), 0.0)), 6),
        }
    candidates = [
        name for name in DESCRIPTOR_ORDER[1:]
        if descriptor_summary[name]["mean_auc_delta_vs_signed_relief"]
        >= SELECTION_MINIMUM_MEAN_AUC_DELTA
        and descriptor_summary[name]["improved_object_count"]
        >= SELECTION_MINIMUM_IMPROVED_OBJECT_COUNT
        and descriptor_summary[name]["maximum_object_auc_regression"]
        <= SELECTION_MAXIMUM_OBJECT_AUC_REGRESSION
    ]
    selected = (
        max(candidates, key=lambda name: descriptor_summary[name]["mean_object_auc"])
        if candidates else "none"
    )
    audit_path = output / "training_relief_descriptor_audit.png"
    report = {
        "schema_version": "lalitha_phase5_training_relief_descriptors_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "Phase 5.17 training cross-view relief descriptor discrimination",
        "scope": {
            "object_ids": list(TRAIN_OBJECTS),
            "split": "train",
            "source_view": SOURCE_VIEW,
            "target_views": list(TARGET_VIEWS),
        },
        "policy": {
            "descriptor_order": list(DESCRIPTOR_ORDER),
            "control_offsets_xy": CONTROL_OFFSETS_XY.astype(int).tolist(),
            "minimum_control_count": MINIMUM_CONTROL_COUNT,
            "minimum_positive_count_for_view": MINIMUM_POSITIVE_COUNT_FOR_VIEW,
            "minimum_control_relief_confidence": MINIMUM_CONTROL_RELIEF_CONFIDENCE,
            "maximum_control_photo_detail_delta": MAXIMUM_CONTROL_PHOTO_DETAIL_DELTA,
            "selection_minimum_mean_auc_delta": SELECTION_MINIMUM_MEAN_AUC_DELTA,
            "selection_minimum_improved_object_count": SELECTION_MINIMUM_IMPROVED_OBJECT_COUNT,
            "selection_maximum_object_auc_regression": SELECTION_MAXIMUM_OBJECT_AUC_REGRESSION,
        },
        "summary": {
            "object_count": len(objects),
            "included_track_count": sum(item["included_track_count"] for item in objects),
            "target_view_eligible_track_count": {
                view: sum(item["target_views"][view]["control_eligible_track_count"] for item in objects)
                for view in TARGET_VIEWS
            },
            "descriptors": descriptor_summary,
            "selected_descriptor": selected,
        },
        "checks": {
            "training_objects_only": all(item["split"] == "train" for item in objects),
            "validation_objects_not_processed": True,
            "test_objects_not_processed": True,
            "baseline_query_prefix_reproduced": all(
                item["baseline_query_prefix_reproduced"] for item in objects
            ),
            "reciprocal_correspondences_unchanged": True,
            "phase5_13_surfaces_and_thresholds_unchanged": True,
            "no_step_files_generated": True,
        },
        "objects": objects,
        "decision": (
            "freeze_selected_descriptor_for_one_validation_discrimination_test"
            if selected != "none"
            else "reject_all_descriptor_candidates_and_research_new_representation_on_training"
        ),
        "claims": {
            "training_descriptor_discrimination_measured": True,
            "validation_relief_improved": False,
            "physical_relief_validated": False,
            "surface_promotion_authorized": False,
            "cad_fusion_authorized": False,
        },
        "limitations": [
            "Accepted VGGT tracks are machine hypotheses, not labelled physical correspondence truth.",
            "Nearby displaced target points are negative controls but can overlap repeated ornament motifs.",
            "Descriptor discrimination does not measure signed relief transfer or a 3D surface.",
            "Top-view evidence may be scarce because the front relief is foreshortened or occluded.",
        ],
        "artifacts": {"audit": _relative(audit_path, root)},
    }
    _render_audit(report, audit_path)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("data/phase5_training_relief_descriptors_v1"))
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    report = run(root, output)
    print(json.dumps({"summary": report["summary"], "decision": report["decision"]}, indent=2))


if __name__ == "__main__":
    main()
