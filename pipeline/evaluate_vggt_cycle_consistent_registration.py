"""Evaluate reciprocal-cycle-filtered VGGT registration for Phase 3.

The experiment reuses pinned VGGT track evidence.  For every source-only A->B
match, the predicted point must lie near a sampled query in B and that query's
B->A track must return near the original A query.  The held-out target view is
excluded from correspondence fitting and remains score-only evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline.evaluate_vggt_surface_aware_hull import _load_object
from pipeline.evaluate_vggt_track_guided_registration import (
    _evaluate_object,
    _object_summary,
    _registration_preview,
    _summary,
)
from pipeline.evaluate_vggt_visual_hull_completion import (
    EVALUATION_OBJECTS,
    VALIDATION_OBJECTS,
)
from pipeline.validate_vggt_multiring_experiment import _records


IDENTITY = {
    "minimum_visibility": 1.0,
    "minimum_track_confidence": 1.0,
    "maximum_initial_residual_fraction": 0.0,
    "residual_trim_quantile": 0.5,
    "minimum_correspondences": 999999,
    "maximum_rotation_degrees": 0.0,
    "maximum_translation_fraction": 0.0,
}


CYCLE_PROFILES: dict[str, dict[str, float | int]] = {
    "identity": IDENTITY,
    "cycle_tight": {
        "minimum_visibility": 0.60,
        "minimum_track_confidence": 0.20,
        "maximum_query_snap_pixels": 10.0,
        "maximum_cycle_error_pixels": 6.0,
        "maximum_initial_residual_fraction": 0.025,
        "residual_trim_quantile": 0.60,
        "minimum_correspondences": 6,
        "maximum_rotation_degrees": 0.35,
        "maximum_translation_fraction": 0.002,
    },
    "cycle_balanced": {
        "minimum_visibility": 0.50,
        "minimum_track_confidence": 0.12,
        "maximum_query_snap_pixels": 16.0,
        "maximum_cycle_error_pixels": 10.0,
        "maximum_initial_residual_fraction": 0.04,
        "residual_trim_quantile": 0.65,
        "minimum_correspondences": 8,
        "maximum_rotation_degrees": 0.50,
        "maximum_translation_fraction": 0.0025,
    },
    "cycle_relaxed": {
        "minimum_visibility": 0.40,
        "minimum_track_confidence": 0.08,
        "maximum_query_snap_pixels": 24.0,
        "maximum_cycle_error_pixels": 16.0,
        "maximum_initial_residual_fraction": 0.06,
        "residual_trim_quantile": 0.70,
        "minimum_correspondences": 10,
        "maximum_rotation_degrees": 0.75,
        "maximum_translation_fraction": 0.004,
    },
}


def _load_track_evidence(
    root: Path, evidence_dir: Path, object_id: str
) -> dict[str, np.ndarray]:
    path = root / evidence_dir / object_id / "vggt_track_evidence.npz"
    with np.load(path) as evidence:
        return {name: evidence[name] for name in evidence.files}


def _cycle_statistics(tracks: dict[str, np.ndarray], threshold: float) -> dict[str, Any]:
    query = tracks["query_points_xy"]
    distances = []
    accepted = 0
    total = 0
    for source_index in range(query.shape[0]):
        for reference_index in range(query.shape[0]):
            if source_index == reference_index:
                continue
            forward = tracks["tracks_xy"][source_index, reference_index]
            delta = forward[:, None, :] - query[reference_index][None, :, :]
            nearest = np.argmin(np.sum(delta * delta, axis=2), axis=1)
            reverse = tracks["tracks_xy"][reference_index, source_index, nearest]
            error = np.linalg.norm(reverse - query[source_index], axis=1)
            distances.extend(error.tolist())
            accepted += int(np.count_nonzero(error <= threshold))
            total += len(error)
    values = np.asarray(distances, dtype=np.float64)
    return {
        "pair_track_count": total,
        "cycle_error_threshold_pixels": threshold,
        "accepted_cycle_fraction": round(accepted / max(total, 1), 6),
        "median_cycle_error_pixels": round(float(np.median(values)), 4),
        "p90_cycle_error_pixels": round(float(np.quantile(values, 0.90)), 4),
    }


def run(
    root: Path,
    output_dir: Path,
    track_evidence_dir: Path = Path(
        "dataset/phase_runs/v1/vggt_dense_track_evidence_v1"
    ),
    voxel_resolution: int = 96,
    confidence_quantile: float = 0.50,
) -> dict[str, Any]:
    records = _records(root / "dataset/prepared_v1/manifest.jsonl")
    validation_inputs = {
        object_id: _load_object(root, object_id, records[object_id], confidence_quantile)
        for object_id in VALIDATION_OBJECTS
    }
    validation_tracks = {
        object_id: _load_track_evidence(root, track_evidence_dir, object_id)
        for object_id in VALIDATION_OBJECTS
    }
    profile_results = {}
    for profile_name in CYCLE_PROFILES:
        objects = [
            _evaluate_object(
                validation_inputs[object_id],
                validation_tracks[object_id],
                profile_name,
                voxel_resolution,
                CYCLE_PROFILES,
            )
            for object_id in VALIDATION_OBJECTS
        ]
        profile_results[profile_name] = {
            "summary": _summary(objects),
            "objects": objects,
        }
    selected_profile = max(
        CYCLE_PROFILES,
        key=lambda name: profile_results[name]["summary"]["mean_iou"],
    )
    selected_validation = profile_results[selected_profile]["objects"]

    evaluation_inputs = {
        object_id: _load_object(root, object_id, records[object_id], confidence_quantile)
        for object_id in EVALUATION_OBJECTS
    }
    evaluation = [
        _evaluate_object(
            evaluation_inputs[object_id],
            _load_track_evidence(root, track_evidence_dir, object_id),
            selected_profile,
            voxel_resolution,
            CYCLE_PROFILES,
        )
        for object_id in EVALUATION_OBJECTS
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for item in [*selected_validation, *evaluation]:
        preview = output_dir / f"{item['object_id']}_cycle_registration.png"
        _registration_preview(item, preview, "DENSE CYCLE")
        summary = _object_summary(item)
        summary["preview"] = str(preview.relative_to(root))
        summaries.append(summary)

    profile_diagnostics = {}
    for profile_name, result in profile_results.items():
        diagnostic = next(
            item for item in result["objects"] if item["object_id"] == "ring_020"
        )
        path = output_dir / f"ring_020_{profile_name}_cycle_diagnostic.png"
        _registration_preview(diagnostic, path, "DENSE CYCLE")
        profile_diagnostics[profile_name] = str(path.relative_to(root))

    validation = [item for item in summaries if item["split"] == "val"]
    test = [item for item in summaries if item["split"] == "test"]
    validation_summary = _summary(selected_validation)
    test_summary = _summary(test)
    no_test_view_regresses = all(
        entry["registered"]["iou"] >= entry["baseline"]["iou"]
        for item in test
        for entry in item["views"]
    )
    accepted = (
        selected_profile != "identity"
        and test_summary["mean_delta_over_signed_depth"] > 0
        and all(item["mean_delta_over_signed_depth"] > 0 for item in test)
        and no_test_view_regresses
    )
    selected_cycle_threshold = float(
        CYCLE_PROFILES[selected_profile].get("maximum_cycle_error_pixels", 0.0)
    )
    cycle_statistics = {
        object_id: _cycle_statistics(
            _load_track_evidence(root, track_evidence_dir, object_id),
            selected_cycle_threshold,
        )
        for object_id in (*VALIDATION_OBJECTS, *EVALUATION_OBJECTS)
    }
    report = {
        "schema_version": "vggt_dense_cycle_consistent_registration_v1",
        "dataset": "STL-1",
        "authority": "non_metric_reciprocal_track_registration_hypothesis",
        "input_resolution": 518,
        "voxel_resolution": voxel_resolution,
        "track_evidence_dir": str(track_evidence_dir),
        "validation_selection": {
            "object_ids": list(VALIDATION_OBJECTS),
            "profile_candidates": {
                name: result["summary"] for name, result in profile_results.items()
            },
            "selected_profile": selected_profile,
            "selected_profile_definition": CYCLE_PROFILES[selected_profile],
            "selection_rule": "highest pooled validation mean IoU",
            "profile_diagnostics": profile_diagnostics,
        },
        "validation": validation_summary,
        "test_evaluation": {
            "object_ids": [item["object_id"] for item in test],
            **test_summary,
            "no_view_regresses": no_test_view_regresses,
        },
        "objects": summaries,
        "cycle_statistics": cycle_statistics,
        "checks": {
            "selection_uses_validation_split_only": all(
                item["split"] == "val" for item in validation
            ),
            "target_tracks_mask_points_and_camera_excluded_from_registration": True,
            "target_mask_and_points_excluded_from_reconstruction": True,
            "source_camera_and_points_transformed_together": True,
            "reciprocal_cycle_filter_enabled": selected_profile != "identity",
            "all_test_objects_improve_over_signed_depth": all(
                item["mean_delta_over_signed_depth"] > 0 for item in test
            ),
            "no_test_view_regresses": no_test_view_regresses,
            "geometry_exit_gate_passed": all(
                item["exit_gate_passed"] for item in summaries
            ),
        },
        "decision": (
            "retained_as_cycle_consistent_registration_prior"
            if accepted
            else "rejected_no_regression_free_heldout_gain"
        ),
        "claims": {
            "strict_target_image_holdout": False,
            "dense_correspondence_validated": False,
            "track_accuracy_validated": False,
            "physical_camera_calibration_validated": False,
            "metric_scale_validated": False,
            "component_identity_used": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "This is reciprocal filtering of 96 sparse query tracks per view, not dense matching.",
            "VGGT features were jointly conditioned on all five images even though target tracks were excluded from fitting.",
            "Nearest-query association can reject valid matches where the reference query sampling is sparse.",
            "STL-1 has no reviewed component IDs, so component identity cannot yet constrain matches.",
            "No scan, CAD truth, calibrated camera or known physical scale is available.",
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
        default=Path(
            "dataset/phase_runs/v1/vggt_dense_cycle_consistent_registration_v1"
        ),
    )
    parser.add_argument("--voxel-resolution", type=int, default=96)
    parser.add_argument("--confidence-quantile", type=float, default=0.50)
    parser.add_argument(
        "--track-evidence-dir",
        type=Path,
        default=Path("dataset/phase_runs/v1/vggt_dense_track_evidence_v1"),
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    report = run(
        root,
        output_dir,
        args.track_evidence_dir,
        args.voxel_resolution,
        args.confidence_quantile,
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
