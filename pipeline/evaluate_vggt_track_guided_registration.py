"""Evaluate VGGT-track-guided source registration before signed-depth carving.

Only tracks between the four source views for each held-out target are used.
Tracker visibility/confidence, world-point confidence and robust 3D residuals
filter correspondences.  Source point maps and cameras are transformed together.
The target mask, target points and target camera remain score-only evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from pipeline.evaluate_vggt_rigid_source_registration import (
    _bound_transform,
    _correct_extrinsic,
    _transform_points,
)
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


REGISTRATION_PROFILES = {
    "identity": {
        "minimum_visibility": 1.0,
        "minimum_track_confidence": 1.0,
        "maximum_initial_residual_fraction": 0.0,
        "residual_trim_quantile": 0.5,
        "minimum_correspondences": 999999,
        "maximum_rotation_degrees": 0.0,
        "maximum_translation_fraction": 0.0,
    },
    "track_strict": {
        "minimum_visibility": 0.75,
        "minimum_track_confidence": 0.35,
        "maximum_initial_residual_fraction": 0.02,
        "residual_trim_quantile": 0.60,
        "minimum_correspondences": 8,
        "maximum_rotation_degrees": 0.5,
        "maximum_translation_fraction": 0.0025,
    },
    "track_balanced": {
        "minimum_visibility": 0.60,
        "minimum_track_confidence": 0.20,
        "maximum_initial_residual_fraction": 0.04,
        "residual_trim_quantile": 0.70,
        "minimum_correspondences": 12,
        "maximum_rotation_degrees": 1.0,
        "maximum_translation_fraction": 0.005,
    },
    "track_permissive": {
        "minimum_visibility": 0.45,
        "minimum_track_confidence": 0.10,
        "maximum_initial_residual_fraction": 0.08,
        "residual_trim_quantile": 0.80,
        "minimum_correspondences": 16,
        "maximum_rotation_degrees": 2.0,
        "maximum_translation_fraction": 0.01,
    },
}


def _pixel_indices(points: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    rounded = np.rint(points).astype(np.int32)
    valid = np.isfinite(points).all(axis=1)
    valid &= (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < size)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < size)
    )
    return rounded, valid


def _weighted_rigid_fit(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.maximum(weights.astype(np.float64), 1e-8)
    weights /= np.sum(weights)
    source_centre = np.sum(source * weights[:, None], axis=0)
    target_centre = np.sum(target * weights[:, None], axis=0)
    source_offset = source - source_centre
    target_offset = target - target_centre
    covariance = source_offset.T @ (target_offset * weights[:, None])
    left, _, right = np.linalg.svd(covariance)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0:
        right[-1] *= -1
        rotation = right.T @ left.T
    translation = target_centre - rotation @ source_centre
    return rotation, translation


def _load_tracks(root: Path, object_id: str) -> dict[str, np.ndarray]:
    path = (
        root
        / "dataset/phase_runs/v1/vggt_track_evidence_v1"
        / object_id
        / "vggt_track_evidence.npz"
    )
    with np.load(path) as evidence:
        return {name: evidence[name] for name in evidence.files}


def _directed_correspondences(
    item: dict[str, Any],
    tracks: dict[str, np.ndarray],
    moving_index: int,
    reference_index: int,
    profile: dict[str, float | int],
    confidence_thresholds: list[float],
    diagonal: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = tracks["query_points_xy"][moving_index]
    predicted = tracks["tracks_xy"][moving_index, reference_index]
    visibility = tracks["visibility"][moving_index, reference_index]
    track_confidence = tracks["confidence"][moving_index, reference_index]
    query_pixels, query_valid = _pixel_indices(query, item["image_size"])
    predicted_pixels, predicted_valid = _pixel_indices(
        predicted, item["image_size"]
    )
    valid = query_valid & predicted_valid
    valid &= visibility >= float(profile["minimum_visibility"])
    valid &= track_confidence >= float(profile["minimum_track_confidence"])

    # Optional reciprocal-cycle guard.  The query sets are selected independently
    # in each view, so the forward prediction is first associated with its nearest
    # query in the reference view.  That reference query must then track back near
    # the original moving-view query.  Existing registration profiles omit these
    # keys and therefore retain their byte-for-byte filtering behaviour.
    maximum_query_snap = float(profile.get("maximum_query_snap_pixels", np.inf))
    maximum_cycle_error = float(profile.get("maximum_cycle_error_pixels", np.inf))
    if np.isfinite(maximum_query_snap) or np.isfinite(maximum_cycle_error):
        reference_queries = tracks["query_points_xy"][reference_index]
        difference = predicted[:, None, :] - reference_queries[None, :, :]
        nearest = np.argmin(np.sum(difference * difference, axis=2), axis=1)
        snap_error = np.linalg.norm(predicted - reference_queries[nearest], axis=1)
        reverse = tracks["tracks_xy"][reference_index, moving_index, nearest]
        reverse_visibility = tracks["visibility"][
            reference_index, moving_index, nearest
        ]
        reverse_confidence = tracks["confidence"][
            reference_index, moving_index, nearest
        ]
        cycle_error = np.linalg.norm(reverse - query, axis=1)
        valid &= snap_error <= maximum_query_snap
        valid &= cycle_error <= maximum_cycle_error
        valid &= reverse_visibility >= float(profile["minimum_visibility"])
        valid &= reverse_confidence >= float(profile["minimum_track_confidence"])
    indices = np.flatnonzero(valid)
    if not len(indices):
        return (
            np.empty((0, 3), np.float64),
            np.empty((0, 3), np.float64),
            np.empty(0, np.float64),
        )
    query_pixel = query_pixels[indices]
    predicted_pixel = predicted_pixels[indices]
    inside = item["masks"][moving_index][query_pixel[:, 1], query_pixel[:, 0]]
    inside &= item["masks"][reference_index][
        predicted_pixel[:, 1], predicted_pixel[:, 0]
    ]
    indices = indices[inside]
    query_pixel = query_pixels[indices]
    predicted_pixel = predicted_pixels[indices]
    source = item["world_points"][moving_index][
        query_pixel[:, 1], query_pixel[:, 0]
    ]
    target = item["world_points"][reference_index][
        predicted_pixel[:, 1], predicted_pixel[:, 0]
    ]
    source_confidence = item["world_points_conf"][moving_index][
        query_pixel[:, 1], query_pixel[:, 0]
    ]
    target_confidence = item["world_points_conf"][reference_index][
        predicted_pixel[:, 1], predicted_pixel[:, 0]
    ]
    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    finite &= np.isfinite(source_confidence) & np.isfinite(target_confidence)
    finite &= source_confidence >= confidence_thresholds[moving_index]
    finite &= target_confidence >= confidence_thresholds[reference_index]
    source = source[finite].astype(np.float64)
    target = target[finite].astype(np.float64)
    indices = indices[finite]
    source_confidence = source_confidence[finite]
    target_confidence = target_confidence[finite]
    residual = np.linalg.norm(source - target, axis=1)
    geometrically_consistent = residual <= diagonal * float(
        profile["maximum_initial_residual_fraction"]
    )
    source = source[geometrically_consistent]
    target = target[geometrically_consistent]
    indices = indices[geometrically_consistent]
    source_confidence = source_confidence[geometrically_consistent]
    target_confidence = target_confidence[geometrically_consistent]
    weights = (
        visibility[indices]
        * track_confidence[indices]
        * np.sqrt(np.maximum(source_confidence * target_confidence, 0))
    )
    if np.isfinite(maximum_query_snap) or np.isfinite(maximum_cycle_error):
        weights *= np.sqrt(
            np.maximum(
                tracks["visibility"][reference_index, moving_index, nearest[indices]]
                * tracks["confidence"][
                    reference_index, moving_index, nearest[indices]
                ],
                0,
            )
        )
    return source, target, weights.astype(np.float64)


def _register_sources(
    item: dict[str, Any],
    tracks: dict[str, np.ndarray],
    target_index: int,
    profile_name: str,
    profiles: dict[str, dict[str, float | int]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    profile = (profiles or REGISTRATION_PROFILES)[profile_name]
    source_indices = [index for index in range(len(VIEW_ORDER)) if index != target_index]
    source_points = np.concatenate(
        [item["candidates"][index] for index in source_indices]
    )
    lower, upper = np.percentile(source_points, (1, 99), axis=0)
    diagonal = float(np.linalg.norm(upper - lower))
    confidence_thresholds = []
    for index in range(len(VIEW_ORDER)):
        confidence = item["world_points_conf"][index]
        values = confidence[item["masks"][index] & np.isfinite(confidence)]
        confidence_thresholds.append(float(np.quantile(values, 0.50)))

    rotations = {index: np.eye(3) for index in source_indices}
    translations = {index: np.zeros(3) for index in source_indices}
    match_records = []
    for moving_index in source_indices:
        source_parts = []
        target_parts = []
        weight_parts = []
        pair_counts = {}
        for reference_index in source_indices:
            if moving_index == reference_index:
                continue
            source, target, weights = _directed_correspondences(
                item,
                tracks,
                moving_index,
                reference_index,
                profile,
                confidence_thresholds,
                diagonal,
            )
            pair_counts[VIEW_ORDER[reference_index]] = int(len(source))
            if len(source):
                source_parts.append(source)
                target_parts.append(target)
                weight_parts.append(weights)
        count = int(sum(len(part) for part in source_parts))
        inliers = 0
        before_median = None
        after_median = None
        if count >= int(profile["minimum_correspondences"]):
            source = np.concatenate(source_parts)
            target = np.concatenate(target_parts)
            weights = np.concatenate(weight_parts)
            before = np.linalg.norm(source - target, axis=1)
            rotation, translation = _weighted_rigid_fit(source, target, weights)
            transformed = _transform_points(source, rotation, translation)
            residual = np.linalg.norm(transformed - target, axis=1)
            trim = residual <= np.quantile(
                residual, float(profile["residual_trim_quantile"])
            )
            if int(np.count_nonzero(trim)) >= int(profile["minimum_correspondences"]):
                rotation, translation = _weighted_rigid_fit(
                    source[trim], target[trim], weights[trim]
                )
                inliers = int(np.count_nonzero(trim))
            rotation, translation = _bound_transform(
                rotation,
                translation,
                float(profile["maximum_rotation_degrees"]),
                diagonal * float(profile["maximum_translation_fraction"]),
            )
            after = np.linalg.norm(
                _transform_points(source, rotation, translation) - target,
                axis=1,
            )
            rotations[moving_index] = rotation
            translations[moving_index] = translation
            before_median = round(float(np.median(before)) / diagonal, 8)
            after_median = round(float(np.median(after)) / diagonal, 8)
        match_records.append(
            {
                "view": VIEW_ORDER[moving_index],
                "pair_correspondences": pair_counts,
                "accepted_correspondences": count,
                "robust_inliers": inliers,
                "median_residual_before_fraction": before_median,
                "median_residual_after_fraction": after_median,
                "rotation_degrees": round(
                    float(
                        np.degrees(
                            np.linalg.norm(
                                Rotation.from_matrix(
                                    rotations[moving_index]
                                ).as_rotvec()
                            )
                        )
                    ),
                    6,
                ),
                "translation_fraction": round(
                    float(np.linalg.norm(translations[moving_index]))
                    / max(diagonal, 1e-12),
                    8,
                ),
            }
        )

    corrected_candidates = [points.copy() for points in item["candidates"]]
    corrected_world = item["world_points"].copy()
    corrected_extrinsic = item["extrinsic"].copy()
    for source_index in source_indices:
        rotation = rotations[source_index]
        translation = translations[source_index]
        corrected_candidates[source_index] = _transform_points(
            item["candidates"][source_index], rotation, translation
        ).astype(np.float32)
        corrected_world[source_index] = _transform_points(
            item["world_points"][source_index], rotation, translation
        ).astype(np.float32)
        corrected_extrinsic[source_index] = _correct_extrinsic(
            item["extrinsic"][source_index], rotation, translation
        ).astype(np.float32)
    return {
        **item,
        "candidates": corrected_candidates,
        "world_points": corrected_world,
        "extrinsic": corrected_extrinsic,
    }, {
        "profile": profile_name,
        "matches": match_records,
        "total_accepted_correspondences": int(
            sum(record["accepted_correspondences"] for record in match_records)
        ),
        "total_robust_inliers": int(
            sum(record["robust_inliers"] for record in match_records)
        ),
    }


def _evaluate_object(
    item: dict[str, Any],
    tracks: dict[str, np.ndarray],
    profile_name: str,
    voxel_resolution: int,
    profiles: dict[str, dict[str, float | int]] | None = None,
) -> dict[str, Any]:
    views = []
    baseline_projections = []
    registered_projections = []
    for target_index, view in enumerate(VIEW_ORDER):
        registered_item, registration = _register_sources(
            item, tracks, target_index, profile_name, profiles
        )
        baseline_voxels = _signed_depth_voxels(
            item, target_index, item["intrinsic"], voxel_resolution
        )
        registered_voxels = _signed_depth_voxels(
            registered_item,
            target_index,
            registered_item["intrinsic"],
            voxel_resolution,
        )
        target = np.uint8(item["masks"][target_index]) * 255
        baseline_projection = _project(
            baseline_voxels,
            item["extrinsic"][target_index],
            item["intrinsic"][target_index],
            item["image_size"],
        )
        registered_projection = _project(
            registered_voxels,
            item["extrinsic"][target_index],
            item["intrinsic"][target_index],
            item["image_size"],
        )
        baseline_projections.append(baseline_projection)
        registered_projections.append(registered_projection)
        views.append(
            {
                "view": view,
                "baseline": _metrics(baseline_projection, target),
                "registered": _metrics(registered_projection, target),
                "registration": registration,
                "baseline_voxels": int(len(baseline_voxels)),
                "registered_voxels": int(len(registered_voxels)),
            }
        )
    baseline_mean = float(np.mean([entry["baseline"]["iou"] for entry in views]))
    registered_mean = float(
        np.mean([entry["registered"]["iou"] for entry in views])
    )
    return {
        **item,
        "profile": profile_name,
        "views": views,
        "baseline_projections": baseline_projections,
        "registered_projections": registered_projections,
        "baseline_mean_iou": round(baseline_mean, 6),
        "mean_iou": round(registered_mean, 6),
        "minimum_iou": round(
            float(np.min([entry["registered"]["iou"] for entry in views])), 6
        ),
        "mean_precision": round(
            float(np.mean([entry["registered"]["precision"] for entry in views])),
            6,
        ),
        "mean_recall": round(
            float(np.mean([entry["registered"]["recall"] for entry in views])), 6
        ),
        "mean_delta_over_signed_depth": round(registered_mean - baseline_mean, 6),
        "mean_correspondence_count": round(
            float(
                np.mean(
                    [
                        entry["registration"]["total_accepted_correspondences"]
                        for entry in views
                    ]
                )
            ),
            3,
        ),
    }


def _summary(items: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "mean_iou": round(float(np.mean([item["mean_iou"] for item in items])), 6),
        "minimum_iou": round(
            float(np.min([item["minimum_iou"] for item in items])), 6
        ),
        "mean_precision": round(
            float(np.mean([item["mean_precision"] for item in items])), 6
        ),
        "mean_recall": round(
            float(np.mean([item["mean_recall"] for item in items])), 6
        ),
        "mean_delta_over_signed_depth": round(
            float(np.mean([item["mean_delta_over_signed_depth"] for item in items])),
            6,
        ),
        "mean_correspondence_count": round(
            float(np.mean([item["mean_correspondence_count"] for item in items])),
            3,
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
        "mean_correspondence_count": item["mean_correspondence_count"],
        "views": item["views"],
        "exit_gate_passed": item["mean_iou"] >= 0.80
        and item["minimum_iou"] >= 0.70,
    }


def _registration_preview(
    item: dict[str, Any], output: Path, method_label: str = "TRACK-GUIDED"
) -> None:
    rows = []
    for index, entry in enumerate(item["views"]):
        target = np.uint8(item["masks"][index]) * 255
        baseline = item["baseline_projections"][index]
        registered = item["registered_projections"][index]
        rows.append(
            np.hstack(
                (
                    _label(item["sources"][index], f"{entry['view'].upper()} SOURCE"),
                    _label(
                        cv2.cvtColor(target, cv2.COLOR_GRAY2BGR),
                        "TARGET MASK",
                        "score only",
                    ),
                    _label(
                        cv2.cvtColor(baseline, cv2.COLOR_GRAY2BGR),
                        "SIGNED DEPTH",
                        "original source geometry",
                    ),
                    _label(
                        cv2.cvtColor(registered, cv2.COLOR_GRAY2BGR),
                        method_label,
                        item["profile"],
                    ),
                    _label(
                        _agreement(item["sources"][index], registered, target),
                        "AGREEMENT",
                        f"IoU {entry['registered']['iou']:.3f} / green match",
                    ),
                )
            )
        )
    if not cv2.imwrite(str(output), np.vstack(rows)):
        raise RuntimeError(f"Could not write {output}")


def _track_pair_preview(
    source: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    query: np.ndarray,
    predicted: np.ndarray,
    visibility: np.ndarray,
    confidence: np.ndarray,
    label: str,
) -> np.ndarray:
    size = source.shape[0]
    canvas = np.hstack((source.copy(), target.copy()))
    pixels, valid = _pixel_indices(predicted, size)
    valid_indices = np.flatnonzero(valid)
    valid_indices = valid_indices[
        mask[pixels[valid_indices, 1], pixels[valid_indices, 0]]
    ]
    score = visibility[valid_indices] * confidence[valid_indices]
    order = valid_indices[np.argsort(score)[::-1][:40]]
    for rank, index in enumerate(order):
        colour = (40, int(220 - rank * 3), 255)
        start = tuple(np.rint(query[index]).astype(int))
        end_pixel = tuple(pixels[index] + np.asarray((size, 0)))
        cv2.circle(canvas, start, 3, colour, -1, cv2.LINE_AA)
        cv2.circle(canvas, end_pixel, 3, colour, -1, cv2.LINE_AA)
        cv2.line(canvas, start, end_pixel, colour, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (245, 245, 245), -1)
    cv2.putText(
        canvas,
        label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _track_preview(
    item: dict[str, Any], tracks: dict[str, np.ndarray], output: Path
) -> None:
    pairs = ((0, 2), (1, 3), (1, 4))
    rows = []
    for source_index, target_index in pairs:
        rows.append(
            _track_pair_preview(
                item["sources"][source_index],
                item["sources"][target_index],
                item["masks"][target_index],
                tracks["query_points_xy"][source_index],
                tracks["tracks_xy"][source_index, target_index],
                tracks["visibility"][source_index, target_index],
                tracks["confidence"][source_index, target_index],
                f"{VIEW_ORDER[source_index].upper()} TO {VIEW_ORDER[target_index].upper()} / TOP TRACK SCORES",
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
    validation_tracks = {
        object_id: _load_tracks(root, object_id) for object_id in VALIDATION_OBJECTS
    }
    profile_results = {}
    for profile_name in REGISTRATION_PROFILES:
        evaluated = [
            _evaluate_object(
                validation_inputs[object_id],
                validation_tracks[object_id],
                profile_name,
                voxel_resolution,
            )
            for object_id in VALIDATION_OBJECTS
        ]
        profile_results[profile_name] = {
            "summary": _summary(evaluated),
            "objects": evaluated,
        }
    selected_profile = max(
        REGISTRATION_PROFILES,
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
        _evaluate_object(
            evaluation_inputs[object_id],
            _load_tracks(root, object_id),
            selected_profile,
            voxel_resolution,
        )
        for object_id in EVALUATION_OBJECTS
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_diagnostics = {}
    for profile_name, result in profile_results.items():
        diagnostic = next(
            item for item in result["objects"] if item["object_id"] == "ring_020"
        )
        path = output_dir / f"ring_020_{profile_name}_track_registration.png"
        _registration_preview(diagnostic, path)
        profile_diagnostics[profile_name] = str(path.relative_to(root))
    summaries = []
    for item in [*selected_validation, *evaluation]:
        preview = output_dir / f"{item['object_id']}_track_guided_registration.png"
        _registration_preview(item, preview)
        summary = _object_summary(item)
        summary["preview"] = str(preview.relative_to(root))
        summaries.append(summary)
    track_previews = {}
    for object_id, item in (
        ("ring_020", validation_inputs["ring_020"]),
        ("ring_015", evaluation_inputs["ring_015"]),
    ):
        path = output_dir / f"{object_id}_track_evidence.png"
        _track_preview(item, _load_tracks(root, object_id), path)
        track_previews[object_id] = str(path.relative_to(root))
    validation = [item for item in summaries if item["split"] == "val"]
    test = [item for item in summaries if item["split"] == "test"]
    test_summary = _summary(test)
    accepted = (
        selected_profile != "identity"
        and test_summary["mean_delta_over_signed_depth"] > 0
        and all(item["mean_delta_over_signed_depth"] > 0 for item in test)
    )
    report = {
        "schema_version": "vggt_track_guided_registration_v1",
        "dataset": "STL-1",
        "authority": "non_metric_learned_track_registration_hypothesis",
        "input_resolution": 518,
        "voxel_resolution": voxel_resolution,
        "validation_selection": {
            "object_ids": list(VALIDATION_OBJECTS),
            "profile_candidates": {
                name: result["summary"] for name, result in profile_results.items()
            },
            "selected_profile": selected_profile,
            "selected_profile_definition": REGISTRATION_PROFILES[selected_profile],
            "selection_rule": "highest pooled validation mean IoU",
            "profile_diagnostics": profile_diagnostics,
        },
        "validation": _summary(selected_validation),
        "test_evaluation": {
            "object_ids": [item["object_id"] for item in test],
            **test_summary,
        },
        "objects": summaries,
        "track_previews": track_previews,
        "checks": {
            "selection_uses_validation_split_only": all(
                item["split"] == "val" for item in validation
            ),
            "target_tracks_mask_points_and_camera_excluded_from_registration": True,
            "target_mask_and_points_excluded_from_reconstruction": True,
            "source_camera_and_points_transformed_together": True,
            "all_test_objects_improve_over_signed_depth": all(
                item["mean_delta_over_signed_depth"] > 0 for item in test
            ),
            "geometry_exit_gate_passed": all(
                item["exit_gate_passed"] for item in summaries
            ),
        },
        "decision": (
            "retained_as_track_guided_registration_prior"
            if accepted
            else "rejected_no_consistent_heldout_gain"
        ),
        "claims": {
            "strict_target_image_holdout": False,
            "track_accuracy_validated": False,
            "physical_camera_calibration_validated": False,
            "metric_scale_validated": False,
            "component_identity_used": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "Track features were jointly conditioned on all five images even though target tracks were excluded from fitting.",
            "STL-1 has no reviewed component IDs, so component identity cannot yet constrain matches.",
            "Tracker confidence and low 3D residual do not prove physical correspondence on repeated ornament.",
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
        default=Path("dataset/phase_runs/v1/vggt_track_guided_registration_v1"),
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
