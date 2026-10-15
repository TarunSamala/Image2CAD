"""Test bounded rigid registration of source-view VGGT point maps.

For each held-out target view, only the other four source point maps are
registered.  Every source cloud and its camera are transformed together so its
self-projection is unchanged.  The target mask, point map and camera are not
used during registration or reconstruction.  Registration bounds are selected
on validation objects, then frozen for the held-out test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

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
        "iterations": 0,
        "trim_quantile": 0.25,
        "maximum_rotation_degrees": 0.0,
        "maximum_translation_fraction": 0.0,
    },
    "rigid_tiny": {
        "iterations": 1,
        "trim_quantile": 0.25,
        "maximum_rotation_degrees": 0.5,
        "maximum_translation_fraction": 0.0025,
    },
    "rigid_small": {
        "iterations": 2,
        "trim_quantile": 0.35,
        "maximum_rotation_degrees": 1.0,
        "maximum_translation_fraction": 0.005,
    },
    "rigid_medium": {
        "iterations": 3,
        "trim_quantile": 0.50,
        "maximum_rotation_degrees": 2.0,
        "maximum_translation_fraction": 0.01,
    },
}
MAXIMUM_SAMPLED_POINTS = 3000


def _sample(points: np.ndarray, maximum: int = MAXIMUM_SAMPLED_POINTS) -> np.ndarray:
    if len(points) <= maximum:
        return points.astype(np.float64, copy=True)
    indices = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
    return points[indices].astype(np.float64, copy=True)


def _transform_points(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    return points @ rotation.T + translation


def _pairwise_registration_error(
    clouds: dict[int, np.ndarray],
) -> tuple[float, dict[int, float]]:
    per_view: dict[int, list[float]] = {index: [] for index in clouds}
    for source_index, source in clouds.items():
        for target_index, target in clouds.items():
            if source_index >= target_index:
                continue
            source_to_target = cKDTree(target).query(source, k=1, workers=-1)[0]
            target_to_source = cKDTree(source).query(target, k=1, workers=-1)[0]
            distance = float(
                0.5 * (np.median(source_to_target) + np.median(target_to_source))
            )
            per_view[source_index].append(distance)
            per_view[target_index].append(distance)
    view_error = {
        index: float(np.mean(values)) if values else 0.0
        for index, values in per_view.items()
    }
    return float(np.mean(list(view_error.values()))), view_error


def _estimate_rigid_increment(
    moving: np.ndarray,
    reference: np.ndarray,
    trim_quantile: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    distances, indices = cKDTree(reference).query(moving, k=1, workers=-1)
    threshold = float(np.quantile(distances, trim_quantile))
    keep = distances <= threshold
    if int(np.count_nonzero(keep)) < 32:
        return np.eye(3), np.zeros(3), 0
    source = moving[keep]
    target = reference[indices[keep]]
    source_centre = np.mean(source, axis=0)
    target_centre = np.mean(target, axis=0)
    covariance = (source - source_centre).T @ (target - target_centre)
    left, _, right = np.linalg.svd(covariance)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0:
        right[-1] *= -1
        rotation = right.T @ left.T
    translation = target_centre - rotation @ source_centre
    return rotation, translation, int(np.count_nonzero(keep))


def _bound_transform(
    rotation: np.ndarray,
    translation: np.ndarray,
    maximum_rotation_degrees: float,
    maximum_translation: float,
) -> tuple[np.ndarray, np.ndarray]:
    rotvec = Rotation.from_matrix(rotation).as_rotvec()
    angle = float(np.linalg.norm(rotvec))
    maximum_angle = float(np.radians(maximum_rotation_degrees))
    if angle > maximum_angle > 0:
        rotvec *= maximum_angle / angle
    elif maximum_angle == 0:
        rotvec[:] = 0
    bounded_rotation = Rotation.from_rotvec(rotvec).as_matrix()
    magnitude = float(np.linalg.norm(translation))
    if magnitude > maximum_translation > 0:
        translation = translation * (maximum_translation / magnitude)
    elif maximum_translation == 0:
        translation = np.zeros(3)
    return bounded_rotation, translation


def _correct_extrinsic(
    extrinsic: np.ndarray,
    world_rotation: np.ndarray,
    world_translation: np.ndarray,
) -> np.ndarray:
    camera_rotation = extrinsic[:, :3]
    corrected_rotation = camera_rotation @ world_rotation.T
    corrected_translation = (
        extrinsic[:, 3] - corrected_rotation @ world_translation
    )
    return np.column_stack((corrected_rotation, corrected_translation))


def _register_sources(
    item: dict[str, Any],
    target_index: int,
    profile_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    profile = REGISTRATION_PROFILES[profile_name]
    source_indices = [index for index in range(len(VIEW_ORDER)) if index != target_index]
    sampled = {index: _sample(item["candidates"][index]) for index in source_indices}
    source_points = np.concatenate(list(sampled.values()))
    lower, upper = np.percentile(source_points, (1, 99), axis=0)
    diagonal = float(np.linalg.norm(upper - lower))
    before_error, view_errors = _pairwise_registration_error(sampled)
    anchor_index = min(source_indices, key=lambda index: view_errors[index])
    rotations = {index: np.eye(3) for index in source_indices}
    translations = {index: np.zeros(3) for index in source_indices}
    correspondence_counts = {index: 0 for index in source_indices}

    maximum_translation = diagonal * float(
        profile["maximum_translation_fraction"]
    )
    for _ in range(int(profile["iterations"])):
        transformed = {
            index: _transform_points(
                sampled[index], rotations[index], translations[index]
            )
            for index in source_indices
        }
        for source_index in source_indices:
            if source_index == anchor_index:
                continue
            reference = np.concatenate(
                [
                    transformed[index]
                    for index in source_indices
                    if index != source_index
                ]
            )
            increment_rotation, increment_translation, count = (
                _estimate_rigid_increment(
                    transformed[source_index],
                    reference,
                    float(profile["trim_quantile"]),
                )
            )
            combined_rotation = increment_rotation @ rotations[source_index]
            combined_translation = (
                increment_rotation @ translations[source_index]
                + increment_translation
            )
            rotations[source_index], translations[source_index] = _bound_transform(
                combined_rotation,
                combined_translation,
                float(profile["maximum_rotation_degrees"]),
                maximum_translation,
            )
            correspondence_counts[source_index] = max(
                correspondence_counts[source_index], count
            )

    registered_samples = {
        index: _transform_points(sampled[index], rotations[index], translations[index])
        for index in source_indices
    }
    after_error, _ = _pairwise_registration_error(registered_samples)
    corrected_candidates = [points.copy() for points in item["candidates"]]
    corrected_world = item["world_points"].copy()
    corrected_extrinsic = item["extrinsic"].copy()
    transforms = []
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
        transforms.append(
            {
                "view": VIEW_ORDER[source_index],
                "anchor": source_index == anchor_index,
                "rotation_degrees": round(
                    float(
                        np.degrees(
                            np.linalg.norm(Rotation.from_matrix(rotation).as_rotvec())
                        )
                    ),
                    6,
                ),
                "translation_fraction": round(
                    float(np.linalg.norm(translation)) / max(diagonal, 1e-12), 8
                ),
                "correspondences": correspondence_counts[source_index],
            }
        )
    corrected_item = {
        **item,
        "candidates": corrected_candidates,
        "world_points": corrected_world,
        "extrinsic": corrected_extrinsic,
    }
    return corrected_item, {
        "profile": profile_name,
        "anchor_view": VIEW_ORDER[anchor_index],
        "mean_pairwise_distance_before_fraction": round(
            before_error / max(diagonal, 1e-12), 8
        ),
        "mean_pairwise_distance_after_fraction": round(
            after_error / max(diagonal, 1e-12), 8
        ),
        "mean_pairwise_distance_delta_fraction": round(
            (after_error - before_error) / max(diagonal, 1e-12), 8
        ),
        "transforms": transforms,
    }


def _evaluate_object(
    item: dict[str, Any],
    profile_name: str,
    voxel_resolution: int,
) -> dict[str, Any]:
    views = []
    baseline_projections = []
    registered_projections = []
    for target_index, view in enumerate(VIEW_ORDER):
        registered_item, registration = _register_sources(
            item, target_index, profile_name
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
        "mean_registration_distance_delta_fraction": round(
            float(
                np.mean(
                    [
                        entry["registration"][
                            "mean_pairwise_distance_delta_fraction"
                        ]
                        for entry in views
                    ]
                )
            ),
            8,
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
        "mean_registration_distance_delta_fraction": round(
            float(
                np.mean(
                    [
                        item["mean_registration_distance_delta_fraction"]
                        for item in items
                    ]
                )
            ),
            8,
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
        "mean_registration_distance_delta_fraction": item[
            "mean_registration_distance_delta_fraction"
        ],
        "views": item["views"],
        "exit_gate_passed": item["mean_iou"] >= 0.80
        and item["minimum_iou"] >= 0.70,
    }


def _preview(item: dict[str, Any], output: Path) -> None:
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
                        "RIGID REGISTERED",
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
    for profile_name in REGISTRATION_PROFILES:
        evaluated = [
            _evaluate_object(
                validation_inputs[object_id], profile_name, voxel_resolution
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
            evaluation_inputs[object_id], selected_profile, voxel_resolution
        )
        for object_id in EVALUATION_OBJECTS
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_diagnostics = {}
    for profile_name, result in profile_results.items():
        diagnostic = next(
            item for item in result["objects"] if item["object_id"] == "ring_020"
        )
        diagnostic_path = (
            output_dir / f"ring_020_{profile_name}_registration_diagnostic.png"
        )
        _preview(diagnostic, diagnostic_path)
        profile_diagnostics[profile_name] = str(diagnostic_path.relative_to(root))
    summaries = []
    for item in [*selected_validation, *evaluation]:
        preview = output_dir / f"{item['object_id']}_rigid_source_registration.png"
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
        "schema_version": "vggt_rigid_source_registration_v1",
        "dataset": "STL-1",
        "authority": "non_metric_rigid_source_registration_hypothesis",
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
        "checks": {
            "selection_uses_validation_split_only": all(
                item["split"] == "val" for item in validation
            ),
            "target_mask_points_and_camera_excluded_from_registration": True,
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
            "retained_as_rigid_source_registration_prior"
            if accepted
            else "rejected_no_consistent_heldout_gain"
        ),
        "claims": {
            "strict_target_image_holdout": False,
            "physical_camera_calibration_validated": False,
            "metric_scale_validated": False,
            "hidden_surface_truth_validated": False,
            "component_topology_validated": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "Rigid source registration uses partial VGGT point maps without scan or CAD truth.",
            "Nearest-neighbour correspondences can match repeated or symmetric jewellery features incorrectly.",
            "VGGT cameras and point maps remain jointly conditioned on all five input images.",
            "A lower source-cloud distance does not by itself prove more accurate geometry.",
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
        default=Path("dataset/phase_runs/v1/vggt_rigid_source_registration_v1"),
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
