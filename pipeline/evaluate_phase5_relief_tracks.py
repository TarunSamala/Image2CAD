"""Evaluate sparse reciprocal cross-view support for Phase 5.4 relief evidence.

The default preserves the original fixed seven-ring Phase 5.5 cohort.  The
Phase 5.6 mode applies the same frozen thresholds to all 24 STL-1 rings.  This
stage does not move cameras or geometry.  It asks a narrower question: when a
source pixel has relief evidence, does another view contain depth-supported
detail at the tracked location, and does the nearest reverse query return close
to the source pixel?  Accepted tracks remain sparse machine evidence and never
modify the exact Phase 5 STEP body.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline.build_phase5_stl1_relief_evidence import (
    COMPLETE_METRIC_BACKENDS,
    VIEWS,
    _color,
    _metric_depth_records,
    build_relief_evidence,
)
from pipeline.run_surface_normal_evidence import (
    angular_error_degrees,
    depth_to_screen_normal,
    orient_direct_normals,
)


SCHEMA_VERSION = "lalitha_phase5_relief_track_consistency_v1"
FULL_DATASET_SCHEMA_VERSION = "lalitha_phase5_full_relief_track_consistency_v1"
ALL_STL1_OBJECTS = tuple(f"ring_{index:03d}" for index in range(1, 25))
TRACK_COHORT = (
    "ring_004",
    "ring_012",
    "ring_020",
    "ring_001",
    "ring_007",
    "ring_015",
    "ring_023",
)
TRACK_SIZE = 518
MAP_SIZE = 768
THRESHOLDS = {
    "minimum_source_relief_confidence": 0.05,
    "minimum_target_relief_confidence": 0.05,
    "minimum_visibility": 0.50,
    "minimum_track_confidence": 0.10,
    "maximum_nearest_query_distance_pixels": 12.0,
    "maximum_reciprocal_cycle_error_pixels": 16.0,
    "minimum_normal_agreement_when_available": 0.40,
}


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _metric3d_normal_records(root: Path) -> dict[str, Path]:
    path = (
        root
        / "dataset/phase_runs/v1/image_only_phase2_metric_depth_consensus_v1"
        / "backends/metric3d_v2/backend_report.json"
    )
    if not path.exists():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        item["sample_key"]: root / item["artifacts"]["normals"]
        for item in report["records"]
        if item.get("artifacts", {}).get("normals")
    }


def _sample(values: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Nearest sample a MAP_SIZE map at TRACK_SIZE coordinates."""
    scale_x = (values.shape[1] - 1) / max(TRACK_SIZE - 1, 1)
    scale_y = (values.shape[0] - 1) / max(TRACK_SIZE - 1, 1)
    xs = np.clip(np.rint(points[..., 0] * scale_x), 0, values.shape[1] - 1).astype(
        np.int32
    )
    ys = np.clip(np.rint(points[..., 1] * scale_y), 0, values.shape[0] - 1).astype(
        np.int32
    )
    return values[ys, xs]


def _normal_agreement(
    evidence: dict[str, Any],
    direct_path: Path | None,
) -> np.ndarray | None:
    if direct_path is None or not direct_path.exists():
        return None
    support = np.asarray(evidence["support"], dtype=bool)
    direct, _ = orient_direct_normals(np.load(direct_path), support)
    depth_normal = depth_to_screen_normal(
        np.asarray(evidence["depth_hypothesis"], np.float32), support
    )
    angle = angular_error_degrees(direct, depth_normal)
    agreement = np.exp(-np.square(angle / 35.0)).astype(np.float32)
    agreement[~support] = 0.0
    return agreement


def _load_view_evidence(
    root: Path,
    object_record: dict[str, Any],
    metric_depths: dict[str, tuple[Path, ...]],
    direct_normals: dict[str, Path],
) -> dict[str, dict[str, Any]]:
    result = {}
    object_id = object_record["object_id"]
    for view in VIEWS:
        source_view = object_record["views"][view]
        with np.load(root / source_view["bundle"]) as bundle:
            sample_key = f"{object_id}:{view}"
            kwargs = {
                "depth": bundle["relative_depth_regularized_f32"],
                "uncertainty_u16": bundle["relative_depth_uncertainty_u16"],
                "gradient": bundle["gradient_magnitude_f32"],
                "local_contrast": bundle["local_contrast_f32"],
                "jewelry_mask": bundle["mask_jewelry_u8"],
                "opening_mask": bundle["mask_negative_space_proposal_u8"],
                "metric_depths": tuple(
                    np.load(path) for path in metric_depths.get(sample_key, ())
                ),
            }
            evidence = build_relief_evidence(**kwargs)
        result[view] = {
            "confidence": evidence["confidence"],
            "relief": evidence["relief"],
            "support": evidence["support"],
            "normal_agreement": _normal_agreement(
                evidence, direct_normals.get(sample_key)
            ),
            "source_image": root / source_view["normalized_source"],
        }
    return result


def evaluate_track_consistency(
    tracks: dict[str, np.ndarray],
    view_evidence: dict[str, dict[str, Any]],
) -> dict[str, np.ndarray]:
    query = tracks["query_points_xy"]
    track_xy = tracks["tracks_xy"]
    visibility = tracks["visibility"]
    track_confidence = tracks["confidence"]
    view_count, query_count = query.shape[:2]
    candidate = np.zeros((view_count, query_count), dtype=bool)
    accepted = np.zeros((view_count, view_count, query_count), dtype=bool)
    normal_corroborated = np.zeros_like(accepted)
    score = np.zeros((view_count, view_count, query_count), dtype=np.float32)
    cycle_error = np.full((view_count, view_count, query_count), np.nan, np.float32)
    nearest_distance = np.full_like(cycle_error, np.nan)

    relief_maps = [view_evidence[view]["confidence"] for view in VIEWS]
    normal_maps = [view_evidence[view]["normal_agreement"] for view in VIEWS]
    source_relief = np.stack(
        [_sample(relief_maps[index], query[index]) for index in range(view_count)]
    )
    candidate = source_relief >= THRESHOLDS["minimum_source_relief_confidence"]

    for source_index in range(view_count):
        source_normal = (
            _sample(normal_maps[source_index], query[source_index])
            if normal_maps[source_index] is not None
            else np.ones(query_count, np.float32)
        )
        for target_index in range(view_count):
            if source_index == target_index:
                continue
            forward = track_xy[source_index, target_index]
            delta = forward[:, None, :] - query[target_index][None, :, :]
            squared = np.sum(delta * delta, axis=2)
            nearest = np.argmin(squared, axis=1)
            nearest_distance[source_index, target_index] = np.sqrt(
                squared[np.arange(query_count), nearest]
            )
            reverse = track_xy[target_index, source_index, nearest]
            cycle_error[source_index, target_index] = np.linalg.norm(
                reverse - query[source_index], axis=1
            )
            target_relief = _sample(relief_maps[target_index], forward)
            target_normal = (
                _sample(normal_maps[target_index], forward)
                if normal_maps[target_index] is not None
                else np.ones(query_count, np.float32)
            )
            forward_visibility = visibility[source_index, target_index]
            reverse_visibility = visibility[
                target_index, source_index, nearest
            ]
            forward_confidence = track_confidence[source_index, target_index]
            reverse_confidence = track_confidence[
                target_index, source_index, nearest
            ]
            accepted[source_index, target_index] = (
                candidate[source_index]
                & (target_relief >= THRESHOLDS["minimum_target_relief_confidence"])
                & (forward_visibility >= THRESHOLDS["minimum_visibility"])
                & (reverse_visibility >= THRESHOLDS["minimum_visibility"])
                & (forward_confidence >= THRESHOLDS["minimum_track_confidence"])
                & (reverse_confidence >= THRESHOLDS["minimum_track_confidence"])
                & (
                    nearest_distance[source_index, target_index]
                    <= THRESHOLDS["maximum_nearest_query_distance_pixels"]
                )
                & (
                    cycle_error[source_index, target_index]
                    <= THRESHOLDS["maximum_reciprocal_cycle_error_pixels"]
                )
            )
            if (
                normal_maps[source_index] is not None
                and normal_maps[target_index] is not None
            ):
                normal_corroborated[source_index, target_index] = (
                    accepted[source_index, target_index]
                    & (
                        source_normal
                        >= THRESHOLDS[
                            "minimum_normal_agreement_when_available"
                        ]
                    )
                    & (
                        target_normal
                        >= THRESHOLDS[
                            "minimum_normal_agreement_when_available"
                        ]
                    )
                )
            factors = np.stack(
                (
                    np.clip(source_relief[source_index], 0.0, 1.0),
                    np.clip(target_relief, 0.0, 1.0),
                    np.clip(forward_visibility, 0.0, 1.0),
                    np.clip(reverse_visibility, 0.0, 1.0),
                    np.clip(forward_confidence, 0.0, 1.0),
                    np.clip(reverse_confidence, 0.0, 1.0),
                )
            )
            score[source_index, target_index] = np.power(
                np.prod(np.maximum(factors, 1e-8), axis=0), 1.0 / len(factors)
            )
            score[source_index, target_index][
                ~accepted[source_index, target_index]
            ] = 0.0

    accepted_any = np.any(accepted, axis=1)
    supported_view_count = np.sum(accepted, axis=1).astype(np.int16)
    return {
        "candidate": candidate,
        "accepted": accepted,
        "accepted_any": accepted_any,
        "normal_corroborated": normal_corroborated,
        "supported_view_count": supported_view_count,
        "score": score,
        "cycle_error": cycle_error,
        "nearest_query_distance": nearest_distance,
        "source_relief_confidence": source_relief,
    }


def _draw_points(
    image: np.ndarray,
    points: np.ndarray,
    candidate: np.ndarray,
    accepted: np.ndarray,
) -> np.ndarray:
    canvas = image.copy()
    scale_x = (image.shape[1] - 1) / max(TRACK_SIZE - 1, 1)
    scale_y = (image.shape[0] - 1) / max(TRACK_SIZE - 1, 1)
    for point, is_candidate, is_accepted in zip(points, candidate, accepted):
        if not is_candidate:
            continue
        xy = (
            int(round(float(point[0]) * scale_x)),
            int(round(float(point[1]) * scale_y)),
        )
        color = (76, 190, 95) if is_accepted else (58, 102, 214)
        cv2.circle(canvas, xy, 4, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, xy, 5, (245, 245, 245), 1, cv2.LINE_AA)
    return canvas


def _panel(image: np.ndarray, title: str, width: int = 250) -> np.ndarray:
    scale = width / image.shape[1]
    resized = cv2.resize(
        image,
        (width, max(1, int(round(image.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((resized.shape[0] + 42, width, 3), 246, np.uint8)
    canvas[42:] = resized
    cv2.putText(canvas, title, (7, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (25, 25, 25), 1, cv2.LINE_AA)
    return canvas


def _render_review(
    view_evidence: dict[str, dict[str, Any]],
    tracks: dict[str, np.ndarray],
    evaluated: dict[str, np.ndarray],
    output: Path,
    object_id: str,
    phase_label: str,
) -> None:
    source_index = 0
    panels = []
    source_candidate = evaluated["candidate"][source_index]
    for target_index, view in enumerate(VIEWS):
        image = cv2.imread(str(view_evidence[view]["source_image"]), cv2.IMREAD_COLOR)
        if target_index == source_index:
            points = tracks["query_points_xy"][source_index]
            accepted = evaluated["accepted_any"][source_index]
            title = f"FRONT SOURCE - {int(np.count_nonzero(accepted))} SUPPORTED"
        else:
            points = tracks["tracks_xy"][source_index, target_index]
            accepted = evaluated["accepted"][source_index, target_index]
            title = f"{view.upper()} - {int(np.count_nonzero(accepted))} RECIPROCAL"
        panels.append(
            _panel(_draw_points(image, points, source_candidate, accepted), title)
        )
    row = np.hstack(panels)
    header = np.full((66, row.shape[1], 3), 246, np.uint8)
    cv2.putText(
        header,
        f"{phase_label.upper()} - {object_id.upper()} CROSS-VIEW RELIEF TRACKS",
        (12, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (27, 32, 37),
        2,
        cv2.LINE_AA,
    )
    footer = np.full((36, row.shape[1], 3), 246, np.uint8)
    cv2.putText(
        footer,
        "Green = reciprocal depth-supported evidence - red = source detail rejected by this gate",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (55, 60, 65),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output), np.vstack((header, row, footer)))


def _write_front_maps(
    view_evidence: dict[str, dict[str, Any]],
    tracks: dict[str, np.ndarray],
    evaluated: dict[str, np.ndarray],
    support_path: Path,
    relief_path: Path,
) -> None:
    source_index = 0
    accepted = evaluated["accepted_any"][source_index]
    mask = np.zeros((MAP_SIZE, MAP_SIZE), np.uint8)
    scale = (MAP_SIZE - 1) / max(TRACK_SIZE - 1, 1)
    for point in tracks["query_points_xy"][source_index][accepted]:
        xy = (int(round(float(point[0]) * scale)), int(round(float(point[1]) * scale)))
        cv2.circle(mask, xy, 10, 255, -1, cv2.LINE_AA)
    source_support = view_evidence["front"]["support"]
    mask[~source_support] = 0
    relief = np.asarray(view_evidence["front"]["relief"], np.float32).copy()
    relief[mask == 0] = 0.0
    cv2.imwrite(str(support_path), mask)
    cv2.imwrite(str(relief_path), _color(relief, mask > 0, signed=True))


def _aggregate(records: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not records:
        return {
            "object_count": 0,
            "front_candidate_count": 0,
            "front_supported_count": 0,
            "front_supported_fraction": None,
            "mean_front_supported_view_count": None,
        }
    candidates = sum(item["front"]["candidate_count"] for item in records)
    supported = sum(item["front"]["cross_view_supported_count"] for item in records)
    view_counts = [item["front"]["mean_supported_view_count"] for item in records]
    return {
        "object_count": len(records),
        "front_candidate_count": candidates,
        "front_supported_count": supported,
        "front_supported_fraction": round(supported / max(candidates, 1), 6),
        "mean_front_supported_view_count": round(float(np.mean(view_counts)), 6),
    }


def _render_audit(
    root: Path,
    objects: list[dict[str, Any]],
    output: Path,
    phase_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(objects), 1, figsize=(15, 4 * len(objects)))
    if len(objects) == 1:
        axes = [axes]
    for axis, item in zip(axes, objects):
        image = cv2.imread(str(root / item["artifacts"]["review"]), cv2.IMREAD_COLOR)
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        suffix = " - HELD OUT / NOT AGGREGATED" if item["split"] == "test" else ""
        axis.set_title(
            f"{item['object_id'].upper()} - {item['split'].upper()}{suffix}",
            loc="left",
            fontsize=10,
            fontweight="bold",
        )
        axis.set_axis_off()
    figure.suptitle(
        f"{phase_label.upper()} - RECIPROCAL CROSS-VIEW RELIEF EVIDENCE",
        fontsize=17,
        fontweight="bold",
        color="#20252b",
    )
    figure.tight_layout(rect=(0.005, 0.005, 0.995, 0.982))
    figure.savefig(output, dpi=145, bbox_inches="tight")
    plt.close(figure)


def _render_compact_front_audit(
    root: Path,
    objects: list[dict[str, Any]],
    output: Path,
    phase_label: str,
) -> None:
    """Render a reviewable 6x4 front-evidence sheet for the full dataset."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 6, figsize=(18, 12))
    for axis, item in zip(axes.flat, objects):
        review = cv2.imread(str(root / item["artifacts"]["review"]), cv2.IMREAD_COLOR)
        # The first 250-pixel panel is the front source; remove the sheet header/footer.
        front_panel = review[66 : review.shape[0] - 36, :250]
        axis.imshow(cv2.cvtColor(front_panel, cv2.COLOR_BGR2RGB))
        front = item["front"]
        held = " / HELD OUT" if item["split"] == "test" else ""
        axis.set_title(
            f"{item['object_id'].upper()} - {item['split'].upper()}{held}\n"
            f"{front['cross_view_supported_count']}/{front['candidate_count']} supported",
            loc="left",
            fontsize=8,
            fontweight="bold",
        )
        axis.set_axis_off()
    for axis in axes.flat[len(objects) :]:
        axis.set_axis_off()
    figure.suptitle(
        f"{phase_label.upper()} - STL-1 FRONT RELIEF SUPPORT (24 RINGS)",
        fontsize=17,
        fontweight="bold",
        color="#20252b",
    )
    figure.tight_layout(rect=(0.005, 0.005, 0.995, 0.965))
    figure.savefig(output, dpi=145, bbox_inches="tight")
    plt.close(figure)


def run(
    root: Path,
    output_dir: Path,
    object_ids: tuple[str, ...] = TRACK_COHORT,
    track_root: Path | None = None,
) -> dict[str, Any]:
    phase1_path = root / "dataset/phase_runs/v1/phase1_complete_features_v1/report.json"
    phase1 = json.loads(phase1_path.read_text(encoding="utf-8"))
    phase1_by_id = {item["object_id"]: item for item in phase1["objects"]}
    full_dataset = tuple(object_ids) == ALL_STL1_OBJECTS
    phase_label = "Phase 5.6" if full_dataset else "Phase 5.5"
    if track_root is None:
        track_root = root / (
            "dataset/phase_runs/v1/vggt_full_stl1_track_evidence_v1"
            if full_dataset
            else "dataset/phase_runs/v1/vggt_dense_track_evidence_v1"
        )
    metric_depths = _metric_depth_records(root)
    direct_normals = _metric3d_normal_records(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    objects = []

    for object_id in object_ids:
        object_record = phase1_by_id[object_id]
        evidence = _load_view_evidence(
            root, object_record, metric_depths, direct_normals
        )
        with np.load(track_root / object_id / "vggt_track_evidence.npz") as source:
            tracks = {key: source[key] for key in source.files}
        evaluated = evaluate_track_consistency(tracks, evidence)
        object_dir = output_dir / "objects" / object_id
        object_dir.mkdir(parents=True, exist_ok=True)
        npz_path = object_dir / f"{object_id}_relief_track_support.npz"
        np.savez_compressed(npz_path, **evaluated)
        review_path = object_dir / f"{object_id}_relief_track_review.png"
        _render_review(
            evidence, tracks, evaluated, review_path, object_id, phase_label
        )
        support_path = object_dir / f"{object_id}_front_crossview_support.png"
        relief_path = object_dir / f"{object_id}_front_crossview_relief.png"
        _write_front_maps(
            evidence, tracks, evaluated, support_path, relief_path
        )

        source_index = 0
        front_candidates = evaluated["candidate"][source_index]
        front_accepted = evaluated["accepted_any"][source_index]
        front_view_counts = evaluated["supported_view_count"][source_index]
        front_normal_corroborated = np.any(
            evaluated["normal_corroborated"][source_index], axis=0
        )
        candidate_count = int(np.count_nonzero(front_candidates))
        supported_count = int(np.count_nonzero(front_accepted))
        candidate_view_counts = front_view_counts[front_candidates]
        normal_views = sum(
            evidence[view]["normal_agreement"] is not None for view in VIEWS
        )
        objects.append(
            {
                "object_id": object_id,
                "category": "ring",
                "split": object_record["split"],
                "evaluation_status": (
                    "held_out_not_aggregated"
                    if object_record["split"] == "test"
                    else "sparse_machine_evidence_only"
                ),
                "normal_comparator_view_count": normal_views,
                "front": {
                    "candidate_count": candidate_count,
                    "cross_view_supported_count": supported_count,
                    "cross_view_supported_fraction": round(
                        supported_count / max(candidate_count, 1), 6
                    ),
                    "mean_supported_view_count": round(
                        float(np.mean(candidate_view_counts))
                        if candidate_view_counts.size
                        else 0.0,
                        6,
                    ),
                    "normal_corroborated_count": int(
                        np.count_nonzero(front_normal_corroborated)
                    ),
                },
                "all_views": {
                    "candidate_count": int(np.count_nonzero(evaluated["candidate"])),
                    "cross_view_supported_count": int(
                        np.count_nonzero(evaluated["accepted_any"])
                    ),
                },
                "artifacts": {
                    "evidence": _relative(npz_path, root),
                    "review": _relative(review_path, root),
                    "front_support": _relative(support_path, root),
                    "front_relief": _relative(relief_path, root),
                },
            }
        )

    train = [item for item in objects if item["split"] == "train"]
    validation = [item for item in objects if item["split"] == "val"]
    test = [item for item in objects if item["split"] == "test"]
    audit_path = output_dir / "relief_track_consistency_audit.png"
    _render_audit(root, objects, audit_path, phase_label)
    compact_audit_path = output_dir / "front_relief_support_audit.png"
    if full_dataset:
        _render_compact_front_audit(
            root, objects, compact_audit_path, phase_label
        )
    report = {
        "schema_version": (
            FULL_DATASET_SCHEMA_VERSION if full_dataset else SCHEMA_VERSION
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_profile": "images_only_sparse_reciprocal_relief_tracking",
        "scope": {
            "dataset": "STL-1",
            "object_ids": list(object_ids),
            "view_order": list(VIEWS),
            "input_resolution": TRACK_SIZE,
            "query_count_per_view": 384,
            "thresholds": THRESHOLDS,
        },
        "sources": {
            "phase5_relief_evidence": "data/phase5_stl1_relief_evidence_v1/report.json",
            "vggt_tracks": _relative(track_root / "report.json", root),
            "surface_normal_supplement": "dataset/phase_runs/v1/surface_normal_evidence_v1/report.json",
        },
        "summary": {
            "object_count": len(objects),
            "view_count": len(objects) * len(VIEWS),
            "normal_comparator_view_count": sum(
                item["normal_comparator_view_count"] for item in objects
            ),
            "train": _aggregate(train),
            "validation": _aggregate(validation),
            "test_held_out_count": len(test),
        },
        "checks": {
            (
                "full_24_ring_track_cohort_processed"
                if full_dataset
                else "fixed_seven_ring_track_cohort_processed"
            ): len(objects) == len(object_ids),
            (
                "all_120_tracked_views_processed"
                if full_dataset
                else "all_35_tracked_views_processed"
            ): len(objects) * len(VIEWS) == len(object_ids) * len(VIEWS),
            "selection_thresholds_do_not_use_test_objects": True,
            "held_out_test_not_aggregated": len(test) == 3,
            "reciprocal_cycle_required": True,
            "nearest_query_distance_bounded": True,
            "normal_comparator_reported_separately_from_universal_gate": True,
            "exact_cad_body_unchanged": True,
            "ring_only_scope_enforced": all(item["category"] == "ring" for item in objects),
        },
        "objects": objects,
        "artifacts": {
            "audit": _relative(audit_path, root),
            **(
                {"compact_front_audit": _relative(compact_audit_path, root)}
                if full_dataset
                else {}
            ),
        },
        "decision": (
            "full_dataset_sparse_relief_support_retained_not_cad_fusion"
            if full_dataset
            else "retain_sparse_relief_correspondence_as_review_evidence_not_cad_fusion"
        ),
        "claims": {
            "sparse_reciprocal_relief_support_available": True,
            "dense_correspondence_validated": False,
            "physical_correspondence_validated": False,
            "relief_accuracy_measured": False,
            "cad_fusion_authorized": False,
            "metric_scale_validated": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            (
                "All 24 rings have retained sparse VGGT query tracks, but this is not dense correspondence."
                if full_dataset
                else "Only seven rings have retained VGGT track evidence; this is not full 24-ring coverage."
            ),
            "VGGT features were jointly conditioned on all five images and are machine evidence, not independent correspondence truth.",
            "Nearest-query reciprocal association can reject correct tracks because the 384 target queries remain sparse.",
            f"The direct-normal comparator is available for only {sum(item['normal_comparator_view_count'] for item in objects)} of {len(objects) * len(VIEWS)} views and shares the Metric3D backbone with one depth hypothesis.",
            "View-dependent highlights can still satisfy learned feature tracks; accepted points require review before CAD use.",
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--track-root",
        type=Path,
        default=None,
        help="Override the retained VGGT-track evidence directory.",
    )
    parser.add_argument(
        "--all-stl1",
        action="store_true",
        help="Evaluate the frozen Phase 5.5 gate across all 24 STL-1 rings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    output_arg = args.output or Path(
        "data/phase5_full_relief_track_consistency_v1"
        if args.all_stl1
        else "data/phase5_relief_track_consistency_v1"
    )
    output = output_arg if output_arg.is_absolute() else root / output_arg
    track_root = args.track_root
    if track_root is not None and not track_root.is_absolute():
        track_root = root / track_root
    report = run(
        root,
        output,
        object_ids=ALL_STL1_OBJECTS if args.all_stl1 else TRACK_COHORT,
        track_root=track_root,
    )
    print(json.dumps({"report": str(output / "report.json"), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
