"""Compare relief-stratified VGGT queries with the frozen Phase 5.6 baseline.

The selection policy is developed on training objects and then evaluated once
on the fixed validation split.  It reuses the Phase 5.5 reciprocal gate without
changing thresholds, cameras or CAD.  Increased spatial coverage remains
machine evidence and cannot authorize relief fusion.
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
    _metric_depth_records,
)
from pipeline.evaluate_phase5_relief_tracks import (
    MAP_SIZE,
    THRESHOLDS,
    _draw_points,
    _load_view_evidence,
    _metric3d_normal_records,
    evaluate_track_consistency,
)
from pipeline.extract_vggt_track_evidence import TRAIN_OBJECTS, VALIDATION_OBJECTS


OBJECT_IDS = ("ring_001", "ring_008", "ring_009", "ring_024")
VARIANTS = {
    "phase5_6_baseline": Path(
        "dataset/phase_runs/v1/vggt_full_stl1_track_evidence_v1"
    ),
    "phase5_7_relief_stratified": Path(
        "dataset/phase_runs/v1/vggt_relief_stratified_pilot_v1"
    ),
    "phase5_7_hybrid": Path(
        "dataset/phase_runs/v1/vggt_relief_hybrid_pilot_v1"
    ),
}
TRAIN_VARIANTS = {
    **VARIANTS,
    "phase5_7_hybrid": Path(
        "dataset/phase_runs/v1/vggt_relief_hybrid_train_v1"
    ),
}
VALIDATION_VARIANTS = {
    "phase5_6_baseline": VARIANTS["phase5_6_baseline"],
    "phase5_7_relief_stratified": Path(
        "dataset/phase_runs/v1/vggt_relief_stratified_validation_v1"
    ),
    "phase5_7_hybrid": Path(
        "dataset/phase_runs/v1/vggt_relief_hybrid_validation_v1"
    ),
}


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _coverage_mask(points: np.ndarray, selected: np.ndarray) -> np.ndarray:
    mask = np.zeros((MAP_SIZE, MAP_SIZE), np.uint8)
    scale = (MAP_SIZE - 1) / 517.0
    for point in points[selected]:
        xy = (
            int(round(float(point[0]) * scale)),
            int(round(float(point[1]) * scale)),
        )
        cv2.circle(mask, xy, 6, 1, -1, cv2.LINE_AA)
    return mask > 0


def _variant_metrics(
    tracks: dict[str, np.ndarray],
    evaluated: dict[str, np.ndarray],
    view_evidence: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidate = evaluated["candidate"][0]
    accepted = evaluated["accepted_any"][0]
    points = tracks["query_points_xy"][0]
    relief_domain = view_evidence["front"]["confidence"] >= THRESHOLDS[
        "minimum_source_relief_confidence"
    ]
    relief_count = max(int(np.count_nonzero(relief_domain)), 1)
    candidate_coverage = _coverage_mask(points, candidate) & relief_domain
    accepted_coverage = _coverage_mask(points, accepted) & relief_domain
    candidate_count = int(np.count_nonzero(candidate))
    supported_count = int(np.count_nonzero(accepted))
    return {
        "query_count_per_view": int(points.shape[0]),
        "front_candidate_count": candidate_count,
        "front_supported_count": supported_count,
        "front_supported_fraction": round(
            supported_count / max(candidate_count, 1), 6
        ),
        "front_relief_domain_pixels": relief_count,
        "front_candidate_spatial_coverage": round(
            np.count_nonzero(candidate_coverage) / relief_count, 6
        ),
        "front_supported_spatial_coverage": round(
            np.count_nonzero(accepted_coverage) / relief_count, 6
        ),
        "accepted_by_target_view": {
            view: int(np.count_nonzero(evaluated["accepted"][0, index]))
            for index, view in enumerate(("front", "top", "iso", "lsv", "rsv"))
            if index != 0
        },
    }


def _preserve_baseline_gate(
    hybrid: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Keep frozen baseline decisions and score only appended queries as new."""
    result = {key: value.copy() for key, value in hybrid.items()}
    baseline_count = baseline["candidate"].shape[1]
    result["candidate"][:, :baseline_count] = baseline["candidate"]
    result["accepted"][:, :, :baseline_count] = baseline["accepted"]
    result["normal_corroborated"][:, :, :baseline_count] = baseline[
        "normal_corroborated"
    ]
    result["score"][:, :, :baseline_count] = baseline["score"]
    result["accepted_any"] = np.any(result["accepted"], axis=1)
    result["supported_view_count"] = np.sum(
        result["accepted"], axis=1
    ).astype(np.int16)
    return result


def _panel(image: np.ndarray, title: str, metrics: dict[str, Any]) -> np.ndarray:
    width = 520
    scale = width / image.shape[1]
    resized = cv2.resize(
        image,
        (width, int(round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((resized.shape[0] + 92, width, 3), 246, np.uint8)
    canvas[92:] = resized
    cv2.putText(
        canvas,
        title,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (28, 31, 35),
        2,
        cv2.LINE_AA,
    )
    detail = (
        f"{metrics['front_supported_count']}/{metrics['front_candidate_count']} supported | "
        f"spatial coverage {metrics['front_supported_spatial_coverage']:.1%}"
    )
    cv2.putText(
        canvas,
        detail,
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (48, 53, 59),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Green = reciprocal support | orange = rejected relief candidate",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.37,
        (70, 74, 79),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _render_audit(
    source: np.ndarray,
    variants: dict[str, dict[str, Any]],
    output: Path,
    heading: str,
) -> None:
    panels = []
    titles = {
        "phase5_6_baseline": "PHASE 5.6 BASELINE - 384 QUERIES",
        "phase5_7_relief_stratified": "STRATIFIED ONLY - 1,024 QUERIES",
        "phase5_7_hybrid": "HYBRID - 384 BASELINE + 640 RELIEF",
    }
    for name, item in variants.items():
        evaluated = item["evaluated"]
        tracks = item["tracks"]
        overlay = _draw_points(
            source,
            tracks["query_points_xy"][0],
            evaluated["candidate"][0],
            evaluated["accepted_any"][0],
        )
        panels.append(_panel(overlay, titles[name], item["metrics"]))
    row = np.hstack(panels)
    header = np.full((70, row.shape[1], 3), 246, np.uint8)
    cv2.putText(
        header,
        heading,
        (12, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (28, 31, 35),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output), np.vstack((header, row)))


def _comparison(public_variants: dict[str, dict[str, Any]]) -> dict[str, float]:
    baseline = public_variants["phase5_6_baseline"]
    dense = public_variants["phase5_7_hybrid"]
    return {
        "query_count_multiplier": round(
            dense["query_count_per_view"] / baseline["query_count_per_view"], 6
        ),
        "candidate_spatial_coverage_delta": round(
            dense["front_candidate_spatial_coverage"]
            - baseline["front_candidate_spatial_coverage"],
            6,
        ),
        "supported_spatial_coverage_delta": round(
            dense["front_supported_spatial_coverage"]
            - baseline["front_supported_spatial_coverage"],
            6,
        ),
        "supported_fraction_delta": round(
            dense["front_supported_fraction"]
            - baseline["front_supported_fraction"],
            6,
        ),
    }


def _render_cohort_audit(paths: list[Path], output: Path) -> None:
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    tile_width = 780
    tiles = []
    for image in images:
        scale = tile_width / image.shape[1]
        tiles.append(
            cv2.resize(
                image,
                (tile_width, int(round(image.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        )
    if len(tiles) % 2:
        tiles.append(np.full_like(tiles[0], 246))
    rows = [np.hstack(tiles[index : index + 2]) for index in range(0, len(tiles), 2)]
    cv2.imwrite(str(output), np.vstack(rows))


def run(
    root: Path,
    output_dir: Path,
    object_ids: tuple[str, ...] = OBJECT_IDS,
    track_roots: dict[str, Path] = VARIANTS,
) -> dict[str, Any]:
    phase1 = json.loads(
        (
            root / "dataset/phase_runs/v1/phase1_complete_features_v1/report.json"
        ).read_text(encoding="utf-8")
    )
    phase1_by_id = {item["object_id"]: item for item in phase1["objects"]}
    expected_split = (
        "val" if tuple(object_ids) == VALIDATION_OBJECTS else "train"
    )
    run_label = (
        "VALIDATION"
        if tuple(object_ids) == VALIDATION_OBJECTS
        else "FULL TRAINING"
        if tuple(object_ids) == TRAIN_OBJECTS
        else "TRAINING PILOT"
    )
    metric_depths = _metric_depth_records(root)
    direct_normals = _metric3d_normal_records(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    objects = []
    audit_paths = []

    for object_id in object_ids:
        object_record = phase1_by_id[object_id]
        if object_record["split"] != expected_split:
            raise RuntimeError(
                f"Phase 5.7 {expected_split} run received {object_record['split']} object"
            )
        view_evidence = _load_view_evidence(
            root, object_record, metric_depths, direct_normals
        )
        variants: dict[str, dict[str, Any]] = {}
        public_variants: dict[str, dict[str, Any]] = {}
        for name, relative_root in track_roots.items():
            track_root = root / relative_root
            with np.load(
                track_root / object_id / "vggt_track_evidence.npz"
            ) as source:
                tracks = {key: source[key] for key in source.files}
            evaluated = evaluate_track_consistency(tracks, view_evidence)
            if name == "phase5_7_hybrid":
                evaluated = _preserve_baseline_gate(
                    evaluated,
                    variants["phase5_6_baseline"]["evaluated"],
                )
            metrics = _variant_metrics(tracks, evaluated, view_evidence)
            variants[name] = {
                "tracks": tracks,
                "evaluated": evaluated,
                "metrics": metrics,
            }
            source_report = json.loads(
                (track_root / object_id / "report.json").read_text(
                    encoding="utf-8"
                )
            )
            public_variants[name] = {
                **metrics,
                "query_selection": source_report["query_selection"],
                "query_region_counts": source_report.get(
                    "query_region_counts", {}
                ),
                "runtime": source_report["runtime"],
            }
        source = cv2.imread(
            str(view_evidence["front"]["source_image"]), cv2.IMREAD_COLOR
        )
        audit_path = output_dir / f"{object_id}_dense_relief_pilot_audit.png"
        _render_audit(
            source,
            variants,
            audit_path,
            f"PHASE 5.7 - RELIEF-STRATIFIED TRACK DENSITY ({run_label})",
        )
        audit_paths.append(audit_path)
        objects.append(
            {
                "object_id": object_id,
                "split": object_record["split"],
                "variants": public_variants,
                "comparison": _comparison(public_variants),
                "baseline_gate_preserved": np.array_equal(
                    variants["phase5_7_hybrid"]["evaluated"]["accepted"][
                        :, :, :384
                    ],
                    variants["phase5_6_baseline"]["evaluated"]["accepted"],
                ),
                "artifacts": {"audit": _relative(audit_path, root)},
            }
        )

    cohort_audit_path = output_dir / (
        "validation_cohort_dense_relief_audit.png"
        if tuple(object_ids) == VALIDATION_OBJECTS
        else "training_cohort_dense_relief_audit.png"
    )
    _render_cohort_audit(audit_paths, cohort_audit_path)
    comparisons = [item["comparison"] for item in objects]
    mean_supported_delta = float(
        np.mean([item["supported_spatial_coverage_delta"] for item in comparisons])
    )
    report = {
        "schema_version": "lalitha_phase5_dense_relief_pilot_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": (
            "Phase 5.7 fixed-policy validation relief-query evaluation"
            if tuple(object_ids) == VALIDATION_OBJECTS
            else (
                "Phase 5.7 full-training hybrid relief-query evaluation"
                if tuple(object_ids) == TRAIN_OBJECTS
                else "Phase 5.7 training-only relief-stratified query pilot"
            )
        ),
        "object_ids": list(object_ids),
        "object_count": len(objects),
        "frozen_thresholds": THRESHOLDS,
        "summary": {
            "mean_candidate_spatial_coverage_delta": round(
                float(
                    np.mean(
                        [
                            item["candidate_spatial_coverage_delta"]
                            for item in comparisons
                        ]
                    )
                ),
                6,
            ),
            "mean_supported_spatial_coverage_delta": round(
                mean_supported_delta, 6
            ),
            "mean_supported_fraction_delta": round(
                float(
                    np.mean(
                        [item["supported_fraction_delta"] for item in comparisons]
                    )
                ),
                6,
            ),
            "all_objects_improve_supported_spatial_coverage": all(
                item["supported_spatial_coverage_delta"] > 0.0
                for item in comparisons
            ),
        },
        "checks": {
            "expected_split_only": all(
                item["split"] == expected_split for item in objects
            ),
            "test_objects_not_processed": set(object_ids).isdisjoint(
                {"ring_007", "ring_015", "ring_023"}
            ),
            "frozen_phase5_5_gate_reused": True,
            "dense_query_count_exceeds_baseline": all(
                item["comparison"]["query_count_multiplier"] > 1.0
                for item in objects
            ),
            "baseline_acceptance_gate_preserved": all(
                item["baseline_gate_preserved"] for item in objects
            ),
            "exact_cad_unchanged": True,
        },
        "objects": objects,
        "decision": (
            (
                "retain_validation_improvement_test_still_held_out"
                if tuple(object_ids) == VALIDATION_OBJECTS
                else (
                    "freeze_training_policy_then_evaluate_validation"
                    if tuple(object_ids) == TRAIN_OBJECTS
                    else "expand_to_full_training_cohort_before_validation"
                )
            )
            if mean_supported_delta > 0.0
            and all(
                item["supported_spatial_coverage_delta"] > 0.0
                for item in comparisons
            )
            else "reject_or_refine_density_policy"
        ),
        "claims": {
            "track_accuracy_validated": False,
            "dense_correspondence_validated": False,
            "physical_correspondence_validated": False,
            "cad_fusion_authorized": False,
            "manufacturing_accuracy_validated": False,
        },
        "artifacts": {"audit": _relative(cohort_audit_path, root)},
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Evaluate the frozen hybrid policy over all 18 training rings.",
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Evaluate the frozen training policy on the three validation rings.",
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    if args.train_only and args.validation_only:
        parser.error("--train-only and --validation-only cannot be combined")
    output_arg = args.output or Path(
        "data/phase5_dense_relief_validation_v1"
        if args.validation_only
        else (
            "data/phase5_dense_relief_train_v1"
            if args.train_only
            else "data/phase5_dense_relief_pilot_v1"
        )
    )
    output = output_arg if output_arg.is_absolute() else root / output_arg
    report = run(
        root,
        output,
        object_ids=(
            VALIDATION_OBJECTS
            if args.validation_only
            else TRAIN_OBJECTS if args.train_only else OBJECT_IDS
        ),
        track_roots=(
            VALIDATION_VARIANTS
            if args.validation_only
            else TRAIN_VARIANTS if args.train_only else VARIANTS
        ),
    )
    print(
        json.dumps(
            {
                "report": str(output / "report.json"),
                "summary": report["summary"],
                "decision": report["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
