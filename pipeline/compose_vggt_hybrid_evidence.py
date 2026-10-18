"""Compose baseline and relief-stratified VGGT queries without rerunning VGGT.

Each retained query track is copied from a pinned source run.  The first 384
queries remain byte-identical to Phase 5.6; 640 non-duplicate stratified queries
are appended.  This changes evidence density only and never modifies geometry.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from pipeline.extract_vggt_track_evidence import (
    RELIEF_QUERY_REGIONS,
    TRAIN_OBJECTS,
    VALIDATION_OBJECTS,
    VIEW_ORDER,
)


PILOT_OBJECTS = ("ring_001", "ring_008", "ring_009", "ring_024")
BASELINE_ROOT = Path("dataset/phase_runs/v1/vggt_full_stl1_track_evidence_v1")
STRATIFIED_ROOT = Path(
    "dataset/phase_runs/v1/vggt_relief_stratified_pilot_v1"
)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        return {key: source[key] for key in source.files}


def _selected_stratified_indices(
    baseline_points: np.ndarray,
    stratified_points: np.ndarray,
    target_count: int,
) -> np.ndarray:
    occupied = {
        (int(round(float(point[0]))), int(round(float(point[1]))))
        for point in baseline_points
    }
    selected = []
    for index, point in enumerate(stratified_points):
        key = (int(round(float(point[0]))), int(round(float(point[1]))))
        if key in occupied:
            continue
        selected.append(index)
        occupied.add(key)
        if len(selected) == target_count:
            break
    if len(selected) != target_count:
        raise RuntimeError(
            f"Only {len(selected)} unique stratified queries for {target_count} slots"
        )
    return np.asarray(selected, np.int32)


def _compose_object(
    root: Path,
    object_id: str,
    output_dir: Path,
    query_count: int,
    stratified_source_root: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    baseline_root = root / BASELINE_ROOT / object_id
    stratified_root = root / stratified_source_root / object_id
    baseline = _load(baseline_root / "vggt_track_evidence.npz")
    stratified = _load(stratified_root / "vggt_track_evidence.npz")
    baseline_count = baseline["query_points_xy"].shape[1]
    append_count = query_count - baseline_count
    if append_count <= 0:
        raise ValueError("hybrid query count must exceed the baseline count")

    selected_by_view = np.stack(
        [
            _selected_stratified_indices(
                baseline["query_points_xy"][view_index],
                stratified["query_points_xy"][view_index],
                append_count,
            )
            for view_index in range(len(VIEW_ORDER))
        ]
    )
    query_points = []
    query_regions = []
    tracks = []
    visibility = []
    confidence = []
    for source_index in range(len(VIEW_ORDER)):
        selected = selected_by_view[source_index]
        query_points.append(
            np.concatenate(
                (
                    baseline["query_points_xy"][source_index],
                    stratified["query_points_xy"][source_index, selected],
                ),
                axis=0,
            )
        )
        query_regions.append(
            np.concatenate(
                (
                    np.full(baseline_count, 4, np.uint8),
                    stratified["query_region"][source_index, selected],
                )
            )
        )
        tracks.append(
            np.concatenate(
                (
                    baseline["tracks_xy"][source_index],
                    stratified["tracks_xy"][source_index][:, selected],
                ),
                axis=1,
            )
        )
        visibility.append(
            np.concatenate(
                (
                    baseline["visibility"][source_index],
                    stratified["visibility"][source_index][:, selected],
                ),
                axis=1,
            )
        )
        confidence.append(
            np.concatenate(
                (
                    baseline["confidence"][source_index],
                    stratified["confidence"][source_index][:, selected],
                ),
                axis=1,
            )
        )
    hybrid = {
        "query_points_xy": np.stack(query_points),
        "query_region": np.stack(query_regions),
        "tracks_xy": np.stack(tracks),
        "visibility": np.stack(visibility),
        "confidence": np.stack(confidence),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "vggt_track_evidence.npz"
    np.savez_compressed(evidence_path, **hybrid)
    baseline_report = json.loads(
        (baseline_root / "report.json").read_text(encoding="utf-8")
    )
    stratified_report = json.loads(
        (stratified_root / "report.json").read_text(encoding="utf-8")
    )
    in_bounds = (
        (hybrid["tracks_xy"][..., 0] >= 0)
        & (hybrid["tracks_xy"][..., 0] < 518)
        & (hybrid["tracks_xy"][..., 1] >= 0)
        & (hybrid["tracks_xy"][..., 1] < 518)
    )
    diagonal = np.arange(len(VIEW_ORDER))
    source_error = np.linalg.norm(
        hybrid["tracks_xy"][diagonal, diagonal] - hybrid["query_points_xy"],
        axis=-1,
    )
    report = {
        "schema_version": "vggt_hybrid_track_evidence_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "object_id": object_id,
        "split": baseline_report["split"],
        "views": list(VIEW_ORDER),
        "image_size": 518,
        "query_count_per_view": query_count,
        "query_selection": "baseline_384_plus_unique_relief_stratified_640",
        "query_region_counts": {
            VIEW_ORDER[view_index]: {
                RELIEF_QUERY_REGIONS[region]: int(
                    np.count_nonzero(hybrid["query_region"][view_index] == region)
                )
                for region in RELIEF_QUERY_REGIONS
            }
            for view_index in range(len(VIEW_ORDER))
        },
        "sources": {
            "baseline_report": str(
                (baseline_root / "report.json").relative_to(root)
            ),
            "stratified_report": str(
                (stratified_root / "report.json").relative_to(root)
            ),
            "checkpoint_sha256": baseline_report["tracker"]["checkpoint_sha256"],
        },
        "runtime": {
            "composition_seconds": round(time.perf_counter() - started, 6),
            "retained_source_gpu_seconds": round(
                float(baseline_report["runtime"]["seconds"])
                + float(stratified_report["runtime"]["seconds"]),
                3,
            ),
            "peak_cuda_bytes": max(
                int(baseline_report["runtime"]["peak_cuda_bytes"]),
                int(stratified_report["runtime"]["peak_cuda_bytes"]),
            ),
        },
        "summary": {
            "mean_visibility": round(float(np.mean(hybrid["visibility"])), 6),
            "mean_confidence": round(float(np.mean(hybrid["confidence"])), 6),
            "in_bounds_fraction": round(float(np.mean(in_bounds)), 6),
            "maximum_source_query_error_pixels": round(
                float(np.max(source_error)), 6
            ),
        },
        "checks": {
            "source_checkpoints_match": baseline_report["tracker"][
                "checkpoint_sha256"
            ]
            == stratified_report["tracker"]["checkpoint_sha256"],
            "baseline_query_points_preserved_exactly": np.array_equal(
                hybrid["query_points_xy"][:, :baseline_count],
                baseline["query_points_xy"],
            ),
            "baseline_tracks_preserved_exactly": np.array_equal(
                hybrid["tracks_xy"][:, :, :baseline_count],
                baseline["tracks_xy"],
            ),
            "all_queries_unique_per_source": all(
                len(np.unique(hybrid["query_points_xy"][index], axis=0))
                == query_count
                for index in range(len(VIEW_ORDER))
            ),
            "all_tracks_in_bounds_above_95_percent": float(np.mean(in_bounds))
            >= 0.95,
            "all_source_queries_preserved": float(np.max(source_error)) <= 0.01,
        },
        "artifacts": {
            "track_evidence": str(evidence_path.relative_to(root)),
        },
        "claims": {
            "track_accuracy_validated": False,
            "physical_correspondence_validated": False,
            "cad_fusion_authorized": False,
            "manufacturing_accuracy_validated": False,
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def run(
    root: Path,
    output_root: Path,
    object_ids: tuple[str, ...] = PILOT_OBJECTS,
    query_count: int = 1024,
    stratified_source_root: Path = STRATIFIED_ROOT,
) -> dict[str, Any]:
    objects = [
        _compose_object(
            root,
            object_id,
            output_root / object_id,
            query_count,
            stratified_source_root,
        )
        for object_id in object_ids
    ]
    report = {
        "schema_version": "vggt_hybrid_track_evidence_batch_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "STL-1",
        "object_ids": list(object_ids),
        "object_count": len(objects),
        "query_count_per_view": query_count,
        "objects": objects,
        "checks": {
            "all_objects_completed": len(objects) == len(object_ids),
            "all_object_checks_pass": all(
                all(item["checks"].values()) for item in objects
            ),
            "test_objects_not_processed": set(object_ids).isdisjoint(
                {"ring_007", "ring_015", "ring_023"}
            ),
        },
        "claims": {
            "dense_correspondence_validated": False,
            "physical_correspondence_validated": False,
            "cad_fusion_authorized": False,
        },
    }
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset/phase_runs/v1/vggt_relief_hybrid_pilot_v1"),
    )
    parser.add_argument("--query-count", type=int, default=1024)
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Compose all 18 fixed training rings instead of the four-ring pilot.",
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Compose the three fixed validation rings after policy freeze.",
    )
    parser.add_argument(
        "--stratified-root",
        type=Path,
        default=STRATIFIED_ROOT,
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = (
        args.output_root
        if args.output_root.is_absolute()
        else root / args.output_root
    )
    if args.train_only and args.validation_only:
        parser.error("--train-only and --validation-only cannot be combined")
    report = run(
        root,
        output,
        object_ids=(
            TRAIN_OBJECTS
            if args.train_only
            else VALIDATION_OBJECTS if args.validation_only else PILOT_OBJECTS
        ),
        query_count=args.query_count,
        stratified_source_root=args.stratified_root,
    )
    print(
        json.dumps(
            {
                "report": str(output / "report.json"),
                "object_count": report["object_count"],
                "checks": report["checks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
