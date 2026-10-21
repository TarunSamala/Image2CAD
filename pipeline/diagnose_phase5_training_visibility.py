"""Audit front-to-target evidence loss on the fixed Phase 5 training split.

The target mask and VGGT visibility are *proxies*, not physical occlusion labels.
This diagnostic does not relax the frozen Phase 5.13 gate or fit a surface.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline.build_phase5_stl1_relief_evidence import _metric_depth_records
from pipeline.evaluate_phase5_dense_relief_pilot import _preserve_baseline_gate
from pipeline.evaluate_phase5_relief_tracks import (
    THRESHOLDS,
    TRACK_SIZE,
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
    _target_topology,
)
from pipeline.extract_vggt_track_evidence import TRAIN_OBJECTS, VIEW_ORDER


def _route_counts(
    source_candidate: np.ndarray,
    target_points: np.ndarray,
    target_jewelry: np.ndarray,
    forward_visibility: np.ndarray,
    target_confidence: np.ndarray,
    accepted: np.ndarray,
    normal_corroborated: np.ndarray,
) -> dict[str, int]:
    """Count cumulative evidence survival without calling a proxy physical truth."""
    if target_points.shape != (len(source_candidate), 2):
        raise ValueError("target point and source query counts differ")
    finite = np.all(np.isfinite(target_points), axis=1)
    inside = finite & np.all(
        (target_points >= 0) & (target_points <= TRACK_SIZE - 1), axis=1
    )
    safe_points = np.where(finite[:, None], target_points, 0.0)
    jewelry = inside & _sample(target_jewelry, safe_points).astype(bool)
    visibility = (
        np.isfinite(forward_visibility)
        & (forward_visibility >= THRESHOLDS["minimum_visibility"])
    )
    depth_support = (
        np.isfinite(target_confidence)
        & (target_confidence >= THRESHOLDS["minimum_target_relief_confidence"])
    )
    stages = {
        "source_relief_candidate": np.asarray(source_candidate, bool),
        "target_projection_in_frame": inside,
        "target_jewelry_mask_proxy": jewelry,
        "vggt_forward_visibility_proxy": visibility,
        "target_relief_support": depth_support,
        "reciprocal_track_accepted": np.asarray(accepted, bool),
        "available_direct_normal_corroborated": np.asarray(normal_corroborated, bool),
    }
    retained = np.ones(len(source_candidate), bool)
    counts = {}
    for name, stage in stages.items():
        retained &= stage
        counts[name] = int(np.count_nonzero(retained))
    return counts


def _normal_diagnostic(
    source_points: np.ndarray,
    target_points: np.ndarray,
    accepted: np.ndarray,
    source_map: np.ndarray | None,
    target_map: np.ndarray | None,
) -> dict[str, Any]:
    """Show which normal proxy removes accepted tracks, without changing it."""
    accepted = np.asarray(accepted, bool)
    pair_available = source_map is not None and target_map is not None
    if not pair_available:
        return {
            "source_normal_map_available": source_map is not None,
            "target_normal_map_available": target_map is not None,
            "normal_pair_available": False,
            "accepted_track_count": int(np.count_nonzero(accepted)),
            "accepted_with_normal_pair_count": 0,
            "source_normal_pass_count": None,
            "target_normal_pass_count": None,
            "both_normal_pass_count": None,
        }
    threshold = THRESHOLDS["minimum_normal_agreement_when_available"]
    safe_target = np.nan_to_num(target_points, nan=0.0, posinf=0.0, neginf=0.0)
    source_ok = _sample(source_map, source_points) >= threshold
    target_ok = _sample(target_map, safe_target) >= threshold
    return {
        "source_normal_map_available": True,
        "target_normal_map_available": True,
        "normal_pair_available": True,
        "accepted_track_count": int(np.count_nonzero(accepted)),
        "accepted_with_normal_pair_count": int(np.count_nonzero(accepted)),
        "source_normal_pass_count": int(np.count_nonzero(accepted & source_ok)),
        "target_normal_pass_count": int(np.count_nonzero(accepted & target_ok)),
        "both_normal_pass_count": int(np.count_nonzero(accepted & source_ok & target_ok)),
    }


def _render_audit(objects: list[dict[str, Any]], output: Path) -> None:
    """Small readable cohort chart; the JSON retains per-ring counts."""
    cell_width, cell_height = 310, 38
    width = cell_width * (len(TARGET_VIEWS) + 1)
    height = 135 + cell_height * (len(objects) + 2)
    canvas = np.full((height, width, 3), 246, np.uint8)
    cv2.putText(canvas, "PHASE 5.18 TRAINING VIEW-ROUTE AUDIT", (16, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (25, 29, 34), 2, cv2.LINE_AA)
    cv2.putText(canvas, "candidate / mask / learned visible / accepted / direct-normal / track-topology triangles",
                (16, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (58, 62, 68), 1, cv2.LINE_AA)
    cv2.putText(canvas, "grey = no direct-normal pair   amber = pair available, no corroborated track   green = corroborated",
                (16, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (58, 62, 68), 1, cv2.LINE_AA)
    for column, label in enumerate(("RING", *TARGET_VIEWS)):
        cv2.putText(canvas, label.upper(), (14 + column * cell_width, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 35, 40), 1, cv2.LINE_AA)
    for row, obj in enumerate(objects):
        y = 136 + row * cell_height
        cv2.putText(canvas, obj["object_id"], (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.44, (45, 48, 53), 1, cv2.LINE_AA)
        for column, view in enumerate(TARGET_VIEWS, 1):
            item = obj["target_views"][view]
            c = item["cumulative_query_counts"]
            values = (
                c["source_relief_candidate"], c["target_jewelry_mask_proxy"],
                c["vggt_forward_visibility_proxy"],
                c["reciprocal_track_accepted"], c["available_direct_normal_corroborated"],
                item["stable_triangle_count"],
            )
            if not item["normal_diagnostic"]["normal_pair_available"]:
                color = (105, 105, 105)
            elif values[-2] == 0:
                color = (45, 125, 180)
            else:
                color = (50, 105, 70)
            cv2.putText(canvas, "/".join(map(str, values)),
                        (14 + column * cell_width, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.39, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Image-mask and learned-visibility proxies only; no physical occlusion or CAD claim",
                (16, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                (58, 62, 68), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"Could not write {output}")


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
        baseline = _load_tracks(root / BASELINE_TRACK_ROOT / object_id / "vggt_track_evidence.npz")
        tracks = _load_tracks(root / HYBRID_TRACK_ROOT / object_id / "vggt_track_evidence.npz")
        if not np.allclose(tracks["query_points_xy"][:, :384], baseline["query_points_xy"]):
            raise RuntimeError(f"{object_id} does not preserve the baseline query prefix")
        evaluated = _preserve_baseline_gate(
            evaluate_track_consistency(tracks, evidence),
            evaluate_track_consistency(baseline, evidence),
        )
        target_views = {}
        source_points = tracks["query_points_xy"][SOURCE_INDEX]
        for target_index, view in enumerate(VIEW_ORDER):
            if target_index == SOURCE_INDEX:
                continue
            with np.load(root / record["views"][view]["bundle"]) as bundle:
                jewelry_mask = bundle["mask_jewelry_u8"] > 0
            points = tracks["tracks_xy"][SOURCE_INDEX, target_index]
            counts = _route_counts(
                evaluated["candidate"][SOURCE_INDEX], points, jewelry_mask,
                tracks["visibility"][SOURCE_INDEX, target_index],
                _sample(
                    evidence[view]["confidence"],
                    np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0),
                ),
                evaluated["accepted"][SOURCE_INDEX, target_index],
                evaluated["normal_corroborated"][SOURCE_INDEX, target_index],
            )
            topology, _ = _target_topology(
                source_points, points,
                evaluated["accepted"][SOURCE_INDEX, target_index],
            )
            target_views[view] = {
                "cumulative_query_counts": counts,
                "stable_triangle_count": topology["orientation_preserved_triangle_count"],
                "normal_corroborated_track_count": int(np.count_nonzero(
                    evaluated["normal_corroborated"][SOURCE_INDEX, target_index]
                )),
                "normal_diagnostic": _normal_diagnostic(
                    source_points, points,
                    evaluated["accepted"][SOURCE_INDEX, target_index],
                    evidence[SOURCE_VIEW]["normal_agreement"],
                    evidence[view]["normal_agreement"],
                ),
            }
        objects.append({"object_id": object_id, "split": "train", "target_views": target_views})
    summary_views = {}
    for view in TARGET_VIEWS:
        summary_views[view] = {
            "cumulative_query_counts": {
                stage: sum(item["target_views"][view]["cumulative_query_counts"][stage]
                           for item in objects)
                for stage in objects[0]["target_views"][view]["cumulative_query_counts"]
            },
            "stable_triangle_count": sum(item["target_views"][view]["stable_triangle_count"]
                                         for item in objects),
            "objects_with_stable_triangles": sum(item["target_views"][view]["stable_triangle_count"] > 0
                                                 for item in objects),
            "normal_diagnostic": {
                "objects_with_normal_pair": sum(
                    item["target_views"][view]["normal_diagnostic"]["normal_pair_available"]
                    for item in objects
                ),
                **{
                    key: sum(
                        item["target_views"][view]["normal_diagnostic"][key] or 0
                        for item in objects
                    )
                    for key in (
                        "accepted_track_count", "accepted_with_normal_pair_count",
                        "source_normal_pass_count", "target_normal_pass_count",
                        "both_normal_pass_count",
                    )
                },
            },
        }
    audit = output / "training_visibility_route_audit.png"
    _render_audit(objects, audit)
    report = {
        "schema_version": "lalitha_phase5_training_visibility_route_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "Phase 5.18 training visibility and correspondence route audit",
        "scope": {"split": "train", "object_ids": list(TRAIN_OBJECTS),
                  "source_view": SOURCE_VIEW, "target_views": list(TARGET_VIEWS)},
        "policy": {"normal_evidence": "original Metric3D direct-normal records only",
                   "not_comparable_to": "Phase 5.8 validation-specific three-model normal corroboration",
                   "gate_changed": False},
        "summary": {"object_count": len(objects), "target_views": summary_views},
        "objects": objects,
        "checks": {"training_objects_only": all(item["split"] == "train" for item in objects),
                   "validation_objects_not_processed": True,
                   "frozen_phase5_13_gate_unchanged": True},
        "claims": {"physical_occlusion_identified": False,
                   "validation_relief_improved": False,
                   "surface_promotion_authorized": False,
                   "cad_fusion_authorized": False},
        "limitations": [
            "Target jewelry mask is segmentation, not physical visibility ground truth.",
            "VGGT visibility and reciprocal tracks are machine hypotheses.",
            "Normal pass counts are defined only where both original direct-normal maps exist; missing maps are not failures of normal agreement.",
            "The validation split has a separate Phase 5.8 three-model normal gate; this training-only route audit does not estimate that gate's outcome.",
            "Track-topology triangles use accepted tracks, not normal-corroborated tracks or measured surfaces.",
            "A missing front-to-top route can reflect occlusion, foreshortening, weak texture, or model failure.",
            "This diagnostic does not rerun or improve the independent Phase 5.13 surface gate.",
        ],
        "artifacts": {"audit": _relative(audit, root)},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("data/phase5_training_visibility_route_v1"))
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    report = run(root, output)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
