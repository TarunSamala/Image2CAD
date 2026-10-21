"""Audit front-region to target-view routes on the fixed training split.

This backtracks to correspondence routing before any relief surface is fit.
Image-space grid cells are not semantic jewellery components. A missing route
is unresolved evidence, never a declaration that a physical surface is absent.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline.build_phase5_stl1_relief_evidence import _metric_depth_records
from pipeline.evaluate_phase5_dense_relief_pilot import _preserve_baseline_gate
from pipeline.evaluate_phase5_relief_tracks import (
    _load_view_evidence,
    _metric3d_normal_records,
    evaluate_track_consistency,
)
from pipeline.evaluate_phase5_training_track_topology import (
    BASELINE_TRACK_ROOT,
    HYBRID_TRACK_ROOT,
    PHASE1_REPORT,
    SOURCE_INDEX,
    TARGET_VIEWS,
    _load_tracks,
    _relative,
)
from pipeline.extract_vggt_track_evidence import TRAIN_OBJECTS, VIEW_ORDER


GRID_SIZE = 3
MINIMUM_REGION_CANDIDATES = 10
MINIMUM_REVIEW_TRACKS = 3


def _grid_ids(points: np.ndarray, bbox_xyxy: tuple[float, float, float, float]) -> np.ndarray:
    """Map 518-pixel front-query coordinates to stable bbox-relative grid IDs."""
    x0, y0, x1, y1 = bbox_xyxy
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Front foreground bbox is empty")
    x = np.clip((points[:, 0] - x0) / (x1 - x0), 0.0, 1.0 - 1e-6)
    y = np.clip((points[:, 1] - y0) / (y1 - y0), 0.0, 1.0 - 1e-6)
    return (np.floor(y * GRID_SIZE).astype(np.int32) * GRID_SIZE
            + np.floor(x * GRID_SIZE).astype(np.int32))


def _region_record(
    region_id: int,
    selected: np.ndarray,
    candidate: np.ndarray,
    accepted_by_view: dict[str, np.ndarray],
) -> dict[str, Any]:
    candidates = int(np.count_nonzero(selected & candidate))
    views = {}
    for view in TARGET_VIEWS:
        accepted = int(np.count_nonzero(selected & candidate & accepted_by_view[view]))
        if candidates < MINIMUM_REGION_CANDIDATES:
            state = "insufficient_source_queries"
        elif accepted >= MINIMUM_REVIEW_TRACKS:
            state = "reciprocal_track_review_evidence"
        else:
            state = "cross_view_identity_unresolved"
        views[view] = {
            "accepted_track_count": accepted,
            "accepted_fraction_of_candidates": round(accepted / max(candidates, 1), 6),
            "state": state,
        }
    return {
        "region_id": f"front_grid_{region_id // GRID_SIZE}_{region_id % GRID_SIZE}",
        "row": region_id // GRID_SIZE,
        "column": region_id % GRID_SIZE,
        "candidate_count": candidates,
        "target_views": views,
        "semantic_label": "unknown_image_space_region",
    }


def _render_audit(summary: dict[str, Any], output: Path) -> None:
    figure, axes = plt.subplots(1, len(TARGET_VIEWS), figsize=(15, 4.7),
                                 facecolor="#f4f4f4")
    for axis, view in zip(axes, TARGET_VIEWS):
        cells = summary["target_views"][view]["regions"]
        fractions = np.asarray(
            [item["accepted_fraction_of_candidates"] for item in cells],
            np.float32,
        ).reshape(GRID_SIZE, GRID_SIZE)
        axis.imshow(fractions, cmap="viridis", vmin=0.0, vmax=0.35)
        axis.set_title(view.upper())
        axis.set_xticks(range(GRID_SIZE), ["left", "centre", "right"])
        axis.set_yticks(range(GRID_SIZE), ["upper", "middle", "lower"])
        for region, item in enumerate(cells):
            row, column = divmod(region, GRID_SIZE)
            axis.text(column, row,
                      f"{item['accepted_track_count']}/{item['candidate_count']}",
                      ha="center", va="center", fontsize=9,
                      color="white" if fractions[row, column] < 0.2 else "black")
    figure.suptitle("PHASE 5.20 - TRAINING FRONT-REGION VIEW ROUTES",
                    fontsize=16, fontweight="bold")
    figure.text(0.5, 0.025,
                "Count = accepted reciprocal tracks / front relief candidates; grid cells are not semantic components or physical visibility truth",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0.01, 0.08, 0.99, 0.91))
    figure.savefig(output, dpi=170, bbox_inches="tight")
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
        baseline = _load_tracks(root / BASELINE_TRACK_ROOT / object_id / "vggt_track_evidence.npz")
        tracks = _load_tracks(root / HYBRID_TRACK_ROOT / object_id / "vggt_track_evidence.npz")
        if not np.allclose(tracks["query_points_xy"][:, :384], baseline["query_points_xy"]):
            raise RuntimeError(f"{object_id} does not preserve baseline queries")
        evaluated = _preserve_baseline_gate(
            evaluate_track_consistency(tracks, evidence),
            evaluate_track_consistency(baseline, evidence),
        )
        source_points = tracks["query_points_xy"][SOURCE_INDEX]
        candidate = evaluated["candidate"][SOURCE_INDEX]
        if not np.any(candidate):
            raise RuntimeError(f"{object_id} has no front relief candidate")
        # Bounding only the source relief candidate is intentional: no hidden
        # or physical component boundary is inferred from a whole-ring mask.
        xy = source_points[candidate]
        bbox = (float(xy[:, 0].min()), float(xy[:, 1].min()),
                float(xy[:, 0].max()), float(xy[:, 1].max()))
        grid_ids = _grid_ids(source_points, bbox)
        accepted_by_view = {
            view: evaluated["accepted"][SOURCE_INDEX, index]
            for index, view in enumerate(VIEW_ORDER) if index != SOURCE_INDEX
        }
        regions = [
            _region_record(region_id, grid_ids == region_id, candidate, accepted_by_view)
            for region_id in range(GRID_SIZE * GRID_SIZE)
        ]
        objects.append({
            "object_id": object_id, "split": "train",
            "front_candidate_bbox_xyxy_track_pixels": [round(value, 3) for value in bbox],
            "candidate_count": int(np.count_nonzero(candidate)),
            "regions": regions,
            "checks": {
                "regions_partition_candidates": sum(item["candidate_count"] for item in regions)
                == int(np.count_nonzero(candidate)),
                "accepted_counts_reproduced": all(
                    sum(item["target_views"][view]["accepted_track_count"] for item in regions)
                    == int(np.count_nonzero(accepted_by_view[view] & candidate))
                    for view in TARGET_VIEWS
                ),
            },
        })
    target_views = {}
    for view in TARGET_VIEWS:
        regions = []
        for region_id in range(GRID_SIZE * GRID_SIZE):
            candidates = sum(item["regions"][region_id]["candidate_count"] for item in objects)
            accepted = sum(
                item["regions"][region_id]["target_views"][view]["accepted_track_count"]
                for item in objects
            )
            regions.append({
                "region_id": f"front_grid_{region_id // GRID_SIZE}_{region_id % GRID_SIZE}",
                "candidate_count": candidates,
                "accepted_track_count": accepted,
                "accepted_fraction_of_candidates": round(accepted / max(candidates, 1), 6),
                "objects_with_review_evidence": sum(
                    item["regions"][region_id]["target_views"][view]["state"]
                    == "reciprocal_track_review_evidence" for item in objects
                ),
            })
        target_views[view] = {
            "regions": regions,
            "accepted_track_count": sum(item["accepted_track_count"] for item in regions),
            "objects_with_any_accepted_track": sum(
                any(region["target_views"][view]["accepted_track_count"] > 0
                    for region in item["regions"])
                for item in objects
            ),
        }
    summary = {
        "object_count": len(objects),
        "front_candidate_count": sum(item["candidate_count"] for item in objects),
        "target_views": target_views,
    }
    audit = output / "training_region_route_audit.png"
    _render_audit(summary, audit)
    report = {
        "schema_version": "lalitha_phase5_training_region_routes_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "Phase 5.20 training image-space front-region route backtrack",
        "scope": {"split": "train", "object_ids": list(TRAIN_OBJECTS),
                  "source_view": "front", "target_views": list(TARGET_VIEWS)},
        "policy": {"grid_size": GRID_SIZE,
                   "minimum_region_candidates": MINIMUM_REGION_CANDIDATES,
                   "minimum_review_tracks": MINIMUM_REVIEW_TRACKS,
                   "frozen_reciprocal_thresholds_reused": True,
                   "source_region_basis": "front relief query bbox, not a semantic component"},
        "summary": summary,
        "objects": objects,
        "checks": {
            "training_only": all(item["split"] == "train" for item in objects),
            "candidate_partition_valid": all(item["checks"]["regions_partition_candidates"] for item in objects),
            "accepted_counts_reproduced": all(item["checks"]["accepted_counts_reproduced"] for item in objects),
            "validation_and_test_untouched": True,
            "phase5_13_surface_gate_unchanged": True,
        },
        "decision": "use_region_route_matrix_to_design_visibility_annotations_on_training_images",
        "claims": {"physical_visibility_validated": False,
                   "semantic_components_identified": False,
                   "matching_accuracy_improved": False,
                   "surface_promotion_authorized": False,
                   "cad_fusion_authorized": False},
        "limitations": [
            "Grid cells pool disparate ornaments and are not component IDs.",
            "Reciprocal VGGT tracks are machine hypotheses, not labelled correspondence truth.",
            "Zero accepted tracks can mean occlusion, foreshortening, missing target detail or model failure.",
            "Counts cannot establish the accuracy of a new route policy or missing geometry.",
        ],
        "artifacts": {"audit": _relative(audit, root)},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("data/phase5_training_region_routes_v1"))
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    report = run(root, output)
    print(json.dumps({"summary": {"object_count": report["summary"]["object_count"],
                                   "front_candidate_count": report["summary"]["front_candidate_count"],
                                   "accepted_by_view": {
                                       view: item["accepted_track_count"]
                                       for view, item in report["summary"]["target_views"].items()
                                   }}}, indent=2))


if __name__ == "__main__":
    main()
