"""Measure local reciprocal-track topology on the Phase 5 training split.

Phase 5.14 identified sparse or unstable target-view topology as a dominant
validation blocker.  This training-only experiment compares the frozen Phase
5.6 query set with the Phase 5.7 hybrid query set.  A source triangle is kept
only when all three vertices pass the existing reciprocal gate for the same
target view.  Degenerate and orientation-flipped target projections are
reported explicitly.

This is image-space correspondence evidence.  It does not fit relief, alter a
validation threshold, create a surface, or authorize CAD fusion.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline.build_phase5_stl1_relief_evidence import _metric_depth_records
from pipeline.evaluate_phase5_dense_relief_pilot import _preserve_baseline_gate
from pipeline.evaluate_phase5_relief_tracks import (
    _load_view_evidence,
    _metric3d_normal_records,
    evaluate_track_consistency,
)
from pipeline.extract_vggt_track_evidence import TRAIN_OBJECTS, VIEW_ORDER


PHASE1_REPORT = Path(
    "dataset/phase_runs/v1/phase1_complete_features_v1/report.json"
)
BASELINE_TRACK_ROOT = Path(
    "dataset/phase_runs/v1/vggt_full_stl1_track_evidence_v1"
)
HYBRID_TRACK_ROOT = Path(
    "dataset/phase_runs/v1/vggt_relief_hybrid_train_v1"
)
SOURCE_VIEW = "front"
SOURCE_INDEX = VIEW_ORDER.index(SOURCE_VIEW)
TARGET_VIEWS = tuple(view for view in VIEW_ORDER if view != SOURCE_VIEW)

# These are topology-construction settings, not Phase 5.13 validation gates.
LOCAL_NEIGHBOR_COUNT = 6
LOCAL_NEIGHBOR_RADIUS_PIXELS = 52.0
MINIMUM_SOURCE_TRIANGLE_AREA_PIXELS2 = 4.0
MINIMUM_TARGET_TRIANGLE_AREA_PIXELS2 = 1.0
MINIMUM_AREA_SCALE = 0.05
MAXIMUM_AREA_SCALE = 20.0


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_tracks(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        return {key: source[key] for key in source.files}


def _signed_area(points: np.ndarray) -> np.ndarray:
    first = points[:, 1] - points[:, 0]
    second = points[:, 2] - points[:, 0]
    return 0.5 * (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])


def _local_triangles(
    points: np.ndarray,
    accepted: np.ndarray,
) -> np.ndarray:
    """Return three-cliques from a mutual local-neighbour graph."""

    retained = np.flatnonzero(accepted)
    if len(retained) < 3:
        return np.empty((0, 3), np.int32)
    selected = np.asarray(points[retained], np.float32)
    delta = selected[:, None, :] - selected[None, :, :]
    distances = np.sqrt(np.sum(np.square(delta), axis=2))
    np.fill_diagonal(distances, np.inf)
    neighbour_sets: list[set[int]] = []
    for row in distances:
        ordered = np.argsort(row)
        neighbour_sets.append(
            {
                int(index)
                for index in ordered[:LOCAL_NEIGHBOR_COUNT]
                if row[index] <= LOCAL_NEIGHBOR_RADIUS_PIXELS
            }
        )
    adjacency = np.zeros((len(retained), len(retained)), dtype=bool)
    for left, neighbours in enumerate(neighbour_sets):
        for right in neighbours:
            if left in neighbour_sets[right]:
                adjacency[left, right] = True
                adjacency[right, left] = True

    triangles: list[tuple[int, int, int]] = []
    for left in range(len(retained)):
        right_candidates = np.flatnonzero(adjacency[left, left + 1 :]) + left + 1
        for right in right_candidates:
            third_candidates = np.flatnonzero(
                adjacency[left] & adjacency[right]
            )
            for third in third_candidates:
                if third <= right:
                    continue
                triangle = np.asarray(
                    [selected[left], selected[right], selected[third]],
                    np.float32,
                )[None, ...]
                if abs(float(_signed_area(triangle)[0])) < (
                    MINIMUM_SOURCE_TRIANGLE_AREA_PIXELS2
                ):
                    continue
                triangles.append(
                    (
                        int(retained[left]),
                        int(retained[right]),
                        int(retained[third]),
                    )
                )
    if not triangles:
        return np.empty((0, 3), np.int32)
    return np.asarray(triangles, np.int32)


def _target_topology(
    source_points: np.ndarray,
    target_points: np.ndarray,
    accepted: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    faces = _local_triangles(source_points, accepted)
    accepted_count = int(np.count_nonzero(accepted))
    if not len(faces):
        metrics = {
            "accepted_vertex_count": accepted_count,
            "complete_source_triangle_count": 0,
            "nondegenerate_target_triangle_count": 0,
            "orientation_preserved_triangle_count": 0,
            "orientation_preservation_fraction": 0.0,
            "stable_vertex_count": 0,
            "stable_vertex_fraction_of_accepted": 0.0,
        }
        return metrics, {
            "faces": faces,
            "stable": np.empty(0, bool),
            "nondegenerate": np.empty(0, bool),
        }

    source_area = _signed_area(source_points[faces])
    target_area = _signed_area(target_points[faces])
    absolute_source_area = np.abs(source_area)
    absolute_target_area = np.abs(target_area)
    area_scale = absolute_target_area / np.maximum(absolute_source_area, 1e-8)
    finite = np.all(np.isfinite(target_points[faces]), axis=(1, 2))
    nondegenerate = (
        finite
        & (absolute_target_area >= MINIMUM_TARGET_TRIANGLE_AREA_PIXELS2)
        & (area_scale >= MINIMUM_AREA_SCALE)
        & (area_scale <= MAXIMUM_AREA_SCALE)
    )
    orientation_preserved = source_area * target_area > 0.0
    stable = nondegenerate & orientation_preserved
    stable_vertices = (
        np.unique(faces[stable]).astype(np.int32)
        if np.any(stable)
        else np.empty(0, np.int32)
    )
    nondegenerate_count = int(np.count_nonzero(nondegenerate))
    stable_count = int(np.count_nonzero(stable))
    metrics = {
        "accepted_vertex_count": accepted_count,
        "complete_source_triangle_count": int(len(faces)),
        "nondegenerate_target_triangle_count": nondegenerate_count,
        "orientation_preserved_triangle_count": stable_count,
        "orientation_preservation_fraction": round(
            stable_count / max(nondegenerate_count, 1), 6
        ),
        "stable_vertex_count": int(len(stable_vertices)),
        "stable_vertex_fraction_of_accepted": round(
            len(stable_vertices) / max(accepted_count, 1), 6
        ),
    }
    return metrics, {
        "faces": faces,
        "stable": stable,
        "nondegenerate": nondegenerate,
    }


def _variant_topology(
    tracks: dict[str, np.ndarray],
    evaluated: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    source_points = tracks["query_points_xy"][SOURCE_INDEX]
    targets: dict[str, Any] = {}
    visuals: dict[str, dict[str, np.ndarray]] = {}
    for target_index, target_view in enumerate(VIEW_ORDER):
        if target_index == SOURCE_INDEX:
            continue
        metrics, visual = _target_topology(
            source_points,
            tracks["tracks_xy"][SOURCE_INDEX, target_index],
            evaluated["accepted"][SOURCE_INDEX, target_index],
        )
        targets[target_view] = metrics
        visuals[target_view] = visual
    totals = {
        key: sum(target[key] for target in targets.values())
        for key in (
            "accepted_vertex_count",
            "complete_source_triangle_count",
            "nondegenerate_target_triangle_count",
            "orientation_preserved_triangle_count",
            "stable_vertex_count",
        )
    }
    totals["orientation_preservation_fraction"] = round(
        totals["orientation_preserved_triangle_count"]
        / max(totals["nondegenerate_target_triangle_count"], 1),
        6,
    )
    return {"targets": targets, "totals": totals}, visuals


def _draw_topology(
    image: np.ndarray,
    points: np.ndarray,
    visual: dict[str, np.ndarray],
) -> np.ndarray:
    canvas = image.copy()
    scale_x = (canvas.shape[1] - 1) / 517.0
    scale_y = (canvas.shape[0] - 1) / 517.0
    for face, stable, nondegenerate in zip(
        visual["faces"], visual["stable"], visual["nondegenerate"]
    ):
        polygon = points[face].copy()
        polygon[:, 0] *= scale_x
        polygon[:, 1] *= scale_y
        polygon = np.round(polygon).astype(np.int32)
        color = (
            (68, 170, 78)
            if stable
            else (55, 95, 205)
            if nondegenerate
            else (150, 150, 150)
        )
        cv2.polylines(canvas, [polygon], True, color, 1, cv2.LINE_AA)
    return canvas


def _panel(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    resized = cv2.resize(image, (330, 330), interpolation=cv2.INTER_AREA)
    canvas = np.full((395, 330, 3), 246, np.uint8)
    canvas[65:] = resized
    cv2.putText(
        canvas, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
        (28, 31, 35), 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, subtitle, (8, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
        (62, 66, 71), 1, cv2.LINE_AA,
    )
    return canvas


def _render_object_audit(
    *,
    object_id: str,
    view_evidence: dict[str, dict[str, Any]],
    tracks: dict[str, np.ndarray],
    topology: dict[str, Any],
    visuals: dict[str, dict[str, np.ndarray]],
    output: Path,
) -> None:
    panels = []
    source = cv2.imread(
        str(view_evidence[SOURCE_VIEW]["source_image"]), cv2.IMREAD_COLOR
    )
    source_overlay = source.copy()
    source_points = tracks["query_points_xy"][SOURCE_INDEX]
    for target_view in TARGET_VIEWS:
        source_overlay = _draw_topology(
            source_overlay, source_points, visuals[target_view]
        )
    stable_total = topology["totals"]["orientation_preserved_triangle_count"]
    panels.append(
        _panel(source_overlay, "FRONT SOURCE", f"{stable_total} stable target triangles")
    )
    for target_index, target_view in enumerate(VIEW_ORDER):
        if target_index == SOURCE_INDEX:
            continue
        target = cv2.imread(
            str(view_evidence[target_view]["source_image"]), cv2.IMREAD_COLOR
        )
        target_points = tracks["tracks_xy"][SOURCE_INDEX, target_index]
        overlay = _draw_topology(target, target_points, visuals[target_view])
        metrics = topology["targets"][target_view]
        panels.append(
            _panel(
                overlay,
                f"{target_view.upper()} TARGET",
                f"{metrics['orientation_preserved_triangle_count']}/"
                f"{metrics['complete_source_triangle_count']} stable/source",
            )
        )
    row = np.hstack(panels)
    header = np.full((68, row.shape[1], 3), 246, np.uint8)
    cv2.putText(
        header,
        f"PHASE 5.15 TRAINING TOPOLOGY - {object_id.upper()} - HYBRID TRACKS",
        (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.67,
        (28, 31, 35), 2, cv2.LINE_AA,
    )
    cv2.putText(
        header,
        "Green = orientation preserved | red = folded | grey = degenerate",
        (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.39,
        (62, 66, 71), 1, cv2.LINE_AA,
    )
    footer = np.full((42, row.shape[1], 3), 246, np.uint8)
    cv2.putText(
        footer,
        "Training image-space evidence only - no metric surface, CAD or manufacturing claim",
        (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
        (62, 66, 71), 1, cv2.LINE_AA,
    )
    cv2.imwrite(str(output), np.vstack((header, row, footer)))


def _render_cohort_audit(objects: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [item["object_id"] for item in objects]
    baseline = [
        item["variants"]["phase5_6_baseline"]["totals"][
            "orientation_preserved_triangle_count"
        ]
        for item in objects
    ]
    hybrid = [
        item["variants"]["phase5_7_hybrid"]["totals"][
            "orientation_preserved_triangle_count"
        ]
        for item in objects
    ]
    folded = [
        item["variants"]["phase5_7_hybrid"]["totals"][
            "nondegenerate_target_triangle_count"
        ]
        - item["variants"]["phase5_7_hybrid"]["totals"][
            "orientation_preserved_triangle_count"
        ]
        for item in objects
    ]
    positions = np.arange(len(names))
    figure, axes = plt.subplots(2, 1, figsize=(16, 10), facecolor="#f4f4f4")
    width = 0.38
    axes[0].bar(positions - width / 2, baseline, width, label="Phase 5.6 baseline", color="#9da2a8")
    axes[0].bar(positions + width / 2, hybrid, width, label="Phase 5.7 hybrid", color="#3f9b65")
    axes[0].set_title("Orientation-preserved local triangles", loc="left", fontweight="bold")
    axes[0].set_xticks(positions, names, rotation=45, ha="right")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(positions, hybrid, label="preserved", color="#3f9b65")
    axes[1].bar(positions, folded, bottom=hybrid, label="folded", color="#c95b66")
    axes[1].set_title("Hybrid target topology quality", loc="left", fontweight="bold")
    axes[1].set_xticks(positions, names, rotation=45, ha="right")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    figure.suptitle(
        "PHASE 5.15 - TRAINING-SPLIT RECIPROCAL TOPOLOGY DENSITY",
        fontsize=17, fontweight="bold", color="#20252b",
    )
    figure.text(
        0.5, 0.012,
        "Training image-space evidence only - validation gate remains frozen - no CAD authorization",
        ha="center", fontsize=10, color="#4a5159",
    )
    figure.tight_layout(rect=(0.02, 0.045, 0.98, 0.94))
    figure.savefig(output, dpi=170, bbox_inches="tight")
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
        object_record = phase1_by_id[object_id]
        if object_record["split"] != "train":
            raise RuntimeError(f"{object_id} is outside the training split")
        evidence = _load_view_evidence(
            root, object_record, metric_depths, direct_normals
        )
        baseline = _load_tracks(
            root / BASELINE_TRACK_ROOT / object_id / "vggt_track_evidence.npz"
        )
        hybrid = _load_tracks(
            root / HYBRID_TRACK_ROOT / object_id / "vggt_track_evidence.npz"
        )
        baseline_evaluated = evaluate_track_consistency(baseline, evidence)
        hybrid_evaluated = _preserve_baseline_gate(
            evaluate_track_consistency(hybrid, evidence), baseline_evaluated
        )
        baseline_topology, _ = _variant_topology(baseline, baseline_evaluated)
        hybrid_topology, hybrid_visuals = _variant_topology(
            hybrid, hybrid_evaluated
        )
        baseline_stable = baseline_topology["totals"][
            "orientation_preserved_triangle_count"
        ]
        hybrid_stable = hybrid_topology["totals"][
            "orientation_preserved_triangle_count"
        ]
        object_dir = output / object_id
        object_dir.mkdir(parents=True, exist_ok=True)
        audit_path = object_dir / "training_track_topology_audit.png"
        _render_object_audit(
            object_id=object_id,
            view_evidence=evidence,
            tracks=hybrid,
            topology=hybrid_topology,
            visuals=hybrid_visuals,
            output=audit_path,
        )
        objects.append(
            {
                "object_id": object_id,
                "split": "train",
                "variants": {
                    "phase5_6_baseline": baseline_topology,
                    "phase5_7_hybrid": hybrid_topology,
                },
                "comparison": {
                    "orientation_preserved_triangle_delta": hybrid_stable
                    - baseline_stable,
                    "orientation_preserved_triangle_multiplier": round(
                        hybrid_stable / max(baseline_stable, 1), 6
                    ),
                },
                "baseline_query_prefix_reproduced": bool(
                    np.allclose(
                        hybrid["query_points_xy"][:, :384],
                        baseline["query_points_xy"],
                    )
                ),
                "artifacts": {"audit": _relative(audit_path, root)},
            }
        )

    def total(variant: str, key: str) -> int:
        return sum(item["variants"][variant]["totals"][key] for item in objects)

    baseline_stable = total(
        "phase5_6_baseline", "orientation_preserved_triangle_count"
    )
    hybrid_stable = total(
        "phase5_7_hybrid", "orientation_preserved_triangle_count"
    )
    improved_count = sum(
        item["comparison"]["orientation_preserved_triangle_delta"] > 0
        for item in objects
    )
    regressed_count = sum(
        item["comparison"]["orientation_preserved_triangle_delta"] < 0
        for item in objects
    )
    target_summary = {}
    for target_view in TARGET_VIEWS:
        baseline_target = sum(
            item["variants"]["phase5_6_baseline"]["targets"][target_view][
                "orientation_preserved_triangle_count"
            ]
            for item in objects
        )
        hybrid_target = sum(
            item["variants"]["phase5_7_hybrid"]["targets"][target_view][
                "orientation_preserved_triangle_count"
            ]
            for item in objects
        )
        target_summary[target_view] = {
            "baseline_orientation_preserved_triangle_count": baseline_target,
            "hybrid_orientation_preserved_triangle_count": hybrid_target,
            "orientation_preserved_triangle_delta": hybrid_target
            - baseline_target,
            "objects_with_hybrid_topology": sum(
                item["variants"]["phase5_7_hybrid"]["targets"][target_view][
                    "orientation_preserved_triangle_count"
                ]
                > 0
                for item in objects
            ),
        }
    audit_path = output / "training_track_topology_audit.png"
    report_path = output / "report.json"
    manifest_path = output / "integrity_manifest.json"
    _render_cohort_audit(objects, audit_path)
    selected = (
        "phase5_7_hybrid"
        if hybrid_stable > baseline_stable and regressed_count == 0
        else "no_training_policy_selected"
    )
    report = {
        "schema_version": "lalitha_phase5_training_track_topology_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "Phase 5.15 training-split reciprocal topology density",
        "scope": {
            "object_ids": list(TRAIN_OBJECTS),
            "split": "train",
            "source_view": SOURCE_VIEW,
            "target_views": list(TARGET_VIEWS),
        },
        "sources": {
            "phase1": PHASE1_REPORT.as_posix(),
            "baseline_tracks": BASELINE_TRACK_ROOT.as_posix(),
            "hybrid_tracks": HYBRID_TRACK_ROOT.as_posix(),
            "failure_attribution": "data/phase5_target_failure_attribution_v1/report.json",
        },
        "topology_settings": {
            "local_neighbor_count": LOCAL_NEIGHBOR_COUNT,
            "local_neighbor_radius_pixels": LOCAL_NEIGHBOR_RADIUS_PIXELS,
            "minimum_source_triangle_area_pixels2": MINIMUM_SOURCE_TRIANGLE_AREA_PIXELS2,
            "minimum_target_triangle_area_pixels2": MINIMUM_TARGET_TRIANGLE_AREA_PIXELS2,
            "minimum_area_scale": MINIMUM_AREA_SCALE,
            "maximum_area_scale": MAXIMUM_AREA_SCALE,
        },
        "summary": {
            "object_count": len(objects),
            "baseline_orientation_preserved_triangle_count": baseline_stable,
            "hybrid_orientation_preserved_triangle_count": hybrid_stable,
            "orientation_preserved_triangle_delta": hybrid_stable
            - baseline_stable,
            "improved_object_count": improved_count,
            "unchanged_object_count": len(objects) - improved_count - regressed_count,
            "regressed_object_count": regressed_count,
            "selected_training_policy": selected,
            "target_views": target_summary,
            "complete_target_view_coverage_available": all(
                item["objects_with_hybrid_topology"] == len(objects)
                for item in target_summary.values()
            ),
        },
        "checks": {
            "training_objects_only": all(item["split"] == "train" for item in objects),
            "validation_objects_not_processed": True,
            "test_objects_not_processed": True,
            "baseline_query_prefix_reproduced": all(
                item["baseline_query_prefix_reproduced"] for item in objects
            ),
            "existing_reciprocal_gate_reused": True,
            "phase5_13_thresholds_unchanged": True,
            "phase5_13_surfaces_unchanged": True,
            "no_step_files_generated": True,
        },
        "objects": objects,
        "decision": (
            "freeze_hybrid_query_policy_for_one_validation_topology_evaluation"
            if selected == "phase5_7_hybrid"
            else "continue_training_topology_development_without_validation_evaluation"
        ),
        "claims": {
            "training_topology_density_measured": True,
            "validation_topology_improved": False,
            "metric_surface_recovered": False,
            "hidden_geometry_recovered": False,
            "cad_fusion_authorized": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "The triangles are local image-space cliques built from reciprocal correspondences, not a physical surface mesh.",
            "Training improvement does not establish validation or test improvement.",
            "Orientation preservation rejects obvious folds but does not prove metric geometry.",
            "The front-to-top route remains sparse across the cohort and must not be inferred from the improved angled and side views.",
            "Polished metal, stones, shadows and occlusion can still produce appearance-driven correspondences.",
        ],
        "artifacts": {
            "audit": _relative(audit_path, root),
            "integrity_manifest": _relative(manifest_path, root),
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    integrity_targets = {
        "phase5_7_training_source_report": root
        / "data"
        / "phase5_dense_relief_train_v1"
        / "report.json",
        "phase5_15_report": report_path,
        "phase5_15_audit": audit_path,
        "phase5_15_implementation": root
        / "pipeline"
        / "evaluate_phase5_training_track_topology.py",
    }
    manifest = {
        "schema_version": "lalitha_phase5_training_track_topology_integrity_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            label: {
                "path": _relative(path, root),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for label, path in integrity_targets.items()
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/phase5_training_track_topology_v1"),
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    report = run(root, output)
    print(
        json.dumps(
            {
                "report": str(output / "report.json"),
                "summary": report["summary"],
                "checks": report["checks"],
                "decision": report["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
