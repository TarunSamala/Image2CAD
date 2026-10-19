"""Apply frozen Phase 5.15 reciprocal topology settings to validation rings.

This is a one-policy, one-split transfer check. It reads the retained training
decision and its integrity manifest, then measures the same local topology on
the three validation rings. It does not tune settings from validation results
or rerun the Phase 5.13 surface gate.
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

from pipeline.build_phase5_stl1_relief_evidence import _metric_depth_records
from pipeline.evaluate_phase5_dense_relief_pilot import _preserve_baseline_gate
from pipeline.evaluate_phase5_relief_tracks import (
    _load_view_evidence,
    _metric3d_normal_records,
    evaluate_track_consistency,
)
from pipeline.evaluate_phase5_training_track_topology import (
    BASELINE_TRACK_ROOT,
    PHASE1_REPORT,
    SOURCE_INDEX,
    SOURCE_VIEW,
    TARGET_VIEWS,
    _draw_topology,
    _load_tracks,
    _panel,
    _relative,
    _sha256,
    _variant_topology,
)
from pipeline.extract_vggt_track_evidence import VALIDATION_OBJECTS, VIEW_ORDER


TRAINING_ROOT = Path("data/phase5_training_track_topology_v1")
VALIDATION_TRACK_ROOT = Path(
    "dataset/phase_runs/v1/vggt_relief_hybrid_validation_v1"
)


def _verify_frozen_training(root: Path) -> dict[str, Any]:
    report_path = root / TRAINING_ROOT / "report.json"
    manifest_path = root / TRAINING_ROOT / "integrity_manifest.json"
    training = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if training["scope"]["split"] != "train":
        raise RuntimeError("Phase 5.15 source is not training-only")
    if training["summary"]["selected_training_policy"] != "phase5_7_hybrid":
        raise RuntimeError("Phase 5.15 did not select the hybrid policy")
    for entry in manifest["files"].values():
        path = root / entry["path"]
        if path.stat().st_size != entry["size_bytes"] or _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Frozen Phase 5.15 artifact changed: {entry['path']}")
    return training


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
    source_image = cv2.imread(
        str(view_evidence[SOURCE_VIEW]["source_image"]), cv2.IMREAD_COLOR
    )
    source_points = tracks["query_points_xy"][SOURCE_INDEX]
    for target_view in TARGET_VIEWS:
        source_image = _draw_topology(
            source_image, source_points, visuals[target_view]
        )
    panels.append(
        _panel(
            source_image,
            "FRONT SOURCE",
            f"{topology['totals']['orientation_preserved_triangle_count']} stable target triangles",
        )
    )
    for target_index, target_view in enumerate(VIEW_ORDER):
        if target_index == SOURCE_INDEX:
            continue
        target_image = cv2.imread(
            str(view_evidence[target_view]["source_image"]), cv2.IMREAD_COLOR
        )
        overlay = _draw_topology(
            target_image,
            tracks["tracks_xy"][SOURCE_INDEX, target_index],
            visuals[target_view],
        )
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
        f"PHASE 5.16 FROZEN VALIDATION TOPOLOGY - {object_id.upper()}",
        (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.66,
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
        "Frozen image-space policy - independent relief and CAD gates remain open",
        (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
        (62, 66, 71), 1, cv2.LINE_AA,
    )
    cv2.imwrite(str(output), np.vstack((header, row, footer)))


def _render_cohort_audit(objects: list[dict[str, Any]], output: Path) -> None:
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
    target_values = {
        view: [
            item["variants"]["phase5_7_hybrid"]["targets"][view][
                "orientation_preserved_triangle_count"
            ]
            for item in objects
        ]
        for view in TARGET_VIEWS
    }
    positions = np.arange(len(objects))
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), facecolor="#f4f4f4")
    axes[0].bar(positions - 0.2, baseline, 0.4, color="#9da2a8", label="baseline")
    axes[0].bar(positions + 0.2, hybrid, 0.4, color="#3f9b65", label="hybrid")
    axes[0].set_xticks(positions, names)
    axes[0].set_title("Stable local triangles after frozen policy transfer", loc="left")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    colors = {"top": "#b44978", "iso": "#3f9b65", "lsv": "#4f82bb", "rsv": "#c48a46"}
    bottom = np.zeros(len(objects))
    for view in TARGET_VIEWS:
        values = target_values[view]
        axes[1].bar(positions, values, bottom=bottom, label=view.upper(), color=colors[view])
        bottom += values
    axes[1].set_xticks(positions, names)
    axes[1].set_title("Hybrid topology by target view", loc="left")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    figure.suptitle("PHASE 5.16 - FROZEN VALIDATION TOPOLOGY", fontsize=17, fontweight="bold")
    figure.text(
        0.5, 0.01,
        "Image-space correspondences only - Phase 5.13 surface and CAD gates unchanged",
        ha="center", fontsize=10,
    )
    figure.tight_layout(rect=(0.02, 0.04, 0.98, 0.94))
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def run(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    output = output.resolve()
    training = _verify_frozen_training(root)
    phase1 = json.loads((root / PHASE1_REPORT).read_text(encoding="utf-8"))
    phase1_by_id = {item["object_id"]: item for item in phase1["objects"]}
    metric_depths = _metric_depth_records(root)
    direct_normals = _metric3d_normal_records(root)
    output.mkdir(parents=True, exist_ok=True)
    objects = []
    for object_id in VALIDATION_OBJECTS:
        object_record = phase1_by_id[object_id]
        if object_record["split"] != "val":
            raise RuntimeError(f"{object_id} is outside the validation split")
        evidence = _load_view_evidence(
            root, object_record, metric_depths, direct_normals
        )
        baseline = _load_tracks(
            root / BASELINE_TRACK_ROOT / object_id / "vggt_track_evidence.npz"
        )
        hybrid = _load_tracks(
            root / VALIDATION_TRACK_ROOT / object_id / "vggt_track_evidence.npz"
        )
        baseline_evaluated = evaluate_track_consistency(baseline, evidence)
        hybrid_evaluated = _preserve_baseline_gate(
            evaluate_track_consistency(hybrid, evidence), baseline_evaluated
        )
        baseline_topology, _ = _variant_topology(baseline, baseline_evaluated)
        hybrid_topology, hybrid_visuals = _variant_topology(
            hybrid, hybrid_evaluated
        )
        object_dir = output / object_id
        object_dir.mkdir(parents=True, exist_ok=True)
        audit_path = object_dir / "validation_track_topology_audit.png"
        _render_object_audit(
            object_id=object_id,
            view_evidence=evidence,
            tracks=hybrid,
            topology=hybrid_topology,
            visuals=hybrid_visuals,
            output=audit_path,
        )
        baseline_count = baseline_topology["totals"][
            "orientation_preserved_triangle_count"
        ]
        hybrid_count = hybrid_topology["totals"][
            "orientation_preserved_triangle_count"
        ]
        objects.append(
            {
                "object_id": object_id,
                "split": "val",
                "variants": {
                    "phase5_6_baseline": baseline_topology,
                    "phase5_7_hybrid": hybrid_topology,
                },
                "orientation_preserved_triangle_delta": hybrid_count - baseline_count,
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

    baseline_total = total("phase5_6_baseline", "orientation_preserved_triangle_count")
    hybrid_total = total("phase5_7_hybrid", "orientation_preserved_triangle_count")
    target_summary = {
        view: {
            "stable_triangle_count": sum(
                item["variants"]["phase5_7_hybrid"]["targets"][view][
                    "orientation_preserved_triangle_count"
                ]
                for item in objects
            ),
            "objects_with_stable_topology": sum(
                item["variants"]["phase5_7_hybrid"]["targets"][view][
                    "orientation_preserved_triangle_count"
                ] > 0
                for item in objects
            ),
        }
        for view in TARGET_VIEWS
    }
    audit_path = output / "validation_track_topology_audit.png"
    _render_cohort_audit(objects, audit_path)
    report = {
        "schema_version": "lalitha_phase5_validation_track_topology_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "Phase 5.16 frozen-policy validation topology transfer",
        "scope": {
            "object_ids": list(VALIDATION_OBJECTS),
            "split": "val",
            "source_view": SOURCE_VIEW,
            "target_views": list(TARGET_VIEWS),
        },
        "sources": {
            "frozen_training_report": f"{TRAINING_ROOT.as_posix()}/report.json",
            "frozen_training_manifest": f"{TRAINING_ROOT.as_posix()}/integrity_manifest.json",
            "baseline_tracks": BASELINE_TRACK_ROOT.as_posix(),
            "hybrid_tracks": VALIDATION_TRACK_ROOT.as_posix(),
        },
        "frozen_topology_settings": training["topology_settings"],
        "summary": {
            "object_count": len(objects),
            "baseline_orientation_preserved_triangle_count": baseline_total,
            "hybrid_orientation_preserved_triangle_count": hybrid_total,
            "orientation_preserved_triangle_delta": hybrid_total - baseline_total,
            "improved_object_count": sum(
                item["orientation_preserved_triangle_delta"] > 0 for item in objects
            ),
            "regressed_object_count": sum(
                item["orientation_preserved_triangle_delta"] < 0 for item in objects
            ),
            "target_views": target_summary,
            "complete_target_view_coverage_available": all(
                item["objects_with_stable_topology"] == len(objects)
                for item in target_summary.values()
            ),
        },
        "checks": {
            "frozen_training_manifest_verified": True,
            "fixed_validation_objects_only": all(item["split"] == "val" for item in objects),
            "baseline_query_prefix_reproduced": all(
                item["baseline_query_prefix_reproduced"] for item in objects
            ),
            "test_objects_not_processed": True,
            "phase5_13_thresholds_unchanged": True,
            "phase5_13_surfaces_unchanged": True,
            "no_step_files_generated": True,
        },
        "objects": objects,
        "decision": "retain_as_image_space_validation_topology_evidence_only",
        "claims": {
            "validation_topology_density_measured": True,
            "independent_surface_gate_improved": False,
            "metric_surface_recovered": False,
            "cad_fusion_authorized": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "Local reciprocal triangles are image-space correspondence evidence, not physical mesh faces.",
            "The existing Phase 5.13 target-view relief gate was not rerun or improved by this topology count.",
            "No metric scale, hidden geometry or manufacturing dimensions are recovered.",
        ],
        "artifacts": {"audit": _relative(audit_path, root)},
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/phase5_validation_track_topology_v1"),
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    report = run(root, output)
    print(json.dumps({"summary": report["summary"], "checks": report["checks"]}, indent=2))


if __name__ == "__main__":
    main()
