"""Attribute Phase 5.13 target-view failures without changing its gate.

The retained Phase 5.13 report is treated as frozen validation evidence.  This
stage classifies why each target evaluation failed and emits a patch action
queue.  It does not recompute scores, tune thresholds, modify surfaces or
authorize CAD fusion.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline.extract_vggt_track_evidence import VALIDATION_OBJECTS


TARGET_REPORT_ROOT = Path("data/phase5_target_reprojection_v1")

BLOCKER_ORDER = (
    ("minimum_tracked_queries", "insufficient_normal_tracks"),
    ("minimum_projected_faces", "no_stable_target_triangle"),
    ("minimum_projected_pixels", "insufficient_projected_area"),
    ("orientation_preserved", "orientation_instability"),
    ("minimum_target_support_fraction", "target_support_gap"),
    ("maximum_target_relief_rmse", "target_relief_rmse_exceeded"),
    ("beats_zero_relief_baseline", "no_relief_transfer_gain"),
)

STRUCTURAL_BLOCKERS = {
    "insufficient_normal_tracks",
    "no_stable_target_triangle",
    "insufficient_projected_area",
    "orientation_instability",
    "target_support_gap",
}


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _blockers(checks: dict[str, bool]) -> list[str]:
    return [label for key, label in BLOCKER_ORDER if not checks.get(key, False)]


def _primary_blocker(checks: dict[str, bool]) -> str:
    failures = _blockers(checks)
    return failures[0] if failures else "gate_passed"


def _action_for_patch(
    state: str,
    blockers: set[str],
) -> tuple[int, str]:
    if state == "multi_target_supported_review":
        return 1, "manual_surface_review_without_cad_promotion"
    if state == "single_target_supported_review":
        return 1, "seek_second_independent_target_view"
    if blockers & STRUCTURAL_BLOCKERS:
        return 2, "improve_cross_view_track_topology_on_training_split"
    if blockers:
        return 3, "develop_view_invariant_relief_on_training_split"
    return 4, "retain_unresolved_until_more_image_evidence"


def _render_audit(report: dict[str, Any], output: Path) -> None:
    summary = report["summary"]
    blocker_counts = summary["primary_blocker_counts"]
    action_counts = summary["recommended_action_counts"]
    state_counts = summary["patch_state_counts"]
    objects = report["objects"]

    figure, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#f4f4f4")
    figure.suptitle(
        "PHASE 5.14 - TARGET REPROJECTION FAILURE ATTRIBUTION",
        fontsize=17,
        fontweight="bold",
        color="#20252b",
    )

    def horizontal_bar(axis: Any, values: dict[str, int], title: str) -> None:
        labels = [label.replace("_", " ") for label in values]
        counts = list(values.values())
        positions = np.arange(len(labels))
        axis.barh(positions, counts, color="#b44978")
        axis.set_yticks(positions, labels=labels, fontsize=8)
        axis.invert_yaxis()
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="x", alpha=0.2)
        for position, count in zip(positions, counts):
            axis.text(count + 0.2, position, str(count), va="center", fontsize=8)

    horizontal_bar(axes[0, 0], blocker_counts, "Primary target-evaluation blockers")
    horizontal_bar(axes[0, 1], state_counts, "Patch validation states")
    horizontal_bar(axes[1, 0], action_counts, "Frozen-gate improvement actions")

    axis = axes[1, 1]
    names = [item["object_id"] for item in objects]
    evaluated = [item["target_evaluation_count"] for item in objects]
    passed = [item["passed_target_evaluation_count"] for item in objects]
    positions = np.arange(len(names))
    axis.bar(positions, evaluated, color="#c8cbd0", label="evaluated")
    axis.bar(positions, passed, color="#3f9b65", label="passed")
    axis.set_xticks(positions, labels=names)
    axis.set_title("Independent target gates by object", loc="left", fontweight="bold")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    for position, total, accepted in zip(positions, evaluated, passed):
        axis.text(position, total + 0.4, f"{accepted}/{total}", ha="center", fontsize=9)

    figure.text(
        0.5,
        0.02,
        "Diagnostic only - Phase 5.13 thresholds and surfaces remain frozen - no CAD or manufacturing authorization",
        ha="center",
        fontsize=10,
        color="#4a5159",
    )
    figure.tight_layout(rect=(0.02, 0.05, 0.98, 0.93))
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def run(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_report = json.loads(
        (root / TARGET_REPORT_ROOT / "report.json").read_text(encoding="utf-8")
    )
    rows: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    primary_blocker_counts: dict[str, int] = {}
    patch_state_counts: dict[str, int] = {}
    recommended_action_counts: dict[str, int] = {}

    for source_object in source_report["objects"]:
        object_id = source_object["object_id"]
        object_report = json.loads(
            (root / source_object["artifacts"]["report"]).read_text(
                encoding="utf-8"
            )
        )
        object_blockers: dict[str, int] = {}
        object_actions: dict[str, int] = {}
        for patch in object_report["patches"]:
            all_blockers: set[str] = set()
            evaluation_diagnostics = []
            for evaluation in patch["target_evaluations"]:
                blockers = _blockers(evaluation["checks"])
                primary = _primary_blocker(evaluation["checks"])
                all_blockers.update(blockers)
                primary_blocker_counts[primary] = (
                    primary_blocker_counts.get(primary, 0) + 1
                )
                object_blockers[primary] = object_blockers.get(primary, 0) + 1
                evaluation_diagnostics.append(
                    {
                        "target_view": evaluation["target_view"],
                        "gate_passed": evaluation[
                            "independent_target_gate_passed"
                        ],
                        "primary_blocker": primary,
                        "all_blockers": blockers,
                    }
                )
            state = patch["independent_target_state"]
            priority, action = _action_for_patch(state, all_blockers)
            patch_state_counts[state] = patch_state_counts.get(state, 0) + 1
            recommended_action_counts[action] = (
                recommended_action_counts.get(action, 0) + 1
            )
            object_actions[action] = object_actions.get(action, 0) + 1
            rows.append(
                {
                    "object_id": object_id,
                    "patch_id": patch["patch_id"],
                    "independent_target_state": state,
                    "query_count": patch["query_count"],
                    "target_evaluation_count": patch[
                        "target_evaluation_count"
                    ],
                    "passed_target_evaluation_count": patch[
                        "passed_target_evaluation_count"
                    ],
                    "priority": priority,
                    "recommended_action": action,
                    "blockers": sorted(all_blockers),
                    "target_diagnostics": evaluation_diagnostics,
                    "surface_promotion_authorized": False,
                    "cad_fusion_authorized": False,
                }
            )
        objects.append(
            {
                "object_id": object_id,
                "split": source_object["split"],
                "eligible_patch_count": source_object[
                    "eligible_patch_count"
                ],
                "target_evaluation_count": source_object[
                    "target_evaluation_count"
                ],
                "passed_target_evaluation_count": source_object[
                    "passed_target_evaluation_count"
                ],
                "primary_blocker_counts": object_blockers,
                "recommended_action_counts": object_actions,
            }
        )

    rows.sort(key=lambda item: (item["priority"], item["object_id"], item["patch_id"]))
    queue_path = output / "patch_action_queue.csv"
    with queue_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "priority",
                "object_id",
                "patch_id",
                "independent_target_state",
                "query_count",
                "target_evaluation_count",
                "passed_target_evaluation_count",
                "recommended_action",
                "blockers",
                "surface_promotion_authorized",
                "cad_fusion_authorized",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        ";".join(row[key]) if key == "blockers" else row[key]
                    )
                    for key in writer.fieldnames
                }
            )

    audit_path = output / "target_reprojection_failure_audit.png"
    report_path = output / "report.json"
    manifest_path = output / "integrity_manifest.json"
    report = {
        "schema_version": "lalitha_phase5_target_failure_attribution_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "Phase 5.14 frozen target-reprojection failure attribution",
        "scope": {
            "object_ids": list(VALIDATION_OBJECTS),
            "split": "val",
            "phase5_13_gate_frozen": True,
        },
        "source": f"{TARGET_REPORT_ROOT.as_posix()}/report.json",
        "summary": {
            "object_count": len(objects),
            "patch_count": len(rows),
            "target_evaluation_count": sum(
                item["target_evaluation_count"] for item in objects
            ),
            "passed_target_evaluation_count": sum(
                item["passed_target_evaluation_count"] for item in objects
            ),
            "primary_blocker_counts": dict(
                sorted(
                    primary_blocker_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            "patch_state_counts": dict(sorted(patch_state_counts.items())),
            "recommended_action_counts": dict(
                sorted(
                    recommended_action_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
        },
        "checks": {
            "fixed_validation_objects_only": all(
                item["split"] == "val" for item in objects
            ),
            "phase5_13_patch_count_reproduced": len(rows)
            == source_report["summary"]["eligible_patch_count"],
            "phase5_13_evaluation_count_reproduced": sum(
                item["target_evaluation_count"] for item in objects
            )
            == source_report["summary"]["target_evaluation_count"],
            "phase5_13_pass_count_reproduced": sum(
                item["passed_target_evaluation_count"] for item in objects
            )
            == source_report["summary"]["passed_target_evaluation_count"],
            "thresholds_unchanged": True,
            "surfaces_unchanged": True,
            "test_objects_not_processed": True,
            "exact_cad_unchanged": True,
        },
        "objects": objects,
        "patch_action_queue": rows,
        "decision": "use_failure_attribution_to_design_training_split_improvements",
        "claims": {
            "physical_accuracy_validated": False,
            "target_gate_improved": False,
            "surface_promotion_authorized": False,
            "cad_fusion_authorized": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "Failure labels explain the frozen Phase 5.13 checks; they do not prove a unique physical root cause.",
            "Recommended actions must be developed on training data before the validation gate is rerun.",
            "A passing target evaluation remains review evidence and does not authorize CAD fusion.",
        ],
        "artifacts": {
            "audit": _relative(audit_path, root),
            "patch_action_queue_csv": _relative(queue_path, root),
            "integrity_manifest": _relative(manifest_path, root),
        },
    }
    _render_audit(report, audit_path)
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    integrity_targets = {
        "phase5_13_source_report": root / TARGET_REPORT_ROOT / "report.json",
        "phase5_14_report": report_path,
        "phase5_14_action_queue": queue_path,
        "phase5_14_audit": audit_path,
        "phase5_14_implementation": root
        / "pipeline"
        / "diagnose_phase5_target_reprojection.py",
    }
    manifest = {
        "schema_version": "lalitha_phase5_target_failure_integrity_v1",
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
        default=Path("data/phase5_target_failure_attribution_v1"),
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
