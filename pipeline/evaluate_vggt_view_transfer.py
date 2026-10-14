"""Learn a view-transfer policy on one ring and test it on held-out rings.

The experiment separates two questions that a single leave-one-view-out IoU
cannot answer:

* which source views project contained surface evidence into a target view;
* whether a source-view policy learned on a training object transfers to
  unseen objects without using their target masks for selection.

This remains non-metric research evidence.  It does not complete hidden
surfaces and it cannot validate CAD or manufacturing accuracy.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree

from pipeline.validate_vggt_multiring_experiment import (
    VIEW_ORDER,
    _agreement,
    _label,
    _metrics,
    _project,
    _records,
    _square,
)


def _paths(
    root: Path, object_dir: Path, record: dict[str, Any], view: str
) -> tuple[Path, Path]:
    manifest_path = object_dir / "input_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return (
            root / manifest["views"][view]["image"],
            root / manifest["views"][view]["mask"],
        )
    return (
        root / record["views"][view]["image_path"],
        root / record["views"][view]["jewelry_mask_path"],
    )


def _filtered_evidence(
    root: Path,
    object_dir: Path,
    record: dict[str, Any],
    confidence_quantile: float,
    support_radius_fraction: float,
    minimum_supported_views: int,
) -> dict[str, Any]:
    report = json.loads((object_dir / "vggt_report.json").read_text(encoding="utf-8"))
    with np.load(object_dir / "vggt_cameras.npz") as cameras:
        extrinsic = cameras["extrinsic"][0].copy()
        intrinsic = cameras["intrinsic"][0].copy()
    with np.load(object_dir / "vggt_dense_evidence.npz") as dense:
        world = dense["world_points"][0]
        confidence = dense["world_points_conf"][0]
    size = int(report["image_size"])

    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    candidates: list[np.ndarray] = []
    for index, view in enumerate(VIEW_ORDER):
        source_path, mask_path = _paths(root, object_dir, record, view)
        source = _square(source_path, size)
        target = _square(mask_path, size, grayscale=True) > 0
        finite = np.isfinite(world[index]).all(axis=-1) & np.isfinite(
            confidence[index]
        )
        values = confidence[index][target & finite]
        threshold = float(np.quantile(values, confidence_quantile))
        valid = target & finite & (confidence[index] >= threshold)
        sources.append(source)
        targets.append(target)
        candidates.append(world[index][valid])

    all_candidates = np.concatenate(candidates, axis=0)
    bounds = np.percentile(all_candidates, (1, 99), axis=0)
    radius = max(
        float(np.linalg.norm(bounds[1] - bounds[0])) * support_radius_fraction,
        1e-7,
    )
    trees = [cKDTree(points) for points in candidates]
    filtered: list[np.ndarray] = []
    for source_index, points in enumerate(candidates):
        support = np.ones(len(points), dtype=np.uint8)
        for target_index, tree in enumerate(trees):
            if source_index == target_index:
                continue
            distances, _ = tree.query(points, k=1, workers=-1)
            support += distances <= radius
        filtered.append(points[support >= minimum_supported_views])

    return {
        "size": size,
        "sources": sources,
        "targets": targets,
        "points": filtered,
        "extrinsic": extrinsic,
        "intrinsic": intrinsic,
    }


def _score_sources(
    evidence: dict[str, Any], target_index: int, source_indices: tuple[int, ...]
) -> tuple[np.ndarray, dict[str, float]]:
    points = np.concatenate(
        [evidence["points"][index] for index in source_indices], axis=0
    )
    prediction = _project(
        points,
        evidence["extrinsic"][target_index],
        evidence["intrinsic"][target_index],
        evidence["size"],
    )
    target = np.uint8(evidence["targets"][target_index]) * 255
    return prediction, _metrics(prediction, target)


def _all_source_subsets(target_index: int) -> list[tuple[int, ...]]:
    available = [index for index in range(len(VIEW_ORDER)) if index != target_index]
    return [
        subset
        for length in range(1, len(available) + 1)
        for subset in itertools.combinations(available, length)
    ]


def _learn_policy(evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    policy: dict[str, dict[str, Any]] = {}
    for target_index, target_view in enumerate(VIEW_ORDER):
        candidates = []
        for source_indices in _all_source_subsets(target_index):
            _, metrics = _score_sources(evidence, target_index, source_indices)
            candidates.append((metrics["iou"], metrics["precision"], source_indices, metrics))
        _, _, selected, metrics = max(candidates, key=lambda row: (row[0], row[1], -len(row[2])))
        all_sources = tuple(
            index for index in range(len(VIEW_ORDER)) if index != target_index
        )
        _, baseline = _score_sources(evidence, target_index, all_sources)
        policy[target_view] = {
            "sources": [VIEW_ORDER[index] for index in selected],
            "training_metrics": metrics,
            "all_source_baseline_metrics": baseline,
            "candidate_subset_count": len(candidates),
        }
    return policy


def _mean(view_metrics: list[dict[str, Any]], key: str) -> float:
    return round(float(np.mean([item[key] for item in view_metrics])), 6)


def _evaluate_object(
    root: Path,
    object_dir: Path,
    output_dir: Path,
    record: dict[str, Any],
    policy: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    evidence = _filtered_evidence(root, object_dir, record, 0.50, 0.025, 2)
    rows = []
    pairwise: dict[str, dict[str, dict[str, float]]] = {}
    baseline_views = []
    policy_views = []
    for target_index, target_view in enumerate(VIEW_ORDER):
        pairwise[target_view] = {}
        for source_index, source_view in enumerate(VIEW_ORDER):
            if source_index == target_index:
                continue
            _, values = _score_sources(evidence, target_index, (source_index,))
            pairwise[target_view][source_view] = values

        baseline_sources = tuple(
            index for index in range(len(VIEW_ORDER)) if index != target_index
        )
        selected_sources = tuple(
            VIEW_ORDER.index(view) for view in policy[target_view]["sources"]
        )
        baseline_prediction, baseline_metrics = _score_sources(
            evidence, target_index, baseline_sources
        )
        policy_prediction, policy_metrics = _score_sources(
            evidence, target_index, selected_sources
        )
        baseline_views.append({"view": target_view, **baseline_metrics})
        policy_views.append(
            {
                "view": target_view,
                "sources": list(policy[target_view]["sources"]),
                **policy_metrics,
            }
        )

        target_u8 = np.uint8(evidence["targets"][target_index]) * 255
        rows.append(
            np.hstack(
                (
                    _label(evidence["sources"][target_index], f"{target_view.upper()} SOURCE"),
                    _label(cv2.cvtColor(target_u8, cv2.COLOR_GRAY2BGR), "TARGET MASK"),
                    _label(
                        cv2.cvtColor(baseline_prediction, cv2.COLOR_GRAY2BGR),
                        "ALL SOURCES",
                        f"IoU {baseline_metrics['iou']:.3f}",
                    ),
                    _label(
                        cv2.cvtColor(policy_prediction, cv2.COLOR_GRAY2BGR),
                        "TRAIN POLICY",
                        f"IoU {policy_metrics['iou']:.3f}",
                    ),
                    _label(
                        _agreement(
                            evidence["sources"][target_index],
                            policy_prediction,
                            target_u8,
                        ),
                        "POLICY AGREEMENT",
                        "+".join(policy[target_view]["sources"]),
                    ),
                )
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    preview = output_dir / f"{record['object_id']}_view_transfer.png"
    if not cv2.imwrite(str(preview), np.vstack(rows)):
        raise RuntimeError(f"Could not write {preview}")

    baseline = {
        "views": baseline_views,
        "mean_iou": _mean(baseline_views, "iou"),
        "mean_precision": _mean(baseline_views, "precision"),
        "mean_recall": _mean(baseline_views, "recall"),
    }
    selected = {
        "views": policy_views,
        "mean_iou": _mean(policy_views, "iou"),
        "mean_precision": _mean(policy_views, "precision"),
        "mean_recall": _mean(policy_views, "recall"),
    }
    return {
        "object_id": record["object_id"],
        "split": record["split"],
        "pairwise_single_source_metrics": pairwise,
        "all_source_baseline": baseline,
        "train_selected_policy": selected,
        "delta": {
            "mean_iou": round(selected["mean_iou"] - baseline["mean_iou"], 6),
            "mean_precision": round(
                selected["mean_precision"] - baseline["mean_precision"], 6
            ),
            "mean_recall": round(
                selected["mean_recall"] - baseline["mean_recall"], 6
            ),
        },
        "artifacts": {"preview": str(preview.relative_to(root))},
    }


def run(root: Path, baseline_root: Path, output_root: Path) -> dict[str, Any]:
    run_report = json.loads(
        (baseline_root / "run_report.json").read_text(encoding="utf-8")
    )
    records = _records(root / "dataset/prepared_v1/manifest.jsonl")
    by_id = {item["object_id"]: item for item in run_report["objects"]}
    training_id = "ring_001"
    training_record = records[training_id]
    training_evidence = _filtered_evidence(
        root, baseline_root / training_id, training_record, 0.50, 0.025, 2
    )
    policy = _learn_policy(training_evidence)
    results = [
        _evaluate_object(
            root,
            baseline_root / item["object_id"],
            output_root / item["object_id"],
            records[item["object_id"]],
            policy,
        )
        for item in run_report["objects"]
    ]
    heldout = [item for item in results if item["split"] == "test"]
    heldout_summary = {
        "object_count": len(heldout),
        "mean_baseline_iou": _mean(
            [{"value": item["all_source_baseline"]["mean_iou"]} for item in heldout],
            "value",
        ),
        "mean_policy_iou": _mean(
            [{"value": item["train_selected_policy"]["mean_iou"]} for item in heldout],
            "value",
        ),
        "mean_iou_delta": _mean(
            [{"value": item["delta"]["mean_iou"]} for item in heldout], "value"
        ),
        "mean_precision_delta": _mean(
            [{"value": item["delta"]["mean_precision"]} for item in heldout],
            "value",
        ),
        "mean_recall_delta": _mean(
            [{"value": item["delta"]["mean_recall"]} for item in heldout],
            "value",
        ),
    }
    accepted = (
        heldout_summary["mean_iou_delta"] > 0
        and heldout_summary["mean_precision_delta"] >= 0
        and all(item["delta"]["mean_iou"] > 0 for item in heldout)
    )
    report = {
        "schema_version": "vggt_view_transfer_policy_v1",
        "dataset": "STL-1",
        "training_object": training_id,
        "training_split": by_id[training_id]["split"],
        "policy": policy,
        "objects": results,
        "heldout": heldout_summary,
        "checks": {
            "policy_selected_on_training_object_only": True,
            "heldout_target_masks_used_for_selection": False,
            "every_policy_excludes_target_view": all(
                target not in item["sources"] for target, item in policy.items()
            ),
            "both_heldout_objects_improved_iou": all(
                item["delta"]["mean_iou"] > 0 for item in heldout
            ),
            "heldout_policy_gate_passed": accepted,
        },
        "decision": (
            "accepted_as_view_selection_candidate"
            if accepted
            else "rejected_as_nontransferable_view_selection"
        ),
        "claims": {
            "camera_pose_accuracy_validated": False,
            "hidden_surface_completion_validated": False,
            "cross_view_geometry_validated": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "The policy is learned from one training object and requires a larger identity-disjoint cohort before generalization can be claimed.",
            "Held-out masks are used only for scoring, but all masks remain machine proposals.",
            "Source selection can remove inconsistent surfaces; it cannot infer geometry never visible in any source view.",
            "All coordinates remain relative and non-metric.",
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("dataset/phase_runs/v1/vggt_multiring_224_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/phase_runs/v1/vggt_view_transfer_v1"),
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    baseline = (
        args.baseline_root
        if args.baseline_root.is_absolute()
        else root / args.baseline_root
    )
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    print(json.dumps(run(root, baseline, output), indent=2))


if __name__ == "__main__":
    main()
