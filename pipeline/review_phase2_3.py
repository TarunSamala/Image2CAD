"""Validate Phase 2.3 review decisions and materialize semantic instances.

The machine evidence graph remains immutable.  This stage creates a new,
versioned reviewed graph whose components can later drive parametric geometry.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


FINAL_DECISIONS = {"accept", "reject", "relabel"}
ACTIONABLE_SEMANTICS = {
    "stone",
    "prong",
    "metal_body",
    "engraving",
    "relief",
    "filigree",
    "hole",
    "cavity_opening",
    "chain_link",
    "clasp",
    "bail",
    "other",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_mask(path: Path, mask: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask):
        raise RuntimeError(f"Could not write reviewed mask: {path}")
    return str(path)


def _observation_index(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    observations = {}
    for view in graph["views"].values():
        for observation in view["observations"]:
            observation_id = observation["observation_id"]
            if observation_id in observations:
                raise ValueError(f"Duplicate observation ID: {observation_id}")
            observations[observation_id] = observation
    return observations


def apply_review(evidence_graph_path: Path, review_manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Reviewed Phase 2.3 output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = _load(evidence_graph_path)
    review = _load(review_manifest_path)
    observations = _observation_index(graph)
    decisions = review.get("decisions", {})
    if set(decisions) != set(observations):
        missing = sorted(set(observations) - set(decisions))
        unexpected = sorted(set(decisions) - set(observations))
        raise ValueError(f"Review observation mismatch; missing={missing}, unexpected={unexpected}")

    unresolved: list[str] = []
    rejected: list[str] = []
    accepted: dict[str, dict[str, Any]] = {}
    for observation_id, decision in decisions.items():
        action = decision.get("decision", "pending")
        semantic = decision.get("semantic", "unknown_detail")
        if action not in FINAL_DECISIONS:
            unresolved.append(observation_id)
            continue
        if action == "reject":
            rejected.append(observation_id)
            continue
        if semantic not in ACTIONABLE_SEMANTICS:
            unresolved.append(observation_id)
            continue
        accepted[observation_id] = {
            **observations[observation_id],
            "state": "human_reviewed",
            "semantic": semantic,
            "review_notes": decision.get("notes", ""),
        }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in accepted.values():
        grouped[observation.get("track_id") or observation["observation_id"]].append(observation)

    conflicts = []
    for track_id, items in grouped.items():
        semantics = sorted({item["semantic"] for item in items})
        if len(semantics) > 1:
            conflicts.append({"track_id": track_id, "semantics": semantics})

    semantic_numbers: Counter[str] = Counter()
    components = []
    if not conflicts:
        for track_id in sorted(grouped):
            items = grouped[track_id]
            semantic = items[0]["semantic"]
            semantic_numbers[semantic] += 1
            component_id = f"{semantic}_{semantic_numbers[semantic]:03d}"
            masks = {}
            for item in items:
                source = cv2.imread(item["mask_path"], cv2.IMREAD_GRAYSCALE)
                if source is None:
                    raise ValueError(f"Cannot decode observation mask: {item['mask_path']}")
                view = item["view"]
                if view in masks:
                    existing = cv2.imread(masks[view], cv2.IMREAD_GRAYSCALE)
                    source = cv2.bitwise_or(existing, source)
                masks[view] = _write_mask(output_dir / "masks" / semantic / component_id / f"{view}.png", source)
                item["component_id"] = component_id
            components.append({
                "component_id": component_id,
                "semantic": semantic,
                "source_track_id": track_id,
                "observation_ids": [item["observation_id"] for item in items],
                "views": sorted(masks),
                "mask_paths": masks,
                "editable_independently": True,
                "geometry_status": "reviewed_2d_evidence_only",
            })

    counts = Counter(component["semantic"] for component in components)
    review_complete = not unresolved and not conflicts
    report = {
        "schema_version": "jewellery_reviewed_evidence_graph_v1",
        "stage": "phase2_3_human_review_gate",
        "source_evidence_graph": str(evidence_graph_path),
        "source_review_manifest": str(review_manifest_path),
        "review": {
            "complete": review_complete,
            "unresolved_observation_ids": unresolved,
            "rejected_observation_ids": rejected,
            "semantic_conflicts": conflicts,
            "accepted_observation_count": len(accepted),
        },
        "components": components,
        "component_counts": dict(sorted(counts.items())),
        "geometry_handoff": {
            "allowed": review_complete and bool(components),
            "reason": "reviewed_components_available" if review_complete and components else "complete semantic review before geometry",
            "individual_stones": [item["component_id"] for item in components if item["semantic"] == "stone"],
            "individual_prongs": [item["component_id"] for item in components if item["semantic"] == "prong"],
            "relief_components": [item["component_id"] for item in components if item["semantic"] in {"relief", "engraving"}],
            "metric_scale_available": False,
            "hidden_geometry_confirmed": False,
        },
    }
    report_path = output_dir / "reviewed_evidence_graph.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-graph", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = apply_review(args.evidence_graph, args.review_manifest, args.output_dir)
    print(json.dumps({
        "review_complete": report["review"]["complete"],
        "components": len(report["components"]),
        "geometry_handoff": report["geometry_handoff"]["allowed"],
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
