"""Review manifest for ambiguous STL-1 Phase 4 component candidates.

Machine proposal masks remain immutable.  Review decisions only attach an
image-space semantic label and never authorize cross-view identity, metric
geometry, stone seats, cavities, or CAD generation.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VIEWS = ("front", "top", "iso", "lsv", "rsv")
DECISIONS = ("pending", "approved", "needs_correction", "rejected")
SEMANTIC_LABELS = {
    "unknown": "Unknown / insufficient image evidence",
    "aperture": "Aperture or open hole",
    "decorative_openwork": "Decorative openwork",
    "structural_opening": "Structural opening",
    "possible_stone_or_setting": "Possible stone or setting region",
}
MANIFEST_RELATIVE_PATH = Path(
    "data/phase4_stl1_component_candidates_v1/review_manifest.json"
)
_LOCK = threading.RLock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _reviewer(value: Any) -> str:
    reviewer = str(value or "").strip()
    if not reviewer:
        raise ValueError("Reviewer is required")
    if len(reviewer) > 120:
        raise ValueError("Reviewer must be 120 characters or fewer")
    return reviewer


def _note(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if len(value) > 1000:
        raise ValueError("Review note must be 1000 characters or fewer")
    return value or None


def refresh_gate(manifest: dict[str, Any]) -> None:
    decision_counts = {decision: 0 for decision in DECISIONS}
    semantic_counts = {label: 0 for label in SEMANTIC_LABELS}
    total = 0
    pending = 0
    possible_stone_count = 0
    for object_record in manifest["objects"].values():
        object_pending = 0
        for view_record in object_record["views"].values():
            for candidate in view_record["candidates"]:
                total += 1
                decision_counts[candidate["decision"]] += 1
                semantic_counts[candidate["semantic_label"]] += 1
                if candidate["decision"] != "approved":
                    pending += 1
                    object_pending += 1
                if (
                    candidate["decision"] == "approved"
                    and candidate["semantic_label"] == "possible_stone_or_setting"
                ):
                    possible_stone_count += 1
        object_record["pending_candidate_count"] = object_pending
        object_record["semantic_review_complete"] = object_pending == 0
    manifest["gate"] = {
        "status": "review_complete" if pending == 0 else "review_required",
        "candidate_count": total,
        "approved_candidate_count": decision_counts["approved"],
        "pending_candidate_count": pending,
        "decision_counts": decision_counts,
        "semantic_counts": semantic_counts,
        "approved_possible_stone_or_setting_count": possible_stone_count,
        "image_space_semantic_review_complete": pending == 0,
        "individual_stones_confirmed": False,
        "cross_view_component_identity_resolved": False,
        "component_graph_generation_authorized": False,
        "reason": "A reviewed image-space label does not establish material, hidden geometry, cross-view identity, seat/cavity geometry, or manufacturing dimensions.",
    }
    manifest["updated_at_utc"] = _now()


def initialize_manifest(root: Path, overwrite: bool = False) -> dict[str, Any]:
    root = root.resolve()
    path = root / MANIFEST_RELATIVE_PATH
    if path.exists() and not overwrite:
        raise FileExistsError(f"Review manifest already exists: {path}")
    report = _read(root / "data/phase4_stl1_component_candidates_v1/report.json")
    objects: dict[str, Any] = {}
    seen: set[str] = set()
    for object_id, object_report in report["objects"].items():
        views: dict[str, Any] = {}
        for view in VIEWS:
            candidates = []
            for proposal in object_report["views"][view]["candidates"]:
                candidate_id = proposal["candidate_id"]
                if candidate_id in seen:
                    raise ValueError(f"Duplicate candidate ID: {candidate_id}")
                seen.add(candidate_id)
                candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "proposal_type": proposal["candidate_type"],
                        "proposal_subtype": proposal["candidate_subtype"],
                        "mask_path": proposal["mask_path"],
                        "mask_sha256": proposal["mask_sha256"],
                        "bbox_px": proposal["bbox_px"],
                        "center_px": proposal["center_px"],
                        "area_px2": proposal["area_px2"],
                        "circularity": proposal["circularity"],
                        "semantic_label": "unknown",
                        "decision": "pending",
                        "reviewer": None,
                        "reviewed_at_utc": None,
                        "note": None,
                        "history": [],
                    }
                )
            views[view] = {
                "overlay_path": object_report["views"][view]["overlay_path"],
                "candidates": candidates,
            }
        objects[object_id] = {
            "split": object_report["split"],
            "views": views,
            "pending_candidate_count": sum(
                len(record["candidates"]) for record in views.values()
            ),
            "semantic_review_complete": False,
        }
    manifest = {
        "schema_version": "lalitha_phase4_candidate_semantic_review_v1",
        "source_report": "data/phase4_stl1_component_candidates_v1/report.json",
        "authority_policy": "Human decisions classify visible image regions only. They do not prove gemstone material, hidden geometry, metric scale, seats, cavities, or CAD correctness.",
        "semantic_labels": SEMANTIC_LABELS,
        "views": list(VIEWS),
        "objects": objects,
        "review_contract": {
            "proposals_immutable": True,
            "identified_reviewer_required": True,
            "history_append_only": True,
            "cross_view_identity_separate_gate": True,
            "cad_authorization_never_implied": True,
        },
        "created_at_utc": _now(),
    }
    refresh_gate(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, manifest)
    return manifest


def load_manifest(root: Path) -> tuple[Path, dict[str, Any]]:
    path = root.resolve() / MANIFEST_RELATIVE_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Phase 4 candidate review manifest is missing: {path}")
    manifest = _read(path)
    refresh_gate(manifest)
    return path, manifest


def apply_candidate_review(
    root: Path,
    object_id: str,
    view: str,
    candidate_id: str,
    semantic_label: str,
    decision: str,
    reviewer: str,
    note: Any = None,
) -> dict[str, Any]:
    if view not in VIEWS:
        raise ValueError(f"Invalid view: {view}")
    if semantic_label not in SEMANTIC_LABELS:
        raise ValueError(f"Invalid semantic label: {semantic_label}")
    if decision not in DECISIONS[1:]:
        raise ValueError(f"Invalid review decision: {decision}")
    reviewer = _reviewer(reviewer)
    note = _note(note)
    with _LOCK:
        path, manifest = load_manifest(root)
        try:
            candidates = manifest["objects"][object_id]["views"][view]["candidates"]
        except KeyError as exc:
            raise ValueError(f"Unknown candidate scope: {object_id}:{view}") from exc
        matches = [item for item in candidates if item["candidate_id"] == candidate_id]
        if len(matches) != 1:
            raise ValueError(f"Unknown candidate ID for {object_id}:{view}: {candidate_id}")
        candidate = matches[0]
        timestamp = _now()
        candidate["history"].append(
            {
                "semantic_label": candidate["semantic_label"],
                "decision": candidate["decision"],
                "reviewer": candidate["reviewer"],
                "reviewed_at_utc": candidate["reviewed_at_utc"],
                "note": candidate["note"],
            }
        )
        candidate.update(
            {
                "semantic_label": semantic_label,
                "decision": decision,
                "reviewer": reviewer,
                "reviewed_at_utc": timestamp,
                "note": note,
            }
        )
        refresh_gate(manifest)
        _write_atomic(path, manifest)
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = initialize_manifest(args.root, overwrite=args.overwrite)
    print(json.dumps(manifest["gate"], indent=2))


if __name__ == "__main__":
    main()
