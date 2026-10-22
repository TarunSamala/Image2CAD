"""Lightweight, auditable Phase 1 human-review state operations.

This module intentionally uses only the Python standard library so the Studio
can record review decisions without importing OpenCV or running inference.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VIEWS = ("front", "side", "top", "angled", "back")
LABELS = (
    "jewelry",
    "metal",
    "shank",
    "stone_visible",
    "setting",
    "prongs",
    "negative_space",
)
COMPONENT_CATALOG = (
    ("shank_001", "shank"),
    ("setting_001", "setting"),
    ("stone_001", "round_gemstone"),
    ("prong_001", "prong"),
    ("prong_002", "prong"),
    ("prong_003", "prong"),
    ("prong_004", "prong"),
)
DECISIONS = ("pending", "approved", "needs_correction", "rejected")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def refresh_gate(manifest: dict[str, Any]) -> None:
    """Refresh the image-ground-truth gate without inventing metric evidence."""
    pending_labels = [
        f"{view}:{label}"
        for view, record in manifest["views"].items()
        for label, review in record["reviews"].items()
        if review["decision"] != "approved"
    ]
    pending_identities = [
        item["component_id"]
        for item in manifest["component_catalog"]
        if item["review"]["decision"] != "approved"
    ]
    camera_missing = [
        view
        for view, record in manifest["camera_records"].items()
        if record["review"]["decision"] != "approved"
    ]
    scale = manifest["physical_scale"]
    scale_missing = not scale["measurements"]
    scale_required = bool(scale.get("required_for_phase1_exit", True))
    scale_blocking = scale_missing and scale_required
    conflicts = list(manifest.get("review_conflicts", []))
    image_ground_truth_complete = not (
        pending_labels or pending_identities or camera_missing or conflicts
    )
    manifest["exit_gate"] = {
        "status": (
            "pass"
            if image_ground_truth_complete and not scale_blocking
            else "blocked"
        ),
        "pending_label_reviews": pending_labels,
        "pending_component_identities": pending_identities,
        "pending_camera_records": camera_missing,
        "physical_scale_missing": scale_missing,
        "physical_scale_required": scale_required,
        "metric_scale_available": not scale_missing,
        "unresolved_conflicts": conflicts,
        "image_ground_truth_complete": image_ground_truth_complete,
        "human_ground_truth_complete": image_ground_truth_complete,
    }
    manifest["status"] = (
        "image_ground_truth_complete"
        if manifest["exit_gate"]["status"] == "pass"
        else "human_review_required"
    )
    manifest["updated_at_utc"] = utc_now()


def load_manifest(root: Path) -> tuple[Path, dict[str, Any]]:
    path = root / "data" / "ring01_ground_truth_v1" / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Initialize Phase 1 first: {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    refresh_gate(manifest)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _validated_reviewer(reviewer: str) -> str:
    reviewer = reviewer.strip()
    if not reviewer:
        raise ValueError("Reviewer is required")
    if len(reviewer) > 120:
        raise ValueError("Reviewer must be 120 characters or fewer")
    return reviewer


def _clean_note(note: str | None) -> str | None:
    return note.strip() if isinstance(note, str) and note.strip() else None


def add_review_comment(
    root: Path,
    scope: str,
    reviewer: str,
    note: str,
    view: str | None = None,
    label: str | None = None,
    component_id: str | None = None,
) -> dict[str, Any]:
    """Attach feedback without changing any human-review decision or gate."""
    reviewer_name = _validated_reviewer(str(reviewer or ""))
    comment = _clean_note(note)
    if not comment or len(comment) > 500:
        raise ValueError("Comment must be 1 to 500 characters")
    path, manifest = load_manifest(root)
    if scope == "label" and view in VIEWS and label in LABELS:
        review = manifest["views"][view]["reviews"][label]
    elif scope == "identity":
        item = next(
            (item for item in manifest["component_catalog"] if item["component_id"] == component_id),
            None,
        )
        if item is None:
            raise ValueError("Unknown component ID")
        review = item["review"]
    elif scope == "camera" and view in VIEWS:
        review = manifest["camera_records"][view]["review"]
    else:
        raise ValueError("Unknown Ring01 review scope")
    review.setdefault("comments", []).append(
        {"reviewer": reviewer_name, "created_at_utc": utc_now(), "text": comment}
    )
    save_manifest(path, manifest)
    return manifest


def apply_label_review(
    root: Path,
    view: str,
    label: str,
    decision: str,
    reviewer: str,
    note: str | None = None,
) -> dict[str, Any]:
    if view not in VIEWS:
        raise ValueError(f"Unknown view: {view}")
    if label not in LABELS:
        raise ValueError(f"Unknown label: {label}")
    if decision not in DECISIONS[1:]:
        raise ValueError(f"Invalid review decision: {decision}")
    path, manifest = load_manifest(root)
    manifest["views"][view]["reviews"][label].update(
        {
            "decision": decision,
            "reviewer": _validated_reviewer(reviewer),
            "reviewed_at_utc": utc_now(),
            "note": _clean_note(note),
        }
    )
    save_manifest(path, manifest)
    return manifest


def apply_identity_review(
    root: Path,
    component_id: str,
    decision: str,
    reviewer: str,
    note: str | None = None,
) -> dict[str, Any]:
    if decision not in DECISIONS[1:]:
        raise ValueError(f"Invalid review decision: {decision}")
    path, manifest = load_manifest(root)
    item = next(
        (
            item
            for item in manifest["component_catalog"]
            if item["component_id"] == component_id
        ),
        None,
    )
    if item is None:
        raise ValueError(f"Unknown component ID: {component_id}")
    item["review"].update(
        {
            "decision": decision,
            "reviewer": _validated_reviewer(reviewer),
            "reviewed_at_utc": utc_now(),
            "note": _clean_note(note),
        }
    )
    item["cross_view_identity"] = (
        "human_approved" if decision == "approved" else "pending_human_review"
    )
    save_manifest(path, manifest)
    return manifest


def apply_camera_review(
    root: Path,
    view: str,
    projection_model: str,
    uncertainty: str,
    decision: str,
    reviewer: str,
    note: str | None = None,
    intrinsics: Any = None,
    extrinsics: Any = None,
) -> dict[str, Any]:
    if view not in VIEWS:
        raise ValueError(f"Unknown view: {view}")
    if projection_model not in {
        "perspective",
        "orthographic",
        "weak_perspective",
        "unknown",
    }:
        raise ValueError(f"Invalid projection model: {projection_model}")
    if decision not in {"approved", "needs_correction"}:
        raise ValueError(f"Invalid camera review decision: {decision}")
    uncertainty = uncertainty.strip()
    if not uncertainty:
        raise ValueError("Camera uncertainty is required")
    path, manifest = load_manifest(root)
    record = manifest["camera_records"][view]
    record.update(
        {
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "projection_model": projection_model,
            "uncertainty": uncertainty,
        }
    )
    record["review"].update(
        {
            "decision": decision,
            "reviewer": _validated_reviewer(reviewer),
            "reviewed_at_utc": utc_now(),
            "note": _clean_note(note),
        }
    )
    save_manifest(path, manifest)
    return manifest
