"""Auditable, dataset-wide Phase 1 human review for STL-1.

The retained machine outputs remain immutable proposals. Human decisions and
optional corrected silhouette masks are stored separately. This module does
not promote relative depth to metric truth or infer missing CAD supervision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import threading
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


VIEWS = ("front", "top", "iso", "lsv", "rsv")
DECISIONS = ("pending", "approved", "needs_correction", "rejected")
YES_NO_UNKNOWN = ("unknown", "yes", "no")
MANIFEST_RELATIVE_PATH = Path(
    "dataset/phase_runs/v1/phase1_human_review_v1/manifest.json"
)
EVIDENCE_TYPES: dict[str, dict[str, str]] = {
    "source_quality": {
        "title": "Source image quality",
        "question": "Is the jewellery visible, in focus and sufficiently detailed?",
        "authority": "observed_source",
    },
    "normalization": {
        "title": "Normalization",
        "question": "Does the normalized image preserve the complete object and detail?",
        "authority": "derived_machine_proposal",
    },
    "view_semantics": {
        "title": "View semantics",
        "question": "Does the assigned front/top/isometric/left/right view label match the image?",
        "authority": "source_filename_hypothesis",
    },
    "silhouette": {
        "title": "Jewellery silhouette",
        "question": "Does the mask include the complete object and exclude the background?",
        "authority": "pseudo_background_difference",
    },
    "edges": {
        "title": "Edges and fine detail",
        "question": "Do the retained edges preserve visible boundaries and sculptural detail?",
        "authority": "opencv_canny_proposal",
    },
    "relative_depth": {
        "title": "Relative depth plausibility",
        "question": "Is the near-to-far ordering visually plausible within this view?",
        "authority": "depth_anything_v2_non_metric_proposal",
    },
    "depth_uncertainty": {
        "title": "Depth uncertainty",
        "question": "Does uncertainty highlight ambiguous boundaries and reflective/detail regions?",
        "authority": "transform_consistency_non_metric_proposal",
    },
}
_LOCK = threading.RLock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_dir = root / ".git"
        head_path = git_dir / "HEAD"
        if not head_path.is_file():
            return None
        head = head_path.read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head if len(head) == 40 else None
        reference = head.removeprefix("ref: ")
        loose_ref = git_dir / reference
        if loose_ref.is_file():
            value = loose_ref.read_text(encoding="utf-8").strip()
            return value if len(value) == 40 else None
        packed_refs = git_dir / "packed-refs"
        if packed_refs.is_file():
            for line in packed_refs.read_text(encoding="utf-8").splitlines():
                if line.startswith(("#", "^")):
                    continue
                fields = line.split(" ", 1)
                if len(fields) == 2 and fields[1] == reference:
                    return fields[0] if len(fields[0]) == 40 else None
        return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _relative_existing(root: Path, value: str | Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Evidence path escapes repository: {value}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"Required Phase 1 evidence is missing: {relative}")
    return relative.as_posix()


def _review_record(path: str, sha256: str, authority: str) -> dict[str, Any]:
    return {
        "proposal_path": path,
        "proposal_sha256": sha256,
        "active_path": path,
        "active_sha256": sha256,
        "corrected_path": None,
        "corrected_sha256": None,
        "authority": authority,
        "decision": "pending",
        "reviewer": None,
        "reviewed_at_utc": None,
        "note": None,
        "history": [],
    }


def _clean_note(note: Any) -> str | None:
    if not isinstance(note, str):
        return None
    note = note.strip()
    if len(note) > 1000:
        raise ValueError("Review note must be 1000 characters or fewer")
    return note or None


def _reviewer(value: Any) -> str:
    value = str(value or "").strip()
    if not value:
        raise ValueError("Reviewer is required")
    if len(value) > 120:
        raise ValueError("Reviewer must be 120 characters or fewer")
    return value


def _decision(value: Any) -> str:
    value = str(value or "")
    if value not in DECISIONS[1:]:
        raise ValueError(f"Invalid review decision: {value}")
    return value


def _choice(value: Any, field: str) -> str:
    value = str(value or "unknown")
    if value not in YES_NO_UNKNOWN:
        raise ValueError(f"Invalid {field}: {value}")
    return value


def _refresh_object(record: dict[str, Any]) -> None:
    pending = []
    for view, view_record in record["views"].items():
        for evidence_type, review in view_record["evidence"].items():
            if review["decision"] != "approved":
                pending.append(f"{view}:{evidence_type}")
    object_review_pending = record["object_review"]["decision"] != "approved"
    record["gate"] = {
        "status": "pass" if not pending and not object_review_pending else "blocked",
        "pending_evidence": pending,
        "pending_evidence_count": len(pending),
        "object_review_pending": object_review_pending,
        "human_review_complete": not pending and not object_review_pending,
    }


def refresh_gate(manifest: dict[str, Any]) -> None:
    pending_evidence = 0
    pending_objects = []
    approved_evidence = 0
    total_evidence = 0
    decision_counts = {decision: 0 for decision in DECISIONS}
    for object_id, record in manifest["objects"].items():
        _refresh_object(record)
        pending_evidence += record["gate"]["pending_evidence_count"]
        if record["gate"]["object_review_pending"]:
            pending_objects.append(object_id)
        for view_record in record["views"].values():
            for review in view_record["evidence"].values():
                total_evidence += 1
                decision_counts[review["decision"]] += 1
                if review["decision"] == "approved":
                    approved_evidence += 1
    automated_failures = [
        key for key, passed in manifest["automated_checks"].items() if not passed
    ]
    passed = not pending_evidence and not pending_objects and not automated_failures
    manifest["gate"] = {
        "status": "pass" if passed else "blocked",
        "object_count": len(manifest["objects"]),
        "view_count": len(manifest["objects"]) * len(VIEWS),
        "evidence_review_count": total_evidence,
        "approved_evidence_count": approved_evidence,
        "pending_evidence_count": pending_evidence,
        "pending_object_reviews": pending_objects,
        "pending_object_review_count": len(pending_objects),
        "decision_counts": decision_counts,
        "automated_check_failures": automated_failures,
        "image_evidence_review_complete": passed,
        "metric_scale_available": False,
        "manufacturing_accuracy_validated": False,
    }
    manifest["status"] = (
        "human_review_complete" if passed else "human_review_required"
    )
    manifest["updated_at_utc"] = utc_now()


def initialize_manifest(root: Path, overwrite: bool = False) -> dict[str, Any]:
    root = root.resolve()
    output_path = root / MANIFEST_RELATIVE_PATH
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Review manifest already exists: {output_path}")
    prepared = _read_jsonl(root / "dataset/prepared_v1/manifest.jsonl")
    complete = _read_json(
        root / "dataset/phase_runs/v1/phase1_complete_features_v1/report.json"
    )
    complete_by_id = {item["object_id"]: item for item in complete["objects"]}
    hash_cache: dict[str, str] = {}

    def evidence(path_value: str, authority: str) -> dict[str, Any]:
        relative = _relative_existing(root, path_value)
        digest = hash_cache.setdefault(relative, _sha256(root / relative))
        return _review_record(relative, digest, authority)

    objects: dict[str, Any] = {}
    for prepared_record in prepared:
        object_id = prepared_record["object_id"]
        if object_id not in complete_by_id:
            raise ValueError(f"Complete Phase 1 report is missing {object_id}")
        complete_record = complete_by_id[object_id]
        views: dict[str, Any] = {}
        for view in VIEWS:
            prepared_view = prepared_record["views"][view]
            complete_view = complete_record["views"][view]
            paths = {
                "source_quality": complete_view["source"],
                "normalization": complete_view["normalized_source"],
                "view_semantics": complete_view["source"],
                "silhouette": prepared_view["jewelry_mask_path"],
                "edges": prepared_view["edge_path"],
                "relative_depth": complete_view["regularized_depth_u16"],
                "depth_uncertainty": complete_view["uncertainty_u16"],
            }
            views[view] = {
                "view": view,
                "view_contract": complete_view["view_contract"],
                "evidence": {
                    kind: evidence(paths[kind], definition["authority"])
                    for kind, definition in EVIDENCE_TYPES.items()
                },
                "automated_checks": {
                    "source_pixel_roundtrip_exact": bool(
                        complete_view["source_pixel_roundtrip_exact"]
                    ),
                    "normalized_pixel_retained_exactly": bool(
                        complete_view["normalized_pixel_retained_exactly"]
                    ),
                    "three_depth_runs_deterministic": bool(
                        complete_view["repeat_deterministic"]
                    ),
                    "feature_matrix_count_is_22": complete_view["feature_count"] == 22,
                },
                "diagnostics": {
                    "horizontal_flip_error_over_depth_span": complete_view[
                        "horizontal_flip_error_over_depth_span"
                    ],
                    "vertical_flip_error_over_depth_span": complete_view[
                        "vertical_flip_error_over_depth_span"
                    ],
                    "metric_depth": False,
                    "cross_view_depth_aligned": False,
                },
            }
        objects[object_id] = {
            "object_id": object_id,
            "source_name": prepared_record.get("source_object_name", object_id),
            "split": prepared_record["split"],
            "views": views,
            "object_review": {
                "decision": "pending",
                "reviewer": None,
                "reviewed_at_utc": None,
                "note": None,
                "component_inventory": {
                    "has_stones": "unknown",
                    "estimated_visible_stone_count": None,
                    "has_prongs": "unknown",
                    "has_sculptural_relief": "unknown",
                    "cross_view_identity_consistent": "unknown",
                },
                "history": [],
            },
        }

    all_view_checks = [
        passed
        for record in objects.values()
        for view_record in record["views"].values()
        for passed in view_record["automated_checks"].values()
    ]
    manifest: dict[str, Any] = {
        "schema_version": "stl1_phase1_human_review_v1",
        "dataset": "STL-1",
        "phase": "phase1_image_evidence_ground_truth",
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "source_commit": _source_commit(root),
        "authority_policy": (
            "Machine artifacts remain immutable proposals. Approval records human "
            "acceptance in image space only. Relative depth stays non-metric, and "
            "no CAD, hidden geometry, material or manufacturing truth is created."
        ),
        "evidence_types": EVIDENCE_TYPES,
        "review_contract": {
            "views": list(VIEWS),
            "evidence_types_per_view": list(EVIDENCE_TYPES),
            "object_component_inventory_required": True,
            "corrected_silhouette_supported": True,
            "original_proposals_are_immutable": True,
        },
        "automated_checks": {
            "twenty_four_objects": len(objects) == 24,
            "five_views_per_object": all(
                set(record["views"]) == set(VIEWS) for record in objects.values()
            ),
            "all_120_views_present": len(objects) * len(VIEWS) == 120,
            "all_source_roundtrips_and_feature_contracts_pass": all(all_view_checks),
        },
        "objects": objects,
        "limitations": [
            "No physical dimensions or calibrated cameras are supplied.",
            "Depth Anything V2 outputs are relative per-view proposals, not millimetres.",
            "STL-1 contains rendered rings and does not provide production CAD or scan truth.",
            "Silhouette approval does not create individual stone, prong, cavity or relief masks.",
            "A separate paired photo-to-CAD dataset is still required for metric reconstruction.",
        ],
    }
    refresh_gate(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_manifest(output_path, manifest)
    return manifest


def manifest_path(root: Path) -> Path:
    return root.resolve() / MANIFEST_RELATIVE_PATH


def load_manifest(root: Path) -> tuple[Path, dict[str, Any]]:
    path = manifest_path(root)
    if not path.is_file():
        raise FileNotFoundError(f"Initialize STL-1 Phase 1 review first: {path}")
    return path, _read_json(path)


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    with _LOCK:
        refresh_gate(manifest)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)


def _object_view_evidence(
    manifest: dict[str, Any], object_id: str, view: str, evidence_type: str
) -> dict[str, Any]:
    if object_id not in manifest["objects"]:
        raise ValueError(f"Unknown STL-1 object: {object_id}")
    if view not in VIEWS:
        raise ValueError(f"Unknown STL-1 view: {view}")
    if evidence_type not in EVIDENCE_TYPES:
        raise ValueError(f"Unknown Phase 1 evidence type: {evidence_type}")
    return manifest["objects"][object_id]["views"][view]["evidence"][evidence_type]


def add_review_comment(
    root: Path,
    object_id: str,
    scope: str,
    reviewer: str,
    note: Any,
    view: str | None = None,
    evidence_type: str | None = None,
) -> dict[str, Any]:
    """Record a scoped comment without approving or rejecting evidence."""
    reviewer_name = _reviewer(reviewer)
    comment = _clean_note(note)
    if not comment:
        raise ValueError("Comment is required")
    path, manifest = load_manifest(root)
    if scope == "evidence":
        record = _object_view_evidence(manifest, object_id, view, evidence_type)
    elif scope == "object" and object_id in manifest["objects"]:
        record = manifest["objects"][object_id]["object_review"]
    else:
        raise ValueError("Unknown STL-1 review scope")
    record.setdefault("comments", []).append(
        {"reviewer": reviewer_name, "created_at_utc": utc_now(), "text": comment}
    )
    save_manifest(path, manifest)
    return manifest


def apply_evidence_review(
    root: Path,
    object_id: str,
    view: str,
    evidence_type: str,
    decision: str,
    reviewer: str,
    note: Any = None,
) -> dict[str, Any]:
    path, manifest = load_manifest(root)
    record = _object_view_evidence(manifest, object_id, view, evidence_type)
    timestamp = utc_now()
    event = {
        "event": "review_decision",
        "decision": _decision(decision),
        "reviewer": _reviewer(reviewer),
        "reviewed_at_utc": timestamp,
        "note": _clean_note(note),
        "active_sha256": record["active_sha256"],
    }
    record["history"].append(event)
    record.update(
        {
            "decision": event["decision"],
            "reviewer": event["reviewer"],
            "reviewed_at_utc": timestamp,
            "note": event["note"],
        }
    )
    save_manifest(path, manifest)
    return manifest


def apply_view_review(
    root: Path,
    object_id: str,
    view: str,
    decision: str,
    reviewer: str,
    note: Any = None,
) -> dict[str, Any]:
    path, manifest = load_manifest(root)
    if object_id not in manifest["objects"]:
        raise ValueError(f"Unknown STL-1 object: {object_id}")
    if view not in VIEWS:
        raise ValueError(f"Unknown STL-1 view: {view}")
    reviewed_decision = _decision(decision)
    reviewer_name = _reviewer(reviewer)
    cleaned_note = _clean_note(note)
    timestamp = utc_now()
    for evidence_type in EVIDENCE_TYPES:
        record = manifest["objects"][object_id]["views"][view]["evidence"][
            evidence_type
        ]
        event = {
            "event": "view_review_decision",
            "decision": reviewed_decision,
            "reviewer": reviewer_name,
            "reviewed_at_utc": timestamp,
            "note": cleaned_note,
            "active_sha256": record["active_sha256"],
        }
        record["history"].append(event)
        record.update(
            {
                "decision": reviewed_decision,
                "reviewer": reviewer_name,
                "reviewed_at_utc": timestamp,
                "note": cleaned_note,
            }
        )
    save_manifest(path, manifest)
    return manifest


def apply_object_review(
    root: Path,
    object_id: str,
    decision: str,
    reviewer: str,
    inventory: dict[str, Any],
    note: Any = None,
) -> dict[str, Any]:
    path, manifest = load_manifest(root)
    if object_id not in manifest["objects"]:
        raise ValueError(f"Unknown STL-1 object: {object_id}")
    if not isinstance(inventory, dict):
        raise ValueError("Component inventory must be an object")
    visible_stones = inventory.get("estimated_visible_stone_count")
    if visible_stones in ("", None):
        visible_stones = None
    else:
        try:
            visible_stones = int(visible_stones)
        except (TypeError, ValueError) as exc:
            raise ValueError("Visible stone count must be a non-negative integer") from exc
        if not 0 <= visible_stones <= 1000:
            raise ValueError("Visible stone count must be between 0 and 1000")
    cleaned_inventory = {
        "has_stones": _choice(inventory.get("has_stones"), "has_stones"),
        "estimated_visible_stone_count": visible_stones,
        "has_prongs": _choice(inventory.get("has_prongs"), "has_prongs"),
        "has_sculptural_relief": _choice(
            inventory.get("has_sculptural_relief"), "has_sculptural_relief"
        ),
        "cross_view_identity_consistent": _choice(
            inventory.get("cross_view_identity_consistent"),
            "cross_view_identity_consistent",
        ),
    }
    timestamp = utc_now()
    event = {
        "event": "object_review",
        "decision": _decision(decision),
        "reviewer": _reviewer(reviewer),
        "reviewed_at_utc": timestamp,
        "note": _clean_note(note),
        "component_inventory": cleaned_inventory,
    }
    record = manifest["objects"][object_id]["object_review"]
    record["history"].append(event)
    record.update(event)
    record.pop("event", None)
    save_manifest(path, manifest)
    return manifest


def save_corrected_silhouette(
    root: Path,
    object_id: str,
    view: str,
    image_bytes: bytes,
    reviewer: str,
    note: Any = None,
) -> dict[str, Any]:
    if not image_bytes:
        raise ValueError("Corrected mask file is empty")
    if len(image_bytes) > 10 * 1024 * 1024:
        raise ValueError("Corrected mask exceeds the 10 MiB limit")
    path, manifest = load_manifest(root)
    record = _object_view_evidence(manifest, object_id, view, "silhouette")
    proposal = root.resolve() / record["proposal_path"]
    try:
        with Image.open(proposal) as proposal_image:
            expected_size = proposal_image.size
        with Image.open(BytesIO(image_bytes)) as uploaded:
            if uploaded.format != "PNG":
                raise ValueError("Corrected mask must be a PNG image")
            if uploaded.size != expected_size:
                raise ValueError(
                    f"Corrected mask size {uploaded.size} does not match {expected_size}"
                )
            uploaded.load()
            grayscale = uploaded.convert("L")
    except OSError as exc:
        raise ValueError("Corrected mask must be a decodable image") from exc
    binary = grayscale.point(lambda value: 255 if value >= 128 else 0, mode="1").convert("L")
    extrema = binary.getextrema()
    if extrema in {(0, 0), (255, 255)}:
        raise ValueError("Corrected mask must contain foreground and background")

    relative = Path(
        "dataset/phase_runs/v1/phase1_human_review_v1/corrections/silhouette"
    ) / object_id / f"{view}.png"
    output = root.resolve() / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.png")
    binary.save(temporary, format="PNG", optimize=True)
    os.replace(temporary, output)
    digest = _sha256(output)
    timestamp = utc_now()
    reviewer_name = _reviewer(reviewer)
    record["history"].append(
        {
            "event": "corrected_mask_saved",
            "reviewer": reviewer_name,
            "reviewed_at_utc": timestamp,
            "note": _clean_note(note),
            "corrected_sha256": digest,
        }
    )
    record.update(
        {
            "active_path": relative.as_posix(),
            "active_sha256": digest,
            "corrected_path": relative.as_posix(),
            "corrected_sha256": digest,
            "decision": "pending",
            "reviewer": reviewer_name,
            "reviewed_at_utc": timestamp,
            "note": "Corrected mask saved; approval required",
        }
    )
    save_manifest(path, manifest)
    return manifest


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "status"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.command == "init":
        manifest = initialize_manifest(args.root, overwrite=args.overwrite)
    else:
        _, manifest = load_manifest(args.root)
    print(json.dumps(manifest["gate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
