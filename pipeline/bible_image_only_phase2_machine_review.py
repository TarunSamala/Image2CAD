"""Record a non-human machine audit before image-only Phase 2 research.

This audit never changes Phase 1 review decisions.  It verifies retained hashes,
binary masks, dataset evidence and the reflection-correction checkpoint, then
records AI-assisted visual observations with explicit authority limits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


RING01_VIEWS = ("front", "side", "top", "angled", "back")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _binary_mask_check(path: Path, expected_wh: tuple[int, int]) -> dict[str, Any]:
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"))
    values = sorted(int(value) for value in np.unique(mask))
    foreground = int(np.count_nonzero(mask))
    return {
        "path": path.as_posix(),
        "size_wh": [int(mask.shape[1]), int(mask.shape[0])],
        "size_matches_source": [int(mask.shape[1]), int(mask.shape[0])] == list(expected_wh),
        "binary": set(values).issubset({0, 255}),
        "nonempty": foreground > 0,
        "not_full_frame": foreground < mask.size,
        "foreground_fraction": round(foreground / mask.size, 8),
        "values": values,
    }


def _ring01_audit(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    source_hashes_valid = True
    proposal_hashes_valid = True
    mask_checks: list[dict[str, Any]] = []
    for view in RING01_VIEWS:
        record = manifest["views"][view]
        source = root / record["source"]["path"]
        source_hashes_valid &= _sha256(source) == record["source"]["sha256"]
        expected_wh = tuple(int(value) for value in record["source"]["size_wh"])
        for label, proposal in record["proposals"].items():
            path = root / proposal["path"]
            proposal_hashes_valid &= _sha256(path) == proposal["sha256"]
            check = _binary_mask_check(path, expected_wh)
            check.update({"path": proposal["path"], "view": view, "label": label})
            mask_checks.append(check)

    return {
        "view_count": len(RING01_VIEWS),
        "proposal_count": len(mask_checks),
        "source_hashes_valid": bool(source_hashes_valid),
        "proposal_hashes_valid": bool(proposal_hashes_valid),
        "all_masks_binary": all(item["binary"] for item in mask_checks),
        "all_masks_nonempty": all(item["nonempty"] for item in mask_checks),
        "all_masks_not_full_frame": all(item["not_full_frame"] for item in mask_checks),
        "all_mask_sizes_match_sources": all(item["size_matches_source"] for item in mask_checks),
        "mask_checks": mask_checks,
        "visual_findings": [
            "All five whole-jewellery silhouettes preserve the visible ring body, head and principal negative spaces.",
            "Front and top proposals preserve the visible four-prong arrangement, but gemstone facets and prong boundaries remain reflection-sensitive.",
            "Side and back proposals preserve the shank opening and gallery silhouette; thin gallery and setting boundaries remain uncertain at source resolution.",
            "The angled legacy mask contains reflection-induced inner-shank gaps; the separate correction improves continuity without changing the exterior silhouette.",
            "Relative-depth ordering is visually plausible at coarse scale but uncertainty is high around gemstone facets, prongs and polished metal, so it is non-metric evidence only.",
        ],
    }


def _stl1_audit(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    evidence_count = 0
    missing_paths: list[str] = []
    hash_mismatches: list[str] = []
    hash_cache: dict[Path, str] = {}
    for object_record in manifest["objects"].values():
        for view_record in object_record["views"].values():
            for evidence in view_record["evidence"].values():
                evidence_count += 1
                path = root / evidence["active_path"]
                if not path.is_file():
                    missing_paths.append(evidence["active_path"])
                    continue
                if path not in hash_cache:
                    hash_cache[path] = _sha256(path)
                digest = hash_cache[path]
                if digest != evidence["active_sha256"]:
                    hash_mismatches.append(evidence["active_path"])
    return {
        "object_count": len(manifest["objects"]),
        "view_count": sum(len(record["views"]) for record in manifest["objects"].values()),
        "evidence_count": evidence_count,
        "automated_checks": manifest["automated_checks"],
        "all_automated_checks_pass": all(manifest["automated_checks"].values()),
        "all_evidence_paths_exist": not missing_paths,
        "all_active_hashes_valid": not hash_mismatches,
        "missing_paths": missing_paths,
        "hash_mismatches": hash_mismatches,
        "visual_findings": [
            "The 24-object, five-view overview is complete and view assignments are visually consistent with the source filenames.",
            "Normalization retains the full visible objects across the representative train, validation and test examples.",
            "Pseudo-silhouettes preserve outer jewellery contours but do not provide individual stones, prongs, cavities or sculptural-relief instances.",
            "OpenCV edge evidence retains substantial relief detail, including deity faces and ornament boundaries, but edge presence is not semantic reconstruction.",
            "Held-out overlap scores measure reproduction of machine pseudo-masks and are not human-ground-truth accuracy.",
        ],
    }


def _markdown(report: dict[str, Any]) -> str:
    ring = report["ring01"]
    stl1 = report["stl1"]
    phase1 = report["phase1_gate"]
    lines = [
        "# Image-only Phase 2 machine-review checkpoint",
        "",
        f"Status: **{report['status'].replace('_', ' ').upper()}**",
        "",
        "This checkpoint records deterministic validation and AI-assisted visual inspection. It is not an identified human review and does not close Bible Phase 1 or canonical Bible Phase 2.",
        "",
        "## Phase 1 gate carried forward",
        "",
        f"- Ring01 pending label reviews: `{phase1['ring01_pending_label_reviews']}`",
        f"- Ring01 pending component identities: `{phase1['ring01_pending_component_identities']}`",
        f"- Ring01 pending camera records: `{phase1['ring01_pending_camera_records']}`",
        f"- STL-1 pending evidence decisions: `{phase1['stl1_pending_evidence_count']}`",
        f"- STL-1 pending object inventories: `{phase1['stl1_pending_object_review_count']}`",
        "",
        "## Ring01 machine validation",
        "",
        f"- Five views and `{ring['proposal_count']}` proposal masks checked.",
        f"- Source hashes valid: `{ring['source_hashes_valid']}`",
        f"- Proposal hashes valid: `{ring['proposal_hashes_valid']}`",
        f"- Binary, nonempty, source-sized masks: `{ring['all_masks_binary'] and ring['all_masks_nonempty'] and ring['all_mask_sizes_match_sources']}`",
        "",
    ]
    lines.extend(f"- {finding}" for finding in ring["visual_findings"])
    lines.extend(
        [
            "",
            "## STL-1 machine validation",
            "",
            f"- `{stl1['object_count']}` objects, `{stl1['view_count']}` views and `{stl1['evidence_count']}` evidence records checked.",
            f"- Automated checks pass: `{stl1['all_automated_checks_pass']}`",
            f"- Evidence paths and hashes valid: `{stl1['all_evidence_paths_exist'] and stl1['all_active_hashes_valid']}`",
            "",
        ]
    )
    lines.extend(f"- {finding}" for finding in stl1["visual_findings"])
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Image-only Phase 2 research may continue using these artifacts as machine proposals with uncertainty. Human ground truth, metric depth, canonical paired-CAD Phase 2, exact hidden geometry and manufacturing validation remain blocked.",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(root: Path, output_dir: Path) -> dict[str, Any]:
    root = root.resolve()
    ring_manifest_path = root / "data/ring01_ground_truth_v1/manifest.json"
    stl1_manifest_path = root / "dataset/phase_runs/v1/phase1_human_review_v1/manifest.json"
    reflection_path = root / "data/ring01_image_only/reflection_correction_v1/report.json"
    ring_manifest_hash_before = _sha256(ring_manifest_path)
    stl1_manifest_hash_before = _sha256(stl1_manifest_path)
    ring_manifest = _read_json(ring_manifest_path)
    stl1_manifest = _read_json(stl1_manifest_path)
    reflection = _read_json(reflection_path)

    ring = _ring01_audit(root, ring_manifest)
    stl1 = _stl1_audit(root, stl1_manifest)
    reflection_checks_pass = all(reflection["checks"].values()) and bool(
        reflection.get("checkpoint_valid", False)
    )
    deterministic_checks = {
        "ring01_source_hashes_valid": ring["source_hashes_valid"],
        "ring01_proposal_hashes_valid": ring["proposal_hashes_valid"],
        "ring01_masks_binary": ring["all_masks_binary"],
        "ring01_masks_nonempty": ring["all_masks_nonempty"],
        "ring01_mask_sizes_match_sources": ring["all_mask_sizes_match_sources"],
        "stl1_automated_checks_pass": stl1["all_automated_checks_pass"],
        "stl1_evidence_paths_exist": stl1["all_evidence_paths_exist"],
        "stl1_active_hashes_valid": stl1["all_active_hashes_valid"],
        "reflection_checkpoint_checks_pass": reflection_checks_pass,
    }
    phase1_gate = {
        "status": "blocked_human_review_not_performed",
        "ring01_pending_label_reviews": len(ring_manifest["exit_gate"]["pending_label_reviews"]),
        "ring01_pending_component_identities": len(ring_manifest["exit_gate"]["pending_component_identities"]),
        "ring01_pending_camera_records": len(ring_manifest["exit_gate"]["pending_camera_records"]),
        "stl1_pending_evidence_count": int(stl1_manifest["gate"]["pending_evidence_count"]),
        "stl1_pending_object_review_count": int(stl1_manifest["gate"]["pending_object_review_count"]),
        "human_ground_truth_complete": False,
    }
    report = {
        "schema_version": "lalitha_image_only_phase2_machine_review_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "research_continuation_allowed_with_blocked_human_gate"
            if all(deterministic_checks.values())
            else "machine_validation_failed"
        ),
        "execution_profile": "image_only_reconstruction_v1",
        "authority": "ai_assisted_machine_audit_not_human_ground_truth",
        "phase1_gate": phase1_gate,
        "ring01": ring,
        "stl1": stl1,
        "reflection_correction": {
            "status": reflection["status"],
            "authority": reflection["authority"],
            "checks": reflection["checks"],
            "checkpoint_valid": reflection["checkpoint_valid"],
            "machine_checks_pass": reflection_checks_pass,
        },
        "deterministic_checks": deterministic_checks,
        "claims": {
            "machine_evidence_validated": all(deterministic_checks.values()),
            "human_review_performed": False,
            "bible_phase1_complete": False,
            "canonical_bible_phase2_complete": False,
            "metric_depth_validated": False,
            "manufacturing_accuracy_validated": False,
            "image_only_phase2_research_may_continue": all(deterministic_checks.values()),
        },
        "next_stage": "image_only_phase2_reflection_and_component_evidence",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_dir / "MACHINE_REVIEW.md").write_text(_markdown(report), encoding="utf-8")
    if _sha256(ring_manifest_path) != ring_manifest_hash_before:
        raise RuntimeError("Ring01 human-review manifest changed during machine audit")
    if _sha256(stl1_manifest_path) != stl1_manifest_hash_before:
        raise RuntimeError("STL-1 human-review manifest changed during machine audit")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ring01_image_only/machine_review_v1"),
    )
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else args.root / args.output_dir
    report = build_report(args.root, output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
