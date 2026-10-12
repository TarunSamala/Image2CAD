"""Audit and prepare external real-image ring datasets without mixing sources.

This stage is deliberately image-only. It preserves published train/test labels,
marks samples with unknown object identity as evaluation-only, and never invents
multi-view grouping, metric depth, camera calibration, or CAD supervision.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from audit_jewellery import (
    _agreement,
    _detail_proposal,
    _dice,
    _grabcut_refine,
    _opencv_mask,
    _overlay,
    _shadow_proposal,
    _transform_mask,
)
from prepare_jewellery_dataset import (
    _background_bgr,
    _foreground_mask,
    _normalized,
)
from train_dataset_phases import _binary_iou, _boundary_f1


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
EXPECTED_COUNTS = {
    "ornasia": {"evaluation_unassigned": 196},
    "segment_your_ring": {"train": 8, "test": 2},
}

MASK_EXIT_GATE = {
    "iou": 0.80,
    "boundary_f1_2px": 0.50,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return cleaned or "sample"


def _natural_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path))]


def _images_under(path: Path) -> list[Path]:
    return sorted(
        (item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES),
        key=_natural_key,
    )


def _is_ring_directory(part: str) -> bool:
    return re.fullmatch(r"rings?(?:[_ -]\d+)?", part.lower()) is not None


def discover_ornasia(root: Path) -> list[dict[str, Any]]:
    """Return original ring photographs only, excluding augmented experiment copies."""

    if not root.is_dir():
        raise FileNotFoundError(f"OrnAsia root not found: {root}")
    ring_images = [
        path
        for path in _images_under(root)
        if any(_is_ring_directory(part) for part in path.parts)
    ]
    raw_images = [
        path
        for path in ring_images
        if any("raw" in part.lower() or "original" in part.lower() for part in path.parts)
    ]
    selected = raw_images or ring_images
    if not selected:
        raise ValueError(f"No Ring/Rings image directory found under {root}")

    # The archive can contain the same raw tree inside more than one wrapper.
    unique: list[Path] = []
    seen_hashes: set[str] = set()
    for path in selected:
        digest = _sha256(path)
        if digest not in seen_hashes:
            seen_hashes.add(digest)
            unique.append(path)
    return [
        {
            "dataset_id": "ornasia",
            "sample_id": f"ornasia_{index:04d}",
            "split": "evaluation_unassigned",
            "source_image": path,
            "ground_truth_mask": None,
            "object_identity": None,
            "view_semantic": None,
        }
        for index, path in enumerate(unique, 1)
    ]


def discover_segment_your_ring(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Segment Your Ring root not found: {root}")
    dataset_root = root / "dataset" if (root / "dataset").is_dir() else root
    records: list[dict[str, Any]] = []
    for split in ("train", "test"):
        image_dir = dataset_root / split / "images"
        mask_dir = dataset_root / split / "masks"
        images = _images_under(image_dir) if image_dir.is_dir() else []
        masks = {_safe_id(path.stem): path for path in _images_under(mask_dir)} if mask_dir.is_dir() else {}
        for image_path in images:
            key = _safe_id(image_path.stem)
            if key not in masks:
                raise ValueError(f"Missing Segment Your Ring mask for {image_path.name}")
            records.append(
                {
                    "dataset_id": "segment_your_ring",
                    "sample_id": f"syr_{split}_{key}",
                    "split": split,
                    "source_image": image_path,
                    "ground_truth_mask": masks[key],
                    "object_identity": None,
                    "view_semantic": None,
                }
            )
    if not records:
        raise ValueError(f"No Segment Your Ring image/mask pairs found under {root}")
    return records


def _read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def _read_binary_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    mask = _read_image(path, cv2.IMREAD_GRAYSCALE)
    if mask.shape != expected_shape:
        mask = cv2.resize(mask, (expected_shape[1], expected_shape[0]), interpolation=cv2.INTER_NEAREST)
    binary = np.where(mask >= 128, 255, 0).astype(np.uint8)
    border = np.concatenate((binary[0], binary[-1], binary[:, 0], binary[:, -1]))
    if np.count_nonzero(border) > border.size / 2:
        binary = cv2.bitwise_not(binary)
    return binary


def _write(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write image: {path}")
    return str(path)


def _metric_set(candidate: np.ndarray, target: np.ndarray) -> dict[str, float]:
    candidate_binary, target_binary = candidate > 0, target > 0
    return {
        "iou": round(_binary_iou(candidate_binary, target_binary), 6),
        "dice": round(_dice(candidate_binary, target_binary), 6),
        "boundary_f1_2px": round(_boundary_f1(candidate_binary, target_binary), 6),
    }


def _mask_geometry(mask: np.ndarray) -> dict[str, Any]:
    foreground = mask > 0
    ys, xs = np.where(foreground)
    if not len(xs):
        return {
            "foreground_fraction": 0.0,
            "bbox_fraction": 0.0,
            "bbox_fill_fraction": 0.0,
            "external_component_count": 0,
            "hole_count": 0,
        }

    height, width = foreground.shape
    bbox_width = int(xs.max() - xs.min() + 1)
    bbox_height = int(ys.max() - ys.min() + 1)
    contours, hierarchy = cv2.findContours(
        foreground.astype(np.uint8),
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    external_components = 0
    holes = 0
    if hierarchy is not None:
        for node in hierarchy[0]:
            if node[3] < 0:
                external_components += 1
            else:
                holes += 1
    foreground_count = int(np.count_nonzero(foreground))
    return {
        "foreground_fraction": round(foreground_count / foreground.size, 6),
        "bbox_fraction": round((bbox_width * bbox_height) / foreground.size, 6),
        "bbox_fill_fraction": round(foreground_count / (bbox_width * bbox_height), 6),
        "external_component_count": external_components,
        "hole_count": holes,
    }


def _mask_quality(
    mask: np.ndarray,
    ground_truth_metrics: dict[str, dict[str, float]] | None,
) -> dict[str, Any]:
    geometry = _mask_geometry(mask)
    warnings: list[str] = []
    if geometry["foreground_fraction"] < 0.01:
        warnings.append("foreground_too_small")
    if geometry["foreground_fraction"] > 0.35:
        warnings.append("foreground_occupies_more_than_35_percent")
    if geometry["bbox_fraction"] > 0.70:
        warnings.append("foreground_bbox_occupies_more_than_70_percent")
    if geometry["bbox_fill_fraction"] > 0.70:
        warnings.append("mask_is_unusually_solid_inside_bbox")

    if ground_truth_metrics is None:
        return {
            "status": "review_required_no_ground_truth",
            "approved": False,
            "basis": "shape_plausibility_only",
            "warnings": warnings,
            "geometry": geometry,
            "exit_gate": MASK_EXIT_GATE,
        }

    refined = ground_truth_metrics["phase2_refined"]
    passed = (
        refined["iou"] >= MASK_EXIT_GATE["iou"]
        and refined["boundary_f1_2px"] >= MASK_EXIT_GATE["boundary_f1_2px"]
    )
    if not passed:
        warnings.append("human_mask_exit_gate_not_met")
    return {
        "status": "approved" if passed else "review_required",
        "approved": passed,
        "basis": "human_mask",
        "warnings": warnings,
        "geometry": geometry,
        "exit_gate": MASK_EXIT_GATE,
    }


def _relative(path: str | Path) -> str:
    candidate = Path(path).resolve()
    try:
        return str(candidate.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(candidate)


def _preview_cell(image: np.ndarray, label: str, width: int = 260, height: int = 220) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    canvas = np.full((height + 34, width, 3), 248, np.uint8)
    x = (width - resized.shape[1]) // 2
    y = 34 + (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    cv2.putText(canvas, label, (7, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (30, 30, 30), 1, cv2.LINE_AA)
    return canvas


def _write_preview(records: list[dict[str, Any]], output: Path, limit: int = 12) -> str | None:
    if not records:
        return None
    if len(records) <= limit:
        selected = records
    else:
        indices = np.linspace(0, len(records) - 1, limit, dtype=int)
        selected = [records[int(index)] for index in indices]
    rows = []
    for record in selected:
        artifacts = record["artifacts"]
        source = _read_image(Path(record["source_path"]))
        normalized = _read_image(Path(artifacts["phase1_normalized"]))
        refined = _read_image(Path(artifacts["phase2_refined_silhouette"]), cv2.IMREAD_GRAYSCALE)
        overlay = _read_image(Path(artifacts["phase2_overlay"]))
        truth_metrics = record.get("ground_truth_metrics") or {}
        score = truth_metrics.get("phase2_refined", {}).get("iou")
        rows.append(
            np.hstack(
                (
                    _preview_cell(source, f"{record['sample_id']} | SOURCE"),
                    _preview_cell(normalized, "PHASE 1 | NORMALIZED"),
                    _preview_cell(refined, "PHASE 2 | REFINED MASK"),
                    _preview_cell(overlay, "PHASE 2 | OVERLAY" + (f" | GT IoU {score:.3f}" if score is not None else "")),
                )
            )
        )
    sheet = np.vstack(rows)
    return _write(output, sheet)


def _process_sample(sample: dict[str, Any], output_root: Path, image_size: int) -> dict[str, Any]:
    source_path: Path = sample["source_image"]
    source = _read_image(source_path)
    background = _background_bgr(source)
    try:
        initial_source = _foreground_mask(source, background)
    except ValueError:
        initial_source = _opencv_mask(source)
    normalized, initial, _, transform = _normalized(source, initial_source, background, image_size)
    # Refinement happens after normalization. Running GrabCut on multi-megapixel
    # source photographs is substantially slower without adding boundary detail
    # to the fixed-resolution evidence product.
    refined = _grabcut_refine(normalized, initial)
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    edges = cv2.bitwise_and(
        cv2.Canny(gray, 45, 120),
        cv2.dilate(refined, np.ones((3, 3), np.uint8)),
    )
    details = cv2.bitwise_and(_detail_proposal(normalized, refined), refined)
    shadow, shadow_threshold = _shadow_proposal(normalized, refined)
    shadow = cv2.bitwise_and(
        shadow,
        cv2.bitwise_not(refined),
    )
    overlay = _overlay(normalized, refined, details, shadow)
    agreement = _agreement(initial, refined)

    artifact_dir = output_root / "artifacts" / sample["dataset_id"] / sample["split"] / sample["sample_id"]
    artifacts = {
        "phase1_normalized": _write(artifact_dir / "phase1_normalized.png", normalized),
        "phase1_silhouette": _write(artifact_dir / "phase1_silhouette.png", initial),
        "phase1_edges": _write(artifact_dir / "phase1_edges.png", edges),
        "phase2_refined_silhouette": _write(artifact_dir / "phase2_refined_silhouette.png", refined),
        "phase2_detail_proposal": _write(artifact_dir / "phase2_detail_proposal.png", details),
        "phase2_shadow_proposal": _write(artifact_dir / "phase2_shadow_proposal.png", shadow),
        "phase2_overlay": _write(artifact_dir / "phase2_overlay.png", overlay),
        "phase2_agreement": _write(artifact_dir / "phase2_agreement.png", agreement),
    }

    target_metrics = None
    target_path = sample.get("ground_truth_mask")
    if target_path is not None:
        target_source = _read_binary_mask(Path(target_path), source.shape[:2])
        target = _transform_mask(target_source, transform, image_size)
        artifacts["ground_truth_mask"] = _write(artifact_dir / "ground_truth_mask.png", target)
        target_metrics = {
            "phase1_initial": _metric_set(initial, target),
            "phase2_refined": _metric_set(refined, target),
        }

    initial_metrics = _metric_set(initial, refined)
    mask_quality = _mask_quality(refined, target_metrics)
    return {
        "schema_version": "external_ring_image_evidence_v1",
        "dataset_id": sample["dataset_id"],
        "sample_id": sample["sample_id"],
        "split": sample["split"],
        "training_allowed": sample["split"] == "train" and target_path is not None,
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path),
        "source_bytes": source_path.stat().st_size,
        "source_size_wh": [source.shape[1], source.shape[0]],
        "ground_truth_mask_path": str(target_path) if target_path is not None else None,
        "ground_truth_mask_sha256": _sha256(Path(target_path)) if target_path is not None else None,
        "object_identity": sample["object_identity"],
        "view_semantic": sample["view_semantic"],
        "camera_calibrated": False,
        "metric_scale_available": False,
        "cad_target_available": False,
        "normalization": transform,
        "measurements": {
            "phase2_foreground_fraction": round(float(np.count_nonzero(refined) / refined.size), 6),
            "phase1_edge_fraction": round(float(np.count_nonzero(edges) / edges.size), 6),
            "shadow_gray_threshold": round(float(shadow_threshold), 3),
        },
        "phase1_to_phase2_agreement": initial_metrics,
        "ground_truth_metrics": target_metrics,
        "mask_quality": mask_quality,
        "artifacts": artifacts,
    }


def _aggregate_ground_truth(records: Iterable[dict[str, Any]], stage: str) -> dict[str, float] | None:
    metrics = [
        record["ground_truth_metrics"][stage]
        for record in records
        if record.get("ground_truth_metrics") is not None
    ]
    if not metrics:
        return None
    return {
        key: round(float(np.mean([metric[key] for metric in metrics])), 6)
        for key in ("iou", "dice", "boundary_f1_2px")
    }


def run_intake(
    ornasia_root: Path,
    syr_root: Path,
    output_root: Path,
    *,
    image_size: int = 512,
    strict_counts: bool = False,
    workers: int = 4,
) -> dict[str, Any]:
    samples = discover_ornasia(ornasia_root) + discover_segment_your_ring(syr_root)
    discovered_split_counts = {
        dataset_id: dict(Counter(sample["split"] for sample in samples if sample["dataset_id"] == dataset_id))
        for dataset_id in sorted({sample["dataset_id"] for sample in samples})
    }
    actual_counts_match = all(
        discovered_split_counts.get(dataset_id, {}) == expected
        for dataset_id, expected in EXPECTED_COUNTS.items()
    )
    if strict_counts and not actual_counts_match:
        raise ValueError(f"Published sample counts do not match: {discovered_split_counts}")
    if output_root.exists() and any((output_root / "artifacts").glob("*")):
        raise FileExistsError(f"Artifact output already exists: {output_root / 'artifacts'}")
    output_root.mkdir(parents=True, exist_ok=True)
    if workers < 1 or workers > 8:
        raise ValueError("workers must be between 1 and 8")
    indexed_records: list[tuple[int, dict[str, Any]]] = []
    failures: list[dict[str, str]] = []
    sample_by_index = {index: sample for index, sample in enumerate(samples)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {
            executor.submit(_process_sample, sample, output_root, image_size): index
            for index, sample in sample_by_index.items()
        }
        completed = 0
        for future in concurrent.futures.as_completed(pending):
            index = pending[future]
            sample = sample_by_index[index]
            try:
                indexed_records.append((index, future.result()))
            except (OSError, RuntimeError, ValueError, cv2.error) as exc:
                failures.append({"sample_id": sample["sample_id"], "error": str(exc)})
            completed += 1
            if completed % 20 == 0 or completed == len(samples):
                print(f"processed {completed}/{len(samples)} images ({len(failures)} failures)", flush=True)
    records = [record for _, record in sorted(indexed_records)]

    manifest_path = output_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            portable = json.loads(json.dumps(record))
            portable["source_path"] = _relative(portable["source_path"])
            if portable["ground_truth_mask_path"]:
                portable["ground_truth_mask_path"] = _relative(portable["ground_truth_mask_path"])
            portable["artifacts"] = {
                key: _relative(value) for key, value in portable["artifacts"].items()
            }
            handle.write(json.dumps(portable, separators=(",", ":")) + "\n")

    by_dataset = Counter(record["dataset_id"] for record in records)
    by_split = {
        dataset_id: dict(Counter(record["split"] for record in records if record["dataset_id"] == dataset_id))
        for dataset_id in sorted(by_dataset)
    }
    previews: dict[str, str] = {}
    for dataset_id in sorted(by_dataset):
        path = _write_preview(
            [record for record in records if record["dataset_id"] == dataset_id],
            output_root / "previews" / f"{dataset_id}_phase1_phase2_preview.jpg",
        )
        if path:
            previews[dataset_id] = _relative(path)

    source_hashes = [record["source_sha256"] for record in records]
    quality_counts = dict(Counter(record["mask_quality"]["status"] for record in records))
    checks = {
        "all_images_decoded": len(records) == len(samples) and not failures,
        "all_masks_nonempty": all(record["measurements"]["phase2_foreground_fraction"] > 0.001 for record in records),
        "all_sources_unique": len(source_hashes) == len(set(source_hashes)),
        "syr_all_pairs_have_ground_truth_masks": all(
            record["ground_truth_metrics"] is not None
            for record in records
            if record["dataset_id"] == "segment_your_ring"
        ),
        "ornasia_never_enabled_for_training": all(
            not record["training_allowed"]
            for record in records
            if record["dataset_id"] == "ornasia"
        ),
        "published_counts_match": actual_counts_match,
        "no_metric_or_cad_claims": all(
            not record["metric_scale_available"] and not record["cad_target_available"]
            for record in records
        ),
        "all_segmentation_outputs_approved": all(
            record["mask_quality"]["approved"] for record in records
        ),
    }
    report = {
        "schema_version": "external_ring_dataset_intake_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "passed" if all(checks.values()) else "review_required",
        "input_contract": "images_only",
        "image_size": image_size,
        "workers": workers,
        "sample_count": len(records),
        "dataset_counts": dict(by_dataset),
        "split_counts": by_split,
        "ground_truth_mask_count": sum(record["ground_truth_metrics"] is not None for record in records),
        "failure_count": len(failures),
        "failures": failures,
        "mask_quality_counts": quality_counts,
        "approved_mask_count": quality_counts.get("approved", 0),
        "review_required_mask_count": len(records) - quality_counts.get("approved", 0),
        "mask_exit_gate": MASK_EXIT_GATE,
        "checks": checks,
        "metrics": {
            "syr_phase1_initial_vs_human_mask": _aggregate_ground_truth(records, "phase1_initial"),
            "syr_phase2_refined_vs_human_mask": _aggregate_ground_truth(records, "phase2_refined"),
        },
        "split_policy": {
            "segment_your_ring": "Preserve publisher train/test directories exactly.",
            "ornasia": "Evaluation-only because physical-object identities and leakage-safe groups are unavailable.",
        },
        "claim_limits": [
            "OrnAsia has category labels but no human masks, view semantics, camera calibration, scale, mesh, or CAD.",
            "Segment Your Ring supplies whole-ring masks only; it has no stones, prongs, cavities, cameras, scale, mesh, or CAD labels.",
            "Phase 1-to-Phase 2 agreement is not accuracy. Only Segment Your Ring rows have mask-ground-truth metrics.",
            "Every OrnAsia mask remains review-required until a human mask or equivalent independent ground truth exists.",
            "SAM confidence is model confidence, not segmentation accuracy; it cannot approve a mask without ground truth.",
        ],
        "algorithms": {
            "phase1_silhouette": "OpenCV border-background difference",
            "phase1_edges": "OpenCV Canny constrained by foreground",
            "phase2_refinement": "OpenCV GrabCut initialized from Phase 1",
            "phase2_proposals": "brightness detail and local cast-shadow heuristics",
        },
        "manifest": _relative(manifest_path),
        "previews": previews,
    }
    (output_root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ornasia-root", type=Path, required=True)
    parser.add_argument("--syr-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("dataset/external_ring_v1"))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--strict-counts", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    report = run_intake(
        args.ornasia_root,
        args.syr_root,
        args.output_root,
        image_size=args.image_size,
        strict_counts=args.strict_counts,
        workers=args.workers,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
