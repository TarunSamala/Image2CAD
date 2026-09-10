"""Phase 2.3 universal jewellery evidence extraction.

This stage preserves image evidence for later geometry reasoning.  It emits
generic, reviewable instances rather than assigning unsupported jewellery
semantics (for example, calling every highlight a gemstone).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class EvidenceInstance:
    observation_id: str
    view: str
    kind: str
    state: str
    mask_path: str
    bbox_xywh: list[int]
    centroid_xy_normalized: list[float]
    area_fraction: float
    circularity: float
    edge_support: float
    highlight_overlap: float
    confidence: float
    uncertainty: list[str]
    track_id: str | None = None


def _read(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def _write(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write image: {path}")
    return str(path)


def _normalize01(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, (1, 99))
    if high <= low:
        return np.zeros(image.shape, np.uint8)
    return np.clip((image.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)


def _geometry_maps(image: np.ndarray, foreground: np.ndarray) -> dict[str, np.ndarray]:
    inside = foreground > 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, 7, 32, 7)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(smooth)

    boundary = cv2.morphologyEx(foreground, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
    internal = cv2.Canny(clahe, 35, 105)
    internal[(~inside) | (boundary > 0)] = 0

    ridges = np.zeros_like(gray)
    valleys = np.zeros_like(gray)
    for size in (5, 9, 15, 25):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        ridges = np.maximum(ridges, cv2.morphologyEx(clahe, cv2.MORPH_TOPHAT, kernel))
        valleys = np.maximum(valleys, cv2.morphologyEx(clahe, cv2.MORPH_BLACKHAT, kernel))
    ridges = _normalize01(ridges)
    valleys = _normalize01(valleys)
    ridges[~inside] = 0
    valleys[~inside] = 0

    laplacian = cv2.Laplacian(clahe, cv2.CV_32F, ksize=3)
    relief = _normalize01(np.abs(laplacian))
    relief[~inside] = 0

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    chroma = np.linalg.norm(lab[:, :, 1:].astype(np.float32) - 128.0, axis=2)
    values = gray[inside]
    bright = float(np.percentile(values, 88)) if values.size else 255.0
    highlight = np.where(inside & (gray >= bright) & (chroma < np.percentile(chroma[inside], 70) if np.any(inside) else False), 255, 0).astype(np.uint8)
    highlight = cv2.morphologyEx(highlight, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    outer = np.zeros_like(foreground)
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(outer, contours, -1, 255, cv2.FILLED)
    negative = cv2.bitwise_and(outer, cv2.bitwise_not(foreground))

    return {
        "internal_edges": internal,
        "ridges": ridges,
        "valleys": valleys,
        "relief_response": relief,
        "highlight_interference": highlight,
        "negative_space": negative,
    }


def _foreground_quality(foreground: np.ndarray) -> dict[str, Any]:
    binary = foreground > 0
    points = cv2.findNonZero(binary.astype(np.uint8))
    if points is None:
        return {"passed": False, "reasons": ["empty_foreground"], "occupancy": 0.0, "bbox_extent": 0.0, "solidity": 0.0}
    x, y, width, height = cv2.boundingRect(points)
    area = int(np.count_nonzero(binary))
    bbox_extent = float(area / max(1, width * height))
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_area = sum(cv2.contourArea(cv2.convexHull(contour)) for contour in contours)
    solidity = float(area / hull_area) if hull_area else 0.0
    occupancy = float(area / binary.size)
    reasons = []
    if occupancy > 0.48:
        reasons.append("foreground_occupancy_unusually_high")
    if bbox_extent > 0.90 and solidity > 0.97:
        reasons.append("foreground_is_nearly_solid_bounding_rectangle")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "occupancy": round(occupancy, 6),
        "bbox_extent": round(bbox_extent, 6),
        "solidity": round(solidity, 6),
        "bbox_xywh": [int(x), int(y), int(width), int(height)],
    }


def _instance_candidates(maps: dict[str, np.ndarray], foreground: np.ndarray) -> np.ndarray:
    edge_fill = cv2.dilate(maps["internal_edges"], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    ridge_threshold = max(48, int(np.percentile(maps["ridges"][foreground > 0], 76))) if np.any(foreground) else 255
    valley_threshold = max(48, int(np.percentile(maps["valleys"][foreground > 0], 76))) if np.any(foreground) else 255
    candidate = np.where(
        (foreground > 0)
        & (edge_fill > 0)
        & ((maps["ridges"] >= ridge_threshold) | (maps["valleys"] >= valley_threshold)),
        255,
        0,
    ).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return candidate


def _extract_instances(
    view: str,
    candidate: np.ndarray,
    maps: dict[str, np.ndarray],
    output_dir: Path,
) -> list[EvidenceInstance]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    image_area = candidate.size
    minimum = max(18, round(image_area * 0.00004))
    maximum = round(image_area * 0.12)
    ranked = [label for label in range(1, count) if minimum <= stats[label, cv2.CC_STAT_AREA] <= maximum]
    ranked.sort(key=lambda label: (centroids[label][1], centroids[label][0]))
    instances: list[EvidenceInstance] = []
    for number, label in enumerate(ranked, start=1):
        mask = np.where(labels == label, 255, 0).astype(np.uint8)
        x, y, width, height, area = (int(value) for value in stats[label])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
        circularity = float(4 * math.pi * area / (perimeter * perimeter)) if perimeter else 0.0
        boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        edge_support = float(np.mean(maps["internal_edges"][boundary] > 0)) if np.any(boundary) else 0.0
        highlight_overlap = float(np.mean(maps["highlight_interference"][mask > 0] > 0))
        confidence = float(np.clip(0.30 + 0.45 * edge_support + 0.25 * min(1.0, area / (minimum * 5)), 0, 1))
        uncertainty = ["generic_detail_not_semantically_confirmed"]
        if highlight_overlap > 0.45:
            uncertainty.append("possible_reflection_or_specular_highlight")
            confidence *= 0.72
        observation_id = f"{view}_detail_{number:03d}"
        mask_path = _write(output_dir / "instances" / view / f"{observation_id}.png", mask)
        instances.append(EvidenceInstance(
            observation_id=observation_id,
            view=view,
            kind="generic_detail",
            state="machine_proposal",
            mask_path=mask_path,
            bbox_xywh=[x, y, width, height],
            centroid_xy_normalized=[round(float(centroids[label][0] / candidate.shape[1]), 6), round(float(centroids[label][1] / candidate.shape[0]), 6)],
            area_fraction=round(float(area / image_area), 8),
            circularity=round(circularity, 6),
            edge_support=round(edge_support, 6),
            highlight_overlap=round(highlight_overlap, 6),
            confidence=round(confidence, 6),
            uncertainty=uncertainty,
        ))
    return instances


def _descriptor(instance: EvidenceInstance) -> np.ndarray:
    return np.asarray([
        instance.centroid_xy_normalized[0],
        instance.centroid_xy_normalized[1],
        min(1.0, math.sqrt(instance.area_fraction) * 8.0),
        instance.circularity,
        instance.edge_support,
    ], np.float32)


def _associate(instances_by_view: dict[str, list[EvidenceInstance]]) -> list[dict[str, Any]]:
    """Create conservative cross-view hypotheses, never claimed as ground truth."""
    view_order = [view for view in ("front", "top", "iso", "lsv", "rsv", "single") if view in instances_by_view]
    if not view_order:
        return []
    anchor = max(view_order, key=lambda view: len(instances_by_view[view]))
    observation_lookup = {
        item.observation_id: item
        for items in instances_by_view.values()
        for item in items
    }
    tracks = [{"track_id": f"detail_track_{i:03d}", "observations": [item.observation_id], "confidence": item.confidence, "state": "machine_hypothesis"} for i, item in enumerate(instances_by_view[anchor], 1)]
    for item, track in zip(instances_by_view[anchor], tracks):
        item.track_id = track["track_id"]
    for view in view_order:
        if view == anchor:
            continue
        available = set(range(len(tracks)))
        for item in sorted(instances_by_view[view], key=lambda value: value.confidence, reverse=True):
            candidates = []
            for index in available:
                representative_id = tracks[index]["observations"][0]
                representative = observation_lookup[representative_id]
                distance = float(np.linalg.norm(_descriptor(item) - _descriptor(representative)))
                candidates.append((distance, index))
            if candidates and min(candidates)[0] <= 0.55:
                distance, index = min(candidates)
                track = tracks[index]
                track["observations"].append(item.observation_id)
                track["confidence"] = round(min(float(track["confidence"]), item.confidence) * max(0.25, 1.0 - distance), 6)
                item.track_id = track["track_id"]
                available.remove(index)
            else:
                track_id = f"detail_track_{len(tracks) + 1:03d}"
                item.track_id = track_id
                tracks.append({"track_id": track_id, "observations": [item.observation_id], "confidence": round(item.confidence * 0.6, 6), "state": "single_view_hypothesis"})
    return tracks


def _overlay(image: np.ndarray, instances: list[EvidenceInstance], maps: dict[str, np.ndarray]) -> np.ndarray:
    result = image.copy()
    result[maps["internal_edges"] > 0] = (35, 185, 35)
    result[maps["negative_space"] > 0] = (185, 80, 185)
    for number, instance in enumerate(instances, start=1):
        mask = _read(Path(instance.mask_path), cv2.IMREAD_GRAYSCALE)
        color = np.asarray(((37 * number) % 220 + 25, (83 * number) % 220 + 25, (151 * number) % 220 + 25), np.float32)
        selected = mask > 0
        result[selected] = (result[selected].astype(np.float32) * 0.45 + color * 0.55).astype(np.uint8)
        x, y, _, _ = instance.bbox_xywh
        cv2.putText(result, str(number), (x, max(12, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (10, 10, 10), 1, cv2.LINE_AA)
    return result


def extract_phase2_3(view_inputs: dict[str, dict[str, Path]], output_dir: Path) -> dict[str, Any]:
    """Extract reviewable per-view evidence and cross-view hypotheses."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Phase 2.3 output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    views: dict[str, Any] = {}
    instances_by_view: dict[str, list[EvidenceInstance]] = {}
    arrays: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    for view, paths in view_inputs.items():
        image = _read(paths["image"])
        foreground = _read(paths["foreground"], cv2.IMREAD_GRAYSCALE)
        if image.shape[:2] != foreground.shape:
            raise ValueError(f"Image and foreground dimensions differ for {view}")
        foreground = np.where(foreground > 0, 255, 0).astype(np.uint8)
        input_quality = _foreground_quality(foreground)
        maps = _geometry_maps(image, foreground)
        candidate = _instance_candidates(maps, foreground)
        instances = _extract_instances(view, candidate, maps, output_dir)
        instances_by_view[view] = instances
        arrays[view] = (image, maps)
        artifacts = {name: _write(output_dir / "views" / view / f"{name}.png", value) for name, value in maps.items()}
        artifacts["instance_candidate_map"] = _write(output_dir / "views" / view / "instance_candidate_map.png", candidate)
        views[view] = {
            "input": {key: str(value) for key, value in paths.items()},
            "artifacts": artifacts,
            "input_mask_quality": input_quality,
            "observations": [],
            "counts": {"machine_detail_proposals": len(instances)},
        }

    tracks = _associate(instances_by_view)
    for view, instances in instances_by_view.items():
        image, maps = arrays[view]
        views[view]["observations"] = [asdict(item) for item in instances]
        views[view]["artifacts"]["review_overlay"] = _write(output_dir / "views" / view / "review_overlay.png", _overlay(image, instances, maps))

    review = {
        "instructions": "Review each machine proposal. Set decision to accept, reject, split, merge, or relabel; do not infer hidden geometry here.",
        "allowed_decisions": ["pending", "accept", "reject", "split", "merge", "relabel"],
        "allowed_semantics": ["unknown_detail", "stone", "prong", "metal_body", "engraving", "relief", "filigree", "hole", "cavity_opening", "chain_link", "clasp", "bail", "other"],
        "decisions": {item.observation_id: {"decision": "pending", "semantic": "unknown_detail", "notes": ""} for values in instances_by_view.values() for item in values},
    }
    review_path = output_dir / "review_manifest.json"
    review_path.write_text(json.dumps(review, indent=2), encoding="utf-8")
    input_masks_pass = all(view["input_mask_quality"]["passed"] for view in views.values())
    proposal_count = sum(len(values) for values in instances_by_view.values())
    report = {
        "schema_version": "jewellery_evidence_graph_v1",
        "stage": "phase2_3_universal_evidence_extraction",
        "scope": "visible_image_evidence_only",
        "views": views,
        "cross_view_tracks": tracks,
        "validation": {
            "input_masks_pass": input_masks_pass,
            "machine_proposals_generated": proposal_count > 0,
            "ready_for_human_instance_review": input_masks_pass and proposal_count > 0,
            "human_review_complete": False,
            "human_ground_truth_available": False,
            "accuracy_claim_allowed": False,
        },
        "limitations": [
            "Generic detail instances are proposals, not confirmed stones or components.",
            "Cross-view tracks are appearance hypotheses until human review or calibrated geometry confirms them.",
            "Hidden geometry and physical dimensions are not inferred by this stage.",
        ],
        "review_manifest": str(review_path),
    }
    report_path = output_dir / "evidence_graph.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True, help="Completed upload-auditor output directory")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    audit = json.loads((args.audit_dir / "audit_report.json").read_text(encoding="utf-8"))
    inputs = {
        view: {
            "image": Path(record["artifacts"]["normalized"]),
            "foreground": Path(record["artifacts"]["phase2_refined_silhouette"]),
        }
        for view, record in audit["views"].items()
    }
    output = args.output_dir or args.audit_dir / "phase2_3"
    report = extract_phase2_3(inputs, output)
    print(json.dumps({"output_dir": str(output), "views": len(report["views"]), "tracks": len(report["cross_view_tracks"])}, indent=2))


if __name__ == "__main__":
    main()
