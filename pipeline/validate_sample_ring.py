"""Validate a single real-photo ring through Phase 1 and Phase 2 refinement."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _predict(
    predictor: SAM2ImagePredictor,
    box: tuple[int, int, int, int],
    points: list[tuple[int, int]],
    labels: list[int],
    device: str,
) -> tuple[np.ndarray, float]:
    context = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    with context:
        masks, scores, _ = predictor.predict(
            point_coords=np.asarray(points, np.float32),
            point_labels=np.asarray(labels, np.int32),
            box=np.asarray(box, np.float32),
            multimask_output=True,
        )
    index = int(np.argmax(scores))
    return masks[index].astype(np.uint8) * 255, float(scores[index])


def _largest_seeded_component(candidate: np.ndarray, seed: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    best_label, best_overlap = 0, 0
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < 40:
            continue
        overlap = np.count_nonzero((labels == label) & (seed > 0))
        if overlap > best_overlap:
            best_label, best_overlap = label, overlap
    return np.where(labels == best_label, 255, 0).astype(np.uint8) if best_label else seed.copy()


def _grabcut_refine(image: np.ndarray, prior: np.ndarray) -> np.ndarray:
    inner = cv2.erode(prior, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    outer = cv2.dilate(prior, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)))
    labels = np.full(prior.shape, cv2.GC_BGD, np.uint8)
    labels[outer > 0] = cv2.GC_PR_BGD
    labels[prior > 0] = cv2.GC_PR_FGD
    labels[inner > 0] = cv2.GC_FGD
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, labels, None, background_model, foreground_model, 5, cv2.GC_INIT_WITH_MASK)
    candidate = np.where((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return _largest_seeded_component(candidate, inner)


def _normalize(image: np.ndarray, mask: np.ndarray, size: int = 768) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    points = cv2.findNonZero(mask)
    if points is None:
        raise RuntimeError("The sample ring silhouette is empty")
    x, y, width, height = cv2.boundingRect(points)
    margin = round(max(width, height) * 0.08)
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(image.shape[1], x + width + margin), min(image.shape[0], y + height + margin)
    crop, crop_mask = image[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    scale = min((size - 80) / crop.shape[1], (size - 80) / crop.shape[0])
    resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(crop_mask, (resized.shape[1], resized.shape[0]), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((size, size, 3), 245, np.uint8)
    normalized_mask = np.zeros((size, size), np.uint8)
    px, py = (size - resized.shape[1]) // 2, (size - resized.shape[0]) // 2
    region = canvas[py : py + resized.shape[0], px : px + resized.shape[1]]
    active = resized_mask > 0
    region[active] = resized[active]
    normalized_mask[py : py + resized.shape[0], px : px + resized.shape[1]] = resized_mask
    return canvas, normalized_mask, {
        "source_bbox_xywh": [x, y, width, height],
        "crop_xyxy": [x0, y0, x1, y1],
        "scale": round(scale, 7),
        "placement_xywh": [px, py, resized.shape[1], resized.shape[0]],
    }


def _shadow_proposal(image: np.ndarray, jewelry: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    points = cv2.findNonZero(jewelry)
    if points is None:
        return np.zeros_like(jewelry), 0.0
    x, y, width, height = cv2.boundingRect(points)
    support = np.zeros_like(jewelry)
    top = min(image.shape[0], y + round(height * 0.68))
    bottom = min(image.shape[0], y + height + round(height * 0.16))
    left = max(0, x + round(width * 0.08))
    right = min(image.shape[1], x + round(width * 0.92))
    support[top:bottom, left:right] = 255
    support = cv2.bitwise_and(support, cv2.bitwise_not(jewelry))
    local_values = gray[support > 0]
    if not len(local_values):
        return np.zeros_like(jewelry), 0.0
    threshold = float(np.percentile(local_values, 34))
    shadow = np.where((gray <= threshold) & (support > 0), 255, 0).astype(np.uint8)
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    shadow = cv2.bitwise_and(shadow, cv2.bitwise_not(jewelry))
    return shadow, threshold


def _support_instances(
    predictor: SAM2ImagePredictor,
    jewelry: np.ndarray,
    image_shape: tuple[int, int],
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    height, width = image_shape
    definitions = {
        "support_nw": ((0.27, 0.18, 0.43, 0.50), (0.36, 0.29)),
        "support_ne": ((0.57, 0.18, 0.73, 0.50), (0.65, 0.29)),
        "support_sw": ((0.27, 0.43, 0.44, 0.77), (0.36, 0.64)),
        "support_se": ((0.57, 0.43, 0.74, 0.77), (0.65, 0.64)),
    }
    instances, scores = {}, {}
    for name, (box_f, point_f) in definitions.items():
        box = tuple(round(value * (width if index % 2 == 0 else height)) for index, value in enumerate(box_f))
        point = (round(point_f[0] * width), round(point_f[1] * height))
        mask, score = _predict(predictor, box, [point], [1], device)
        instances[name] = cv2.bitwise_and(mask, jewelry)
        scores[name] = round(score, 6)
    return instances, scores


def _edges(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 135, L2gradient=True)
    return cv2.bitwise_and(edges, cv2.dilate(mask, np.ones((5, 5), np.uint8)))


def _fit(value: np.ndarray, width: int, height: int) -> np.ndarray:
    if value.ndim == 2:
        value = cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)
    scale = min(width / value.shape[1], height / value.shape[0])
    resized = cv2.resize(value, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 248, np.uint8)
    x, y = (width - resized.shape[1]) // 2, (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _cell(value: np.ndarray, label: str, width: int = 360, height: int = 255) -> np.ndarray:
    canvas = np.full((height + 34, width, 3), 248, np.uint8)
    cv2.putText(canvas, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (25, 25, 25), 1, cv2.LINE_AA)
    canvas[34:] = _fit(value, width, height)
    return canvas


def _instance_visual(image: np.ndarray, instances: dict[str, np.ndarray]) -> np.ndarray:
    result = image.copy().astype(np.float32)
    colors = ((45, 65, 230), (45, 190, 230), (210, 80, 180), (80, 190, 70))
    for color, (_, mask) in zip(colors, instances.items()):
        active = mask > 0
        result[active] = result[active] * 0.35 + np.asarray(color, np.float32) * 0.65
    return result.astype(np.uint8)


def _overlay(image: np.ndarray, jewelry: np.ndarray, shadow: np.ndarray, instances: dict[str, np.ndarray]) -> np.ndarray:
    result = image.copy().astype(np.float32)
    result[shadow > 0] = result[shadow > 0] * 0.35 + np.asarray((180, 65, 180), np.float32) * 0.65
    boundary = cv2.morphologyEx(jewelry, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    result[boundary] = (40, 210, 40)
    result = _instance_visual(result.astype(np.uint8), instances).astype(np.float32)
    return result.astype(np.uint8)


def _comparison(
    source: np.ndarray,
    initial: np.ndarray,
    normalized: np.ndarray,
    edges: np.ndarray,
    refined: np.ndarray,
    shadow: np.ndarray,
    instances: dict[str, np.ndarray],
    overlay: np.ndarray,
) -> np.ndarray:
    row1 = np.hstack((
        _cell(source, "REFERENCE PHOTO"),
        _cell(initial, "PHASE 1 - SAM SILHOUETTE"),
        _cell(normalized, "PHASE 1 - NORMALIZED"),
        _cell(edges, "PHASE 1 - EDGES"),
    ))
    row2 = np.hstack((
        _cell(refined, "PHASE 2 - REFINED SILHOUETTE"),
        _cell(shadow, "PHASE 2 - SHADOW PROPOSAL"),
        _cell(_instance_visual(source, instances), "PHASE 2 - 4 SUPPORT PROPOSALS"),
        _cell(overlay, "PHASE 2 - FINAL OVERLAY"),
    ))
    header = np.full((92, row1.shape[1], 3), 250, np.uint8)
    cv2.putText(header, "SAMPLE RING - PHASE 1 / PHASE 2 VALIDATION", (24, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (72, 42, 12), 2, cv2.LINE_AA)
    cv2.putText(header, "Single real photograph | prompted segmentation | component proposals require manual review", (24, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 80, 80), 1, cv2.LINE_AA)
    return np.vstack((header, row1, row2))


def run(input_path: Path, output_dir: Path, checkpoint: Path, device_name: str) -> dict[str, Any]:
    source_hash = _sha256(input_path)
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(input_path)
    height, width = image.shape[:2]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        str(checkpoint),
        device=device_name,
        apply_postprocessing=False,
    )
    predictor = SAM2ImagePredictor(model, max_hole_area=0, max_sprinkle_area=0)
    predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    box = (round(width * 0.13), round(height * 0.16), round(width * 0.90), round(height * 0.80))
    positives = [
        (round(width * 0.50), round(height * 0.44)),
        (round(width * 0.23), round(height * 0.46)),
        (round(width * 0.80), round(height * 0.45)),
        (round(width * 0.50), round(height * 0.66)),
    ]
    negatives = [
        (round(width * 0.05), round(height * 0.10)),
        (round(width * 0.95), round(height * 0.10)),
        (round(width * 0.08), round(height * 0.90)),
        (round(width * 0.92), round(height * 0.90)),
    ]
    initial, sam_score = _predict(predictor, box, positives + negatives, [1] * len(positives) + [0] * len(negatives), device_name)
    refined = _grabcut_refine(image, initial)
    instances, support_scores = _support_instances(predictor, refined, (height, width), device_name)
    shadow, shadow_threshold = _shadow_proposal(image, refined)
    edges = _edges(image, refined)
    normalized, normalized_mask, transform = _normalize(image, refined)
    overlay = _overlay(image, refined, shadow, instances)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "phase1_silhouette": output_dir / "phase1_silhouette.png",
        "phase1_edges": output_dir / "phase1_edges.png",
        "phase1_normalized": output_dir / "phase1_normalized.png",
        "phase1_normalized_mask": output_dir / "phase1_normalized_mask.png",
        "phase2_refined_silhouette": output_dir / "phase2_refined_silhouette.png",
        "phase2_shadow_proposal": output_dir / "phase2_shadow_proposal.png",
        "phase2_support_overlay": output_dir / "phase2_support_overlay.png",
        "phase2_final_overlay": output_dir / "phase2_final_overlay.png",
        "comparison": output_dir / "sample_ring_phase_comparison.png",
    }
    images = {
        "phase1_silhouette": initial,
        "phase1_edges": edges,
        "phase1_normalized": normalized,
        "phase1_normalized_mask": normalized_mask,
        "phase2_refined_silhouette": refined,
        "phase2_shadow_proposal": shadow,
        "phase2_support_overlay": _instance_visual(image, instances),
        "phase2_final_overlay": overlay,
    }
    for name, value in images.items():
        cv2.imwrite(str(artifacts[name]), value)
    support_paths = {}
    for name, mask in instances.items():
        path = output_dir / "support_instances" / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), mask)
        support_paths[name] = str(path)
    comparison = _comparison(image, initial, normalized, edges, refined, shadow, instances, overlay)
    cv2.imwrite(str(artifacts["comparison"]), comparison)

    initial_area, refined_area = np.count_nonzero(initial), np.count_nonzero(refined)
    report = {
        "sample": "ring_02_real_photo",
        "source": str(input_path),
        "source_sha256": source_hash,
        "source_preserved": source_hash == _sha256(input_path),
        "input_view_count": 1,
        "device": device_name,
        "phase1": {
            "method": "SAM 2.1 Tiny prompted silhouette plus normalization",
            "sam_score": round(sam_score, 6),
            "prompt_box_xyxy": list(box),
            "normalization": transform,
        },
        "phase2": {
            "method": "narrow-band GrabCut, masked Canny edges, local shadow proposal, prompted support instances",
            "initial_area_px2": int(initial_area),
            "refined_area_px2": int(refined_area),
            "refined_to_initial_area_ratio": round(refined_area / max(1, initial_area), 6),
            "shadow_area_px2": int(np.count_nonzero(shadow)),
            "shadow_gray_threshold": round(shadow_threshold, 3),
            "support_instance_count": len(instances),
            "support_sam_scores": support_scores,
        },
        "validation": {
            "silhouette_nonempty": refined_area > 0,
            "normalization_complete": normalized.shape[:2] == (768, 768),
            "four_support_proposals_present": all(np.count_nonzero(mask) > 0 for mask in instances.values()),
            "shadow_disjoint_from_jewelry": np.count_nonzero((shadow > 0) & (refined > 0)) == 0,
            "pixel_accuracy_validated": False,
            "multi_view_consistency_validated": False,
        },
        "interpretation": {
            "prongs": "Four outer corner structures are support proposals, not confirmed gemstone prongs.",
            "shadow": "Low-confidence local cast-shadow proposal requiring manual review.",
            "silhouette": "Prompted machine prediction; no human mask exists for IoU validation.",
        },
        "artifacts": {name: str(path) for name, path in artifacts.items()} | {"support_instances": support_paths},
    }
    (output_dir / "sample_ring_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset/Sample_test/ring_02.jpeg"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/Sample_test/phase_validation"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/sam2.1_hiera_tiny.pt"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    report = run(args.input, args.output_dir, args.checkpoint, args.device)
    print(json.dumps({
        "source_preserved": report["source_preserved"],
        "phase1_sam_score": report["phase1"]["sam_score"],
        "phase2": report["phase2"],
        "validation": report["validation"],
        "comparison": report["artifacts"]["comparison"],
    }, indent=2))


if __name__ == "__main__":
    main()
