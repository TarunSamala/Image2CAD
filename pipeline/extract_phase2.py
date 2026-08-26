"""Phase 2: SAM-assisted semantic component proposals for jewellery views.

SAM is class-agnostic, so Phase 1 geometry supplies reproducible prompts and
view-specific regions.  Outputs are explicitly marked as predictions or
derived proposals rather than manufacturing ground truth.
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


VIEW_NAMES = ("front", "side", "top", "angled", "back")
COLORS = {
    "shank": (66, 135, 245),
    "stone": (255, 220, 40),
    "setting": (255, 145, 35),
    "prongs": (255, 55, 70),
    "shadow": (200, 60, 220),
}


def _predict(
    predictor: SAM2ImagePredictor,
    box: tuple[int, int, int, int],
    points: list[tuple[float, float]] | None = None,
    labels: list[int] | None = None,
    multimask: bool = True,
) -> tuple[np.ndarray, float]:
    point_array = np.asarray(points, dtype=np.float32) if points else None
    label_array = np.asarray(labels, dtype=np.int32) if labels else None
    masks, scores, _ = predictor.predict(
        point_coords=point_array,
        point_labels=label_array,
        box=np.asarray(box, dtype=np.float32),
        multimask_output=multimask,
    )
    index = int(np.argmax(scores))
    return masks[index].astype(np.uint8) * 255, float(scores[index])


def _regions(view: str, bbox: dict[str, int]) -> dict[str, Any]:
    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
    cx = x + w * 0.5
    if view in ("front", "top"):
        head = (x + 0.32 * w, y, x + 0.68 * w, y + h)
        stone = (x + 0.34 * w, y + 0.08 * h, x + 0.66 * w, y + 0.86 * h)
        shank_points = [(x + 0.13 * w, y + 0.56 * h), (x + 0.87 * w, y + 0.56 * h)]
    elif view in ("side", "back"):
        head = (x + 0.25 * w, y, x + 0.75 * w, y + 0.34 * h)
        stone = (x + 0.34 * w, y, x + 0.66 * w, y + 0.20 * h)
        shank_points = [(x + 0.13 * w, y + 0.55 * h), (x + 0.87 * w, y + 0.55 * h), (cx, y + 0.91 * h)]
    else:
        head = (x + 0.25 * w, y, x + 0.78 * w, y + 0.52 * h)
        stone = (x + 0.35 * w, y + 0.04 * h, x + 0.68 * w, y + 0.34 * h)
        shank_points = [(x + 0.14 * w, y + 0.50 * h), (x + 0.70 * w, y + 0.82 * h)]

    def integer_box(values: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        return tuple(round(value) for value in values)  # type: ignore[return-value]

    stone_box = integer_box(stone)
    return {
        "whole_box": (x, y, x + w - 1, y + h - 1),
        "head_box": integer_box(head),
        "stone_box": stone_box,
        "stone_center": ((stone_box[0] + stone_box[2]) / 2, (stone_box[1] + stone_box[3]) / 2),
        "shank_points": shank_points,
    }


def _box_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    result = np.zeros(shape, np.uint8)
    x0, y0, x1, y1 = box
    result[max(0, y0) : min(shape[0], y1 + 1), max(0, x0) : min(shape[1], x1 + 1)] = 255
    return result


def _component_record(mask: np.ndarray, confidence: float, source: str, parent: str | None = None) -> dict[str, Any]:
    points = cv2.findNonZero(mask)
    bbox = None
    if points is not None:
        x, y, w, h = cv2.boundingRect(points)
        bbox = {"x": x, "y": y, "width": w, "height": h}
    return {
        "area_px2": int(np.count_nonzero(mask)),
        "bbox_px": bbox,
        "confidence": round(max(0.0, min(1.0, confidence)), 5),
        "source": source,
        "parent": parent,
        "status": "predicted" if source.startswith("sam2") else "derived_proposal",
    }


def extract_view(
    predictor: SAM2ImagePredictor,
    image_path: Path,
    phase1: dict[str, Any],
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = bgr.shape[:2]
    phase1_mask_path = Path(phase1["artifacts"]["mask"])
    if not phase1_mask_path.is_absolute():
        phase1_mask_path = Path.cwd() / phase1_mask_path
    phase1_mask = cv2.imread(str(phase1_mask_path), cv2.IMREAD_GRAYSCALE)
    if phase1_mask is None:
        raise FileNotFoundError(phase1["artifacts"]["mask"])
    regions = _regions(phase1["view"], phase1["segmentation"]["bbox_px"])

    predictor.set_image(rgb)
    context = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    with context:
        jewelry_sam, jewelry_score = _predict(predictor, regions["whole_box"], multimask=False)
        stone_sam, stone_score = _predict(
            predictor, regions["stone_box"], [regions["stone_center"]], [1], multimask=True
        )
        shank_sam, shank_score = _predict(
            predictor,
            regions["whole_box"],
            regions["shank_points"] + [regions["stone_center"]],
            [1] * len(regions["shank_points"]) + [0],
            multimask=True,
        )

    # Phase 1 remains the permissive evidence mask. SAM removes much of the
    # soft shadow; semantic masks are constrained to that measured support.
    jewelry = cv2.bitwise_and(jewelry_sam, phase1_mask)
    if np.count_nonzero(jewelry) < np.count_nonzero(phase1_mask) * 0.55:
        jewelry = phase1_mask.copy()
        jewelry_score *= 0.65
    stone_allowed = cv2.bitwise_and(jewelry, _box_mask((h, w), regions["stone_box"]))
    stone = cv2.bitwise_and(stone_sam, stone_allowed)
    shank = cv2.bitwise_and(shank_sam, jewelry)
    shank = cv2.bitwise_and(shank, cv2.bitwise_not(cv2.dilate(stone, np.ones((3, 3), np.uint8))))

    head_support = cv2.bitwise_and(jewelry, _box_mask((h, w), regions["head_box"]))
    setting = cv2.bitwise_and(head_support, cv2.bitwise_not(stone))
    outside_head = cv2.bitwise_and(jewelry, cv2.bitwise_not(_box_mask((h, w), regions["head_box"])))
    shank = cv2.bitwise_or(shank, outside_head)
    shank = cv2.bitwise_and(shank, cv2.bitwise_not(stone))

    # Prongs are a hierarchical subset of the setting at this stage. A
    # jewellery-finetuned detector will split individual prongs in Phase 2.1.
    stone_dilated = cv2.dilate(stone, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    prongs = cv2.bitwise_and(setting, stone_dilated)
    shadow = cv2.bitwise_and(phase1_mask, cv2.bitwise_not(jewelry_sam))
    metal = cv2.bitwise_and(jewelry, cv2.bitwise_not(stone))

    masks = {"jewelry": jewelry, "metal": metal, "shank": shank, "stone": stone, "setting": setting, "prongs": prongs, "shadow": shadow}
    view = phase1["view"]
    for name, mask in masks.items():
        path = output_dir / "masks" / name / f"ring01_{view}_{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), mask)

    overlay = bgr.astype(np.float32)
    for name in ("shank", "setting", "stone", "prongs", "shadow"):
        active = masks[name] > 0
        color = np.asarray(COLORS[name][::-1], dtype=np.float32)
        overlay[active] = overlay[active] * 0.45 + color * 0.55
    overlay_path = output_dir / "overlays" / f"ring01_{view}_components.png"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(overlay_path), overlay.astype(np.uint8))

    phase1_area = max(1, int(np.count_nonzero(phase1_mask)))
    return {
        "view": view,
        "source": str(image_path),
        "prompt_regions": {key: value for key, value in regions.items() if key != "shank_points"},
        "components": {
            "jewelry": _component_record(jewelry, jewelry_score, "sam2_box_prompt"),
            "metal": _component_record(metal, min(jewelry_score, 0.84), "derived_from_jewelry_minus_stone", "jewelry"),
            "shank": _component_record(shank, shank_score, "sam2_points_and_geometry", "metal"),
            "stone": _component_record(stone, stone_score, "sam2_box_and_point_prompt", "jewelry"),
            "setting": _component_record(setting, min(stone_score, jewelry_score) * 0.82, "derived_head_region", "metal"),
            "prongs": _component_record(prongs, min(stone_score, jewelry_score) * 0.66, "derived_setting_near_stone", "setting"),
            "shadow": _component_record(shadow, 0.55, "phase1_minus_sam2_jewelry"),
        },
        "validation": {
            "phase1_coverage": round(np.count_nonzero(cv2.bitwise_and(jewelry, phase1_mask)) / phase1_area, 5),
            "stone_inside_jewelry": bool(np.count_nonzero(cv2.bitwise_and(stone, cv2.bitwise_not(jewelry))) == 0),
            "prongs_inside_setting": bool(np.count_nonzero(cv2.bitwise_and(prongs, cv2.bitwise_not(setting))) == 0),
        },
        "artifacts": {"overlay": str(overlay_path)},
    }


def run(input_dir: Path, phase1_path: Path, output_dir: Path, checkpoint: Path, device: str) -> dict[str, Any]:
    phase1_report = json.loads(phase1_path.read_text(encoding="utf-8"))
    selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if selected_device == "auto":
        selected_device = "cpu"
    model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_t.yaml", str(checkpoint), device=selected_device, apply_postprocessing=True
    )
    predictor = SAM2ImagePredictor(model, max_hole_area=16, max_sprinkle_area=16)
    views = {}
    for name in VIEW_NAMES:
        views[name] = extract_view(
            predictor,
            input_dir / f"ring01_{name}.png",
            phase1_report["views"][name],
            output_dir,
            selected_device,
        )
        if selected_device == "cuda":
            torch.cuda.empty_cache()
    report = {
        "sample": "ring01",
        "stage": "phase2_semantic_component_proposals",
        "model": "sam2.1_hiera_tiny",
        "device": selected_device,
        "view_count": len(views),
        "taxonomy": {"jewelry": ["metal", "stone"], "metal": ["shank", "setting"], "setting": ["prongs"]},
        "warning": "SAM 2 is class-agnostic; labels are geometry-assisted proposals requiring validation.",
        "views": views,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase2_components.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/ring01_reference_images"))
    parser.add_argument("--phase1", type=Path, default=Path("data/ring01_phase1/phase1_features.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/ring01_phase2"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/sam2.1_hiera_tiny.pt"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    report = run(args.input_dir, args.phase1, args.output_dir, args.checkpoint, args.device)
    print(json.dumps({name: view["components"] for name, view in report["views"].items()}, indent=2))


if __name__ == "__main__":
    main()
