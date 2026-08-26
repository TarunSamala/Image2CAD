"""Phase 2.2: image-guided boundary and component refinement.

This stage is deliberately reversible.  It refines Phase 2.1 masks with
GrabCut and edge-aware watershed without modifying any earlier artifacts.
The output is not described as ground truth: occluded geometry and reflective
boundaries are recorded as manual-review items.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


VIEWS = ("front", "side", "top", "angled", "back")
COMPONENTS = ("jewelry", "metal", "shank", "stone_visible", "stone_amodal", "setting", "prongs", "shadow")
COLORS = {
    "shank": (66, 135, 245),
    "stone_visible": (255, 220, 40),
    "setting": (255, 145, 35),
    "prongs": (255, 55, 70),
    "shadow": (200, 60, 220),
}


def _read_mask(base: Path, view: str, component: str) -> np.ndarray:
    path = base / "masks" / component / f"ring01_{view}_{component}.png"
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise FileNotFoundError(path)
    return np.where(value > 127, 255, 0).astype(np.uint8)


def _kernel(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _grabcut(image: np.ndarray, prior: np.ndarray, iterations: int = 5) -> np.ndarray:
    """Refine only a narrow band around a prior mask."""
    if np.count_nonzero(prior) < 10:
        return prior.copy()
    scale = max(1, round(min(image.shape[:2]) / 150))
    inner = cv2.erode(prior, _kernel(2 * scale + 1), iterations=1)
    outer = cv2.dilate(prior, _kernel(4 * scale + 1), iterations=1)
    labels = np.full(prior.shape, cv2.GC_BGD, np.uint8)
    labels[outer > 0] = cv2.GC_PR_BGD
    labels[prior > 0] = cv2.GC_PR_FGD
    labels[inner > 0] = cv2.GC_FGD
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image, labels, None, bg_model, fg_model, iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return prior.copy()
    result = np.where((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    # Never permit image glare to remove the reliable interior seed.
    return cv2.bitwise_or(result, inner)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    count, labels = cv2.connectedComponents((mask > 0).astype(np.uint8))
    return [np.where(labels == index, 255, 0).astype(np.uint8) for index in range(1, count)]


def _retain_seeded(candidate: np.ndarray, seed: np.ndarray) -> np.ndarray:
    pieces = _components(candidate)
    if not pieces:
        return seed.copy()
    best = max(pieces, key=lambda part: np.count_nonzero(cv2.bitwise_and(part, seed)))
    if np.count_nonzero(cv2.bitwise_and(best, seed)) == 0:
        return seed.copy()
    return best


def _closed_outer(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(mask)
    if contours:
        cv2.drawContours(result, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return result


def _refine_stone(image: np.ndarray, prior: np.ndarray) -> np.ndarray:
    candidate = _retain_seeded(_grabcut(image, prior), cv2.erode(prior, _kernel(3)))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, _kernel(5), iterations=2)
    candidate = _closed_outer(candidate)
    # Constrain the fit to the local prior band to prevent shiny metal merging.
    return cv2.bitwise_and(candidate, cv2.dilate(prior, _kernel(7)))


def _refine_prongs(image: np.ndarray, prior: np.ndarray, force_four: bool) -> tuple[np.ndarray, list[dict[str, Any]]]:
    pieces = _components(prior)
    result = np.zeros_like(prior)
    records: list[dict[str, Any]] = []
    for index, piece in enumerate(pieces, start=1):
        refined = _retain_seeded(_grabcut(image, piece, 4), cv2.erode(piece, _kernel(3)))
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, _kernel(3))
        refined = _retain_seeded(refined, piece)
        if np.count_nonzero(refined) < 8:
            refined = piece
        result = cv2.bitwise_or(result, refined)
        moments = cv2.moments(refined, binaryImage=True)
        records.append({
            "prong_id": index,
            "area_px2": int(np.count_nonzero(refined)),
            "center_px": {
                "x": round(moments["m10"] / moments["m00"], 3) if moments["m00"] else None,
                "y": round(moments["m01"] / moments["m00"], 3) if moments["m00"] else None,
            },
            "method": "local_grabcut_with_topology_seed",
        })
    if force_four and len(pieces) != 4:
        return prior.copy(), []
    return result, records


def _partition_metal(image: np.ndarray, metal: np.ndarray, shank: np.ndarray, setting: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Move the internal boundary using image gradients while retaining two parts."""
    markers = np.zeros(metal.shape, np.int32)
    markers[metal == 0] = 1
    shank_seed = cv2.erode(shank, _kernel(5))
    setting_seed = cv2.erode(setting, _kernel(5))
    if not np.any(shank_seed) or not np.any(setting_seed):
        return shank.copy(), setting.copy()
    markers[shank_seed > 0] = 2
    markers[setting_seed > 0] = 3
    cv2.watershed(image.copy(), markers)
    shank_out = np.where(markers == 2, 255, 0).astype(np.uint8)
    setting_out = np.where(markers == 3, 255, 0).astype(np.uint8)
    shank_out = cv2.bitwise_and(shank_out, metal)
    setting_out = cv2.bitwise_and(setting_out, metal)
    unresolved = cv2.bitwise_and(metal, cv2.bitwise_not(cv2.bitwise_or(shank_out, setting_out)))
    if np.any(unresolved):
        distance_shank = cv2.distanceTransform(cv2.bitwise_not(shank_seed), cv2.DIST_L2, 3)
        distance_setting = cv2.distanceTransform(cv2.bitwise_not(setting_seed), cv2.DIST_L2, 3)
        shank_out[(unresolved > 0) & (distance_shank <= distance_setting)] = 255
        setting_out[(unresolved > 0) & (distance_setting < distance_shank)] = 255
    return shank_out, setting_out


def _boundary(mask: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, _kernel(3))


def _edge_support(mask: np.ndarray, gray: np.ndarray) -> float:
    edge = cv2.Canny(gray, 55, 145)
    edge = cv2.dilate(edge, _kernel(3))
    boundary = _boundary(mask)
    total = np.count_nonzero(boundary)
    return float(np.count_nonzero(cv2.bitwise_and(boundary, edge)) / max(1, total))


def _save_mask(output: Path, view: str, component: str, mask: np.ndarray) -> str:
    path = output / "masks" / component / f"ring01_{view}_{component}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask)
    return str(path)


def _metrics(mask: np.ndarray, gray: np.ndarray) -> dict[str, Any]:
    points = cv2.findNonZero(mask)
    bbox = None
    if points is not None:
        x, y, width, height = cv2.boundingRect(points)
        bbox = {"x": x, "y": y, "width": width, "height": height}
    return {
        "area_px2": int(np.count_nonzero(mask)),
        "bbox_px": bbox,
        "boundary_edge_support": round(_edge_support(mask, gray), 5),
    }


def _write_review(image: np.ndarray, masks: dict[str, np.ndarray], view: str, output: Path) -> str:
    overlay = image.copy()
    for name in ("shank", "setting", "stone_visible", "prongs", "shadow"):
        active = masks[name] > 0
        color = np.asarray(COLORS[name][::-1], np.uint8)
        overlay[active] = (overlay[active].astype(np.float32) * 0.5 + color * 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(masks[name], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, tuple(int(v) for v in color), 1, cv2.LINE_AA)
    legend_height = 30
    canvas = np.full((image.shape[0] + legend_height, image.shape[1] * 2, 3), 248, np.uint8)
    canvas[legend_height:, : image.shape[1]] = image
    canvas[legend_height:, image.shape[1] :] = overlay
    cv2.putText(canvas, f"{view.upper()} | ORIGINAL", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, "PHASE 2.2 REVIEW (not ground truth)", (image.shape[1] + 6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (20, 20, 20), 1, cv2.LINE_AA)
    path = output / "review" / f"ring01_{view}_annotation_review.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)
    return str(path)


def run(input_dir: Path, phase2_1_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    views: dict[str, Any] = {}
    for view in VIEWS:
        source_path = input_dir / f"ring01_{view}.png"
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(source_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        prior = {name: _read_mask(phase2_1_dir, view, name) for name in COMPONENTS}

        jewelry = _retain_seeded(_grabcut(image, prior["jewelry"]), cv2.erode(prior["jewelry"], _kernel(3)))
        stone_amodal = _refine_stone(image, prior["stone_amodal"])
        prongs, instances = _refine_prongs(image, prior["prongs"], view in ("front", "top"))
        jewelry = cv2.bitwise_or(jewelry, cv2.bitwise_or(stone_amodal, prongs))
        stone_visible = cv2.bitwise_and(stone_amodal, cv2.bitwise_not(prongs))
        metal = cv2.bitwise_and(jewelry, cv2.bitwise_not(stone_visible))
        shank_seed = cv2.bitwise_and(prior["shank"], metal)
        setting_seed = cv2.bitwise_and(cv2.bitwise_or(prior["setting"], prongs), metal)
        shank, setting = _partition_metal(image, metal, shank_seed, setting_seed)
        setting = cv2.bitwise_or(setting, prongs)
        shank = cv2.bitwise_and(metal, cv2.bitwise_not(setting))

        shadow = _grabcut(image, prior["shadow"], 3)
        shadow = cv2.morphologyEx(shadow, cv2.MORPH_OPEN, _kernel(3))
        shadow = cv2.bitwise_and(shadow, cv2.bitwise_not(jewelry))
        masks = {
            "jewelry": jewelry,
            "metal": metal,
            "shank": shank,
            "stone_visible": stone_visible,
            "stone_amodal": stone_amodal,
            "setting": setting,
            "prongs": prongs,
            "shadow": shadow,
        }
        artifacts = {name: _save_mask(output_dir, view, name, mask) for name, mask in masks.items()}
        artifacts["review"] = _write_review(image, masks, view, output_dir)
        review = ["shadow boundary versus reflective metal"]
        if view in ("side", "back", "angled"):
            review.append("occluded prong count and hidden stone/setting geometry")
        if view == "angled":
            review.append("foreshortened gemstone outline")
        views[view] = {
            "components": {name: _metrics(mask, gray) for name, mask in masks.items()},
            "prong_instances": instances,
            "manual_review_required": review,
            "artifacts": artifacts,
        }

    report = {
        "sample": "ring01",
        "stage": "phase2_2_boundary_component_refinement",
        "source_stage": str(phase2_1_dir / "phase2_refined.json"),
        "reversible": True,
        "algorithms": ["OpenCV GrabCut narrow-band refinement", "edge-aware watershed partition", "connected-component topology constraints"],
        "ground_truth_status": "unavailable_pending_human_annotation",
        "ground_truth_metrics": {"iou": None, "dice": None, "boundary_f1": None},
        "views": views,
    }
    (output_dir / "phase2_2_components.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/ring01_reference_images"))
    parser.add_argument("--phase2-1-dir", type=Path, default=Path("data/ring01_phase2_refined"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/ring01_phase2_2"))
    args = parser.parse_args()
    report = run(args.input_dir, args.phase2_1_dir, args.output_dir)
    print(json.dumps({"output": str(args.output_dir), "views": list(report["views"]), "reversible": True}, indent=2))


if __name__ == "__main__":
    main()
