"""Phase 2.1: topology-constrained repair of Phase 2 jewellery masks.

The SAM outputs remain untouched.  This stage writes a separate refinement
directory and uses explicit Ring01 topology: one stone, one shank, one setting,
and four prongs visible in the front/top reference views.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


VIEWS = ("front", "side", "top", "angled", "back")
COLORS = {
    "shank": (66, 135, 245),
    "stone_visible": (255, 220, 40),
    "setting": (255, 145, 35),
    "prongs": (255, 55, 70),
    "shadow": (200, 60, 220),
}


def _read_mask(base: Path, view: str, component: str) -> np.ndarray:
    path = base / "masks" / component / f"ring01_{view}_{component}.png"
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask


def _filled_stone(mask: np.ndarray, view: str) -> np.ndarray:
    points = cv2.findNonZero(mask)
    if points is None:
        return mask.copy()
    result = np.zeros_like(mask)
    if view in ("front", "top", "angled"):
        x, y, w, h = cv2.boundingRect(points)
        center = (round(x + (w - 1) / 2), round(y + (h - 1) / 2))
        axes = (max(1, round((w - 1) / 2)), max(1, round((h - 1) / 2)))
        cv2.ellipse(result, center, axes, 0, 0, 360, 255, -1)
    else:
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(result, hull, 255)
    return result


def _four_prongs(setting: np.ndarray, stone: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    points = cv2.findNonZero(stone)
    if points is None:
        return np.zeros_like(stone), []
    x, y, w, h = cv2.boundingRect(points)
    cx, cy = x + (w - 1) / 2, y + (h - 1) / 2
    radius = max(4, round(max(w, h) * 0.115))
    expected = [
        (cx - 0.45 * w, cy - 0.50 * h),
        (cx + 0.45 * w, cy - 0.50 * h),
        (cx - 0.45 * w, cy + 0.50 * h),
        (cx + 0.45 * w, cy + 0.50 * h),
    ]
    result = np.zeros_like(stone)
    records = []
    for index, (px, py) in enumerate(expected, start=1):
        roi = np.zeros_like(stone)
        center = (round(px), round(py))
        cv2.circle(roi, center, radius, 255, -1)
        candidate = cv2.bitwise_and(setting, roi)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
        contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        prong = np.zeros_like(stone)
        if contours:
            nearest = min(
                contours,
                key=lambda contour: (cv2.boundingRect(contour)[0] + cv2.boundingRect(contour)[2] / 2 - px) ** 2
                + (cv2.boundingRect(contour)[1] + cv2.boundingRect(contour)[3] / 2 - py) ** 2,
            )
            cv2.fillConvexPoly(prong, cv2.convexHull(nearest), 255)
            prong = cv2.dilate(prong, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        if np.count_nonzero(prong) < 12:
            cv2.circle(prong, center, max(3, radius - 2), 255, -1)
        prong = cv2.bitwise_and(prong, roi)
        result = cv2.bitwise_or(result, prong)
        moments = cv2.moments(prong, binaryImage=True)
        records.append({
            "prong_id": index,
            "center_px": {
                "x": round(moments["m10"] / moments["m00"], 3) if moments["m00"] else center[0],
                "y": round(moments["m01"] / moments["m00"], 3) if moments["m00"] else center[1],
            },
            "area_px2": int(np.count_nonzero(prong)),
            "source": "setting_region_plus_four_prong_topology",
        })
    return result, records


def _save_mask(output: Path, view: str, component: str, mask: np.ndarray) -> str:
    path = output / "masks" / component / f"ring01_{view}_{component}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask)
    return str(path)


def _metrics(mask: np.ndarray) -> dict[str, Any]:
    points = cv2.findNonZero(mask)
    bbox = None
    if points is not None:
        x, y, w, h = cv2.boundingRect(points)
        bbox = {"x": x, "y": y, "width": w, "height": h}
    return {"area_px2": int(np.count_nonzero(mask)), "bbox_px": bbox}


def run(input_dir: Path, phase2_dir: Path, output_dir: Path) -> dict[str, Any]:
    source_report = json.loads((phase2_dir / "phase2_components.json").read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    views = {}
    for view in VIEWS:
        bgr = cv2.imread(str(input_dir / f"ring01_{view}.png"), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(input_dir / f"ring01_{view}.png")
        jewelry = _read_mask(phase2_dir, view, "jewelry")
        baseline_setting = _read_mask(phase2_dir, view, "setting")
        baseline_shadow = _read_mask(phase2_dir, view, "shadow")
        amodal_stone = _filled_stone(_read_mask(phase2_dir, view, "stone"), view)

        prong_records: list[dict[str, Any]] = []
        if view in ("front", "top"):
            prongs, prong_records = _four_prongs(baseline_setting, amodal_stone)
        else:
            prongs = _read_mask(phase2_dir, view, "prongs")

        # The stone is complete geometrically, while the visible mask accounts
        # for metal prongs occluding its perimeter.
        stone_visible = cv2.bitwise_and(amodal_stone, cv2.bitwise_not(prongs))
        jewelry = cv2.bitwise_or(jewelry, cv2.bitwise_or(amodal_stone, prongs))
        metal = cv2.bitwise_and(jewelry, cv2.bitwise_not(stone_visible))

        head_box = source_report["views"][view]["prompt_regions"]["head_box"]
        x0, y0, x1, y1 = (int(value) for value in head_box)
        head_support = np.zeros_like(jewelry)
        head_support[max(0, y0) : min(jewelry.shape[0], y1 + 1), max(0, x0) : min(jewelry.shape[1], x1 + 1)] = 255
        setting = cv2.bitwise_and(metal, head_support)
        setting = cv2.bitwise_or(setting, prongs)
        shank = cv2.bitwise_and(metal, cv2.bitwise_not(setting))
        shadow = cv2.bitwise_and(baseline_shadow, cv2.bitwise_not(jewelry))

        masks = {
            "jewelry": jewelry,
            "metal": metal,
            "shank": shank,
            "stone_visible": stone_visible,
            "stone_amodal": amodal_stone,
            "setting": setting,
            "prongs": prongs,
            "shadow": shadow,
        }
        artifacts = {name: _save_mask(output_dir, view, name, mask) for name, mask in masks.items()}

        overlay = bgr.astype(np.float32)
        for name in ("shank", "setting", "stone_visible", "prongs", "shadow"):
            active = masks[name] > 0
            color = np.asarray(COLORS[name][::-1], dtype=np.float32)
            overlay[active] = overlay[active] * 0.45 + color * 0.55
        overlay_path = output_dir / "overlays" / f"ring01_{view}_refined.png"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(overlay_path), overlay.astype(np.uint8))
        artifacts["overlay"] = str(overlay_path)

        views[view] = {
            "components": {name: _metrics(mask) for name, mask in masks.items()},
            "prong_instances": prong_records,
            "artifacts": artifacts,
        }

    report = {
        "sample": "ring01",
        "stage": "phase2_1_topology_refinement",
        "source_stage": str(phase2_dir / "phase2_components.json"),
        "reversible": True,
        "rules": [
            "closed amodal gemstone silhouette",
            "disjoint shank and setting",
            "four topology-guided prongs in front/top views",
            "shadow excludes refined jewellery",
        ],
        "views": views,
    }
    (output_dir / "phase2_refined.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/ring01_reference_images"))
    parser.add_argument("--phase2-dir", type=Path, default=Path("data/ring01_phase2"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/ring01_phase2_refined"))
    args = parser.parse_args()
    report = run(args.input_dir, args.phase2_dir, args.output_dir)
    print(json.dumps({"output": str(args.output_dir), "views": list(report["views"]), "reversible": True}, indent=2))


if __name__ == "__main__":
    main()
