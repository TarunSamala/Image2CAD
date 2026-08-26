"""Phase 1: deterministic multi-view jewellery feature extraction.

Produces masks, edges, contour geometry, symmetry scores, and diagnostic
overlays.  This stage deliberately uses classical CV so its measurements are
reproducible and can later supervise SAM/depth/CAD fitting stages.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


VIEW_NAMES = ("front", "side", "top", "angled", "back")


def _remove_border_and_noise(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = mask.shape
    candidates: list[tuple[int, int]] = []
    for label in range(1, count):
        x, y, cw, ch, area = stats[label]
        touches_border = x <= 3 or y <= 3 or x + cw >= w - 3 or y + ch >= h - 3
        if not touches_border and area >= max(12, int(h * w * 0.00025)):
            candidates.append((label, int(area)))
    if not candidates:
        return np.zeros_like(mask)

    largest_label, largest_area = max(candidates, key=lambda item: item[1])
    lx, ly, lw, lh, _ = stats[largest_label]
    margin_x, margin_y = int(lw * 0.18), int(lh * 0.18)
    result = np.zeros_like(mask)
    for label, area in candidates:
        x, y, cw, ch, _ = stats[label]
        overlaps_object_region = not (
            x + cw < lx - margin_x or x > lx + lw + margin_x
            or y + ch < ly - margin_y or y > ly + lh + margin_y
        )
        if label == largest_label or (overlaps_object_region and area >= largest_area * 0.01):
            result[labels == label] = 255
    return result


def _segment(image: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu_threshold, raw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    raw[:4, :] = 0
    raw[-4:, :] = 0
    raw[:, :4] = 0
    raw[:, -4:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel, iterations=2)
    return _remove_border_and_noise(raw), float(otsu_threshold)


def _symmetry_iou(mask: np.ndarray, axis: int) -> float:
    points = cv2.findNonZero(mask)
    if points is None:
        return 0.0
    x, y, w, h = cv2.boundingRect(points)
    roi = mask[y : y + h, x : x + w]
    flipped = cv2.flip(roi, axis)
    intersection = np.count_nonzero(cv2.bitwise_and(roi, flipped))
    union = np.count_nonzero(cv2.bitwise_or(roi, flipped))
    return round(intersection / union, 5) if union else 0.0


def extract_view(path: Path, output_dir: Path, view_name: str) -> dict[str, Any]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    h, w = image.shape[:2]
    mask, otsu_threshold = _segment(image)
    points = cv2.findNonZero(mask)
    if points is None:
        raise RuntimeError(f"No jewellery foreground detected in {path}")

    x, y, bw, bh = cv2.boundingRect(points)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_records = []
    total_perimeter = 0.0
    for index, contour in enumerate(contours):
        perimeter = cv2.arcLength(contour, True)
        area = abs(cv2.contourArea(contour))
        if area < 8 or perimeter < 8:
            continue
        epsilon = max(0.5, perimeter * 0.0025)
        polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        parent = int(hierarchy[0][index][3]) if hierarchy is not None else -1
        contour_records.append({
            "index": index,
            "parent": parent,
            "is_hole": parent >= 0,
            "area_px2": round(area, 3),
            "perimeter_px": round(perimeter, 3),
            "polygon_px": polygon.astype(int).tolist(),
        })
        total_perimeter += perimeter

    moments = cv2.moments(mask, binaryImage=True)
    cx = moments["m10"] / moments["m00"] if moments["m00"] else x + bw / 2
    cy = moments["m01"] / moments["m00"] if moments["m00"] else y + bh / 2
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 45, 135, L2gradient=True)
    support = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    edges = cv2.bitwise_and(edges, support)

    overlay = image.copy()
    cv2.drawContours(overlay, contours, -1, (0, 220, 0), 1)
    cv2.rectangle(overlay, (x, y), (x + bw - 1, y + bh - 1), (0, 0, 255), 1)
    cv2.drawMarker(overlay, (round(cx), round(cy)), (255, 0, 0), cv2.MARKER_CROSS, 11, 1)

    mask_path = output_dir / "masks" / f"ring01_{view_name}_mask.png"
    edge_path = output_dir / "edges" / f"ring01_{view_name}_edges.png"
    overlay_path = output_dir / "overlays" / f"ring01_{view_name}_overlay.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    edge_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(edge_path), edges)
    cv2.imwrite(str(overlay_path), overlay)

    foreground_area = int(np.count_nonzero(mask))
    return {
        "view": view_name,
        "source": str(path),
        "image_size_px": {"width": w, "height": h},
        "segmentation": {
            "method": "otsu_morphology_connected_components_v1",
            "threshold": otsu_threshold,
            "foreground_area_px2": foreground_area,
            "foreground_fraction": round(foreground_area / (w * h), 6),
            "bbox_px": {"x": x, "y": y, "width": bw, "height": bh},
            "centroid_px": {"x": round(cx, 3), "y": round(cy, 3)},
            "limitations": ["soft cast shadows can remain connected to the object mask"],
        },
        "geometry": {
            "aspect_ratio": round(bw / bh, 6),
            "total_contour_perimeter_px": round(total_perimeter, 3),
            "horizontal_symmetry_iou": _symmetry_iou(mask, 1),
            "vertical_symmetry_iou": _symmetry_iou(mask, 0),
            "edge_pixel_count": int(np.count_nonzero(edges)),
            "contours": contour_records,
        },
        "artifacts": {"mask": str(mask_path), "edges": str(edge_path), "overlay": str(overlay_path)},
    }


def run(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    views = {}
    for name in VIEW_NAMES:
        path = input_dir / f"ring01_{name}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        views[name] = extract_view(path, output_dir, name)
    report = {
        "sample": "ring01",
        "stage": "phase1_classical_cv",
        "view_count": len(views),
        "coordinate_convention": "pixel origin is top-left; x right; y down",
        "measurements_are_metric": False,
        "views": views,
    }
    report_path = output_dir / "phase1_features.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/ring01_reference_images"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/ring01_phase1"))
    args = parser.parse_args()
    report = run(args.input_dir, args.output_dir)
    summary = {
        name: {
            "bbox_px": record["segmentation"]["bbox_px"],
            "horizontal_symmetry_iou": record["geometry"]["horizontal_symmetry_iou"],
        }
        for name, record in report["views"].items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
