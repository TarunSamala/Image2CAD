"""Reusable multi-region image-to-geometry objective for Phase 3.

The objective deliberately avoids jewellery-category assumptions.  A caller
provides semantic component masks and optional visibility confidence; this
module supplies image-space metrics, regions of interest and a weighted score.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_COMPONENTS = ("jewelry", "shank", "stone_amodal", "setting", "prongs")


@dataclass(frozen=True)
class ObjectiveWeights:
    silhouette: float = 0.28
    detail_region: float = 0.26
    components: float = 0.22
    boundary: float = 0.16
    negative_space: float = 0.08


def read_component_masks(base: Path, sample: str, views: tuple[str, ...], components=DEFAULT_COMPONENTS):
    result: dict[str, dict[str, np.ndarray]] = {}
    for view in views:
        result[view] = {}
        for component in components:
            path = base / "masks" / component / f"{sample}_{view}_{component}.png"
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(path)
            result[view][component] = np.where(mask > 127, 255, 0).astype(np.uint8)
    return result


def bbox(mask: np.ndarray, padding: int = 0) -> tuple[int, int, int, int]:
    points = cv2.findNonZero(mask)
    if points is None:
        raise ValueError("cannot derive a region from an empty mask")
    x, y, width, height = cv2.boundingRect(points)
    return (
        max(0, x - padding),
        max(0, y - padding),
        min(mask.shape[1] - 1, x + width - 1 + padding),
        min(mask.shape[0] - 1, y + height - 1 + padding),
    )


def expand_region(region: tuple[int, int, int, int], shape: tuple[int, int], fraction: float = 0.12):
    x0, y0, x1, y1 = region
    pad = max(3, round(max(x1 - x0 + 1, y1 - y0 + 1) * fraction))
    return max(0, x0 - pad), max(0, y0 - pad), min(shape[1] - 1, x1 + pad), min(shape[0] - 1, y1 + pad)


def crop(mask: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = region
    return mask[y0 : y1 + 1, x0 : x1 + 1]


def iou(first: np.ndarray, second: np.ndarray) -> float:
    first_on, second_on = first > 0, second > 0
    return float(np.count_nonzero(first_on & second_on) / max(1, np.count_nonzero(first_on | second_on)))


def boundary_f1(first: np.ndarray, second: np.ndarray, tolerance: int = 2, external_only: bool = False) -> float:
    mode = cv2.RETR_EXTERNAL if external_only else cv2.RETR_LIST
    first_edge, second_edge = np.zeros_like(first), np.zeros_like(second)
    first_contours, _ = cv2.findContours(first, mode, cv2.CHAIN_APPROX_NONE)
    second_contours, _ = cv2.findContours(second, mode, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(first_edge, first_contours, -1, 255, 1)
    cv2.drawContours(second_edge, second_contours, -1, 255, 1)
    kernel = np.ones((3, 3), np.uint8)
    first_near = cv2.dilate(first_edge, kernel, iterations=tolerance)
    second_near = cv2.dilate(second_edge, kernel, iterations=tolerance)
    precision = np.count_nonzero(cv2.bitwise_and(second_edge, first_near)) / max(1, np.count_nonzero(second_edge))
    recall = np.count_nonzero(cv2.bitwise_and(first_edge, second_near)) / max(1, np.count_nonzero(first_edge))
    return float(2 * precision * recall / max(1e-9, precision + recall))


def enclosed_space(mask: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    """Return holes enclosed by foreground inside a semantic detail region."""
    local = crop(mask, region)
    inverse = cv2.bitwise_not(local)
    flood = inverse.copy()
    flood_mask = np.zeros((local.shape[0] + 2, local.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 0)
    holes = np.where(flood > 0, 255, 0).astype(np.uint8)
    return cv2.morphologyEx(holes, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def detail_regions(targets: dict[str, dict[str, np.ndarray]]) -> dict[str, tuple[int, int, int, int]]:
    regions = {}
    for view, masks in targets.items():
        union = cv2.bitwise_or(masks["setting"], masks["stone_amodal"])
        union = cv2.bitwise_or(union, masks["prongs"])
        regions[view] = expand_region(bbox(union), masks["jewelry"].shape, 0.13)
    return regions


def evaluate_regions(
    targets: dict[str, dict[str, np.ndarray]],
    rendered: dict[str, dict[str, np.ndarray]],
    regions: dict[str, tuple[int, int, int, int]],
    visibility: dict[str, dict[str, float]],
    view_weights: dict[str, float],
    weights: ObjectiveWeights = ObjectiveWeights(),
) -> tuple[float, dict[str, dict[str, float]]]:
    total, total_weight = 0.0, 0.0
    metrics: dict[str, dict[str, float]] = {}
    for view, target in targets.items():
        output = rendered[view]
        region = regions[view]
        silhouette = iou(target["jewelry"], output["jewelry"])
        detail = iou(crop(target["jewelry"], region), crop(output["jewelry"], region))
        boundary = boundary_f1(crop(target["jewelry"], region), crop(output["jewelry"], region), tolerance=2)
        component_scores, component_weight = 0.0, 0.0
        for component in ("shank", "stone_amodal", "setting", "prongs"):
            confidence = visibility[view].get(component, 1.0)
            component_scores += iou(target[component], output[component]) * confidence
            component_weight += confidence
        components = component_scores / max(1e-9, component_weight)
        target_holes, output_holes = enclosed_space(target["jewelry"], region), enclosed_space(output["jewelry"], region)
        if np.any(target_holes) or np.any(output_holes):
            negative = iou(target_holes, output_holes)
            negative_confidence = visibility[view].get("negative_space", 1.0)
        else:
            negative, negative_confidence = 0.0, 0.0
        numerator = (
            weights.silhouette * silhouette
            + weights.detail_region * detail
            + weights.components * components
            + weights.boundary * boundary
            + weights.negative_space * negative * negative_confidence
        )
        denominator = 1.0 - weights.negative_space * (1.0 - negative_confidence)
        score = numerator / denominator
        weight = view_weights.get(view, 1.0)
        total += score * weight
        total_weight += weight
        metrics[view] = {
            "objective": round(score, 6),
            "silhouette_iou": round(silhouette, 6),
            "detail_region_iou": round(detail, 6),
            "component_mean_iou": round(components, 6),
            "detail_boundary_f1": round(boundary, 6),
            "negative_space_iou": round(negative, 6),
        }
    return total / max(1e-9, total_weight), metrics
