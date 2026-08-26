"""Structural, boundary, and cross-view validation for Phase 2.2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


VIEWS = ("front", "side", "top", "angled", "back")


def _mask(base: Path, view: str, component: str) -> np.ndarray:
    path = base / "masks" / component / f"ring01_{view}_{component}.png"
    result = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if result is None:
        raise FileNotFoundError(path)
    return result


def validate(base: Path) -> dict:
    source = json.loads((base / "phase2_2_components.json").read_text(encoding="utf-8"))
    checks: list[bool] = []
    views = {}
    normalized_stones = {}
    for view in VIEWS:
        masks = {name: _mask(base, view, name) for name in ("jewelry", "metal", "shank", "stone_visible", "stone_amodal", "setting", "prongs", "shadow")}
        shank_setting = int(np.count_nonzero(cv2.bitwise_and(masks["shank"], masks["setting"])))
        shadow_jewelry = int(np.count_nonzero(cv2.bitwise_and(masks["shadow"], masks["jewelry"])))
        stone_outside = int(np.count_nonzero(cv2.bitwise_and(masks["stone_visible"], cv2.bitwise_not(masks["stone_amodal"]))))
        metal_outside = int(np.count_nonzero(cv2.bitwise_and(masks["metal"], cv2.bitwise_not(masks["jewelry"]))))
        prong_regions = cv2.connectedComponents((masks["prongs"] > 0).astype(np.uint8))[0] - 1
        contours, _ = cv2.findContours(masks["stone_amodal"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        closed_stone = len(contours) == 1 and np.count_nonzero(masks["stone_amodal"]) > 0
        points = cv2.findNonZero(masks["stone_amodal"])
        if points is not None:
            _, _, width, height = cv2.boundingRect(points)
            normalized_stones[view] = {"width_fraction": width / masks["stone_amodal"].shape[1], "height_fraction": height / masks["stone_amodal"].shape[0]}
        exact_prongs = prong_regions == 4 if view in ("front", "top") else None
        record = {
            "shank_setting_overlap_px2": shank_setting,
            "shank_setting_disjoint_pass": shank_setting == 0,
            "shadow_jewelry_overlap_px2": shadow_jewelry,
            "shadow_disjoint_pass": shadow_jewelry == 0,
            "stone_visible_outside_amodal_px2": stone_outside,
            "stone_containment_pass": stone_outside == 0,
            "metal_outside_jewelry_px2": metal_outside,
            "metal_containment_pass": metal_outside == 0,
            "closed_single_stone_pass": closed_stone,
            "prong_connected_regions": int(prong_regions),
            "four_prong_pass": exact_prongs,
            "boundary_edge_support": {name: source["views"][view]["components"][name]["boundary_edge_support"] for name in ("jewelry", "stone_amodal", "prongs")},
        }
        views[view] = record
        checks.extend([record["shank_setting_disjoint_pass"], record["shadow_disjoint_pass"], record["stone_containment_pass"], record["metal_containment_pass"], record["closed_single_stone_pass"]])
        if exact_prongs is not None:
            checks.append(exact_prongs)

    front_top_width_delta = abs(normalized_stones["front"]["width_fraction"] - normalized_stones["top"]["width_fraction"])
    front_top_height_delta = abs(normalized_stones["front"]["height_fraction"] - normalized_stones["top"]["height_fraction"])
    cross_view = {
        "front_top_stone_width_fraction_delta": round(front_top_width_delta, 5),
        "front_top_stone_height_fraction_delta": round(front_top_height_delta, 5),
        "front_top_scale_consistency_pass": front_top_width_delta <= 0.08 and front_top_height_delta <= 0.08,
        "note": "This is a consistency check, not metric scale validation.",
    }
    checks.append(cross_view["front_top_scale_consistency_pass"])
    report = {
        "stage": "phase2_2_validation",
        "structural_and_consistency_passed": all(checks),
        "pixel_accuracy_validated": False,
        "ground_truth_metrics_available": False,
        "views": views,
        "cross_view": cross_view,
        "blocking_manual_gate": "Human-reviewed masks are required before IoU, Dice, or boundary F1 can validate pixel accuracy.",
    }
    (base / "phase2_2_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-2-dir", type=Path, default=Path("data/ring01_phase2_2"))
    args = parser.parse_args()
    print(json.dumps(validate(args.phase2_2_dir), indent=2))


if __name__ == "__main__":
    main()
