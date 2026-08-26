"""Strict validation and baseline comparison for Phase 2.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


VIEWS = ("front", "side", "top", "angled", "back")


def validate(base: Path) -> dict:
    results = {}
    checks = []
    for view in VIEWS:
        def mask(component: str) -> np.ndarray:
            path = base / "masks" / component / f"ring01_{view}_{component}.png"
            value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if value is None:
                raise FileNotFoundError(path)
            return value

        shank, setting = mask("shank"), mask("setting")
        amodal, prongs = mask("stone_amodal"), mask("prongs")
        shadow, jewelry = mask("shadow"), mask("jewelry")
        overlap = np.count_nonzero(cv2.bitwise_and(shank, setting))
        contours, _ = cv2.findContours(amodal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull_mask = np.zeros_like(amodal)
        for contour in contours:
            cv2.fillConvexPoly(hull_mask, cv2.convexHull(contour), 255)
        solidity = np.count_nonzero(amodal) / max(1, np.count_nonzero(hull_mask))
        prong_regions = cv2.connectedComponents((prongs > 0).astype(np.uint8))[0] - 1
        shadow_overlap = np.count_nonzero(cv2.bitwise_and(shadow, jewelry))
        exact_prongs = prong_regions == 4 if view in ("front", "top") else None
        record = {
            "shank_setting_overlap_px2": int(overlap),
            "shank_setting_disjoint_pass": overlap == 0,
            "amodal_stone_solidity": round(solidity, 5),
            "closed_stone_pass": solidity >= 0.95,
            "prong_connected_regions": int(prong_regions),
            "four_prong_pass": exact_prongs,
            "shadow_jewelry_overlap_px2": int(shadow_overlap),
            "shadow_disjoint_pass": shadow_overlap == 0,
        }
        results[view] = record
        checks.extend([record["shank_setting_disjoint_pass"], record["closed_stone_pass"], record["shadow_disjoint_pass"]])
        if exact_prongs is not None:
            checks.append(exact_prongs)
    report = {
        "stage": "phase2_1_strict_validation",
        "passed": all(checks),
        "views": results,
        "remaining_manual_checks": ["pixel-accurate prong boundaries", "shadow versus reflective underside", "amodal stone shape against human annotation"],
    }
    (base / "phase2_refined_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-dir", type=Path, default=Path("data/ring01_phase2_refined"))
    args = parser.parse_args()
    print(json.dumps(validate(args.phase2_dir), indent=2))


if __name__ == "__main__":
    main()
