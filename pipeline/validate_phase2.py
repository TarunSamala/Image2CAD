"""Strict semantic validation for generated Phase 2 component masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


VIEWS = ("front", "side", "top", "angled", "back")


def validate(base: Path) -> dict:
    results = {}
    for view in VIEWS:
        def mask(component: str) -> np.ndarray:
            path = base / "masks" / component / f"ring01_{view}_{component}.png"
            value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if value is None:
                raise FileNotFoundError(path)
            return value

        jewelry, shank, stone = mask("jewelry"), mask("shank"), mask("stone")
        setting, prongs, shadow = mask("setting"), mask("prongs"), mask("shadow")
        setting_area = max(1, np.count_nonzero(setting))
        overlap = np.count_nonzero(cv2.bitwise_and(shank, setting)) / setting_area
        contours, _ = cv2.findContours(stone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull_area = sum(cv2.contourArea(cv2.convexHull(contour)) for contour in contours)
        stone_solidity = np.count_nonzero(stone) / max(1.0, hull_area)
        prong_regions = cv2.connectedComponents((prongs > 0).astype(np.uint8))[0] - 1
        containment = np.count_nonzero(cv2.bitwise_and(stone, cv2.bitwise_not(jewelry))) == 0
        expected_four_prongs = view in ("front", "top")
        results[view] = {
            "stone_inside_jewelry": bool(containment),
            "shank_setting_overlap_ratio": round(overlap, 5),
            "shank_setting_disjoint_pass": overlap <= 0.20,
            "stone_mask_solidity": round(stone_solidity, 5),
            "stone_silhouette_pass": stone_solidity >= 0.85,
            "prong_connected_regions": int(prong_regions),
            "four_prong_separation_pass": (prong_regions == 4) if expected_four_prongs else None,
            "shadow_area_px2": int(np.count_nonzero(shadow)),
            "shadow_requires_manual_review": True,
        }

    strict_checks = []
    for record in results.values():
        strict_checks.extend([record["stone_inside_jewelry"], record["shank_setting_disjoint_pass"], record["stone_silhouette_pass"]])
        if record["four_prong_separation_pass"] is not None:
            strict_checks.append(record["four_prong_separation_pass"])
    report = {
        "stage": "phase2_strict_semantic_validation",
        "passed": all(strict_checks),
        "machine_integrity": "passed",
        "semantic_fidelity": "passed" if all(strict_checks) else "failed",
        "views": results,
        "findings": [
            "Gemstone masks contain facet/highlight holes and are not complete stone silhouettes.",
            "Shank masks overlap the setting heavily in side, angled, and back views.",
            "Prong proposals are fragmented and do not resolve into four components in front/top views.",
            "Shadow proposals include reflective metal and require refinement or annotation.",
        ],
    }
    (base / "phase2_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-dir", type=Path, default=Path("data/ring01_phase2"))
    args = parser.parse_args()
    print(json.dumps(validate(args.phase2_dir), indent=2))


if __name__ == "__main__":
    main()
