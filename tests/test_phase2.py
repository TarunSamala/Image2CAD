"""Quality gates for generated Ring01 Phase 2 semantic proposals."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import cv2
import numpy as np


class Phase2ExtractionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = Path("data/ring01_phase2")
        cls.report = json.loads((cls.base / "phase2_components.json").read_text(encoding="utf-8"))

    def test_all_views_and_component_masks_exist(self) -> None:
        self.assertEqual(set(self.report["views"]), {"front", "side", "top", "angled", "back"})
        for view in self.report["views"]:
            for component in ("jewelry", "metal", "shank", "stone", "setting", "prongs", "shadow"):
                path = self.base / "masks" / component / f"ring01_{view}_{component}.png"
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                self.assertIsNotNone(mask, path)
                self.assertGreater(np.count_nonzero(mask), 0, path)

    def test_semantic_hierarchy_and_coverage(self) -> None:
        for view, record in self.report["views"].items():
            validation = record["validation"]
            components = record["components"]
            self.assertTrue(validation["stone_inside_jewelry"], view)
            self.assertTrue(validation["prongs_inside_setting"], view)
            self.assertGreater(validation["phase1_coverage"], 0.70, view)
            self.assertGreater(components["stone"]["area_px2"], 100, view)
            self.assertLess(components["stone"]["area_px2"], components["jewelry"]["area_px2"] * 0.50, view)
            self.assertGreater(components["shank"]["area_px2"], components["jewelry"]["area_px2"] * 0.20, view)
            self.assertLessEqual(components["prongs"]["area_px2"], components["setting"]["area_px2"], view)


if __name__ == "__main__":
    unittest.main()
