"""Regression tests for Phase 2.1 topology refinement outputs."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import cv2


class Phase2RefinedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = Path("data/ring01_phase2_refined")
        cls.report = json.loads((cls.base / "phase2_refined_validation.json").read_text(encoding="utf-8"))

    def test_strict_machine_checks_pass(self) -> None:
        self.assertTrue(self.report["passed"])
        for view, record in self.report["views"].items():
            self.assertTrue(record["shank_setting_disjoint_pass"], view)
            self.assertTrue(record["closed_stone_pass"], view)
            self.assertTrue(record["shadow_disjoint_pass"], view)

    def test_front_and_top_have_four_prong_regions(self) -> None:
        for view in ("front", "top"):
            self.assertEqual(self.report["views"][view]["prong_connected_regions"], 4)
            self.assertTrue(self.report["views"][view]["four_prong_pass"])

    def test_all_refined_masks_are_readable(self) -> None:
        components = ("jewelry", "metal", "shank", "stone_visible", "stone_amodal", "setting", "prongs", "shadow")
        for view in self.report["views"]:
            for component in components:
                path = self.base / "masks" / component / f"ring01_{view}_{component}.png"
                self.assertIsNotNone(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), path)


if __name__ == "__main__":
    unittest.main()
