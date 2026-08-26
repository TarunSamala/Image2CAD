"""Regression tests for Phase 2.2 boundary refinement outputs."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import cv2


class Phase22Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = Path("data/ring01_phase2_2")
        cls.report = json.loads((cls.base / "phase2_2_validation.json").read_text(encoding="utf-8"))

    def test_structural_and_consistency_checks_pass(self) -> None:
        self.assertTrue(self.report["structural_and_consistency_passed"])

    def test_pixel_accuracy_is_not_claimed_without_gold_masks(self) -> None:
        self.assertFalse(self.report["pixel_accuracy_validated"])
        self.assertFalse(self.report["ground_truth_metrics_available"])

    def test_front_and_top_retain_four_prongs(self) -> None:
        for view in ("front", "top"):
            self.assertEqual(self.report["views"][view]["prong_connected_regions"], 4)

    def test_masks_and_review_images_are_readable(self) -> None:
        components = ("jewelry", "metal", "shank", "stone_visible", "stone_amodal", "setting", "prongs", "shadow")
        for view in self.report["views"]:
            for component in components:
                path = self.base / "masks" / component / f"ring01_{view}_{component}.png"
                self.assertIsNotNone(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), path)
            review = self.base / "review" / f"ring01_{view}_annotation_review.png"
            self.assertIsNotNone(cv2.imread(str(review), cv2.IMREAD_COLOR), review)


if __name__ == "__main__":
    unittest.main()
