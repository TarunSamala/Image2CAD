"""Regression tests for the Phase 3.3.2 prong-visibility correction."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import cv2


class Phase332Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = Path("data/ring01_phase3_3_2")
        cls.report = json.loads((cls.base / "phase3_3_2_validation.json").read_text(encoding="utf-8"))
        cls.previous = json.loads(
            Path("data/ring01_phase3_3_1/phase3_3_1_validation.json").read_text(encoding="utf-8")
        )

    def test_visibility_checkpoint_passes(self) -> None:
        self.assertTrue(self.report["checkpoint_valid"])
        self.assertTrue(all(self.report["checks"].values()))

    def test_four_tips_are_distinct_and_off_center_axis(self) -> None:
        for view in ("isometric", "opposite", "gem_face"):
            visibility = self.report["visibility"][view]
            self.assertEqual(len(visibility["projected_tip_positions"]), 4)
            self.assertTrue(visibility["four_distinct_tip_positions"])
            self.assertTrue(visibility["no_tip_on_center_axis"])
            self.assertGreater(visibility["minimum_pair_separation_cap_diameters"], 1.0)

    def test_geometry_is_preserved_byte_for_byte(self) -> None:
        self.assertFalse(self.report["geometry_changed"])
        source = Path(self.report["source_step"])
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        self.assertEqual(digest, self.report["source_step_sha256"])

    def test_phase331_metrics_are_preserved(self) -> None:
        expected = {
            "mean_silhouette_iou": self.previous["mean_silhouette_iou"],
            "mean_detail_region_iou": self.previous["mean_detail_region_iou"],
            "mean_external_boundary_f1_2px": self.previous["mean_external_boundary_f1_2px"],
        }
        self.assertEqual(self.report["preserved_exact_metrics"], expected)

    def test_corrected_preview_is_readable(self) -> None:
        preview = Path(self.report["preview"])
        image = cv2.imread(str(preview), cv2.IMREAD_COLOR)
        self.assertIsNotNone(image)
        self.assertEqual(tuple(image.shape[:2]), (890, 840))

    def test_manufacturing_accuracy_is_not_claimed(self) -> None:
        self.assertFalse(self.report["manufacturing_accuracy_validated"])


if __name__ == "__main__":
    unittest.main()
