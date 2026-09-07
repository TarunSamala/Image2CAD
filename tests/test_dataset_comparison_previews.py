"""Validation for generated dataset comparison sheets."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import cv2


class DatasetComparisonPreviewsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path("dataset/phase_runs/v1/comparisons")
        cls.report = json.loads((cls.root / "preview_report.json").read_text(encoding="utf-8"))

    def test_report_covers_complete_dataset(self) -> None:
        self.assertEqual(self.report["object_count"], 24)
        self.assertEqual(self.report["view_count"], 120)
        self.assertEqual(self.report["evaluation_target"], "prepared_v1 pseudo-silhouettes")

    def test_all_preview_images_are_high_resolution(self) -> None:
        self.assertEqual(len(self.report["previews"]), 3)
        for path in self.report["previews"].values():
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            self.assertIsNotNone(image, path)
            self.assertGreaterEqual(image.shape[1], 1500, path)
            self.assertGreaterEqual(image.shape[0], 1000, path)

    def test_prediction_legend_distinguishes_error_types(self) -> None:
        colors = self.report["prediction_color_key_bgr"]
        self.assertEqual(set(colors), {"target_and_prediction", "prediction_only", "target_only"})
        self.assertEqual(len({tuple(value) for value in colors.values()}), 3)


if __name__ == "__main__":
    unittest.main()
