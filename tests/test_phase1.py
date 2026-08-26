"""Quality gates for the deterministic Ring01 Phase 1 extraction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pipeline.extract_phase1 import VIEW_NAMES, run


class Phase1ExtractionTest(unittest.TestCase):
    def test_all_views_have_plausible_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = run(Path("data/ring01_reference_images"), Path(directory))
            self.assertEqual(set(report["views"]), set(VIEW_NAMES))
            for record in report["views"].values():
                segmentation = record["segmentation"]
                geometry = record["geometry"]
                self.assertGreater(segmentation["foreground_fraction"], 0.02)
                self.assertLess(segmentation["foreground_fraction"], 0.40)
                self.assertGreater(len(geometry["contours"]), 0)
                self.assertGreater(geometry["edge_pixel_count"], 100)

            self.assertGreater(report["views"]["front"]["geometry"]["aspect_ratio"], 1.5)
            self.assertGreater(report["views"]["top"]["geometry"]["aspect_ratio"], 1.5)
            self.assertLess(report["views"]["side"]["geometry"]["aspect_ratio"], 1.0)
            self.assertLess(report["views"]["back"]["geometry"]["aspect_ratio"], 1.0)
            self.assertGreater(report["views"]["side"]["geometry"]["horizontal_symmetry_iou"], 0.85)
            self.assertLess(report["views"]["angled"]["geometry"]["horizontal_symmetry_iou"], 0.70)


if __name__ == "__main__":
    unittest.main()
