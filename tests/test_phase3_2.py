"""Regression tests for the Phase 3.2 topology rebuild."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import cv2


class Phase32Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = Path("data/ring01_phase3_2")
        cls.build = json.loads((cls.base / "phase3_2_build.json").read_text(encoding="utf-8"))
        cls.report = json.loads((cls.base / "phase3_2_validation.json").read_text(encoding="utf-8"))

    def test_topology_and_render_validation_pass(self) -> None:
        self.assertTrue(self.report["passed"])
        self.assertTrue(self.build["topology"]["open_gallery"])
        self.assertTrue(self.build["topology"]["faceted_stone"])
        self.assertEqual(self.build["topology"]["prong_count"], 4)

    def test_manufacturing_accuracy_is_not_claimed(self) -> None:
        self.assertFalse(self.report["manufacturing_accuracy_validated"])

    def test_outputs_are_readable(self) -> None:
        for name in ("ring01_phase3_2.step", "ring01_phase3_2.stl", "ring01_phase3_2.obj", "ring01_phase3_2.glb"):
            path = self.base / name
            self.assertTrue(path.is_file() and path.stat().st_size > 0, path)
        preview = self.base / "previews" / "ring01_phase3_2_clean_preview.png"
        self.assertIsNotNone(cv2.imread(str(preview), cv2.IMREAD_COLOR), preview)


if __name__ == "__main__":
    unittest.main()
