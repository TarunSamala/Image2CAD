"""Regression tests for Phase 3.1 fitting and optimized CAD outputs."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


class Phase31Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = Path("data/ring01_phase3_1")
        cls.report = json.loads((cls.base / "phase3_1_validation.json").read_text(encoding="utf-8"))

    def test_refinement_passes(self) -> None:
        self.assertTrue(self.report["passed"])
        self.assertGreater(self.report["objective"]["optimized"], self.report["objective"]["initial"])

    def test_manufacturing_accuracy_is_not_claimed(self) -> None:
        self.assertFalse(self.report["manufacturing_accuracy_validated"])

    def test_optimized_cad_exports_exist(self) -> None:
        for name in ("ring01_phase3_semantic.step", "ring01_phase3_semantic.stl", "ring01_phase3_semantic.obj", "ring01_phase3_semantic.glb"):
            path = self.base / "semantic" / name
            self.assertTrue(path.is_file() and path.stat().st_size > 0, path)

    def test_exported_cad_was_rerendered(self) -> None:
        self.assertTrue(self.report["checks"]["exported_cad_all_view_iou_pass"])
        self.assertTrue(self.report["checks"]["exported_cad_matches_proxy_pass"])


if __name__ == "__main__":
    unittest.main()
