"""Regression tests for Phase 3 coarse multi-view reconstruction."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import cv2


class Phase3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = Path("data/ring01_phase3")
        cls.report = json.loads((cls.base / "phase3_validation.json").read_text(encoding="utf-8"))

    def test_coarse_reconstruction_passes(self) -> None:
        self.assertTrue(self.report["coarse_reconstruction_passed"])

    def test_manufacturing_accuracy_is_not_claimed(self) -> None:
        self.assertFalse(self.report["manufacturing_accuracy_validated"])

    def test_mesh_exports_exist(self) -> None:
        for extension in ("stl", "obj", "glb"):
            path = self.base / "meshes" / f"ring01_phase3_visual_hull.{extension}"
            self.assertTrue(path.is_file() and path.stat().st_size > 0, path)
        preview = self.base / "previews" / "ring01_phase3_mesh_preview.png"
        self.assertIsNotNone(cv2.imread(str(preview), cv2.IMREAD_COLOR), preview)

    def test_all_reprojection_artifacts_exist(self) -> None:
        for view in ("front", "side", "top", "angled", "back"):
            for folder, suffix in (("reprojections", "silhouette"), ("comparisons", "compare")):
                path = self.base / folder / f"ring01_{view}_phase3_{suffix}.png"
                self.assertIsNotNone(cv2.imread(str(path), cv2.IMREAD_COLOR), path)

    def test_semantic_proxy_is_coherent_and_editable(self) -> None:
        semantic = json.loads((self.base / "semantic" / "phase3_semantic_proxy.json").read_text(encoding="utf-8"))
        self.assertTrue(semantic["validation"]["metal_is_single_solid"])
        self.assertTrue(semantic["validation"]["stone_is_single_solid"])
        self.assertEqual(semantic["topology"]["prong_count"], 4)
        for name in ("ring01_phase3_semantic.step", "ring01_phase3_semantic.stl", "ring01_phase3_semantic.obj", "ring01_phase3_semantic.glb"):
            path = self.base / "semantic" / name
            self.assertTrue(path.is_file() and path.stat().st_size > 0, path)


if __name__ == "__main__":
    unittest.main()
