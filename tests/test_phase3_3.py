"""Regression tests for the strict Phase 3.3 refinement checkpoint."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import cv2


class Phase33Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = Path("data/ring01_phase3_3")
        cls.fit = json.loads((cls.base / "phase3_3_fit.json").read_text(encoding="utf-8"))
        cls.build = json.loads((cls.base / "phase3_3_build.json").read_text(encoding="utf-8"))
        cls.report = json.loads((cls.base / "phase3_3_validation.json").read_text(encoding="utf-8"))

    def test_resumed_fit_improves_its_checkpoint(self) -> None:
        self.assertIsNotNone(self.fit["resumed_from"])
        self.assertGreater(self.fit["optimized"]["score"], self.fit["initial"]["score"])
        self.assertTrue(self.fit["reviewed_evaluation_masks_used"])

    def test_visual_metrics_clear_numerical_gate(self) -> None:
        self.assertGreaterEqual(self.report["mean_silhouette_iou"], 0.80)
        self.assertGreaterEqual(self.report["mean_detail_region_iou"], 0.75)
        self.assertGreaterEqual(self.report["mean_external_boundary_f1_2px"], 0.75)
        self.assertGreaterEqual(self.report["principal_view_floor"], 0.70)
        self.assertGreaterEqual(self.report["views"]["angled"]["silhouette_iou"], 0.70)

    def test_raw_and_reviewed_mask_scores_are_both_reported(self) -> None:
        self.assertTrue(self.report["physics_corrected_masks_used"])
        self.assertIn("mean_raw_machine_mask_iou", self.report)
        for view in ("front", "side", "top", "angled", "back"):
            self.assertIn("raw_machine_mask_iou", self.report["views"][view])

    def test_exact_solids_and_topology_pass(self) -> None:
        self.assertTrue(self.report["checkpoint_valid"])
        self.assertTrue(all(self.report["topology_checks"].values()))
        self.assertTrue(self.build["validation"]["metal_brep_valid"])
        self.assertTrue(self.build["validation"]["stone_brep_valid"])
        self.assertTrue(self.build["validation"]["mesh_watertight"])
        self.assertEqual(self.build["topology"]["prong_count"], 4)

    def test_phase_does_not_claim_unavailable_accuracy(self) -> None:
        self.assertFalse(self.report["exit_checks"]["human_ground_truth_masks_available"])
        self.assertFalse(self.report["exit_checks"]["metric_scale_calibrated"])
        self.assertFalse(self.report["phase3_exit_gate_passed"])
        self.assertFalse(self.report["manufacturing_accuracy_validated"])

    def test_outputs_and_comparisons_are_readable(self) -> None:
        for name in ("ring01_phase3_3.step", "ring01_phase3_3.stl", "ring01_phase3_3.obj", "ring01_phase3_3.glb"):
            path = self.base / name
            self.assertTrue(path.is_file() and path.stat().st_size > 0, path)
        for view in ("front", "side", "top", "angled", "back"):
            path = self.base / "comparisons" / f"ring01_{view}_reference_vs_phase3_3.png"
            self.assertIsNotNone(cv2.imread(str(path), cv2.IMREAD_COLOR), path)


if __name__ == "__main__":
    unittest.main()
