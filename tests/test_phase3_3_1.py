"""Regression tests for the Phase 3.3.1 cubic-claw checkpoint."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import cv2
import trimesh


class Phase331Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = Path("data/ring01_phase3_3_1")
        cls.previous = json.loads(
            Path("data/ring01_phase3_3/phase3_3_validation.json").read_text(encoding="utf-8")
        )
        cls.fit = json.loads((cls.base / "phase3_3_1_fit.json").read_text(encoding="utf-8"))
        cls.build = json.loads((cls.base / "phase3_3_1_build.json").read_text(encoding="utf-8"))
        cls.report = json.loads((cls.base / "phase3_3_1_validation.json").read_text(encoding="utf-8"))

    def test_four_individually_addressable_cubic_claws(self) -> None:
        topology = self.build["topology"]
        self.assertEqual(topology["prong_count"], 4)
        self.assertEqual(topology["prong_curve_version"], "cubic_claw_v1")
        self.assertEqual(
            topology["individual_prong_ids"],
            ["prong_ne", "prong_nw", "prong_sw", "prong_se"],
        )
        self.assertEqual(len(set(topology["individual_prong_ids"])), 4)

    def test_each_claw_has_a_real_inward_hook(self) -> None:
        for claw in self.build["topology"]["prongs"]:
            crest = claw["control_2"]
            tip = claw["tip"]
            self.assertGreater(crest[2], tip[2])
            self.assertGreater(math.hypot(crest[0], crest[1]), math.hypot(tip[0], tip[1]))

    def test_exact_solids_are_connected_and_valid(self) -> None:
        validation = self.build["validation"]
        self.assertEqual(validation["metal_solid_count"], 1)
        self.assertEqual(validation["stone_solid_count"], 1)
        self.assertEqual(validation["assembly_solid_count"], 2)
        self.assertTrue(validation["metal_brep_valid"])
        self.assertTrue(validation["stone_brep_valid"])
        self.assertTrue(validation["mesh_watertight"])
        self.assertTrue(self.report["checkpoint_valid"])

    def test_component_export_has_no_remaining_zero_area_faces(self) -> None:
        mesh = trimesh.load(self.base / "ring01_phase3_3_1_prongs.stl", force="mesh")
        self.assertGreater(len(mesh.faces), 0)
        self.assertTrue((trimesh.triangles.area(mesh.triangles) > 1e-10).all())

    def test_visual_metrics_do_not_regress_from_phase33(self) -> None:
        self.assertGreaterEqual(self.report["mean_silhouette_iou"], self.previous["mean_silhouette_iou"])
        self.assertGreaterEqual(self.report["mean_detail_region_iou"], self.previous["mean_detail_region_iou"])
        self.assertGreaterEqual(
            self.report["mean_external_boundary_f1_2px"],
            self.previous["mean_external_boundary_f1_2px"],
        )

    def test_unreviewed_instance_masks_cannot_drive_geometry(self) -> None:
        self.assertEqual(self.fit["objective"]["individual_prongs_diagnostic_only"], 0.0)
        self.assertIn("human-reviewed masks", self.fit["selection_policy"])
        self.assertIn("instance_mean_iou", self.fit["optimized"]["metrics"])

    def test_ninety_percent_and_manufacturing_accuracy_are_not_claimed(self) -> None:
        self.assertFalse(self.report["target_90_reached"])
        self.assertFalse(self.report["phase3_exit_gate_passed"])
        self.assertFalse(self.report["manufacturing_accuracy_validated"])
        self.assertFalse(self.report["exit_checks"]["metric_scale_calibrated"])

    def test_outputs_and_all_comparisons_are_readable(self) -> None:
        for suffix in ("step", "stl", "obj", "glb"):
            path = self.base / f"ring01_phase3_3_1.{suffix}"
            self.assertTrue(path.is_file() and path.stat().st_size > 0, path)
        for view in ("front", "side", "top", "angled", "back"):
            path = self.base / "comparisons" / f"ring01_{view}_reference_vs_phase3_3_1.png"
            self.assertIsNotNone(cv2.imread(str(path), cv2.IMREAD_COLOR), path)


if __name__ == "__main__":
    unittest.main()
