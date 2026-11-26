"""Regression gates for the 24 paired STL-1 pseudo-target meshes."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "dataset/phase_runs/v1/stl1_paired_mesh_audit_v1/report.json"
PAIRING = ROOT / "dataset/phase_runs/v1/stl1_mesh_image_pairing_v1/report.json"
BENCHMARK = ROOT / "dataset/phase_runs/v1/stl1_paired_mesh_benchmark_v1/report.json"


class STL1PairedMeshEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        cls.pairing = json.loads(PAIRING.read_text(encoding="utf-8"))
        cls.benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))

    def test_every_object_has_one_unique_mesh_and_five_images(self) -> None:
        self.assertEqual(self.audit["object_count"], 24)
        self.assertEqual(self.audit["view_count"], 120)
        self.assertEqual(self.audit["splits"], {"train": 18, "val": 3, "test": 3})
        self.assertTrue(all(self.audit["checks"].values()))

    def test_paired_meshes_remain_pseudo_ground_truth(self) -> None:
        quality = self.audit["mesh_quality"]
        self.assertEqual(quality["watertight_count"], 0)
        self.assertEqual(quality["volume_count"], 0)
        self.assertEqual(quality["single_component_count"], 0)
        self.assertIn(
            "manufacturing-accurate CAD ground truth",
            self.audit["claim_policy"]["forbidden"],
        )

    def test_image_pairing_is_supported_without_claiming_3d_truth(self) -> None:
        self.assertEqual(self.pairing["object_count"], 24)
        self.assertEqual(self.pairing["pairing_status_counts"]["front_supported"], 24)
        self.assertFalse(self.pairing["claim_policy"]["validates_3d_accuracy"])
        self.assertFalse(self.pairing["claim_policy"]["validates_manufacturing_accuracy"])

    def test_test_split_is_evaluation_only(self) -> None:
        self.assertEqual(self.benchmark["object_count"], 24)
        self.assertEqual(self.benchmark["splits"]["train"]["object_count"], 18)
        self.assertEqual(self.benchmark["splits"]["val"]["object_count"], 3)
        self.assertEqual(self.benchmark["splits"]["test"]["object_count"], 3)
        self.assertFalse(self.benchmark["test_policy"]["used_for_parameter_selection"])
        self.assertTrue(self.benchmark["test_policy"]["result_is_evaluation_only"])
        self.assertFalse(self.benchmark["claims"]["uses_exact_CAD_ground_truth"])
        self.assertFalse(self.benchmark["claims"]["metric_scale_validated"])
        self.assertFalse(self.benchmark["claims"]["manufacturing_accuracy_validated"])


if __name__ == "__main__":
    unittest.main()
