"""Regression gates for the STL-1 Phase 5.2 head-support envelopes."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


class Phase5STL1HeadSupportFitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.output = cls.root / "data" / "phase5_stl1_exact_head_support_fit_v1"
        cls.report = json.loads(
            (cls.output / "report.json").read_text(encoding="utf-8")
        )

    def test_dataset_split_and_claim_boundary(self) -> None:
        summary = self.report["summary"]
        self.assertEqual(summary["object_count"], 24)
        self.assertEqual(summary["train_scored_count"], 18)
        self.assertEqual(summary["validation_scored_count"], 3)
        self.assertEqual(summary["test_held_out_count"], 3)
        self.assertTrue(self.report["checks"]["held_out_test_not_scored"])
        self.assertFalse(
            self.report["claims"]["manufacturing_accuracy_validated"]
        )

    def test_validation_projection_gates(self) -> None:
        summary = self.report["summary"]
        self.assertGreaterEqual(summary["validation_mean_front_profile_iou"], 0.95)
        self.assertGreaterEqual(summary["validation_mean_top_envelope_iou"], 0.95)

    def test_all_exact_geometry_and_exports_pass(self) -> None:
        checks = self.report["checks"]
        self.assertTrue(checks["all_24_support_candidates_generated"])
        self.assertTrue(checks["all_head_supports_valid"])
        self.assertTrue(checks["all_48_shoulders_valid"])
        self.assertTrue(checks["all_metals_valid_single_solids"])
        self.assertTrue(checks["all_step_exports_present"])
        for item in self.report["objects"]:
            self.assertEqual(item["profile_control_count"], {"front": 96, "top": 96})
            self.assertTrue((self.root / item["artifacts"]["step"]).is_file())
            if item["split"] == "val":
                self.assertTrue(
                    (self.root / item["artifacts"]["geometry_preview"]).is_file()
                )
            if item["split"] == "test":
                self.assertIsNone(item["metrics"])

    def test_audit_is_present(self) -> None:
        self.assertTrue((self.root / self.report["artifacts"]["audit"]).is_file())


if __name__ == "__main__":
    unittest.main()
