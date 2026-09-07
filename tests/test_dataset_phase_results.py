"""Regression gates for the versioned prepared_v1 training experiment."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


class DatasetPhaseResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path("dataset/phase_runs/v1")
        cls.summary = json.loads((cls.root / "summary.json").read_text(encoding="utf-8"))
        cls.test_report = json.loads((cls.root / "phase2_2_test.json").read_text(encoding="utf-8"))
        cls.coverage = json.loads((cls.root / "phase_coverage.json").read_text(encoding="utf-8"))

    def test_all_phase1_dataset_inputs_pass(self) -> None:
        report = json.loads((self.root / "phase1_features.json").read_text(encoding="utf-8"))
        self.assertEqual(report["object_count"], 24)
        self.assertEqual(report["view_count"], 120)
        self.assertTrue(all(report["checks"].values()))

    def test_held_out_silhouette_gate_passes(self) -> None:
        metrics = self.test_report["metrics"]
        self.assertEqual(metrics["sample_count"], 15)
        self.assertGreaterEqual(metrics["mean"]["iou"], 0.90)
        self.assertGreaterEqual(metrics["mean"]["dice"], 0.94)
        self.assertGreaterEqual(metrics["mean"]["boundary_f1_2px"], 0.90)

    def test_predictions_and_checkpoint_exist(self) -> None:
        self.assertTrue((self.root / "checkpoints" / "whole_jewellery_unet.pt").is_file())
        for object_id in ("ring_007", "ring_015", "ring_023"):
            self.assertTrue((self.root / "test_predictions" / f"{object_id}_review.png").is_file())
            for view in ("front", "top", "iso", "lsv", "rsv"):
                self.assertTrue((self.root / "test_predictions" / f"{object_id}_{view}.png").is_file())

    def test_later_phase_limitations_remain_explicit(self) -> None:
        self.assertEqual(self.coverage["highest_executed_dataset_phase"], "phase2_2")
        self.assertFalse(self.coverage["dataset_reaches_latest_phase_gate"])
        self.assertFalse(self.coverage["manufacturing_accuracy_validated"])
        self.assertIn("phase3", self.coverage["not_trainable_from_this_dataset"])
        self.assertIn("phase3_3_2", self.coverage["not_trainable_from_this_dataset"])


if __name__ == "__main__":
    unittest.main()
