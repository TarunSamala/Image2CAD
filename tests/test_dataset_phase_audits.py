"""Validation for Ring01-style dataset phase audit sheets."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import cv2


class DatasetPhaseAuditsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path("dataset/phase_runs/v1/phase_audits")
        cls.report = json.loads((cls.root / "phase_audit_report.json").read_text(encoding="utf-8"))

    def test_every_object_and_view_has_an_audit(self) -> None:
        self.assertEqual(self.report["object_count"], 24)
        self.assertEqual(self.report["object_summary_count"], 24)
        self.assertEqual(self.report["view_audit_count"], 120)
        for object_id, summary in self.report["object_summaries"].items():
            self.assertTrue(Path(summary).is_file(), summary)
            for view in ("front", "top", "iso", "lsv", "rsv"):
                path = self.root / object_id / "views" / f"{object_id}_{view}_audit.png"
                self.assertTrue(path.is_file(), path)

    def test_sheets_are_readable_and_high_resolution(self) -> None:
        for path in self.report["object_summaries"].values():
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            self.assertIsNotNone(image, path)
            self.assertGreaterEqual(image.shape[1], 1400, path)
            self.assertGreaterEqual(image.shape[0], 1000, path)
        sample = cv2.imread(
            str(self.root / "ring_007" / "views" / "ring_007_front_audit.png"),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(sample)
        self.assertGreaterEqual(sample.shape[1], 1000)

    def test_report_keeps_evaluation_and_cad_limits_explicit(self) -> None:
        self.assertGreaterEqual(self.report["metrics_by_split"]["test"]["iou"], 0.90)
        self.assertIn("Only test-split", self.report["evaluation_note"])
        self.assertIn("no paired CAD", self.report["later_phase_status"])
        self.assertFalse(self.report["manufacturing_accuracy_validated"])


if __name__ == "__main__":
    unittest.main()
