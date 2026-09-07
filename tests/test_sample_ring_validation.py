"""Regression checks for the new real-photo sample validation artifacts."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import cv2
import numpy as np


class SampleRingValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = Path("dataset/Sample_test/ring_02.jpeg")
        cls.root = Path("dataset/Sample_test/phase_validation")
        cls.report = json.loads((cls.root / "sample_ring_validation.json").read_text(encoding="utf-8"))

    def test_source_is_preserved(self) -> None:
        digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.assertEqual(digest, self.report["source_sha256"])
        self.assertTrue(self.report["source_preserved"])

    def test_phase1_and_phase2_checks_pass(self) -> None:
        validation = self.report["validation"]
        self.assertTrue(validation["silhouette_nonempty"])
        self.assertTrue(validation["normalization_complete"])
        self.assertTrue(validation["four_support_proposals_present"])
        self.assertTrue(validation["shadow_disjoint_from_jewelry"])
        self.assertFalse(validation["pixel_accuracy_validated"])
        self.assertFalse(validation["multi_view_consistency_validated"])

    def test_artifacts_are_readable(self) -> None:
        for name, path in self.report["artifacts"].items():
            if name == "support_instances":
                for instance_path in path.values():
                    mask = cv2.imread(instance_path, cv2.IMREAD_GRAYSCALE)
                    self.assertIsNotNone(mask, instance_path)
                    self.assertGreater(np.count_nonzero(mask), 0, instance_path)
                continue
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            self.assertIsNotNone(image, path)
        comparison = cv2.imread(self.report["artifacts"]["comparison"], cv2.IMREAD_COLOR)
        self.assertGreaterEqual(comparison.shape[1], 1400)
        self.assertGreaterEqual(comparison.shape[0], 600)

    def test_component_interpretation_is_not_overclaimed(self) -> None:
        self.assertIn("not confirmed", self.report["interpretation"]["prongs"])
        self.assertIn("manual review", self.report["interpretation"]["shadow"])
        self.assertIn("no human mask", self.report["interpretation"]["silhouette"])


if __name__ == "__main__":
    unittest.main()
