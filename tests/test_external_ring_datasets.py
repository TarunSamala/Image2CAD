"""Tests for provenance-safe external ring dataset intake."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from intake_external_ring_datasets import (
    discover_ornasia,
    discover_segment_your_ring,
    run_intake,
)


def _ring_image(path: Path, radius: int, color: tuple[int, int, int]) -> None:
    image = np.full((160, 160, 3), 245, np.uint8)
    cv2.circle(image, (80, 82), radius, color, 13, cv2.LINE_AA)
    cv2.circle(image, (80, 35), 14, color, -1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(path)


def _ring_mask(path: Path, radius: int) -> None:
    mask = np.zeros((160, 160), np.uint8)
    cv2.circle(mask, (80, 82), radius, 255, 13, cv2.LINE_AA)
    cv2.circle(mask, (80, 35), 14, 255, -1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask):
        raise RuntimeError(path)


class ExternalRingDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.ornasia = self.root / "ornasia"
        self.syr = self.root / "syr" / "dataset"
        _ring_image(self.ornasia / "Raw Dataset" / "Ring" / "ring_a.jpg", 49, (50, 70, 90))
        _ring_image(self.ornasia / "Raw Dataset" / "Ring" / "ring_b.jpg", 53, (70, 90, 110))
        # This augmented copy must not enter the intake.
        _ring_image(self.ornasia / "Computer Vision Experiment" / "Ring" / "ring_aug.jpg", 45, (90, 110, 130))
        for split, name, radius in (("train", "ring1", 47), ("test", "ring_test_1", 51)):
            _ring_image(self.syr / split / "images" / f"{name}.jpg", radius, (45, 65, 85))
            _ring_mask(self.syr / split / "masks" / f"{name}.jpg", radius)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_discovery_preserves_source_boundaries_and_splits(self) -> None:
        ornasia = discover_ornasia(self.ornasia)
        syr = discover_segment_your_ring(self.root / "syr")
        self.assertEqual(len(ornasia), 2)
        self.assertTrue(all(item["split"] == "evaluation_unassigned" for item in ornasia))
        self.assertEqual([item["split"] for item in syr], ["train", "test"])
        self.assertTrue(all(item["ground_truth_mask"] is not None for item in syr))

    def test_intake_builds_phase_evidence_and_truth_metrics(self) -> None:
        output = self.root / "output"
        report = run_intake(self.ornasia, self.root / "syr", output, image_size=192)
        records = [
            json.loads(line)
            for line in (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(report["sample_count"], 4)
        self.assertEqual(report["dataset_counts"], {"ornasia": 2, "segment_your_ring": 2})
        self.assertEqual(report["ground_truth_mask_count"], 2)
        self.assertEqual(report["status"], "review_required")
        self.assertFalse(report["checks"]["published_counts_match"])
        self.assertFalse(report["checks"]["all_segmentation_outputs_approved"])
        self.assertTrue(report["checks"]["ornasia_never_enabled_for_training"])
        self.assertTrue(report["checks"]["syr_all_pairs_have_ground_truth_masks"])
        self.assertIsNotNone(report["metrics"]["syr_phase2_refined_vs_human_mask"])
        self.assertTrue((output / "previews" / "ornasia_phase1_phase2_preview.jpg").is_file())
        self.assertTrue((output / "previews" / "segment_your_ring_phase1_phase2_preview.jpg").is_file())
        for record in records:
            self.assertFalse(record["metric_scale_available"])
            self.assertFalse(record["cad_target_available"])
            for artifact in record["artifacts"].values():
                self.assertTrue(Path(artifact).is_file(), artifact)
            self.assertIn(record["mask_quality"]["status"], {
                "approved",
                "review_required",
                "review_required_no_ground_truth",
            })
        ornasia_records = [record for record in records if record["dataset_id"] == "ornasia"]
        self.assertTrue(all(not record["mask_quality"]["approved"] for record in ornasia_records))
        self.assertTrue(all(
            record["mask_quality"]["basis"] == "shape_plausibility_only"
            for record in ornasia_records
        ))
        test_record = next(record for record in records if record["split"] == "test")
        self.assertFalse(test_record["training_allowed"])


if __name__ == "__main__":
    unittest.main()
