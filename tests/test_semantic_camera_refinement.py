"""Regression gates for semantic camera diagnostics and rejected snapping."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import cv2
import numpy as np


class SemanticCameraRefinementTest(unittest.TestCase):
    prior_root = Path("dataset/phase_runs/v1/semantic_camera_priors_v1")
    refinement_root = Path(
        "dataset/phase_runs/v1/vggt_semantic_camera_refinement_v1"
    )

    @classmethod
    def setUpClass(cls) -> None:
        cls.priors = json.loads(
            (cls.prior_root / "report.json").read_text(encoding="utf-8")
        )
        cls.refinement = json.loads(
            (cls.refinement_root / "report.json").read_text(encoding="utf-8")
        )

    def test_named_view_diagnostic_is_useful_but_not_calibrated(self) -> None:
        cohort = self.priors["cohort"]
        self.assertEqual(self.priors["object_count"], 3)
        self.assertLess(cohort["mean_vggt_pairwise_angle_error_deg"], 15.0)
        self.assertGreater(cohort["mean_semantic_holdout_iou"], 0.60)
        self.assertLess(cohort["minimum_semantic_holdout_iou"], 0.50)
        self.assertEqual(cohort["accepted_initialization_count"], 0)
        self.assertFalse(any(self.priors["claims"].values()))

    def test_naive_semantic_snap_is_rejected_for_every_ring(self) -> None:
        cohort = self.refinement["cohort"]
        self.assertLess(cohort["mean_iou_delta"], 0.0)
        self.assertLess(cohort["mean_precision_delta"], 0.0)
        self.assertEqual(cohort["accepted_initialization_count"], 0)
        self.assertEqual(cohort["geometry_exit_gate_count"], 0)
        self.assertFalse(any(self.refinement["claims"].values()))
        for item in self.refinement["objects"]:
            self.assertEqual(item["decision"], "semantic_camera_snap_rejected")

    def test_corrected_rotations_and_review_images_are_valid(self) -> None:
        for item in self.refinement["objects"]:
            object_id = item["object_id"]
            camera_path = (
                self.refinement_root
                / object_id
                / "semantic_corrected_cameras.npz"
            )
            with np.load(camera_path) as cameras:
                rotation = cameras["extrinsic"][0, :, :, :3]
                self.assertTrue(np.isfinite(rotation).all())
                np.testing.assert_allclose(
                    np.linalg.det(rotation), np.ones(5), atol=1e-4
                )
            preview = cv2.imread(
                str(
                    self.refinement_root
                    / object_id
                    / f"{object_id}_semantic_camera_reprojection.png"
                )
            )
            self.assertIsNotNone(preview)
            self.assertGreater(preview.shape[0], 1000)

    def test_semantic_holdout_previews_are_readable(self) -> None:
        for item in self.priors["objects"]:
            object_id = item["object_id"]
            preview = cv2.imread(
                str(
                    self.prior_root
                    / object_id
                    / f"{object_id}_semantic_camera_holdout.png"
                )
            )
            self.assertIsNotNone(preview)
            self.assertGreater(preview.shape[1], 350)


if __name__ == "__main__":
    unittest.main()
