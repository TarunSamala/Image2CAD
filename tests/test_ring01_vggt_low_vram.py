"""Regression checks for the Ring01 VGGT low-VRAM research experiment."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import cv2
import numpy as np
import trimesh


class Ring01VGGTLowVramTest(unittest.TestCase):
    root = Path("dataset/phase_runs/v1/ring01_vggt_dense_224_v1")

    def test_run_completed_inside_four_gib_without_accuracy_claims(self) -> None:
        report = json.loads((self.root / "vggt_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "completed_machine_hypothesis")
        self.assertEqual(report["image_count"], 5)
        self.assertEqual(report["image_size"], 224)
        self.assertLess(report["peak_cuda_bytes"], 4 * 1024**3)
        self.assertEqual(
            report["checkpoint_sha256"],
            "d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0",
        )
        self.assertFalse(any(report["claims"].values()))
        self.assertEqual(report["decision"], "research_only")

    def test_camera_and_dense_arrays_are_finite(self) -> None:
        cameras = np.load(self.root / "vggt_cameras.npz")
        dense = np.load(self.root / "vggt_dense_evidence.npz")
        self.assertEqual(cameras["extrinsic"].shape, (1, 5, 3, 4))
        self.assertEqual(cameras["intrinsic"].shape, (1, 5, 3, 3))
        self.assertEqual(dense["depth"].shape, (1, 5, 224, 224, 1))
        self.assertEqual(dense["world_points"].shape, (1, 5, 224, 224, 3))
        self.assertTrue(all(np.isfinite(dense[key]).all() for key in dense.files))

    def test_visual_review_rejects_cloud_as_reconstruction_geometry(self) -> None:
        review = json.loads(
            (self.root / "machine_visual_review.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            review["decision"], "rejected_as_cross_view_reconstruction_geometry"
        )
        self.assertIn("CAD surface generation", review["rejected_uses"])
        self.assertEqual(review["authority"], "ai_assisted_review_not_human_ground_truth")

    def test_review_previews_and_point_cloud_are_readable(self) -> None:
        for filename in (
            "ring01_vggt_dense_review.png",
            "ring01_vggt_fused_points_review.png",
        ):
            image = cv2.imread(str(self.root / filename))
            self.assertIsNotNone(image)
            self.assertGreater(min(image.shape[:2]), 800)
        cloud = trimesh.load(self.root / "ring01_vggt_fused_points.ply", process=False)
        self.assertGreater(len(cloud.vertices), 20_000)


if __name__ == "__main__":
    unittest.main()
