"""Tests for the reusable one-image and five-view audit program."""

from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np
import trimesh

from audit_jewellery import FIVE_VIEWS, _expects_central_void, _small_object_prompt, run_audit


class UploadedJewelleryAuditTest(unittest.TestCase):
    def test_small_object_prompt_rejects_long_background_fold(self) -> None:
        image = np.full((420, 620, 3), 215, np.uint8)
        cv2.line(image, (0, 360), (619, 300), (165, 165, 165), 22, cv2.LINE_AA)
        cv2.circle(image, (320, 190), 62, (45, 65, 85), 16, cv2.LINE_AA)
        cv2.circle(image, (320, 120), 22, (55, 75, 95), -1, cv2.LINE_AA)
        proposal = _small_object_prompt(image)
        self.assertIsNotNone(proposal)
        points = cv2.findNonZero(proposal)
        self.assertIsNotNone(points)
        x, y, width, height = cv2.boundingRect(points)
        self.assertLess(width, image.shape[1] * 0.55)
        self.assertLess(height, image.shape[0] * 0.55)
        self.assertLess(abs((x + width / 2) - 320), 90)
        self.assertLess(abs((y + height / 2) - 170), 90)

    def test_central_void_detector_distinguishes_hole_from_solid_centre(self) -> None:
        background = np.full((240, 240, 3), 205, np.uint8)
        proposal = np.zeros((240, 240), np.uint8)
        cv2.circle(proposal, (120, 120), 66, 255, 12, cv2.LINE_AA)
        box = np.asarray([40, 40, 200, 200], np.float32)
        self.assertTrue(_expects_central_void(background, proposal, box))
        solid = background.copy()
        cv2.circle(solid, (120, 120), 28, (40, 60, 80), -1, cv2.LINE_AA)
        self.assertFalse(_expects_central_void(solid, proposal, box))

    def test_single_image_runs_through_phase23(self) -> None:
        source = Path("dataset/STL-1/Ring 1/R1 - Front.png")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "single"
            report = run_audit({"single": source}, output, segmenter_name="opencv")
            self.assertEqual(report["input_mode"], "single_image")
            self.assertEqual(report["phase3"]["status"], "not_run_single_image")
            self.assertTrue(Path(report["artifacts"]["summary"]).is_file())
            self.assertEqual(report["phase2_3"]["status"], "machine_proposals_pending_review")
            self.assertGreater(report["phase2_3"]["observation_count"], 0)
            self.assertTrue(Path(report["phase2_3"]["evidence_graph"]).is_file())
            self.assertTrue(Path(report["phase2_3"]["review_manifest"]).is_file())
            self.assertEqual(report["views"]["single"]["source_sha256"], report["views"]["single"]["copied_sha256"])
            self.assertFalse(report["accuracy_scope"]["manufacturing_accuracy_validated"])

    def test_five_views_export_valid_phase3_artifacts(self) -> None:
        source = Path("dataset/STL-1/Ring 1")
        names = {
            "front": "R1 - Front.png", "top": "R1 - Top.png", "iso": "R1 - iso.png",
            "lsv": "R1 - LSV.png", "rsv": "R1 - RSV.png",
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "five"
            report = run_audit({view: source / names[view] for view in FIVE_VIEWS}, output, segmenter_name="opencv", resolution=64)
            self.assertEqual(report["input_mode"], "five_view")
            self.assertEqual(report["phase3"]["status"], "experimental_non_metric_pass")
            mesh = trimesh.load(report["phase3"]["exports"]["stl"], force="mesh")
            self.assertTrue(mesh.is_watertight)
            with zipfile.ZipFile(report["phase3"]["exports"]["3mf"]) as archive:
                self.assertIn("3D/3dmodel.model", archive.namelist())
            preview = cv2.imread(report["phase3"]["exports"]["comparison"])
            self.assertIsNotNone(preview)
            loaded = json.loads((output / "audit_report.json").read_text(encoding="utf-8"))
            self.assertEqual(set(loaded["views"]), set(FIVE_VIEWS))
            self.assertGreater(loaded["phase2_3"]["cross_view_track_count"], 0)

    def test_partial_multiview_set_is_rejected(self) -> None:
        source = Path("dataset/STL-1/Ring 1/R1 - Front.png")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "exactly one image or all five"):
                run_audit({"front": source, "top": source}, Path(directory) / "invalid", segmenter_name="opencv")

    def test_corrupt_image_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "corrupt.png"
            source.write_bytes(b"this is not a decodable image")
            with self.assertRaisesRegex(ValueError, "Cannot decode image"):
                run_audit(
                    {"single": source}, Path(directory) / "invalid", segmenter_name="opencv"
                )


if __name__ == "__main__":
    unittest.main()
