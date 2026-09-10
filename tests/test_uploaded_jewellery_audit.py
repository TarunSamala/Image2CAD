"""Tests for the reusable one-image and five-view audit program."""

from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import trimesh

from audit_jewellery import FIVE_VIEWS, run_audit


class UploadedJewelleryAuditTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
