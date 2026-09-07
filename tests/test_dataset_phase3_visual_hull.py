"""Regression checks for dataset Phase 3 visual-hull exports."""

from __future__ import annotations

import json
import unittest
import zipfile
from pathlib import Path

import cv2
import trimesh


class DatasetPhase3VisualHullTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path("dataset/phase_runs/v1/phase3_visual_hull")
        cls.report = json.loads((cls.root / "phase3_batch_report.json").read_text(encoding="utf-8"))

    def test_all_dataset_objects_have_exports(self) -> None:
        self.assertEqual(self.report["object_count"], 24)
        self.assertTrue(self.report["all_exports_readable"])
        for record in self.report["objects"]:
            self.assertTrue(Path(record["exports"]["stl"]).is_file())
            self.assertTrue(Path(record["exports"]["3mf"]).is_file())
            self.assertTrue(Path(record["exports"]["preview"]).is_file())

    def test_meshes_are_watertight_and_previews_readable(self) -> None:
        self.assertTrue(self.report["all_meshes_watertight"])
        for record in self.report["objects"]:
            mesh = trimesh.load(record["exports"]["stl"], force="mesh")
            self.assertTrue(mesh.is_watertight, record["object_id"])
            self.assertGreater(len(mesh.faces), 100)
            preview = cv2.imread(record["exports"]["preview"], cv2.IMREAD_COLOR)
            self.assertIsNotNone(preview, record["object_id"])
            self.assertGreaterEqual(preview.shape[1], 1000)
            self.assertGreaterEqual(preview.shape[0], 1000)

    def test_3mf_packages_have_required_parts(self) -> None:
        required = {"[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model"}
        for record in self.report["objects"]:
            with zipfile.ZipFile(record["exports"]["3mf"]) as archive:
                self.assertTrue(required.issubset(archive.namelist()), record["object_id"])

    def test_accuracy_scope_is_not_overclaimed(self) -> None:
        self.assertEqual(self.report["phase3_status"], "experimental_non_metric_pass")
        self.assertIn("not_eligible", self.report["phase3_3_status"])
        self.assertFalse(self.report["manufacturing_accuracy_validated"])
        for record in self.report["objects"]:
            self.assertFalse(record["scale"]["calibrated"])
            self.assertFalse(record["camera_calibrated"])
            self.assertFalse(record["phase3_3_eligible"])


if __name__ == "__main__":
    unittest.main()
