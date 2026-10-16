import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from pipeline.build_phase5_stl1_relief_evidence import build_relief_evidence


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "data/phase5_stl1_relief_evidence_v1/report.json"


class Phase5ReliefEvidenceTests(unittest.TestCase):
    def test_brightness_only_detail_is_not_treated_as_strong_relief(self) -> None:
        size = 96
        yy, xx = np.mgrid[:size, :size]
        mask = ((xx - 48) ** 2 + (yy - 48) ** 2) < 38**2
        flat_depth = np.full((size, size), 0.5, np.float32)
        gradient = np.zeros((size, size), np.float32)
        gradient[:, 46:50] = 1.0
        result = build_relief_evidence(
            depth=flat_depth,
            uncertainty_u16=np.zeros((size, size), np.uint16),
            gradient=gradient,
            local_contrast=gradient,
            jewelry_mask=mask.astype(np.uint8) * 255,
            opening_mask=np.zeros((size, size), np.uint8),
        )
        self.assertLess(float(np.max(result["confidence"])), 0.16)
        self.assertEqual(float(np.max(np.abs(result["relief"]))), 0.0)

    def test_openings_are_excluded_and_metric_models_are_recorded(self) -> None:
        size = 96
        yy, xx = np.mgrid[:size, :size]
        mask = ((xx - 48) ** 2 + (yy - 48) ** 2) < 40**2
        opening = ((xx - 48) ** 2 + (yy - 48) ** 2) < 8**2
        bump = np.exp(-((xx - 48) ** 2 + (yy - 48) ** 2) / 240.0).astype(
            np.float32
        )
        result = build_relief_evidence(
            depth=bump,
            uncertainty_u16=np.zeros((size, size), np.uint16),
            gradient=cv2.Laplacian(bump, cv2.CV_32F),
            local_contrast=bump - cv2.GaussianBlur(bump, (0, 0), 4.0),
            jewelry_mask=mask.astype(np.uint8) * 255,
            opening_mask=opening.astype(np.uint8) * 255,
            metric_depths=(bump, bump * 0.98, bump * 1.02),
        )
        self.assertTrue(result["multi_model"])
        self.assertFalse(np.any(result["support"][opening]))
        self.assertEqual(float(np.max(np.abs(result["relief"][opening]))), 0.0)

    def test_retained_report_covers_stl1_without_promoting_relief(self) -> None:
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(report["summary"]["object_count"], 24)
        self.assertEqual(report["summary"]["view_count"], 120)
        self.assertEqual(report["summary"]["front_relief_obj_count"], 24)
        self.assertEqual(report["summary"]["test_held_out_count"], 3)
        self.assertTrue(all(report["checks"].values()))
        self.assertFalse(report["claims"]["raw_brightness_embossed_directly"])
        self.assertFalse(report["claims"]["relief_mesh_fused_into_exact_cad"])
        self.assertFalse(report["claims"]["relief_accuracy_measured"])

        test_ids = {
            item["object_id"]
            for item in report["objects"]
            if item["evaluation_status"] == "held_out_not_aggregated"
        }
        self.assertEqual(test_ids, {"ring_007", "ring_015", "ring_023"})
        for item in report["objects"]:
            mesh = item["front_relief_mesh"]
            self.assertGreater(mesh["vertex_count"], 0)
            self.assertGreater(mesh["face_count"], 0)
            self.assertEqual(mesh["topology"], "open_surface_not_solid")
            self.assertTrue((ROOT / mesh["path"]).exists())


if __name__ == "__main__":
    unittest.main()
