"""Contract tests for the image-only surface-normal evidence checkpoint."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from pipeline.run_surface_normal_evidence import (
    COMPLETE_DEPTH_BACKENDS,
    angular_error_degrees,
    depth_to_screen_normal,
    normalize_vectors,
    orient_direct_normals,
)


class SurfaceNormalEvidenceTest(unittest.TestCase):
    root = Path.cwd()
    output = root / "dataset" / "phase_runs" / "v1" / "surface_normal_evidence_v1"

    def test_vector_helpers_preserve_unit_length_and_orientation(self) -> None:
        mask = np.ones((16, 16), dtype=bool)
        depth = np.tile(np.linspace(0, 1, 16, dtype=np.float32), (16, 1))
        proxy = depth_to_screen_normal(depth, mask)
        self.assertTrue(np.allclose(np.linalg.norm(proxy, axis=-1), 1, atol=1e-5))
        direct = np.zeros((16, 16, 3), dtype=np.float32)
        direct[..., 2] = -1
        oriented, flipped = orient_direct_normals(direct, mask)
        self.assertTrue(flipped)
        self.assertTrue(np.all(oriented[..., 2] > 0))
        self.assertTrue(np.allclose(angular_error_degrees(oriented, oriented), 0))
        self.assertTrue(np.allclose(normalize_vectors(oriented), oriented))

    def test_retained_report_covers_all_selected_views_without_accuracy_claim(self) -> None:
        report = json.loads((self.output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["schema_version"], "image2cad_surface_normal_evidence_v1")
        self.assertEqual(report["sample_count"], 20)
        self.assertEqual(report["object_count"], 4)
        self.assertEqual(
            report["backends"]["depth_gradient_proxies"]["backends"],
            list(COMPLETE_DEPTH_BACKENDS),
        )
        self.assertEqual(report["backends"]["dsine"]["status"], "not_run")
        self.assertTrue(report["claims"]["per_pixel_normal_hypotheses_available"])
        self.assertTrue(report["claims"]["multi_depth_normal_disagreement_measured"])
        self.assertFalse(report["claims"]["normal_accuracy_measured"])
        self.assertFalse(report["claims"]["metric_surface_orientation_validated"])
        self.assertFalse(report["claims"]["manufacturing_accuracy_validated"])
        self.assertFalse(report["coordinate_contract"]["cross_view_frame_alignment"])

    def test_source_arrays_and_review_sheets_are_readable(self) -> None:
        report = json.loads((self.output / "report.json").read_text(encoding="utf-8"))
        for sample in report["samples"]:
            direct = np.load(self.root / sample["direct_normal"]["source_path"])
            self.assertEqual(direct.shape[-1], 3)
            self.assertTrue(np.isfinite(direct).all())
            self.assertEqual(
                set(sample["depth_normal_proxy_sources"]),
                set(COMPLETE_DEPTH_BACKENDS),
            )
            for path in sample["depth_normal_proxy_sources"].values():
                depth = np.load(self.root / path)
                self.assertEqual(depth.shape, direct.shape[:2])
                self.assertTrue(np.isfinite(depth).all())
            self.assertRegex(sample["depth_normal_consensus_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(sample["normal_disagreement_sha256"], r"^[0-9a-f]{64}$")
        for relative in report["review_sheets"].values():
            image = cv2.imread(str(self.root / relative), cv2.IMREAD_COLOR)
            self.assertIsNotNone(image)
            self.assertGreaterEqual(image.shape[0], 900)
            self.assertGreaterEqual(image.shape[1], 1200)


if __name__ == "__main__":
    unittest.main()
