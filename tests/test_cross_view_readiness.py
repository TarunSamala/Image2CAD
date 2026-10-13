"""Contract tests for public cross-view backend readiness evidence."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.run_cross_view_readiness import BACKENDS, build_report


class CrossViewReadinessTest(unittest.TestCase):
    root = Path.cwd()
    retained = root / "dataset/phase_runs/v1/cross_view_backend_readiness_v1/report.json"

    def test_backend_licence_contract_is_explicit(self) -> None:
        self.assertTrue(BACKENDS["vggt"]["public_product_eligible"])
        self.assertFalse(BACKENDS["vggt_omega"]["public_product_eligible"])
        self.assertFalse(BACKENDS["dust3r"]["public_product_eligible"])
        self.assertFalse(BACKENDS["mast3r"]["public_product_eligible"])
        for backend in BACKENDS.values():
            self.assertRegex(backend["source_revision"], r"^[0-9a-f]{40}$")
            self.assertRegex(backend["license_sha256"], r"^[0-9a-f]{64}$")

    def test_checkpoint_and_acceptance_are_both_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            checkpoint.write_bytes(b"test checkpoint sentinel")
            with patch(
                "pipeline.run_cross_view_readiness.runtime_probe",
                return_value={
                    "torch_version": "test",
                    "cuda_available": True,
                    "gpu_name": "test",
                    "gpu_free_gib": 12.0,
                    "gpu_total_gib": 12.0,
                },
            ):
                blocked = build_report(self.root, checkpoint, False)
                ready = build_report(self.root, checkpoint, True, "a" * 40)
        self.assertEqual(blocked["candidates"]["vggt"]["status"], "blocked")
        self.assertEqual(ready["candidates"]["vggt"]["status"], "ready")
        self.assertEqual(ready["summary"]["locally_ready"], ["vggt"])
        self.assertEqual(ready["code_commit"], "a" * 40)

    def test_retained_report_makes_no_geometry_claim(self) -> None:
        report = json.loads(self.retained.read_text(encoding="utf-8"))
        self.assertEqual(report["schema_version"], "cross_view_backend_readiness_v1")
        self.assertEqual(report["summary"]["candidate_count"], 4)
        self.assertEqual(report["summary"]["public_product_candidates"], ["vggt"])
        self.assertEqual(report["summary"]["locally_ready_count"], 0)
        self.assertRegex(report["code_commit"], r"^[0-9a-f]{40}$")
        self.assertFalse(any(report["claims"].values()))


if __name__ == "__main__":
    unittest.main()
