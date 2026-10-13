"""Contract tests for the Bible Phase 3/B2 geometry benchmark."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pipeline.run_phase3_geometry_benchmark import BIBLE_PREREQUISITE_BLOCKERS, _manifest
from pipeline.vggt_geometry_adapter import (
    COMMERCIAL_CHECKPOINT_LICENSE,
    PINNED_VGGT_COMMIT,
    probe_vggt,
)


class Phase3GeometryBenchmarkContractTest(unittest.TestCase):
    def test_retained_stl1_benchmark_is_complete_and_honest(self) -> None:
        report = __import__("json").loads(
            Path("dataset/phase_runs/v1/phase3_geometry_benchmark_v1/report.json")
            .read_text(encoding="utf-8")
        )
        totals = report["colmap"]["database_totals"]
        self.assertEqual(report["selected_object_count"], 24)
        self.assertEqual(report["selected_view_count"], 120)
        self.assertGreater(totals["keypoints_rows"], 100_000)
        self.assertEqual(totals["two_view_geometries_nonempty_records"], 0)
        self.assertEqual(report["colmap"]["completed_sparse_model_count"], 0)
        self.assertEqual(report["decision"], "research_only")
        self.assertFalse(any(report["claims"].values()))

    def test_stl1_manifest_has_b2_five_view_contract(self) -> None:
        records = _manifest(Path("dataset/prepared_v1/manifest.jsonl"))
        self.assertEqual(len(records), 24)
        self.assertTrue(all(set(record["views"]) == {"front", "top", "iso", "lsv", "rsv"} for record in records))

    def test_bible_prerequisites_prevent_accuracy_promotion(self) -> None:
        self.assertIn("calibrated_camera_truth_missing", BIBLE_PREREQUISITE_BLOCKERS)
        self.assertIn("paired_cad_or_scan_truth_missing", BIBLE_PREREQUISITE_BLOCKERS)

    def test_vggt_probe_never_downloads_or_approves_an_undeclared_checkpoint(self) -> None:
        probe = probe_vggt(None, None)
        self.assertEqual(probe["status"], "blocked")
        self.assertIn("local_checkpoint_not_supplied", probe["blockers"])
        self.assertIn("checkpoint_licence_not_approved_for_requested_use", probe["blockers"])
        self.assertEqual(probe["pinned_code_commit"], PINNED_VGGT_COMMIT)

    def test_commercial_checkpoint_still_requires_runtime_and_real_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "missing"
            probe = probe_vggt(checkpoint, COMMERCIAL_CHECKPOINT_LICENSE)
        self.assertEqual(probe["status"], "blocked")
        self.assertIn("local_checkpoint_not_found", probe["blockers"])

    def test_low_vram_override_is_never_implicit(self) -> None:
        probe = probe_vggt(None, None, allow_low_vram_attempt=True)
        self.assertTrue(probe["low_vram_attempt"])
        self.assertEqual(probe["status"], "blocked")
        self.assertIn("local_checkpoint_not_supplied", probe["blockers"])


if __name__ == "__main__":
    unittest.main()
