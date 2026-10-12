"""Tests for the non-human image-only Phase 2 machine-review checkpoint."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from bible_image_only_phase2_machine_review import build_report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BibleImageOnlyPhase2MachineReviewTest(unittest.TestCase):
    def test_machine_audit_passes_without_claiming_human_review(self) -> None:
        root = Path.cwd()
        ring_manifest = root / "data/ring01_ground_truth_v1/manifest.json"
        stl1_manifest = root / "dataset/phase_runs/v1/phase1_human_review_v1/manifest.json"
        hashes_before = (_sha256(ring_manifest), _sha256(stl1_manifest))
        with tempfile.TemporaryDirectory() as directory:
            report = build_report(root, Path(directory))
            self.assertEqual(
                report["status"],
                "research_continuation_allowed_with_blocked_human_gate",
            )
            self.assertTrue(all(report["deterministic_checks"].values()))
            self.assertTrue(report["claims"]["machine_evidence_validated"])
            self.assertTrue(report["claims"]["image_only_phase2_research_may_continue"])
            self.assertFalse(report["claims"]["human_review_performed"])
            self.assertFalse(report["claims"]["bible_phase1_complete"])
            self.assertFalse(report["claims"]["canonical_bible_phase2_complete"])
            self.assertFalse(report["claims"]["manufacturing_accuracy_validated"])
            self.assertEqual(report["ring01"]["proposal_count"], 35)
            self.assertEqual(report["stl1"]["object_count"], 24)
            self.assertEqual(report["stl1"]["view_count"], 120)
            self.assertEqual(report["stl1"]["evidence_count"], 840)
            self.assertTrue((Path(directory) / "report.json").is_file())
            self.assertTrue((Path(directory) / "MACHINE_REVIEW.md").is_file())
        self.assertEqual(hashes_before, (_sha256(ring_manifest), _sha256(stl1_manifest)))


if __name__ == "__main__":
    unittest.main()
