"""Regression gates for STL-1 Phase 5.3 exact openwork trials."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


class Phase5STL1OpenworkFitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.output = cls.root / "data" / "phase5_stl1_exact_openwork_fit_v1"
        cls.report = json.loads(
            (cls.output / "report.json").read_text(encoding="utf-8")
        )

    def test_all_phase4_front_candidates_were_trialed(self) -> None:
        summary = self.report["summary"]
        self.assertEqual(summary["object_count"], 24)
        self.assertEqual(summary["input_candidate_count"], 168)
        self.assertEqual(
            summary["input_candidate_count"],
            summary["accepted_opening_count"]
            + summary["rejected_opening_count"],
        )
        self.assertTrue(self.report["checks"]["all_front_candidates_trialed"])

    def test_validation_projection_improves_without_scoring_test(self) -> None:
        summary = self.report["summary"]
        self.assertGreater(summary["validation_mean_openwork_iou"], 0.98)
        self.assertGreater(summary["validation_mean_iou_delta"], 0.0)
        self.assertGreater(
            summary["validation_mean_openwork_boundary_f1"], 0.95
        )
        self.assertEqual(summary["test_held_out_count"], 3)
        self.assertTrue(self.report["checks"]["held_out_test_not_scored"])

    def test_geometry_and_claim_boundaries_pass(self) -> None:
        checks = self.report["checks"]
        self.assertTrue(checks["all_heads_valid_single_solids"])
        self.assertTrue(checks["all_metals_valid_single_solids"])
        self.assertTrue(checks["all_step_exports_present"])
        self.assertTrue(checks["all_validation_geometry_previews_present"])
        self.assertTrue(checks["candidate_semantics_not_promoted"])
        self.assertFalse(self.report["claims"]["openings_semantically_reviewed"])
        self.assertFalse(self.report["claims"]["relief_reconstructed"])
        self.assertFalse(
            self.report["claims"]["manufacturing_accuracy_validated"]
        )

    def test_artifacts_and_stable_ids_are_present(self) -> None:
        self.assertTrue((self.root / self.report["artifacts"]["audit"]).is_file())
        seen: set[str] = set()
        for item in self.report["objects"]:
            self.assertTrue((self.root / item["artifacts"]["step"]).is_file())
            for candidate in item["opening_candidates"]:
                component_id = candidate["component_id"]
                self.assertNotIn(component_id, seen)
                seen.add(component_id)
                self.assertTrue(component_id.startswith(f"{item['object_id']}_front_"))
            if item["split"] == "test":
                self.assertIsNone(item["metrics"])
        self.assertEqual(len(seen), 168)


if __name__ == "__main__":
    unittest.main()
