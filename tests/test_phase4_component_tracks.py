"""Regression gates for non-metric Phase 4 component-track evidence."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np


class Phase4ComponentTrackTest(unittest.TestCase):
    root = Path("data/phase4_component_tracks_v1")

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads((cls.root / "report.json").read_text(encoding="utf-8"))

    def test_track_gate_preserves_machine_evidence_boundary(self) -> None:
        self.assertEqual(
            self.report["schema_version"],
            "lalitha_phase4_component_track_evidence_v1",
        )
        self.assertTrue(all(self.report["checks"].values()))
        self.assertTrue(
            self.report["gate"]["resolved_identity_assignments_machine_supported"]
        )
        self.assertFalse(self.report["gate"]["side_back_individual_identity_resolved"])
        self.assertFalse(self.report["gate"]["cross_view_identity_reviewed"])
        self.assertFalse(self.report["gate"]["metric_geometry_validated"])
        self.assertFalse(self.report["gate"]["geometry_generation_authorized"])
        self.assertFalse(any(self.report["claims"].values()))

    def test_vggt_selects_the_existing_top_and_angled_identity_order(self) -> None:
        for view in ("top", "angled"):
            comparison = self.report["comparisons"][view]
            self.assertEqual(comparison["best_assignment_indices"], [1, 2, 3, 4])
            self.assertTrue(comparison["identity_is_geometry_optimal"])
            matrix = np.asarray(comparison["distance_matrix_fraction"])
            self.assertEqual(matrix.shape, (4, 4))
            self.assertTrue(
                all(np.argmin(matrix[row]) == row for row in range(4)),
                view,
            )
            self.assertGreater(comparison["permutation_margin_fraction"], 0)

    def test_top_is_strong_and_angled_is_moderate_evidence(self) -> None:
        self.assertEqual(
            self.report["comparisons"]["top"]["evidence_level"],
            "strong_machine_support",
        )
        self.assertEqual(
            self.report["comparisons"]["angled"]["evidence_level"],
            "moderate_machine_support",
        )
        self.assertGreater(
            self.report["comparisons"]["top"]["permutation_margin_fraction"],
            self.report["comparisons"]["angled"]["permutation_margin_fraction"],
        )

    def test_all_resolved_components_have_robust_point_descriptors(self) -> None:
        for view in self.report["resolved_views"]:
            for component_id in self.report["component_ids"]:
                descriptor = self.report["descriptors"][view][component_id]
                self.assertGreaterEqual(descriptor["point_count"], 8)
                self.assertGreater(descriptor["median_cluster_radius"], 0)
                self.assertEqual(
                    descriptor["units"], "vggt_model_relative_non_metric"
                )

    def test_audit_artifact_is_retained(self) -> None:
        audit = Path(self.report["artifacts"]["audit"])
        self.assertTrue(audit.is_file())
        self.assertGreater(audit.stat().st_size, 100_000)


if __name__ == "__main__":
    unittest.main()
