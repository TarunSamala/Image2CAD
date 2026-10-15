"""Regression gates for scoped Phase 4 candidate semantic review."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pipeline.phase4_candidate_review import (
    apply_candidate_review,
    initialize_manifest,
    load_manifest,
)


class Phase4CandidateReviewTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        target = self.root / "data/phase4_stl1_component_candidates_v1"
        target.mkdir(parents=True)
        source = Path("data/phase4_stl1_component_candidates_v1/report.json")
        (target / "report.json").write_text(
            source.read_text(encoding="utf-8"), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_initial_manifest_covers_every_candidate_and_blocks_geometry(self) -> None:
        manifest = initialize_manifest(self.root)
        gate = manifest["gate"]
        self.assertEqual(gate["candidate_count"], 266)
        self.assertEqual(gate["pending_candidate_count"], 266)
        self.assertEqual(gate["approved_candidate_count"], 0)
        self.assertFalse(gate["individual_stones_confirmed"])
        self.assertFalse(gate["cross_view_component_identity_resolved"])
        self.assertFalse(gate["component_graph_generation_authorized"])
        self.assertTrue(manifest["review_contract"]["proposals_immutable"])
        self.assertTrue(manifest["review_contract"]["history_append_only"])

    def test_review_updates_only_selected_record_and_retains_history(self) -> None:
        manifest = initialize_manifest(self.root)
        candidate = manifest["objects"]["ring_001"]["views"]["front"][
            "candidates"
        ][0]
        candidate_id = candidate["candidate_id"]
        before_hash = candidate["mask_sha256"]
        reviewed = apply_candidate_review(
            self.root,
            "ring_001",
            "front",
            candidate_id,
            "aperture",
            "approved",
            "Test Reviewer",
            "Visible opening in the source image",
        )
        selected = reviewed["objects"]["ring_001"]["views"]["front"][
            "candidates"
        ][0]
        self.assertEqual(selected["semantic_label"], "aperture")
        self.assertEqual(selected["decision"], "approved")
        self.assertEqual(selected["mask_sha256"], before_hash)
        self.assertEqual(len(selected["history"]), 1)
        self.assertEqual(selected["history"][0]["decision"], "pending")
        self.assertEqual(reviewed["gate"]["approved_candidate_count"], 1)
        self.assertEqual(reviewed["gate"]["pending_candidate_count"], 265)
        _, reloaded = load_manifest(self.root)
        self.assertEqual(reloaded["gate"]["approved_candidate_count"], 1)

    def test_possible_stone_label_never_authorizes_cad_or_confirms_material(self) -> None:
        manifest = initialize_manifest(self.root)
        candidate_id = manifest["objects"]["ring_014"]["views"]["front"][
            "candidates"
        ][0]["candidate_id"]
        reviewed = apply_candidate_review(
            self.root,
            "ring_014",
            "front",
            candidate_id,
            "possible_stone_or_setting",
            "approved",
            "Test Reviewer",
        )
        gate = reviewed["gate"]
        self.assertEqual(gate["approved_possible_stone_or_setting_count"], 1)
        self.assertFalse(gate["individual_stones_confirmed"])
        self.assertFalse(gate["cross_view_component_identity_resolved"])
        self.assertFalse(gate["component_graph_generation_authorized"])

    def test_invalid_scope_and_unidentified_review_fail_closed(self) -> None:
        initialize_manifest(self.root)
        with self.assertRaisesRegex(ValueError, "Reviewer is required"):
            apply_candidate_review(
                self.root,
                "ring_001",
                "front",
                "ring_001_front_negative_space_001",
                "aperture",
                "approved",
                "",
            )
        with self.assertRaisesRegex(ValueError, "Unknown candidate ID"):
            apply_candidate_review(
                self.root,
                "ring_001",
                "front",
                "candidate_999",
                "aperture",
                "approved",
                "Test Reviewer",
            )


if __name__ == "__main__":
    unittest.main()
