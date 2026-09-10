"""Regression tests for Phase 2.3 semantic review and component IDs."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from review_phase2_3 import apply_review


class Phase23ReviewTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        mask = np.zeros((32, 32), np.uint8)
        mask[8:16, 8:16] = 255
        mask_path = root / "proposal.png"
        cv2.imwrite(str(mask_path), mask)
        graph = {
            "views": {
                "front": {"observations": [{"observation_id": "front_detail_001", "view": "front", "mask_path": str(mask_path), "track_id": "detail_track_001"}]},
                "iso": {"observations": [{"observation_id": "iso_detail_001", "view": "iso", "mask_path": str(mask_path), "track_id": "detail_track_001"}]},
            }
        }
        review = {
            "decisions": {
                "front_detail_001": {"decision": "pending", "semantic": "unknown_detail", "notes": ""},
                "iso_detail_001": {"decision": "pending", "semantic": "unknown_detail", "notes": ""},
            }
        }
        graph_path, review_path = root / "graph.json", root / "review.json"
        graph_path.write_text(json.dumps(graph), encoding="utf-8")
        review_path.write_text(json.dumps(review), encoding="utf-8")
        return graph_path, review_path

    def test_pending_review_blocks_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph, review = self._fixture(root)
            report = apply_review(graph, review, root / "output")
            self.assertFalse(report["review"]["complete"])
            self.assertFalse(report["geometry_handoff"]["allowed"])
            self.assertEqual(len(report["review"]["unresolved_observation_ids"]), 2)

    def test_reviewed_track_becomes_one_addressable_stone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph, review_path = self._fixture(root)
            review = json.loads(review_path.read_text(encoding="utf-8"))
            for decision in review["decisions"].values():
                decision.update({"decision": "relabel", "semantic": "stone"})
            review_path.write_text(json.dumps(review), encoding="utf-8")
            report = apply_review(graph, review_path, root / "output")
            self.assertTrue(report["review"]["complete"])
            self.assertTrue(report["geometry_handoff"]["allowed"])
            self.assertEqual(report["geometry_handoff"]["individual_stones"], ["stone_001"])
            self.assertEqual(report["components"][0]["views"], ["front", "iso"])
            self.assertTrue(all(Path(path).is_file() for path in report["components"][0]["mask_paths"].values()))

    def test_conflicting_semantics_in_one_track_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph, review_path = self._fixture(root)
            review = json.loads(review_path.read_text(encoding="utf-8"))
            review["decisions"]["front_detail_001"].update({"decision": "relabel", "semantic": "stone"})
            review["decisions"]["iso_detail_001"].update({"decision": "relabel", "semantic": "prong"})
            review_path.write_text(json.dumps(review), encoding="utf-8")
            report = apply_review(graph, review_path, root / "output")
            self.assertFalse(report["review"]["complete"])
            self.assertFalse(report["geometry_handoff"]["allowed"])
            self.assertEqual(len(report["review"]["semantic_conflicts"]), 1)


if __name__ == "__main__":
    unittest.main()
