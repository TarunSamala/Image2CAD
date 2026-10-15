"""Regression gates for review-gated Ring01 prong-instance evidence."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import cv2
import numpy as np


class Phase4ProngInstanceTest(unittest.TestCase):
    root = Path("data/phase4_prong_instances_v1")

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads((cls.root / "report.json").read_text(encoding="utf-8"))

    def test_partial_instance_gate_is_honest(self) -> None:
        self.assertEqual(
            self.report["schema_version"],
            "lalitha_phase4_prong_instance_evidence_v1",
        )
        self.assertEqual(self.report["summary"]["resolved_views"], ["front", "top", "angled"])
        self.assertEqual(self.report["summary"]["ambiguous_views"], ["side", "back"])
        self.assertTrue(self.report["gate"]["partial_instance_evidence_available"])
        self.assertFalse(self.report["gate"]["all_views_individually_resolved"])
        self.assertFalse(self.report["gate"]["cross_view_identity_reviewed"])
        self.assertFalse(self.report["gate"]["geometry_generation_authorized"])
        self.assertFalse(any(self.report["claims"].values()))

    def test_every_view_preserves_the_aggregate_without_overlap(self) -> None:
        self.assertTrue(all(self.report["checks"].values()))
        for view, record in self.report["views"].items():
            self.assertEqual(record["aggregate_union_iou"], 1.0, view)
            self.assertEqual(record["pairwise_overlap_px2"], 0, view)
            self.assertEqual(
                set(record["component_observations"]),
                {"prong_001", "prong_002", "prong_003", "prong_004"},
            )

    def test_resolved_masks_are_unique_disjoint_and_match_the_aggregate(self) -> None:
        for view in ("front", "top", "angled"):
            record = self.report["views"][view]
            paths = [
                Path(record["component_observations"][component_id]["visible_mask"])
                for component_id in self.report["component_ids"]
            ]
            self.assertEqual(len(paths), len(set(paths)))
            masks = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0 for path in paths]
            self.assertTrue(all(mask is not None for mask in masks))
            occupancy = np.sum(np.stack(masks, axis=0), axis=0)
            self.assertLessEqual(int(occupancy.max()), 1)
            aggregate = cv2.imread(
                f"data/ring01_phase2_refined/masks/prongs/ring01_{view}_prongs.png",
                cv2.IMREAD_GRAYSCALE,
            ) > 0
            np.testing.assert_array_equal(occupancy > 0, aggregate)

    def test_side_and_back_retain_two_explicit_overlap_groups(self) -> None:
        for view in ("side", "back"):
            observations = self.report["views"][view]["component_observations"]
            self.assertEqual(
                observations["prong_001"]["visible_mask"],
                observations["prong_003"]["visible_mask"],
            )
            self.assertEqual(
                observations["prong_002"]["visible_mask"],
                observations["prong_004"]["visible_mask"],
            )
            self.assertNotEqual(
                observations["prong_001"]["visible_mask"],
                observations["prong_002"]["visible_mask"],
            )
            self.assertTrue(
                all(item["aggregate_instance_mask"] for item in observations.values())
            )
            self.assertTrue(self.report["views"][view]["ambiguity"])

    def test_angled_clockwise_identity_anchor_matches_review_sheet(self) -> None:
        observations = self.report["views"]["angled"]["component_observations"]
        p1 = observations["prong_001"]["center_px"]
        p2 = observations["prong_002"]["center_px"]
        p3 = observations["prong_003"]["center_px"]
        p4 = observations["prong_004"]["center_px"]
        self.assertLess(p1["y"], min(p2["y"], p3["y"], p4["y"]))
        self.assertGreater(p2["x"], max(p1["x"], p3["x"], p4["x"]))
        self.assertLess(p3["x"], min(p1["x"], p2["x"], p4["x"]))
        self.assertGreater(p4["y"], max(p1["y"], p2["y"], p3["y"]))


if __name__ == "__main__":
    unittest.main()
