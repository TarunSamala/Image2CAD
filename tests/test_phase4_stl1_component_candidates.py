"""Regression gates for the STL-1 Phase 4 candidate inventory."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import cv2
import numpy as np


class Phase4Stl1ComponentCandidateTest(unittest.TestCase):
    root = Path("data/phase4_stl1_component_candidates_v1")

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads((cls.root / "report.json").read_text(encoding="utf-8"))

    def test_complete_dataset_coverage_and_fail_closed_gate(self) -> None:
        self.assertEqual(self.report["object_count"], 24)
        self.assertEqual(self.report["view_count"], 120)
        self.assertEqual(len(self.report["objects"]), 24)
        self.assertTrue(all(self.report["checks"].values()))
        self.assertTrue(
            self.report["gate"]["dataset_wide_negative_space_inventory_available"]
        )
        self.assertFalse(self.report["gate"]["individual_stones_confirmed"])
        self.assertFalse(
            self.report["gate"]["cross_view_component_identity_resolved"]
        )
        self.assertFalse(
            self.report["gate"]["component_graph_generation_authorized"]
        )
        self.assertFalse(any(self.report["claims"].values()))

    def test_candidates_have_stable_unique_ids_and_readable_masks(self) -> None:
        ids: list[str] = []
        counted = 0
        allowed = {
            "structural_opening",
            "circular_aperture",
            "decorative_openwork",
        }
        for object_id, record in self.report["objects"].items():
            self.assertEqual(
                set(record["views"]), {"front", "top", "iso", "lsv", "rsv"}
            )
            self.assertEqual(record["confirmed_stone_count"], 0)
            for view_name, view in record["views"].items():
                self.assertFalse(view["cross_view_identity_resolved"])
                self.assertEqual(view["candidate_count"], len(view["candidates"]))
                occupancy = None
                for candidate in view["candidates"]:
                    self.assertTrue(
                        candidate["candidate_id"].startswith(
                            f"{object_id}_{view_name}_"
                        )
                    )
                    self.assertIn(candidate["candidate_subtype"], allowed)
                    self.assertFalse(candidate["is_confirmed_stone"])
                    mask = cv2.imread(candidate["mask_path"], cv2.IMREAD_GRAYSCALE)
                    self.assertIsNotNone(mask, candidate["mask_path"])
                    active = mask > 0
                    self.assertEqual(
                        int(np.count_nonzero(active)), candidate["area_px2"]
                    )
                    occupancy = (
                        active.astype(np.uint8)
                        if occupancy is None
                        else occupancy + active
                    )
                    ids.append(candidate["candidate_id"])
                    counted += 1
                if occupancy is not None:
                    self.assertLessEqual(int(occupancy.max()), 1)
        self.assertEqual(counted, self.report["candidate_count"])
        self.assertEqual(len(ids), len(set(ids)))

    def test_audit_is_retained(self) -> None:
        audit = Path(self.report["artifacts"]["front_audit"])
        image = cv2.imread(str(audit), cv2.IMREAD_COLOR)
        self.assertIsNotNone(image)
        self.assertGreater(image.shape[0], 1000)
        self.assertGreater(image.shape[1], 800)


if __name__ == "__main__":
    unittest.main()
