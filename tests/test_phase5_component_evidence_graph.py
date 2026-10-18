import json
from pathlib import Path
import unittest

import numpy as np

from pipeline.build_phase5_component_evidence_graph import (
    _foreground_bbox,
    _primary_zone,
    _zone_for_point,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data/phase5_component_evidence_graph_v1/report.json"
VALIDATION_IDS = {"ring_004", "ring_012", "ring_020"}
TEST_IDS = {"ring_007", "ring_015", "ring_023"}


class Phase5ComponentEvidenceGraphTests(unittest.TestCase):
    def test_spatial_zones_are_deterministic_and_location_only(self) -> None:
        bbox = (100, 100, 700, 700)
        self.assertEqual(_zone_for_point((400, 150), bbox), "upper_head")
        self.assertEqual(_zone_for_point((150, 400), bbox), "left_head")
        self.assertEqual(_zone_for_point((400, 400), bbox), "central_head")
        self.assertEqual(_zone_for_point((650, 400), bbox), "right_head")
        self.assertEqual(_zone_for_point((400, 650), bbox), "lower_head")
        self.assertEqual(
            _primary_zone(
                {
                    "upper_head": 0,
                    "left_head": 2,
                    "central_head": 4,
                    "right_head": 1,
                    "lower_head": 0,
                }
            ),
            "central_head",
        )

    def test_foreground_bbox_is_exact(self) -> None:
        mask = np.zeros((20, 30), np.uint8)
        mask[3:15, 7:22] = 1
        self.assertEqual(_foreground_bbox(mask), (7, 3, 21, 14))

    def test_graph_retains_claim_boundaries_and_validation_split(self) -> None:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertEqual(set(report["scope"]["object_ids"]), VALIDATION_IDS)
        self.assertTrue(TEST_IDS.isdisjoint(report["scope"]["object_ids"]))
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["summary"]["relief_component_count"], 30)
        self.assertEqual(report["summary"]["multi_view_component_count"], 3)
        self.assertEqual(report["summary"]["single_view_component_count"], 22)
        self.assertEqual(
            report["summary"]["normal_unresolved_component_count"], 5
        )
        self.assertFalse(report["claims"]["semantic_components_validated"])
        self.assertFalse(report["claims"]["stones_confirmed"])
        self.assertFalse(report["claims"]["faces_confirmed"])
        self.assertFalse(report["claims"]["prongs_confirmed"])
        self.assertFalse(report["claims"]["cad_fusion_authorized"])
        self.assertTrue((ROOT / report["artifacts"]["audit"]).is_file())

        for item in report["objects"]:
            self.assertTrue(item["visibility_region_count_reproduced"])
            self.assertTrue(item["all_component_ids_unique"])
            graph_path = ROOT / item["artifacts"]["graph"]
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
            self.assertFalse(graph["claims"]["semantic_components_validated"])
            self.assertTrue(
                all(
                    component["semantic_label"]
                    == "unknown_ornamental_relief"
                    and not component["cad_fusion_authorized"]
                    for component in graph["relief_components"]
                )
            )
            parent_edges = {
                edge["source_id"]
                for edge in graph["edges"]
                if edge["relationship"] == "surface_evidence_for"
            }
            self.assertEqual(
                parent_edges,
                {
                    component["component_id"]
                    for component in graph["relief_components"]
                },
            )


if __name__ == "__main__":
    unittest.main()
