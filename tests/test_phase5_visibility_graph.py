import json
from pathlib import Path
import unittest

from pipeline.build_phase5_visibility_graph import (
    _build_regions,
    _classification,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data/phase5_visibility_graph_v1/report.json"
VALIDATION_IDS = {"ring_004", "ring_012", "ring_020"}
TEST_IDS = {"ring_007", "ring_015", "ring_023"}


class Phase5VisibilityGraphTests(unittest.TestCase):
    def test_classification_requires_distinct_corroborated_views(self) -> None:
        self.assertEqual(
            _classification(["iso", "lsv"], ["iso", "lsv"]),
            "multi_view_normal_corroborated",
        )
        self.assertEqual(
            _classification(["iso", "lsv"], ["iso"]),
            "single_view_normal_corroborated",
        )
        self.assertEqual(
            _classification(["iso"], []),
            "reciprocal_normal_unresolved",
        )
        self.assertEqual(
            _classification([], []), "unobserved_source_candidate"
        )

    def test_regions_link_only_nearby_queries_with_a_shared_view(self) -> None:
        nodes = [
            {
                "node_id": "a",
                "source_xy": [10.0, 10.0],
                "accepted_target_views": ["iso"],
                "normal_corroborated_target_views": ["iso"],
                "classification": "single_view_normal_corroborated",
            },
            {
                "node_id": "b",
                "source_xy": [20.0, 10.0],
                "accepted_target_views": ["iso", "lsv"],
                "normal_corroborated_target_views": ["lsv"],
                "classification": "single_view_normal_corroborated",
            },
            {
                "node_id": "c",
                "source_xy": [21.0, 11.0],
                "accepted_target_views": ["rsv"],
                "normal_corroborated_target_views": ["rsv"],
                "classification": "single_view_normal_corroborated",
            },
            {
                "node_id": "unsupported",
                "source_xy": [12.0, 12.0],
                "accepted_target_views": [],
                "normal_corroborated_target_views": [],
                "classification": "unobserved_source_candidate",
            },
        ]
        regions = _build_regions(nodes)
        self.assertEqual(len(regions), 2)
        self.assertEqual(regions[0]["query_node_ids"], ["a", "b"])
        self.assertEqual(regions[1]["query_node_ids"], ["c"])

    def test_retained_graph_is_validation_only_and_reproduces_inputs(self) -> None:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertEqual(set(report["scope"]["object_ids"]), VALIDATION_IDS)
        self.assertTrue(TEST_IDS.isdisjoint(report["scope"]["object_ids"]))
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(
            report["summary"]["reciprocally_supported_query_count"], 736
        )
        self.assertEqual(
            report["summary"]["normal_corroborated_query_count"], 521
        )
        self.assertEqual(
            report["summary"]["query_class_counts"][
                "multi_view_normal_corroborated"
            ],
            45,
        )
        self.assertGreater(
            report["summary"]["query_class_counts"][
                "unobserved_source_candidate"
            ],
            0,
        )
        self.assertFalse(report["claims"]["hidden_geometry_recovered"])
        self.assertFalse(report["claims"]["cad_fusion_authorized"])
        self.assertTrue((ROOT / report["artifacts"]["audit"]).is_file())
        for item in report["objects"]:
            self.assertTrue(item["phase5_7_supported_count_reproduced"])
            self.assertTrue(item["phase5_8_normal_count_reproduced"])
            self.assertTrue((ROOT / item["artifacts"]["graph"]).is_file())
            self.assertTrue((ROOT / item["artifacts"]["audit"]).is_file())


if __name__ == "__main__":
    unittest.main()
