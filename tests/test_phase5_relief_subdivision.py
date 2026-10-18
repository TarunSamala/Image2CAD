import json
from pathlib import Path
import unittest

import numpy as np

from pipeline.subdivide_phase5_component_evidence import (
    MAX_PATCH_DIAMETER_TRACK_PIXELS,
    _edge_barrier_fraction,
    _split_oversized_group,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data/phase5_relief_subdivision_v1/report.json"
VALIDATION_IDS = {"ring_004", "ring_012", "ring_020"}
TEST_IDS = {"ring_007", "ring_015", "ring_023"}


class Phase5ReliefSubdivisionTests(unittest.TestCase):
    def test_edge_barrier_sampling_detects_crossing(self) -> None:
        edge = np.zeros((30, 30), np.uint8)
        edge[:, 15] = 255
        crossing = _edge_barrier_fraction(edge, (2.0, 15.0), (27.0, 15.0))
        clear = _edge_barrier_fraction(edge, (2.0, 4.0), (12.0, 4.0))
        self.assertGreater(crossing, 0.0)
        self.assertEqual(clear, 0.0)

    def test_oversized_transitive_chain_is_bounded(self) -> None:
        nodes = [
            {
                "node_id": f"q{index}",
                "source_xy": [float(index * 25), 10.0],
            }
            for index in range(12)
        ]
        groups = _split_oversized_group(nodes)
        self.assertGreater(len(groups), 1)
        for group in groups:
            points = np.asarray([item["source_xy"] for item in group])
            diagonal = float(
                np.linalg.norm(points.max(axis=0) - points.min(axis=0))
            )
            self.assertLessEqual(diagonal, MAX_PATCH_DIAMETER_TRACK_PIXELS)

    def test_retained_subdivision_is_complete_and_validation_only(self) -> None:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertEqual(set(report["scope"]["object_ids"]), VALIDATION_IDS)
        self.assertTrue(TEST_IDS.isdisjoint(report["scope"]["object_ids"]))
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["summary"]["parent_component_count"], 30)
        self.assertEqual(report["summary"]["patch_count"], 83)
        self.assertEqual(report["summary"]["patch_increase"], 53)
        self.assertEqual(report["summary"]["fit_review_eligible_patch_count"], 40)
        self.assertGreater(report["summary"]["singleton_patch_count"], 0)
        self.assertFalse(report["claims"]["semantic_components_validated"])
        self.assertFalse(report["claims"]["surface_geometry_validated"])
        self.assertFalse(report["claims"]["cad_fusion_authorized"])
        self.assertTrue((ROOT / report["artifacts"]["audit"]).is_file())

        for item in report["objects"]:
            self.assertTrue(item["source_queries_partitioned_exactly_once"])
            self.assertTrue(item["phase5_10_parent_count_reproduced"])
            graph_path = ROOT / item["artifacts"]["graph"]
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
            self.assertFalse(graph["claims"]["cad_fusion_authorized"])
            all_ids = [
                query_id
                for patch in graph["patches"]
                for query_id in patch["query_node_ids"]
            ]
            self.assertEqual(len(all_ids), len(set(all_ids)))
            for patch in graph["patches"]:
                self.assertEqual(
                    patch["semantic_label"],
                    "unknown_ornamental_relief_patch",
                )
                self.assertFalse(patch["cad_fusion_authorized"])
                if patch["future_surface_fit_review_eligible"]:
                    self.assertGreaterEqual(patch["query_count"], 3)
                    self.assertTrue(patch["normal_corroborated_target_views"])


if __name__ == "__main__":
    unittest.main()
