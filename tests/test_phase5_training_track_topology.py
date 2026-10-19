import json
import hashlib
from pathlib import Path
import unittest

import numpy as np

from pipeline.evaluate_phase5_training_track_topology import _local_triangles
from pipeline.extract_vggt_track_evidence import TRAIN_OBJECTS


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data/phase5_training_track_topology_v1/report.json"
VALIDATION_IDS = {"ring_004", "ring_012", "ring_020"}
TEST_IDS = {"ring_007", "ring_015", "ring_023"}


class Phase5TrainingTrackTopologyTests(unittest.TestCase):
    def test_local_triangles_require_three_accepted_vertices(self) -> None:
        points = np.asarray(
            ((10.0, 10.0), (30.0, 10.0), (20.0, 30.0), (100.0, 100.0)),
            np.float32,
        )
        accepted = np.asarray((True, True, True, False))
        triangles = _local_triangles(points, accepted)
        self.assertEqual(triangles.shape, (1, 3))
        self.assertEqual(set(triangles[0]), {0, 1, 2})
        accepted[2] = False
        self.assertEqual(_local_triangles(points, accepted).shape, (0, 3))

    def test_retained_report_is_training_only_and_reproducible(self) -> None:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        object_ids = set(report["scope"]["object_ids"])
        self.assertEqual(object_ids, set(TRAIN_OBJECTS))
        self.assertTrue(VALIDATION_IDS.isdisjoint(object_ids))
        self.assertTrue(TEST_IDS.isdisjoint(object_ids))
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["summary"]["object_count"], 18)
        self.assertEqual(
            report["summary"]["baseline_orientation_preserved_triangle_count"],
            2252,
        )
        self.assertEqual(
            report["summary"]["hybrid_orientation_preserved_triangle_count"],
            6913,
        )
        self.assertEqual(report["summary"]["improved_object_count"], 18)
        self.assertEqual(report["summary"]["regressed_object_count"], 0)
        self.assertEqual(
            report["summary"]["selected_training_policy"],
            "phase5_7_hybrid",
        )
        self.assertFalse(
            report["summary"]["complete_target_view_coverage_available"]
        )
        self.assertEqual(
            report["summary"]["target_views"]["top"][
                "objects_with_hybrid_topology"
            ],
            2,
        )
        self.assertFalse(report["claims"]["validation_topology_improved"])
        self.assertFalse(report["claims"]["cad_fusion_authorized"])
        self.assertTrue((ROOT / report["artifacts"]["audit"]).is_file())
        manifest_path = ROOT / report["artifacts"]["integrity_manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["files"]), 4)
        for item in manifest["files"].values():
            artifact = ROOT / item["path"]
            self.assertEqual(
                hashlib.sha256(artifact.read_bytes()).hexdigest(),
                item["sha256"],
            )
            self.assertEqual(artifact.stat().st_size, item["size_bytes"])
        self.assertEqual(len(report["objects"]), 18)
        for item in report["objects"]:
            self.assertEqual(item["split"], "train")
            self.assertTrue(item["baseline_query_prefix_reproduced"])
            self.assertGreater(
                item["comparison"]["orientation_preserved_triangle_delta"], 0
            )
            self.assertTrue((ROOT / item["artifacts"]["audit"]).is_file())


if __name__ == "__main__":
    unittest.main()
