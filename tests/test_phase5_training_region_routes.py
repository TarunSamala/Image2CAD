import json
from pathlib import Path
import unittest

import numpy as np

from pipeline.audit_phase5_training_region_routes import _grid_ids, _region_record
from pipeline.extract_vggt_track_evidence import TRAIN_OBJECTS


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data/phase5_training_region_routes_v1/report.json"


class Phase5TrainingRegionRouteTests(unittest.TestCase):
    def test_grid_ids_cover_bbox_edges(self) -> None:
        points = np.asarray(((0, 0), (50, 50), (100, 100), (100, 0), (0, 100)), np.float32)
        self.assertEqual(_grid_ids(points, (0, 0, 100, 100)).tolist(), [0, 4, 8, 2, 6])
        with self.assertRaises(ValueError):
            _grid_ids(points, (0, 0, 0, 100))

    def test_region_states_do_not_infer_occlusion(self) -> None:
        candidate = np.ones(10, bool)
        selected = np.ones(10, bool)
        accepted = {view: np.zeros(10, bool) for view in ("top", "iso", "lsv", "rsv")}
        accepted["iso"][:3] = True
        record = _region_record(4, selected, candidate, accepted)
        self.assertEqual(record["target_views"]["iso"]["state"], "reciprocal_track_review_evidence")
        self.assertEqual(record["target_views"]["top"]["state"], "cross_view_identity_unresolved")
        self.assertEqual(record["region_id"], "front_grid_1_1")

    def test_retained_report_is_training_only_and_conservative(self) -> None:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertEqual(set(report["scope"]["object_ids"]), set(TRAIN_OBJECTS))
        self.assertEqual(report["summary"]["object_count"], 18)
        self.assertEqual(report["summary"]["front_candidate_count"], 16128)
        self.assertTrue(all(report["checks"].values()))
        self.assertFalse(any(report["claims"].values()))
        self.assertTrue((ROOT / report["artifacts"]["audit"]).is_file())
        for item in report["objects"]:
            self.assertEqual(sum(cell["candidate_count"] for cell in item["regions"]), item["candidate_count"])
        for view, summary in report["summary"]["target_views"].items():
            self.assertEqual(sum(cell["accepted_track_count"] for cell in summary["regions"]), summary["accepted_track_count"], view)


if __name__ == "__main__":
    unittest.main()
