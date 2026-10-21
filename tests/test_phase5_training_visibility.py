import json
from pathlib import Path
import unittest

import numpy as np

from pipeline.diagnose_phase5_training_visibility import (
    _normal_diagnostic,
    _route_counts,
)
from pipeline.extract_vggt_track_evidence import TRAIN_OBJECTS


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data/phase5_training_visibility_route_v1/report.json"


class Phase5TrainingVisibilityTests(unittest.TestCase):
    def test_route_separates_mask_visibility_and_track_loss(self) -> None:
        candidate = np.ones(5, bool)
        points = np.asarray(
            ((20, 20), (30, 30), (40, 40), (50, 50), (np.nan, 60)),
            np.float32,
        )
        mask = np.ones((518, 518), bool)
        mask[30, 30] = False
        visibility = np.asarray((1, 1, 0, 1, 1), np.float32)
        confidence = np.asarray((1, 1, 1, 0, 1), np.float32)
        accepted = np.asarray((1, 1, 1, 1, 1), bool)
        normal = np.asarray((1, 1, 1, 1, 1), bool)
        counts = _route_counts(candidate, points, mask, visibility,
                               confidence, accepted, normal)
        self.assertEqual(list(counts.values()), [5, 4, 3, 2, 1, 1, 1])

    def test_retained_report_stays_training_only_and_diagnostic(self) -> None:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertEqual(set(report["scope"]["object_ids"]), set(TRAIN_OBJECTS))
        self.assertEqual(report["summary"]["object_count"], 18)
        self.assertTrue(all(report["checks"].values()))
        self.assertFalse(report["claims"]["physical_occlusion_identified"])
        self.assertFalse(report["claims"]["cad_fusion_authorized"])
        self.assertTrue((ROOT / report["artifacts"]["audit"]).is_file())
        for view in report["scope"]["target_views"]:
            counts = report["summary"]["target_views"][view]["cumulative_query_counts"]
            self.assertEqual(list(counts.values()), sorted(counts.values(), reverse=True))

    def test_normal_diagnostic_separates_source_and_target(self) -> None:
        points = np.asarray(((100, 100), (200, 200)), np.float32)
        source = np.zeros((518, 518), np.float32)
        target = np.zeros((518, 518), np.float32)
        source[100, 100] = 0.8
        target[200, 200] = 0.8
        result = _normal_diagnostic(points, points, np.ones(2, bool), source, target)
        self.assertEqual(result["source_normal_pass_count"], 1)
        self.assertEqual(result["target_normal_pass_count"], 1)
        self.assertEqual(result["both_normal_pass_count"], 0)

    def test_missing_normal_map_is_unavailable_not_agreement_failure(self) -> None:
        points = np.asarray(((100, 100),), np.float32)
        result = _normal_diagnostic(points, points, np.ones(1, bool), None, None)
        self.assertFalse(result["normal_pair_available"])
        self.assertEqual(result["accepted_track_count"], 1)
        self.assertIsNone(result["both_normal_pass_count"])


if __name__ == "__main__":
    unittest.main()
