import json
from pathlib import Path
import unittest

import numpy as np

from pipeline.evaluate_phase5_training_relief_descriptors import (
    _controls,
    _evaluate_view,
)
from pipeline.extract_vggt_track_evidence import TRAIN_OBJECTS


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data/phase5_training_relief_descriptors_v1/report.json"


class Phase5TrainingReliefDescriptorTests(unittest.TestCase):
    def test_controls_require_matching_confidence_and_detail(self) -> None:
        point = np.asarray(((259.0, 259.0),), np.float32)
        support = np.ones((518, 518), dtype=bool)
        confidence = np.full((518, 518), 0.1, np.float32)
        photo = np.full((518, 518), 0.5, np.float32)
        _, valid = _controls(
            point,
            {"support": support, "confidence": confidence, "photo_detail": photo},
        )
        self.assertEqual(int(np.count_nonzero(valid)), valid.size)
        confidence[:, :] = 0.0
        _, rejected = _controls(
            point,
            {"support": support, "confidence": confidence, "photo_detail": photo},
        )
        self.assertFalse(np.any(rejected))

    def test_equal_positive_and_control_errors_score_as_chance(self) -> None:
        point = np.asarray(((259.0, 259.0),), np.float32)
        constant = np.full((518, 518), 0.5, np.float32)
        maps = {
            "support": np.ones((518, 518), dtype=bool),
            "confidence": np.full((518, 518), 0.1, np.float32),
            "signed_relief_baseline": constant,
            "unsigned_depth_detail": constant,
            "photo_detail": constant,
            "depth_photo_fusion": constant,
        }
        result = _evaluate_view(
            point, point, np.asarray((True,)), maps, maps
        )
        self.assertEqual(result["control_eligible_track_count"], 1)
        for descriptor in result["descriptors"].values():
            self.assertEqual(descriptor["paired_discrimination_auc"], 0.5)

    def test_retained_training_decision_rejects_biased_candidates(self) -> None:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertEqual(set(report["scope"]["object_ids"]), set(TRAIN_OBJECTS))
        self.assertEqual(report["summary"]["object_count"], 18)
        self.assertEqual(report["summary"]["included_track_count"], 3259)
        self.assertEqual(report["summary"]["selected_descriptor"], "none")
        self.assertAlmostEqual(
            report["summary"]["descriptors"]["signed_relief_baseline"]["mean_object_auc"],
            0.606918,
        )
        self.assertAlmostEqual(
            report["summary"]["descriptors"]["photo_detail"]["mean_object_auc"],
            0.516295,
        )
        self.assertEqual(
            report["summary"]["target_view_eligible_track_count"]["top"], 2
        )
        self.assertTrue(all(report["checks"].values()))
        self.assertFalse(report["claims"]["validation_relief_improved"])
        self.assertFalse(report["claims"]["cad_fusion_authorized"])
        self.assertTrue((ROOT / report["artifacts"]["audit"]).is_file())
        self.assertTrue(all(item["included_track_count"] > 0 for item in report["objects"]))


if __name__ == "__main__":
    unittest.main()
