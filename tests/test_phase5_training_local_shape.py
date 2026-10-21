import json
from pathlib import Path
import unittest

import numpy as np

from pipeline.evaluate_phase5_training_local_shape import (
    _paired_score,
    _shape_maps,
    _weighted_blur,
)
from pipeline.extract_vggt_track_evidence import TRAIN_OBJECTS


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data/phase5_training_local_shape_v1/report.json"


class Phase5TrainingLocalShapeTests(unittest.TestCase):
    def test_blur_does_not_pull_unsupported_background_into_patch(self) -> None:
        values = np.zeros((64, 64), np.float32)
        support = np.zeros((64, 64), bool)
        values[24:40, 24:40] = 0.8
        support[24:40, 24:40] = True
        blurred = _weighted_blur(values, support, 4.0)
        self.assertAlmostEqual(float(blurred[24, 24]), 0.8, places=5)

    def test_shape_channels_and_paired_tie(self) -> None:
        maps = {
            "signed_relief_baseline": np.full((64, 64), 0.4, np.float32),
            "support": np.ones((64, 64), bool),
        }
        shapes = _shape_maps(maps)
        self.assertEqual(shapes["multiscale_shape_fusion"].shape, (64, 64, 6))
        source = np.asarray(((0.4, 0.0),), np.float32)
        controls = np.asarray((((0.4, 0.0), (0.4, 0.0)),), np.float32)
        result = _paired_score(
            source, source, controls, np.ones((1, 2), bool), (0.5, 0.5)
        )
        self.assertEqual(result["paired_discrimination_auc"], 0.5)

    def test_report_reproduces_frozen_controls_and_no_cad_claim(self) -> None:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertEqual(set(report["scope"]["object_ids"]), set(TRAIN_OBJECTS))
        self.assertEqual(report["summary"]["included_track_count"], 3259)
        self.assertAlmostEqual(
            report["summary"]["descriptors"]["signed_relief_baseline"]["mean_object_auc"],
            0.606918,
        )
        self.assertTrue(all(report["checks"].values()))
        self.assertFalse(report["claims"]["cad_fusion_authorized"])
        self.assertTrue((ROOT / report["artifacts"]["audit"]).is_file())


if __name__ == "__main__":
    unittest.main()
