from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np

from pipeline.run_saraswati_blind_benchmark import _foreground_mask, _surface_metrics
from pipeline.refine_saraswati_blind_relief import carve_front_relief
from pipeline.sweep_saraswati_relief import _non_regressing
from pipeline.refine_saraswati_multiview_relief import carve_axis_relief


class SaraswatiBlindBenchmarkTest(unittest.TestCase):
    def test_axis_relief_preserves_minimum_local_thickness(self) -> None:
        volume = np.zeros((8, 8, 8), dtype=bool)
        volume[2:6, 2:6, 2:6] = True
        depth = np.ones((8, 8), dtype=np.float32)
        valid = np.zeros((8, 8), dtype=bool)
        valid[2:6, 2:6] = True
        carved, operation = carve_axis_relief(
            volume, depth, valid, "front", 0.2, "low"
        )
        self.assertTrue(np.any(carved))
        self.assertGreater(operation["removed_voxels"], 0)
        for row, column in zip(*np.where(valid)):
            self.assertGreaterEqual(np.count_nonzero(carved[row, :, column]), 2)

    def test_relief_acceptance_requires_every_metric_to_avoid_regression(self) -> None:
        baseline = {
            "chamfer_l1_normalized": 0.04,
            "hausdorff_95_normalized": 0.05,
            "f_score_1pct": 0.30,
            "f_score_2pct": 0.58,
            "f_score_5pct": 0.93,
        }
        improved = dict(baseline, chamfer_l1_normalized=0.039, f_score_1pct=0.31)
        regressed = dict(improved, f_score_2pct=0.57)
        self.assertTrue(_non_regressing(improved, baseline))
        self.assertFalse(_non_regressing(regressed, baseline))

    def test_foreground_mask_separates_object_from_border_background(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "view.png"
            image = np.full((96, 96, 3), 170, dtype=np.uint8)
            cv2.circle(image, (48, 48), 24, (120, 55, 15), -1)
            cv2.imwrite(str(path), image)
            mask, metadata = _foreground_mask(path)
            self.assertTrue(mask[48, 48])
            self.assertFalse(mask[0, 0])
            self.assertGreater(metadata["mask_occupancy"], 0.1)

    def test_identical_surface_samples_have_perfect_scores(self) -> None:
        points = np.asarray([
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
        ], dtype=np.float64)
        metrics = _surface_metrics(points, points.copy())
        self.assertEqual(metrics["chamfer_l1_normalized"], 0.0)
        self.assertEqual(metrics["surface_f_scores"]["1pct"]["f_score"], 1.0)

    def test_bounded_relief_carves_only_front_of_valid_columns(self) -> None:
        volume = np.zeros((5, 10, 5), dtype=bool)
        volume[:, 1:9, :] = True
        depth = np.ones((5, 5), dtype=np.float32)
        valid = np.ones((5, 5), dtype=bool)
        carved, report = carve_front_relief(volume, depth, valid, 0.1)
        self.assertLess(np.count_nonzero(carved), np.count_nonzero(volume))
        self.assertTrue(np.all(carved[:, 8, :]))
        self.assertGreater(report["removed_voxels"], 0)


if __name__ == "__main__":
    unittest.main()
