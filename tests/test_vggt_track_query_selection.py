import unittest

import cv2
import numpy as np

from pipeline.extract_vggt_track_evidence import (
    IMAGE_SIZE,
    TEST_OBJECTS,
    TRAIN_VALIDATION_OBJECTS,
    _select_query_points,
    _select_relief_hybrid_query_points,
    _select_relief_stratified_query_points,
)


class VGGTTrackQuerySelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
        cv2.circle(
            self.mask,
            (IMAGE_SIZE // 2, IMAGE_SIZE // 2),
            205,
            1,
            -1,
        )
        yy, xx = np.mgrid[:IMAGE_SIZE, :IMAGE_SIZE]
        image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
        image[..., 0] = np.uint8((xx * 3 + yy) % 255)
        image[..., 1] = np.uint8((xx + yy * 2) % 255)
        image[..., 2] = np.uint8((xx * 2 + yy * 3) % 255)
        self.image = image
        confidence = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.float32)
        relief = (
            self.mask.astype(bool)
            & (xx > IMAGE_SIZE // 3)
            & (xx < IMAGE_SIZE * 2 // 3)
            & (yy > IMAGE_SIZE // 3)
            & (yy < IMAGE_SIZE * 2 // 3)
        )
        confidence[relief] = 0.75
        self.confidence = confidence

    def test_selection_is_deterministic_unique_and_inside_mask(self) -> None:
        points_a, regions_a = _select_relief_stratified_query_points(
            self.image, self.mask, self.confidence, 512
        )
        points_b, regions_b = _select_relief_stratified_query_points(
            self.image, self.mask, self.confidence, 512
        )
        np.testing.assert_array_equal(points_a, points_b)
        np.testing.assert_array_equal(regions_a, regions_b)
        self.assertEqual(points_a.shape, (512, 2))
        self.assertEqual(len(np.unique(points_a, axis=0)), 512)
        xy = np.rint(points_a).astype(np.int32)
        self.assertTrue(np.all(self.mask[xy[:, 1], xy[:, 0]] > 0))

    def test_selection_retains_each_declared_spatial_region(self) -> None:
        _, regions = _select_relief_stratified_query_points(
            self.image, self.mask, self.confidence, 512
        )
        for region in range(4):
            self.assertGreater(np.count_nonzero(regions == region), 0)
        self.assertGreater(
            np.count_nonzero(regions == 0), np.count_nonzero(regions == 1)
        )

    def test_train_validation_cohort_excludes_fixed_test_rings(self) -> None:
        self.assertEqual(len(TRAIN_VALIDATION_OBJECTS), 21)
        self.assertTrue(set(TEST_OBJECTS).isdisjoint(TRAIN_VALIDATION_OBJECTS))
        self.assertEqual(set(TEST_OBJECTS), {"ring_007", "ring_015", "ring_023"})

    def test_hybrid_preserves_every_frozen_baseline_query(self) -> None:
        baseline = _select_query_points(self.image, self.mask, 384)
        hybrid, regions = _select_relief_hybrid_query_points(
            self.image, self.mask, self.confidence, 1024
        )
        np.testing.assert_array_equal(hybrid[:384], baseline)
        self.assertTrue(np.all(regions[:384] == 4))
        self.assertEqual(len(np.unique(hybrid, axis=0)), 1024)


if __name__ == "__main__":
    unittest.main()
