"""Regression gates for VGGT resolution and view-transfer experiments."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import cv2


class VGGTResolutionAndTransferTest(unittest.TestCase):
    resolution_root = Path(
        "dataset/phase_runs/v1/vggt_resolution_ladder_v1"
    )
    transfer_root = Path("dataset/phase_runs/v1/vggt_view_transfer_v1")
    completion_root = Path(
        "dataset/phase_runs/v1/vggt_visual_hull_completion_v1"
    )
    surface_root = Path("dataset/phase_runs/v1/vggt_surface_aware_hull_v1")
    signed_depth_root = Path(
        "dataset/phase_runs/v1/vggt_signed_depth_carving_v1"
    )

    @classmethod
    def setUpClass(cls) -> None:
        cls.resolution = json.loads(
            (cls.resolution_root / "report.json").read_text(encoding="utf-8")
        )
        cls.transfer = json.loads(
            (cls.transfer_root / "report.json").read_text(encoding="utf-8")
        )
        cls.completion = json.loads(
            (cls.completion_root / "report.json").read_text(encoding="utf-8")
        )
        cls.surface = json.loads(
            (cls.surface_root / "report.json").read_text(encoding="utf-8")
        )
        cls.signed_depth = json.loads(
            (cls.signed_depth_root / "report.json").read_text(encoding="utf-8")
        )

    def test_resolution_selection_uses_heldout_results_and_stays_honest(self) -> None:
        self.assertEqual(self.resolution["selected_resolution"], 518)
        self.assertEqual(
            self.resolution["selected_configuration"],
            "vggt_resolution_518_v1",
        )
        selection = self.resolution["validation_selection"]
        self.assertEqual(
            selection["object_ids"], ["ring_004", "ring_012", "ring_020"]
        )
        self.assertEqual(selection["selected_resolution"], 518)
        self.assertGreater(selection["selected_mean_iou"], 0.48)
        self.assertTrue(
            self.resolution["checks"]["selection_uses_validation_split_only"]
        )
        self.assertTrue(
            self.resolution["checks"]["all_selection_objects_are_validation_split"]
        )
        self.assertGreater(
            self.resolution["heldout_change_from_224"]["mean_iou"], 0.30
        )
        self.assertTrue(self.resolution["checks"]["selected_run_below_four_gib"])
        self.assertFalse(
            self.resolution["checks"]["all_selected_objects_pass_geometry_gate"]
        )
        self.assertFalse(any(self.resolution["claims"].values()))

    def test_518_run_completes_all_objects_after_cuda_cleanup(self) -> None:
        report = json.loads(
            Path("dataset/phase_runs/v1/vggt_resolution_518_v1/run_report.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(report["object_count"], 3)
        self.assertTrue(
            all(
                item["status"] == "completed_machine_hypothesis"
                for item in report["objects"]
            )
        )
        self.assertLess(
            max(item["peak_cuda_bytes"] for item in report["objects"]),
            4 * 1024**3,
        )

    def test_view_policy_is_leak_controlled_and_rejected(self) -> None:
        self.assertEqual(self.transfer["training_object"], "ring_001")
        self.assertTrue(
            self.transfer["checks"]["policy_selected_on_training_object_only"]
        )
        self.assertFalse(
            self.transfer["checks"]["heldout_target_masks_used_for_selection"]
        )
        self.assertTrue(
            self.transfer["checks"]["every_policy_excludes_target_view"]
        )
        self.assertLess(self.transfer["heldout"]["mean_iou_delta"], 0)
        self.assertEqual(
            self.transfer["decision"],
            "rejected_as_nontransferable_view_selection",
        )
        self.assertFalse(any(self.transfer["claims"].values()))

    def test_resolution_and_transfer_previews_are_readable(self) -> None:
        for path in (
            self.resolution_root / "vggt_resolution_ladder.png",
            self.resolution_root / "vggt_224_vs_518_reprojection.jpg",
            self.transfer_root / "ring_007" / "ring_007_view_transfer.png",
        ):
            image = cv2.imread(str(path))
            self.assertIsNotNone(image, path)
            self.assertGreater(min(image.shape[:2]), 700)

    def test_visual_hull_selection_is_validation_only_and_improves_completion(self) -> None:
        selection = self.completion["validation_selection"]
        self.assertEqual(
            selection["object_ids"], ["ring_004", "ring_012", "ring_020"]
        )
        self.assertEqual(selection["selected_minimum_source_views"], 4)
        self.assertTrue(
            self.completion["checks"]["selection_uses_validation_split_only"]
        )
        self.assertTrue(
            self.completion["checks"]["target_silhouette_excluded_from_each_hull"]
        )
        self.assertTrue(
            self.completion["checks"]["all_objects_improve_over_point_baseline"]
        )
        self.assertGreater(self.completion["validation"]["mean_iou"], 0.73)
        self.assertEqual(
            self.completion["test_evaluation"]["object_ids"],
            ["ring_007", "ring_015", "ring_023"],
        )
        self.assertGreater(self.completion["test_evaluation"]["mean_iou"], 0.73)
        self.assertGreater(
            self.completion["test_evaluation"]["mean_iou_delta_over_point_baseline"],
            0.24,
        )
        self.assertFalse(self.completion["checks"]["geometry_exit_gate_passed"])
        self.assertFalse(any(self.completion["claims"].values()))

    def test_visual_hull_previews_are_readable(self) -> None:
        for object_id in ("ring_004", "ring_020", "ring_007", "ring_015", "ring_023"):
            path = self.completion_root / f"{object_id}_visual_hull_completion.png"
            image = cv2.imread(str(path))
            self.assertIsNotNone(image, path)
            self.assertGreater(min(image.shape[:2]), 1800)

    def test_surface_aware_hull_is_validation_selected_and_improves_test(self) -> None:
        selection = self.surface["validation_selection"]
        self.assertEqual(
            selection["object_ids"], ["ring_004", "ring_012", "ring_020"]
        )
        self.assertEqual(selection["selected_surface_distance_fraction"], 0.04)
        self.assertTrue(
            self.surface["checks"]["selection_uses_validation_split_only"]
        )
        self.assertTrue(
            self.surface["checks"][
                "target_mask_and_points_excluded_from_reconstruction"
            ]
        )
        self.assertGreater(self.surface["validation"]["mean_iou"], 0.75)
        self.assertGreater(self.surface["validation"]["minimum_iou"], 0.56)
        self.assertEqual(
            self.surface["test_evaluation"]["object_ids"],
            ["ring_007", "ring_015", "ring_023"],
        )
        self.assertGreater(self.surface["test_evaluation"]["mean_iou"], 0.75)
        self.assertGreater(self.surface["test_evaluation"]["minimum_iou"], 0.50)
        self.assertGreater(
            self.surface["test_evaluation"]["mean_delta_over_visual_hull"], 0.01
        )
        self.assertFalse(self.surface["checks"]["geometry_exit_gate_passed"])
        self.assertFalse(any(self.surface["claims"].values()))

    def test_surface_aware_previews_are_readable(self) -> None:
        for object_id in ("ring_004", "ring_020", "ring_007", "ring_015", "ring_023"):
            path = self.surface_root / f"{object_id}_surface_aware_hull.png"
            image = cv2.imread(str(path))
            self.assertIsNotNone(image, path)
            self.assertGreater(min(image.shape[:2]), 2400)

    def test_signed_depth_is_validation_selected_and_improves_full_test(self) -> None:
        selection = self.signed_depth["validation_selection"]
        self.assertEqual(
            selection["object_ids"], ["ring_004", "ring_012", "ring_020"]
        )
        self.assertEqual(selection["selected_depth_tolerance_fraction"], 0.0025)
        self.assertEqual(selection["selected_allowed_source_contradictions"], 1)
        self.assertTrue(
            self.signed_depth["checks"]["selection_uses_validation_split_only"]
        )
        self.assertTrue(
            self.signed_depth["checks"][
                "target_mask_and_points_excluded_from_reconstruction"
            ]
        )
        self.assertTrue(
            self.signed_depth["checks"][
                "all_test_objects_improve_over_surface_aware_hull"
            ]
        )
        self.assertGreater(self.signed_depth["validation"]["mean_iou"], 0.78)
        self.assertGreater(self.signed_depth["validation"]["minimum_iou"], 0.57)
        self.assertEqual(
            self.signed_depth["test_evaluation"]["object_ids"],
            ["ring_007", "ring_015", "ring_023"],
        )
        self.assertGreater(self.signed_depth["test_evaluation"]["mean_iou"], 0.78)
        self.assertGreater(
            self.signed_depth["test_evaluation"]["minimum_iou"], 0.52
        )
        self.assertGreater(
            self.signed_depth["test_evaluation"][
                "mean_delta_over_surface_aware_hull"
            ],
            0.02,
        )
        self.assertFalse(
            self.signed_depth["checks"]["geometry_exit_gate_passed"]
        )
        self.assertFalse(any(self.signed_depth["claims"].values()))

    def test_signed_depth_previews_are_readable(self) -> None:
        for object_id in ("ring_004", "ring_020", "ring_007", "ring_015", "ring_023"):
            path = self.signed_depth_root / f"{object_id}_signed_depth_carving.png"
            image = cv2.imread(str(path))
            self.assertIsNotNone(image, path)
            self.assertGreater(min(image.shape[:2]), 2400)


if __name__ == "__main__":
    unittest.main()
