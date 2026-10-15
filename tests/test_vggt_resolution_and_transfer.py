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
    normal_root = Path("dataset/phase_runs/v1/vggt_normal_consistency_v1")
    negative_space_root = Path("dataset/phase_runs/v1/vggt_negative_space_v1")
    camera_alignment_root = Path(
        "dataset/phase_runs/v1/vggt_source_camera_alignment_v1"
    )
    rigid_registration_root = Path(
        "dataset/phase_runs/v1/vggt_rigid_source_registration_v1"
    )
    track_evidence_root = Path(
        "dataset/phase_runs/v1/vggt_track_evidence_v1"
    )
    track_registration_root = Path(
        "dataset/phase_runs/v1/vggt_track_guided_registration_v1"
    )
    dense_track_evidence_root = Path(
        "dataset/phase_runs/v1/vggt_dense_track_evidence_v1"
    )
    dense_cycle_root = Path(
        "dataset/phase_runs/v1/vggt_dense_cycle_consistent_registration_v1"
    )
    projection_noise_root = Path(
        "dataset/phase_runs/v1/vggt_projection_noise_v1"
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
        cls.normal = json.loads(
            (cls.normal_root / "report.json").read_text(encoding="utf-8")
        )
        cls.negative_space = json.loads(
            (cls.negative_space_root / "report.json").read_text(encoding="utf-8")
        )
        cls.camera_alignment = json.loads(
            (cls.camera_alignment_root / "report.json").read_text(encoding="utf-8")
        )
        cls.rigid_registration = json.loads(
            (cls.rigid_registration_root / "report.json").read_text(
                encoding="utf-8"
            )
        )
        cls.track_evidence = json.loads(
            (cls.track_evidence_root / "report.json").read_text(encoding="utf-8")
        )
        cls.track_registration = json.loads(
            (cls.track_registration_root / "report.json").read_text(
                encoding="utf-8"
            )
        )
        cls.dense_track_evidence = json.loads(
            (cls.dense_track_evidence_root / "report.json").read_text(
                encoding="utf-8"
            )
        )
        cls.dense_cycle = json.loads(
            (cls.dense_cycle_root / "report.json").read_text(encoding="utf-8")
        )
        cls.projection_noise = json.loads(
            (cls.projection_noise_root / "report.json").read_text(
                encoding="utf-8"
            )
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

    def test_raw_normal_carving_is_rejected_without_overstating_accuracy(self) -> None:
        selection = self.normal["validation_selection"]
        self.assertEqual(
            selection["object_ids"], ["ring_004", "ring_012", "ring_020"]
        )
        self.assertEqual(selection["selected_normal_cosine_threshold"], 0.0)
        self.assertEqual(
            self.normal["decision"], "rejected_no_heldout_gain_over_signed_depth"
        )
        self.assertTrue(
            self.normal["checks"][
                "target_mask_points_and_normals_excluded_from_reconstruction"
            ]
        )
        self.assertTrue(
            self.normal["checks"]["unassessed_voxels_retained_as_uncertain"]
        )
        self.assertEqual(
            self.normal["test_evaluation"]["mean_delta_over_signed_depth"], 0.0
        )
        self.assertFalse(self.normal["checks"]["geometry_exit_gate_passed"])
        self.assertFalse(any(self.normal["claims"].values()))

    def test_structural_negative_space_tradeoff_is_rejected_as_default(self) -> None:
        selection = self.negative_space["validation_selection"]
        self.assertEqual(
            selection["object_ids"], ["ring_004", "ring_012", "ring_020"]
        )
        self.assertEqual(selection["selected_hole_margin_pixels"], 4)
        self.assertGreater(self.negative_space["validation"]["mean_iou"], 0.783)
        self.assertGreater(
            self.negative_space["test_evaluation"]["minimum_iou"], 0.54
        )
        self.assertLess(
            self.negative_space["test_evaluation"]["mean_delta_over_signed_depth"],
            0,
        )
        self.assertEqual(
            self.negative_space["decision"],
            "rejected_no_heldout_gain_over_signed_depth",
        )
        self.assertTrue(
            self.negative_space["checks"][
                "target_mask_points_and_holes_excluded_from_reconstruction"
            ]
        )
        self.assertFalse(
            self.negative_space["checks"]["geometry_exit_gate_passed"]
        )
        self.assertFalse(any(self.negative_space["claims"].values()))

    def test_rejected_constraint_previews_are_readable(self) -> None:
        for root, suffix in (
            (self.normal_root, "normal_consistency"),
            (self.negative_space_root, "negative_space"),
        ):
            for object_id in ("ring_020", "ring_015"):
                path = root / f"{object_id}_{suffix}.png"
                image = cv2.imread(str(path))
                self.assertIsNotNone(image, path)
                self.assertGreater(min(image.shape[:2]), 2400)

    def test_source_camera_image_alignment_is_rejected_without_target_leakage(self) -> None:
        selection = self.camera_alignment["validation_selection"]
        self.assertEqual(
            selection["object_ids"], ["ring_004", "ring_012", "ring_020"]
        )
        self.assertEqual(selection["selected_profile"], "identity")
        self.assertLess(
            selection["profile_candidates"]["translation_4px"]["mean_iou"],
            selection["profile_candidates"]["identity"]["mean_iou"],
        )
        self.assertEqual(
            self.camera_alignment["test_evaluation"]["object_ids"],
            ["ring_007", "ring_015", "ring_023"],
        )
        self.assertEqual(
            self.camera_alignment["test_evaluation"][
                "mean_delta_over_signed_depth"
            ],
            0.0,
        )
        self.assertEqual(
            self.camera_alignment["decision"],
            "rejected_no_consistent_heldout_gain",
        )
        self.assertTrue(
            self.camera_alignment["checks"][
                "target_mask_and_points_excluded_from_alignment_and_reconstruction"
            ]
        )
        self.assertTrue(self.camera_alignment["checks"]["target_camera_unchanged"])
        self.assertFalse(
            self.camera_alignment["checks"]["geometry_exit_gate_passed"]
        )
        self.assertFalse(any(self.camera_alignment["claims"].values()))

    def test_source_camera_alignment_previews_are_readable(self) -> None:
        for object_id in ("ring_020", "ring_015"):
            path = (
                self.camera_alignment_root
                / f"{object_id}_source_camera_alignment.png"
            )
            image = cv2.imread(str(path))
            self.assertIsNotNone(image, path)
            self.assertGreater(min(image.shape[:2]), 2400)

    def test_rigid_source_registration_is_bounded_and_rejected(self) -> None:
        selection = self.rigid_registration["validation_selection"]
        self.assertEqual(
            selection["object_ids"], ["ring_004", "ring_012", "ring_020"]
        )
        self.assertEqual(selection["selected_profile"], "identity")
        candidates = selection["profile_candidates"]
        self.assertLess(
            candidates["rigid_medium"][
                "mean_registration_distance_delta_fraction"
            ],
            0,
        )
        self.assertLess(
            candidates["rigid_medium"]["mean_iou"],
            candidates["identity"]["mean_iou"],
        )
        self.assertEqual(
            self.rigid_registration["test_evaluation"]["object_ids"],
            ["ring_007", "ring_015", "ring_023"],
        )
        self.assertEqual(
            self.rigid_registration["decision"],
            "rejected_no_consistent_heldout_gain",
        )
        checks = self.rigid_registration["checks"]
        self.assertTrue(
            checks["target_mask_points_and_camera_excluded_from_registration"]
        )
        self.assertTrue(checks["source_camera_and_points_transformed_together"])
        self.assertFalse(checks["geometry_exit_gate_passed"])
        self.assertFalse(any(self.rigid_registration["claims"].values()))

    def test_rigid_registration_previews_are_readable(self) -> None:
        paths = (
            self.rigid_registration_root
            / "ring_020_identity_registration_diagnostic.png",
            self.rigid_registration_root
            / "ring_020_rigid_medium_registration_diagnostic.png",
            self.rigid_registration_root
            / "ring_015_rigid_source_registration.png",
        )
        for path in paths:
            image = cv2.imread(str(path))
            self.assertIsNotNone(image, path)
            self.assertGreater(min(image.shape[:2]), 2400)

    def test_vggt_track_evidence_is_complete_and_non_metric(self) -> None:
        self.assertEqual(self.track_evidence["object_count"], 7)
        self.assertEqual(
            self.track_evidence["object_ids"],
            [
                "ring_004",
                "ring_012",
                "ring_020",
                "ring_001",
                "ring_007",
                "ring_015",
                "ring_023",
            ],
        )
        checks = self.track_evidence["checks"]
        self.assertTrue(checks["all_objects_completed"])
        self.assertTrue(checks["all_tracks_in_bounds_above_95_percent"])
        self.assertTrue(checks["all_source_queries_preserved"])
        self.assertLess(
            max(item["runtime"]["peak_cuda_bytes"] for item in self.track_evidence["objects"]),
            4 * 1024**3,
        )
        self.assertFalse(any(self.track_evidence["claims"].values()))

    def test_track_guided_registration_is_rejected_on_fixed_test_split(self) -> None:
        selection = self.track_registration["validation_selection"]
        self.assertEqual(
            selection["object_ids"], ["ring_004", "ring_012", "ring_020"]
        )
        self.assertEqual(selection["selected_profile"], "track_strict")
        self.assertGreater(
            self.track_registration["validation"]["mean_delta_over_signed_depth"],
            0,
        )
        self.assertEqual(
            self.track_registration["test_evaluation"]["object_ids"],
            ["ring_007", "ring_015", "ring_023"],
        )
        self.assertLess(
            self.track_registration["test_evaluation"]["mean_delta_over_signed_depth"],
            0,
        )
        self.assertEqual(
            self.track_registration["decision"],
            "rejected_no_consistent_heldout_gain",
        )
        checks = self.track_registration["checks"]
        self.assertTrue(
            checks["target_tracks_mask_points_and_camera_excluded_from_registration"]
        )
        self.assertTrue(checks["source_camera_and_points_transformed_together"])
        self.assertFalse(checks["all_test_objects_improve_over_signed_depth"])
        self.assertFalse(checks["geometry_exit_gate_passed"])
        self.assertFalse(any(self.track_registration["claims"].values()))

    def test_track_registration_previews_are_readable(self) -> None:
        paths = (
            self.track_registration_root / "ring_020_track_evidence.png",
            self.track_registration_root / "ring_015_track_evidence.png",
            self.track_registration_root / "ring_020_track_guided_registration.png",
            self.track_registration_root / "ring_007_track_guided_registration.png",
        )
        for path in paths:
            image = cv2.imread(str(path))
            self.assertIsNotNone(image, path)
            self.assertGreater(min(image.shape[:2]), 1000)

    def test_dense_cycle_registration_selects_identity_on_validation(self) -> None:
        self.assertEqual(self.dense_track_evidence["object_count"], 7)
        self.assertEqual(
            self.dense_track_evidence["objects"][0]["query_count_per_view"],
            384,
        )
        self.assertLess(
            max(
                item["runtime"]["peak_cuda_bytes"]
                for item in self.dense_track_evidence["objects"]
            ),
            4 * 1024**3,
        )
        selection = self.dense_cycle["validation_selection"]
        self.assertEqual(
            selection["object_ids"], ["ring_004", "ring_012", "ring_020"]
        )
        self.assertEqual(selection["selected_profile"], "identity")
        self.assertLess(
            selection["profile_candidates"]["cycle_tight"][
                "mean_delta_over_signed_depth"
            ],
            0,
        )
        self.assertEqual(
            self.dense_cycle["test_evaluation"]["object_ids"],
            ["ring_007", "ring_015", "ring_023"],
        )
        self.assertEqual(
            self.dense_cycle["test_evaluation"]["mean_delta_over_signed_depth"],
            0.0,
        )
        self.assertEqual(
            self.dense_cycle["decision"],
            "rejected_no_regression_free_heldout_gain",
        )
        self.assertFalse(self.dense_cycle["checks"]["reciprocal_cycle_filter_enabled"])
        self.assertFalse(self.dense_cycle["checks"]["geometry_exit_gate_passed"])
        self.assertFalse(any(self.dense_cycle["claims"].values()))

    def test_dense_cycle_previews_are_readable(self) -> None:
        for object_id in ("ring_020", "ring_015"):
            path = self.dense_cycle_root / f"{object_id}_cycle_registration.png"
            image = cv2.imread(str(path))
            self.assertIsNotNone(image, path)
            self.assertGreater(min(image.shape[:2]), 1000)

    def test_projection_noise_cleanup_is_renderer_only_and_transfers(self) -> None:
        selection = self.projection_noise["validation_selection"]
        self.assertEqual(
            selection["object_ids"], ["ring_004", "ring_012", "ring_020"]
        )
        self.assertEqual(
            selection["selected_profile"], "close3_components_ge_4"
        )
        self.assertGreater(
            self.projection_noise["validation"]["mean_delta_over_raw"], 0
        )
        test = self.projection_noise["test_evaluation"]
        self.assertEqual(
            test["object_ids"], ["ring_007", "ring_015", "ring_023"]
        )
        self.assertGreater(test["mean_delta_over_raw"], 0.007)
        self.assertEqual(test["regressed_view_count"], 0)
        self.assertEqual(test["minimum_iou"], 0.529146)
        checks = self.projection_noise["checks"]
        self.assertTrue(checks["selection_uses_validation_split_only"])
        self.assertTrue(checks["target_mask_not_used_for_cleanup_decision"])
        self.assertTrue(checks["voxel_geometry_unchanged"])
        self.assertTrue(checks["all_test_objects_improve_over_raw"])
        self.assertTrue(checks["no_test_view_regresses"])
        self.assertTrue(checks["structural_hole_guard_passed"])
        self.assertFalse(checks["geometry_exit_gate_passed"])
        self.assertEqual(
            self.projection_noise["decision"],
            "retained_as_projection_renderer_only",
        )
        self.assertFalse(any(self.projection_noise["claims"].values()))

    def test_projection_noise_previews_are_readable(self) -> None:
        for object_id in (
            "ring_004",
            "ring_020",
            "ring_007",
            "ring_015",
            "ring_023",
        ):
            path = self.projection_noise_root / f"{object_id}_projection_noise.png"
            image = cv2.imread(str(path))
            self.assertIsNotNone(image, path)
            self.assertGreater(min(image.shape[:2]), 2800)


if __name__ == "__main__":
    unittest.main()
