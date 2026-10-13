"""Regression gates for the representative multi-ring VGGT experiment."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import cv2

from pipeline.run_vggt_multiring_experiment import DEFAULT_OBJECTS, VIEW_ORDER


class VGGTMultiringValidationTest(unittest.TestCase):
    root = Path("dataset/phase_runs/v1/vggt_multiring_224_v1")

    @classmethod
    def setUpClass(cls) -> None:
        cls.run_report = json.loads(
            (cls.root / "run_report.json").read_text(encoding="utf-8")
        )
        cls.validation = json.loads(
            (cls.root / "validation_report.json").read_text(encoding="utf-8")
        )

    def test_default_cohort_spans_train_and_held_out_test_rings(self) -> None:
        self.assertEqual(DEFAULT_OBJECTS, ("ring_001", "ring_007", "ring_023"))
        self.assertEqual(len(VIEW_ORDER), 5)
        self.assertEqual(self.run_report["object_count"], 3)
        self.assertEqual(self.run_report["view_count"], 15)
        self.assertEqual(
            [item["split"] for item in self.run_report["objects"]],
            ["train", "test", "test"],
        )

    def test_every_dense_run_completed_below_four_gib(self) -> None:
        for item in self.run_report["objects"]:
            self.assertEqual(item["status"], "completed_machine_hypothesis")
            self.assertLess(item["peak_cuda_bytes"], 4 * 1024**3)
            report = json.loads(
                (self.root / item["object_id"] / "vggt_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["image_size"], 224)
            self.assertEqual(report["image_count"], 5)
            self.assertFalse(any(report["claims"].values()))

    def test_leave_one_view_out_gate_honestly_rejects_every_object(self) -> None:
        self.assertEqual(self.validation["accepted_object_count"], 0)
        self.assertFalse(any(self.validation["claims"].values()))
        for item in self.validation["objects"]:
            self.assertEqual(
                item["decision"], "rejected_as_cross_view_reconstruction_geometry"
            )
            self.assertLess(item["mean_leave_one_view_out_iou"], 0.80)
            detail = json.loads(
                (self.root / item["object_id"] / "cross_view_validation.json")
                .read_text(encoding="utf-8")
            )
            self.assertTrue(detail["checks"]["no_self_view_in_reprojection"])
            self.assertFalse(detail["checks"]["cross_view_exit_gate_passed"])
            self.assertEqual(len(detail["leave_one_view_out_reprojection"]["views"]), 5)
            reprojection = detail["leave_one_view_out_reprojection"]
            self.assertIn("mean_partial_surface_precision", reprojection)
            self.assertIn("mean_surface_completion_recall", reprojection)
            self.assertIn("not an isolated camera-pose score", reprojection["interpretation"]["iou"])

    def test_review_artifacts_are_readable(self) -> None:
        for object_id in DEFAULT_OBJECTS:
            for suffix in (
                "cross_view_filter_review.png",
                "cross_view_reprojection.png",
            ):
                image = cv2.imread(str(self.root / object_id / f"{object_id}_{suffix}"))
                self.assertIsNotNone(image)
                self.assertGreater(min(image.shape[:2]), 700)

    def test_object_centric_candidate_is_retained_but_not_selected(self) -> None:
        comparison_root = Path(
            "dataset/phase_runs/v1/vggt_multiring_refinement_comparison_v1"
        )
        report = json.loads(
            (comparison_root / "report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            report["decision"], "object_centric_rejected_no_cohort_improvement"
        )
        self.assertEqual(report["selected_configuration"], "prepared_v1_full_frame")
        self.assertLess(report["cohort"]["delta"], 0)
        self.assertEqual(report["cohort"]["object_centric_accepted_count"], 0)
        self.assertFalse(any(report["claims"].values()))
        image = cv2.imread(str(comparison_root / "vggt_refinement_comparison.jpg"))
        self.assertIsNotNone(image)
        self.assertGreater(image.shape[0], 2000)


if __name__ == "__main__":
    unittest.main()
