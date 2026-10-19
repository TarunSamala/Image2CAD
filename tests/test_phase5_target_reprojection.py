import json
from pathlib import Path
import unittest

import numpy as np

from pipeline.validate_phase5_local_relief_reprojection import (
    _signed_triangle_area,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data/phase5_target_reprojection_v1/report.json"
VALIDATION_IDS = {"ring_004", "ring_012", "ring_020"}
TEST_IDS = {"ring_007", "ring_015", "ring_023"}


class Phase5TargetReprojectionTests(unittest.TestCase):
    def test_signed_area_preserves_orientation_information(self) -> None:
        counterclockwise = np.asarray(
            [[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]], np.float32
        )
        clockwise = counterclockwise[[0, 2, 1]]
        self.assertGreater(_signed_triangle_area(counterclockwise), 0.0)
        self.assertLess(_signed_triangle_area(clockwise), 0.0)

    def test_retained_report_is_independent_validation_only(self) -> None:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertEqual(set(report["scope"]["object_ids"]), VALIDATION_IDS)
        self.assertTrue(TEST_IDS.isdisjoint(report["scope"]["object_ids"]))
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["summary"]["eligible_patch_count"], 40)
        self.assertEqual(report["summary"]["target_evaluation_count"], 56)
        self.assertEqual(
            report["summary"]["passed_target_evaluation_count"], 5
        )
        self.assertEqual(
            report["summary"]["multi_target_supported_patch_count"], 0
        )
        self.assertEqual(
            report["summary"]["single_target_supported_patch_count"], 5
        )
        self.assertEqual(
            report["summary"]["target_inconsistent_patch_count"], 26
        )
        self.assertEqual(report["summary"]["unresolved_patch_count"], 9)
        self.assertTrue(
            report["claims"]["independent_target_image_evidence_evaluated"]
        )
        self.assertFalse(report["claims"]["surface_promotion_authorized"])
        self.assertFalse(report["claims"]["cad_fusion_authorized"])
        self.assertFalse(
            report["claims"]["manufacturing_accuracy_validated"]
        )
        self.assertTrue((ROOT / report["artifacts"]["audit"]).is_file())

    def test_patch_states_partition_every_eligible_surface(self) -> None:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        for item in report["objects"]:
            self.assertTrue(item["phase5_12_eligible_count_reproduced"])
            object_report = json.loads(
                (ROOT / item["artifacts"]["report"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                object_report["eligible_patch_count"],
                len(object_report["patches"]),
            )
            state_count = sum(
                object_report[key]
                for key in (
                    "multi_target_supported_patch_count",
                    "single_target_supported_patch_count",
                    "target_inconsistent_patch_count",
                    "unresolved_patch_count",
                )
            )
            self.assertEqual(state_count, object_report["eligible_patch_count"])
            self.assertFalse(
                object_report["claims"]["surface_promotion_authorized"]
            )
            self.assertTrue((ROOT / item["artifacts"]["audit"]).is_file())
            for patch in object_report["patches"]:
                self.assertFalse(patch["surface_promotion_authorized"])
                self.assertFalse(patch["cad_fusion_authorized"])
                for evaluation in patch["target_evaluations"]:
                    if evaluation["independent_target_gate_passed"]:
                        self.assertTrue(all(evaluation["checks"].values()))


if __name__ == "__main__":
    unittest.main()
