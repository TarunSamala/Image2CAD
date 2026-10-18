import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRAIN_REPORT = ROOT / "data/phase5_dense_relief_train_v1/report.json"
VALIDATION_REPORT = ROOT / "data/phase5_dense_relief_validation_v1/report.json"
TRAIN_HYBRID_REPORT = (
    ROOT / "dataset/phase_runs/v1/vggt_relief_hybrid_train_v1/report.json"
)
VALIDATION_HYBRID_REPORT = (
    ROOT / "dataset/phase_runs/v1/vggt_relief_hybrid_validation_v1/report.json"
)
TEST_IDS = {"ring_007", "ring_015", "ring_023"}
VALIDATION_IDS = {"ring_004", "ring_012", "ring_020"}


class Phase5DenseReliefValidationTests(unittest.TestCase):
    def test_complete_training_cohort_freezes_a_no_regression_policy(self) -> None:
        report = json.loads(TRAIN_REPORT.read_text(encoding="utf-8"))
        hybrid = json.loads(TRAIN_HYBRID_REPORT.read_text(encoding="utf-8"))
        self.assertEqual(report["object_count"], 18)
        self.assertEqual(hybrid["object_count"], 18)
        self.assertTrue(all(report["checks"].values()))
        self.assertTrue(all(hybrid["checks"].values()))
        self.assertTrue(TEST_IDS.isdisjoint(report["object_ids"]))
        self.assertTrue(VALIDATION_IDS.isdisjoint(report["object_ids"]))
        self.assertTrue(
            report["summary"]["all_objects_improve_supported_spatial_coverage"]
        )
        self.assertEqual(
            report["decision"],
            "freeze_training_policy_then_evaluate_validation",
        )
        self.assertFalse(report["claims"]["cad_fusion_authorized"])
        self.assertTrue((ROOT / report["artifacts"]["audit"]).is_file())

    def test_frozen_policy_improves_each_validation_ring_without_test_leakage(self) -> None:
        train = json.loads(TRAIN_REPORT.read_text(encoding="utf-8"))
        report = json.loads(VALIDATION_REPORT.read_text(encoding="utf-8"))
        hybrid = json.loads(VALIDATION_HYBRID_REPORT.read_text(encoding="utf-8"))
        self.assertEqual(set(report["object_ids"]), VALIDATION_IDS)
        self.assertEqual(set(hybrid["object_ids"]), VALIDATION_IDS)
        self.assertTrue(all(report["checks"].values()))
        self.assertTrue(all(hybrid["checks"].values()))
        self.assertTrue(TEST_IDS.isdisjoint(report["object_ids"]))
        self.assertEqual(report["frozen_thresholds"], train["frozen_thresholds"])
        self.assertTrue(
            report["summary"]["all_objects_improve_supported_spatial_coverage"]
        )
        self.assertGreater(
            report["summary"]["mean_supported_spatial_coverage_delta"], 0.0
        )
        self.assertEqual(
            report["decision"],
            "retain_validation_improvement_test_still_held_out",
        )
        self.assertFalse(report["claims"]["dense_correspondence_validated"])
        self.assertFalse(report["claims"]["physical_correspondence_validated"])
        self.assertFalse(report["claims"]["cad_fusion_authorized"])
        self.assertTrue((ROOT / report["artifacts"]["audit"]).is_file())
        for item in report["objects"]:
            self.assertEqual(item["split"], "val")
            self.assertTrue(item["baseline_gate_preserved"])
            self.assertGreater(
                item["comparison"]["supported_spatial_coverage_delta"], 0.0
            )


if __name__ == "__main__":
    unittest.main()
