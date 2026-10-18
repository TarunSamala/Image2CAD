import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data/phase5_normal_corroboration_v1/report.json"
DEPTH_REPORT = (
    ROOT / "dataset/phase_runs/v1/phase5_validation_metric_depth_v1/report.json"
)
NORMAL_REPORT = (
    ROOT / "dataset/phase_runs/v1/phase5_validation_surface_normal_v1/report.json"
)
VALIDATION_IDS = {"ring_004", "ring_012", "ring_020"}
TEST_IDS = {"ring_007", "ring_015", "ring_023"}


class Phase5NormalCorroborationTests(unittest.TestCase):
    def test_validation_depth_and_normal_inputs_cover_all_fifteen_views(self) -> None:
        depth = json.loads(DEPTH_REPORT.read_text(encoding="utf-8"))
        normal = json.loads(NORMAL_REPORT.read_text(encoding="utf-8"))
        self.assertEqual(depth["sample_count"], 15)
        self.assertEqual(depth["backend_count"], 3)
        self.assertTrue(depth["same_view_alignment"]["complete"])
        self.assertFalse(depth["claims"]["depth_accuracy_measured"])
        self.assertEqual(normal["sample_count"], 15)
        self.assertEqual(normal["object_count"], 3)
        self.assertFalse(normal["claims"]["normal_accuracy_measured"])
        self.assertEqual(set(normal["review_sheets"]), VALIDATION_IDS)
        for relative in normal["review_sheets"].values():
            self.assertTrue((ROOT / relative).is_file())

    def test_tracks_are_annotated_without_changing_phase5_7_or_cad(self) -> None:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertEqual(set(report["scope"]["object_ids"]), VALIDATION_IDS)
        self.assertTrue(TEST_IDS.isdisjoint(report["scope"]["object_ids"]))
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["summary"]["phase5_7_supported_count"], 736)
        self.assertEqual(report["summary"]["normal_corroborated_count"], 521)
        self.assertGreater(report["summary"]["normal_unresolved_count"], 0)
        self.assertEqual(report["summary"]["top_target_accepted_count"], 1)
        self.assertFalse(report["claims"]["normal_accuracy_validated"])
        self.assertFalse(report["claims"]["cad_fusion_authorized"])
        self.assertFalse(report["claims"]["manufacturing_accuracy_validated"])
        self.assertTrue((ROOT / report["artifacts"]["audit"]).is_file())
        for item in report["objects"]:
            self.assertTrue(item["phase5_7_count_reproduced"])
            self.assertGreater(item["normal_corroborated_count"], 0)
            self.assertGreater(item["normal_unresolved_count"], 0)


if __name__ == "__main__":
    unittest.main()
