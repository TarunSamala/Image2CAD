import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "data/phase5_full_relief_track_consistency_v1/report.json"
PHASE55_REPORT_PATH = ROOT / "data/phase5_relief_track_consistency_v1/report.json"
TRACK_REPORT_PATH = (
    ROOT
    / "dataset/phase_runs/v1/vggt_full_stl1_track_evidence_v1/report.json"
)


class Phase5FullReliefTrackTests(unittest.TestCase):
    def test_full_track_source_covers_all_stl1_rings(self) -> None:
        report = json.loads(TRACK_REPORT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(report["object_count"], 24)
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(
            {item["object_id"] for item in report["objects"]},
            {f"ring_{index:03d}" for index in range(1, 25)},
        )

    def test_frozen_gate_is_evaluated_across_fixed_split(self) -> None:
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        phase55 = json.loads(PHASE55_REPORT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(report["summary"]["object_count"], 24)
        self.assertEqual(report["summary"]["view_count"], 120)
        self.assertEqual(report["summary"]["train"]["object_count"], 18)
        self.assertEqual(report["summary"]["validation"]["object_count"], 3)
        self.assertEqual(report["summary"]["test_held_out_count"], 3)
        self.assertEqual(
            report["scope"]["thresholds"], phase55["scope"]["thresholds"]
        )
        self.assertTrue(all(report["checks"].values()))

    def test_test_split_is_generated_but_never_aggregated_or_approved(self) -> None:
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        test = [item for item in report["objects"] if item["split"] == "test"]
        self.assertEqual(
            {item["object_id"] for item in test},
            {"ring_007", "ring_015", "ring_023"},
        )
        self.assertTrue(
            all(item["evaluation_status"] == "held_out_not_aggregated" for item in test)
        )
        self.assertFalse(report["claims"]["dense_correspondence_validated"])
        self.assertFalse(report["claims"]["physical_correspondence_validated"])
        self.assertFalse(report["claims"]["cad_fusion_authorized"])
        self.assertFalse(report["claims"]["metric_scale_validated"])
        self.assertFalse(report["claims"]["manufacturing_accuracy_validated"])

    def test_every_reported_artifact_exists(self) -> None:
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        for path in report["artifacts"].values():
            self.assertTrue((ROOT / path).is_file(), path)
        for item in report["objects"]:
            for path in item["artifacts"].values():
                self.assertTrue((ROOT / path).is_file(), path)


if __name__ == "__main__":
    unittest.main()
