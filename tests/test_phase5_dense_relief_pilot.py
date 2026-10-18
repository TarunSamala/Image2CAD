import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
HYBRID_REPORT = (
    ROOT / "dataset/phase_runs/v1/vggt_relief_hybrid_pilot_v1/report.json"
)
PILOT_REPORT = ROOT / "data/phase5_dense_relief_pilot_v1/report.json"
TEST_IDS = {"ring_007", "ring_015", "ring_023"}


class Phase5DenseReliefPilotTests(unittest.TestCase):
    def test_hybrid_preserves_baseline_tracks_exactly(self) -> None:
        report = json.loads(HYBRID_REPORT.read_text(encoding="utf-8"))
        self.assertEqual(report["object_count"], 4)
        self.assertTrue(all(report["checks"].values()))
        self.assertTrue(TEST_IDS.isdisjoint(report["object_ids"]))
        for item in report["objects"]:
            self.assertTrue(all(item["checks"].values()))
            self.assertEqual(item["query_count_per_view"], 1024)
            self.assertFalse(item["claims"]["cad_fusion_authorized"])
            self.assertTrue((ROOT / item["artifacts"]["track_evidence"]).is_file())

    def test_training_pilot_improves_every_ring_without_cad_claim(self) -> None:
        report = json.loads(PILOT_REPORT.read_text(encoding="utf-8"))
        self.assertEqual(report["object_count"], 4)
        self.assertTrue(all(report["checks"].values()))
        self.assertTrue(
            report["summary"]["all_objects_improve_supported_spatial_coverage"]
        )
        self.assertGreater(
            report["summary"]["mean_supported_spatial_coverage_delta"], 0.0
        )
        self.assertFalse(report["claims"]["dense_correspondence_validated"])
        self.assertFalse(report["claims"]["physical_correspondence_validated"])
        self.assertFalse(report["claims"]["cad_fusion_authorized"])
        self.assertTrue((ROOT / report["artifacts"]["audit"]).is_file())
        for item in report["objects"]:
            self.assertTrue(item["baseline_gate_preserved"])
            self.assertGreater(
                item["comparison"]["supported_spatial_coverage_delta"], 0.0
            )


if __name__ == "__main__":
    unittest.main()
