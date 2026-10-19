import json
from pathlib import Path
import unittest

from pipeline.evaluate_phase5_validation_track_topology import _verify_frozen_training
from pipeline.extract_vggt_track_evidence import VALIDATION_OBJECTS


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data/phase5_validation_track_topology_v1/report.json"


class Phase5ValidationTrackTopologyTests(unittest.TestCase):
    def test_frozen_training_artifacts_match_manifest(self) -> None:
        training = _verify_frozen_training(ROOT)
        self.assertEqual(
            training["summary"]["selected_training_policy"],
            "phase5_7_hybrid",
        )

    def test_retained_validation_report_keeps_split_and_claims(self) -> None:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertEqual(
            set(report["scope"]["object_ids"]), set(VALIDATION_OBJECTS)
        )
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["summary"]["object_count"], 3)
        self.assertEqual(
            report["summary"]["baseline_orientation_preserved_triangle_count"],
            363,
        )
        self.assertEqual(
            report["summary"]["hybrid_orientation_preserved_triangle_count"],
            1017,
        )
        self.assertEqual(report["summary"]["improved_object_count"], 3)
        self.assertEqual(report["summary"]["regressed_object_count"], 0)
        self.assertEqual(
            report["summary"]["target_views"]["top"]["stable_triangle_count"],
            0,
        )
        self.assertFalse(
            report["summary"]["complete_target_view_coverage_available"]
        )
        self.assertFalse(report["claims"]["independent_surface_gate_improved"])
        self.assertFalse(report["claims"]["cad_fusion_authorized"])
        self.assertTrue((ROOT / report["artifacts"]["audit"]).is_file())
        for item in report["objects"]:
            self.assertEqual(item["split"], "val")
            self.assertGreater(item["orientation_preserved_triangle_delta"], 0)
            self.assertTrue(item["baseline_query_prefix_reproduced"])
            self.assertTrue((ROOT / item["artifacts"]["audit"]).is_file())


if __name__ == "__main__":
    unittest.main()
