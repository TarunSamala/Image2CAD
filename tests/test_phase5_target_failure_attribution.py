import csv
import hashlib
import json
from pathlib import Path
import unittest

from pipeline.diagnose_phase5_target_reprojection import (
    _action_for_patch,
    _primary_blocker,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data/phase5_target_failure_attribution_v1/report.json"
VALIDATION_IDS = {"ring_004", "ring_012", "ring_020"}
TEST_IDS = {"ring_007", "ring_015", "ring_023"}


class Phase5TargetFailureAttributionTests(unittest.TestCase):
    def test_primary_blocker_uses_structural_first_order(self) -> None:
        checks = {
            "minimum_tracked_queries": False,
            "minimum_projected_faces": False,
            "minimum_projected_pixels": False,
            "minimum_target_support_fraction": False,
            "maximum_target_relief_rmse": False,
            "beats_zero_relief_baseline": False,
            "orientation_preserved": False,
        }
        self.assertEqual(
            _primary_blocker(checks), "insufficient_normal_tracks"
        )
        checks["minimum_tracked_queries"] = True
        self.assertEqual(
            _primary_blocker(checks), "no_stable_target_triangle"
        )

    def test_actions_do_not_promote_single_target_evidence(self) -> None:
        self.assertEqual(
            _action_for_patch("single_target_supported_review", set()),
            (1, "seek_second_independent_target_view"),
        )
        self.assertEqual(
            _action_for_patch(
                "target_inconsistent_reject", {"orientation_instability"}
            ),
            (2, "improve_cross_view_track_topology_on_training_split"),
        )

    def test_retained_attribution_reproduces_frozen_gate(self) -> None:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertEqual(set(report["scope"]["object_ids"]), VALIDATION_IDS)
        self.assertTrue(TEST_IDS.isdisjoint(report["scope"]["object_ids"]))
        self.assertTrue(report["scope"]["phase5_13_gate_frozen"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["summary"]["patch_count"], 40)
        self.assertEqual(report["summary"]["target_evaluation_count"], 56)
        self.assertEqual(
            report["summary"]["passed_target_evaluation_count"], 5
        )
        self.assertEqual(
            report["summary"]["primary_blocker_counts"][
                "insufficient_normal_tracks"
            ],
            12,
        )
        self.assertEqual(
            report["summary"]["primary_blocker_counts"][
                "no_relief_transfer_gain"
            ],
            12,
        )
        self.assertFalse(report["claims"]["target_gate_improved"])
        self.assertFalse(report["claims"]["surface_promotion_authorized"])
        self.assertFalse(report["claims"]["cad_fusion_authorized"])
        self.assertTrue((ROOT / report["artifacts"]["audit"]).is_file())

        queue_path = ROOT / report["artifacts"]["patch_action_queue_csv"]
        with queue_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 40)
        self.assertTrue(
            all(row["cad_fusion_authorized"] == "False" for row in rows)
        )
        self.assertTrue(
            all(row["surface_promotion_authorized"] == "False" for row in rows)
        )

        manifest_path = ROOT / report["artifacts"]["integrity_manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["files"]), 5)
        for item in manifest["files"].values():
            artifact = ROOT / item["path"]
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            self.assertEqual(digest, item["sha256"])
            self.assertEqual(artifact.stat().st_size, item["size_bytes"])


if __name__ == "__main__":
    unittest.main()
