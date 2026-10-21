import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pipeline.build_phase5_training_observability_review import _checked_bbox, run
from pipeline.extract_vggt_track_evidence import TRAIN_OBJECTS


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data/phase5_training_observability_review_v1/report.json"


class Phase5TrainingObservabilityReviewTests(unittest.TestCase):
    def test_bbox_requires_valid_normalized_order(self) -> None:
        self.assertEqual(_checked_bbox([0.2, 0.3, 0.6, 0.8]), (0.2, 0.3, 0.6, 0.8))
        for invalid in ([0.5, 0.1, 0.5, 0.9], [-0.1, 0, 0.5, 0.5], [0, 0, 1.2, 1]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _checked_bbox(invalid)

    def test_review_is_training_only_and_not_track_truth(self) -> None:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertEqual(report["summary"]["object_count"], 3)
        self.assertEqual(report["summary"]["top_without_recognizable_face_detail"], 3)
        self.assertEqual(report["summary"]["iso_with_partial_recognizable_face_detail"], 3)
        self.assertTrue(report["summary"]["top_tracks_can_exist_without_reviewed_face_detail"])
        self.assertTrue(set(report["scope"]["object_ids"]).issubset(TRAIN_OBJECTS))
        self.assertTrue(all(report["checks"].values()))
        self.assertFalse(any(report["claims"].values()))
        self.assertTrue((ROOT / report["artifacts"]["preview"]).is_file())
        for row in report["observations"]:
            self.assertEqual(row["track_count_scope"], "whole front relief query set, not the annotated face bbox")
            self.assertEqual(set(row["target_views"]), {"top", "iso"})
            self.assertEqual(set(row["sources"]), {"front", "top", "iso"})

    def test_rebuild_rechecks_phase1_source_hashes(self) -> None:
        with TemporaryDirectory() as temporary:
            report = run(ROOT, Path(temporary))
            self.assertEqual(report["summary"]["object_count"], 3)
            self.assertTrue(report["checks"]["source_files_match_phase1"])


if __name__ == "__main__":
    unittest.main()
