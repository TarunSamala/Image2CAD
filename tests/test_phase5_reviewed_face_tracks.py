import json
from pathlib import Path
import unittest

from pipeline.audit_phase5_reviewed_face_tracks import _review_bbox_in_track_pixels


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data/phase5_reviewed_face_tracks_v1/report.json"


class Phase5ReviewedFaceTracksTests(unittest.TestCase):
    def test_source_box_maps_through_prepared_crop_and_placement(self) -> None:
        view = {
            "source_size_wh": [100, 100],
            "prepared_size_wh": [100, 100],
            "transform": {
                "source_crop_xyxy": [10, 20, 90, 100],
                "resize_scale": 0.5,
                "placement_xywh": [5, 6, 40, 40],
            },
        }
        self.assertEqual(_review_bbox_in_track_pixels([0.2, 0.3, 0.4, 0.5], view, 100),
                         (10.0, 11.0, 20.0, 21.0))
        with self.assertRaises(ValueError):
            _review_bbox_in_track_pixels([0.5, 0, 0.5, 1], view, 100)

    def test_retained_audit_separates_face_region_from_whole_front(self) -> None:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertEqual(report["summary"]["object_count"], 3)
        self.assertEqual(report["summary"]["face_bbox_candidate_count"], 156)
        self.assertEqual(report["summary"]["top_accepted_inside_review_boxes"], 0)
        self.assertEqual(report["summary"]["iso_accepted_inside_review_boxes"], 66)
        self.assertEqual(report["summary"]["top_accepted_whole_front"], 21)
        self.assertTrue(all(report["checks"].values()))
        self.assertFalse(any(report["claims"].values()))
        self.assertTrue((ROOT / report["artifacts"]["audit"]).is_file())
        for row in report["objects"]:
            self.assertTrue(all(row["checks"].values()))
            for target in row["target_views"].values():
                self.assertEqual(target["accepted_inside_review_bbox"],
                                 len(target["accepted_target_points_xy"]))


if __name__ == "__main__":
    unittest.main()
