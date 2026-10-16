import json
from pathlib import Path
import unittest

import numpy as np

from pipeline.evaluate_phase5_relief_tracks import (
    TRACK_SIZE,
    VIEWS,
    evaluate_track_consistency,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "data/phase5_relief_track_consistency_v1/report.json"


class Phase5ReliefTrackTests(unittest.TestCase):
    def test_perfect_reciprocal_tracks_preserve_supported_detail(self) -> None:
        query = np.asarray(
            [[120.0, 120.0], [220.0, 180.0], [310.0, 290.0]],
            dtype=np.float32,
        )
        query_points = np.stack([query for _ in VIEWS])
        tracks_xy = np.stack(
            [np.stack([query for _ in VIEWS]) for _ in VIEWS]
        )
        shape = (len(VIEWS), len(VIEWS), len(query))
        tracks = {
            "query_points_xy": query_points,
            "tracks_xy": tracks_xy,
            "visibility": np.ones(shape, np.float32),
            "confidence": np.ones(shape, np.float32),
        }
        evidence = {}
        for view in VIEWS:
            confidence = np.full((768, 768), 0.25, np.float32)
            evidence[view] = {
                "confidence": confidence,
                "normal_agreement": None,
            }
        result = evaluate_track_consistency(tracks, evidence)
        self.assertTrue(np.all(result["candidate"]))
        self.assertTrue(np.all(result["accepted_any"]))
        self.assertTrue(
            np.all(result["supported_view_count"] == len(VIEWS) - 1)
        )
        self.assertFalse(np.any(result["normal_corroborated"]))

    def test_low_target_relief_is_rejected(self) -> None:
        point = np.asarray([[[TRACK_SIZE / 2, TRACK_SIZE / 2]]] * len(VIEWS), np.float32)
        tracks_xy = np.broadcast_to(
            point[:, None, :, :], (len(VIEWS), len(VIEWS), 1, 2)
        ).copy()
        shape = (len(VIEWS), len(VIEWS), 1)
        tracks = {
            "query_points_xy": point,
            "tracks_xy": tracks_xy,
            "visibility": np.ones(shape, np.float32),
            "confidence": np.ones(shape, np.float32),
        }
        evidence = {
            view: {
                "confidence": np.full((768, 768), 0.25 if index == 0 else 0.0, np.float32),
                "normal_agreement": None,
            }
            for index, view in enumerate(VIEWS)
        }
        result = evaluate_track_consistency(tracks, evidence)
        self.assertTrue(result["candidate"][0, 0])
        self.assertFalse(result["accepted_any"][0, 0])

    def test_retained_report_is_sparse_and_does_not_authorize_cad(self) -> None:
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(report["summary"]["object_count"], 7)
        self.assertEqual(report["summary"]["view_count"], 35)
        self.assertEqual(report["summary"]["test_held_out_count"], 3)
        self.assertTrue(all(report["checks"].values()))
        self.assertGreater(
            report["summary"]["validation"]["front_supported_count"], 0
        )
        self.assertFalse(report["claims"]["dense_correspondence_validated"])
        self.assertFalse(report["claims"]["cad_fusion_authorized"])
        self.assertFalse(report["claims"]["relief_accuracy_measured"])
        test_ids = {
            item["object_id"]
            for item in report["objects"]
            if item["evaluation_status"] == "held_out_not_aggregated"
        }
        self.assertEqual(test_ids, {"ring_007", "ring_015", "ring_023"})
        for item in report["objects"]:
            for path in item["artifacts"].values():
                self.assertTrue((ROOT / path).exists())


if __name__ == "__main__":
    unittest.main()
