from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from pipeline.fit_phase5_stl1_ring_bodies import run


REPO_ROOT = Path(__file__).resolve().parents[1]


class Phase5Stl1RingBodyFitTests(unittest.TestCase):
    def test_exact_profiled_ring_bodies_preserve_held_out_split(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase5_ring_body_") as temp:
            output_dir = Path(temp) / "output"
            report = run(REPO_ROOT, output_dir)

            self.assertEqual(report["summary"]["object_count"], 24)
            self.assertEqual(report["summary"]["train_scored_count"], 18)
            self.assertEqual(report["summary"]["validation_scored_count"], 3)
            self.assertEqual(report["summary"]["test_held_out_count"], 3)
            self.assertGreater(report["summary"]["train_mean_shank_iou"], 0.90)
            self.assertGreater(report["summary"]["validation_mean_shank_iou"], 0.90)
            self.assertGreater(report["summary"]["train_mean_boundary_f1"], 0.95)
            self.assertGreater(report["summary"]["validation_mean_boundary_f1"], 0.95)
            self.assertTrue(all(report["checks"].values()))

            held_out = [item for item in report["objects"] if item["split"] == "test"]
            self.assertEqual(len(held_out), 3)
            self.assertTrue(all(item["metrics"] is None for item in held_out))
            self.assertTrue(all(item["evaluation_status"] == "held_out_not_scored" for item in held_out))

            for item in report["objects"]:
                self.assertEqual(item["category"], "ring")
                self.assertTrue(item["exact_geometry"]["valid"])
                self.assertEqual(item["exact_geometry"]["solid_count"], 1)
                self.assertEqual(item["profile_control_count"], 32)
                self.assertTrue(Path(item["artifacts"]["step"]).is_file())


if __name__ == "__main__":
    unittest.main()
