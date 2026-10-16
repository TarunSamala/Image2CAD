from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from pipeline.propose_phase5_stl1_ring_parameters import run


REPO_ROOT = Path(__file__).resolve().parents[1]


class Phase5Stl1RingParameterTests(unittest.TestCase):
    def test_all_dataset_objects_are_ring_only_and_scale_normalized(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase5_stl1_ring_parameters_") as temp:
            output_dir = Path(temp) / "output"
            report = run(REPO_ROOT, output_dir)

            self.assertEqual(report["summary"]["object_count"], 24)
            self.assertEqual(report["summary"]["view_count"], 120)
            self.assertEqual(
                report["summary"]["split_counts"],
                {"train": 18, "val": 3, "test": 3},
            )
            self.assertTrue(all(report["checks"].values()))
            self.assertTrue((output_dir / "stl1_ring_parameter_audit.png").is_file())

            for item in report["objects"]:
                self.assertEqual(item["category"], "ring")
                self.assertEqual(set(item["views"]), {"front", "top", "iso", "lsv", "rsv"})
                proposal = item["normalized_parameter_proposal"]
                self.assertEqual(proposal["parameter_space"], "scale_normalized_ring_units")
                self.assertIsNone(proposal["unit"])
                self.assertIsNone(proposal["absolute_dimensions"])
                self.assertFalse(proposal["cad_authorized"])
                self.assertFalse(proposal["manufacturing_authorized"])
                self.assertGreater(proposal["shank"]["median_thickness_units"], 0.0)

            retained = json.loads(
                (REPO_ROOT / "data" / "phase5_stl1_ring_parameters_v1" / "report.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(retained["summary"], report["summary"])


if __name__ == "__main__":
    unittest.main()
