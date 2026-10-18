import json
from pathlib import Path
import unittest

import numpy as np

from pipeline.build_phase5_local_relief_surfaces import (
    _rasterize_triangles,
    _triangle_area,
    _triangulate,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data/phase5_local_relief_surfaces_v1/report.json"
VALIDATION_IDS = {"ring_004", "ring_012", "ring_020"}
TEST_IDS = {"ring_007", "ring_015", "ring_023"}


class Phase5LocalReliefSurfaceTests(unittest.TestCase):
    def test_square_triangulation_and_rasterization_are_finite(self) -> None:
        points = np.asarray(
            [[2.0, 2.0], [12.0, 2.0], [12.0, 12.0], [2.0, 12.0]],
            np.float32,
        )
        values = np.asarray([-0.5, 0.0, 0.5, 0.25], np.float32)
        faces = _triangulate(points)
        self.assertEqual(faces.shape, (2, 3))
        self.assertTrue(
            all(_triangle_area(points[face]) >= 1.0 for face in faces)
        )
        rendered, mask = _rasterize_triangles(
            points, values, faces, (16, 16)
        )
        self.assertGreater(int(np.count_nonzero(mask)), 0)
        self.assertTrue(np.all(np.isfinite(rendered[mask])))

    def test_retained_surface_trials_are_validation_only_and_reversible(self) -> None:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertEqual(set(report["scope"]["object_ids"]), VALIDATION_IDS)
        self.assertTrue(TEST_IDS.isdisjoint(report["scope"]["object_ids"]))
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["summary"]["eligible_patch_count"], 40)
        self.assertEqual(report["summary"]["generated_mesh_count"], 40)
        self.assertEqual(report["summary"]["failed_mesh_count"], 0)
        self.assertGreater(report["summary"]["vertex_count"], 0)
        self.assertGreater(report["summary"]["face_count"], 0)
        self.assertFalse(report["claims"]["physical_relief_accuracy_validated"])
        self.assertFalse(report["claims"]["semantic_accuracy_validated"])
        self.assertFalse(report["claims"]["metric_scale_validated"])
        self.assertFalse(report["claims"]["cad_fusion_authorized"])
        self.assertFalse(report["claims"]["manufacturing_accuracy_validated"])
        self.assertTrue((ROOT / report["artifacts"]["audit"]).is_file())

        for item in report["objects"]:
            self.assertTrue(item["phase5_11_eligible_count_reproduced"])
            self.assertTrue(item["all_generated_geometry_finite"])
            self.assertEqual(
                item["eligible_patch_count"], item["generated_mesh_count"]
            )
            object_report = json.loads(
                (ROOT / item["artifacts"]["report"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(object_report["claims"]["cad_fusion_authorized"])
            self.assertEqual(
                object_report["eligible_patch_count"],
                len(object_report["trials"]),
            )
            for trial in object_report["trials"]:
                self.assertTrue(trial["mesh_generated"])
                self.assertTrue(trial["finite_geometry"])
                self.assertEqual(trial["topology"], "open_surface_not_solid")
                self.assertFalse(trial["semantic_claim_authorized"])
                self.assertFalse(trial["cad_fusion_authorized"])
                artifact = ROOT / trial["artifact"]["obj"]
                text = artifact.read_text(encoding="utf-8")
                self.assertIn("not millimetres", text)
                self.assertIn("open evidence surface", text.lower())
                self.assertIn("\nv ", text)
                self.assertIn("\nf ", text)


if __name__ == "__main__":
    unittest.main()
