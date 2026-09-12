"""Tests for the model-agnostic 2D-to-3D benchmark contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from benchmark_2d_to_3d import (
    Hardware,
    _connected_body_count,
    build_plan,
    load_registry,
    rank_reports,
    validate_manifest,
)


class TwoDToThreeDBenchmarkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry_path = Path("benchmarks/2d_to_3d/models.json")
        self.registry = load_registry(self.registry_path)

    def test_registry_is_broad_unique_and_command_free(self) -> None:
        models = self.registry["models"]
        self.assertGreaterEqual(len(models), 15)
        self.assertEqual(len({model["id"] for model in models}), len(models))
        for model in models:
            self.assertNotIn("command", model)
            self.assertIn(model["adapter"], {"existing_pipeline", "external_import"})
            self.assertTrue(model["official_repo"])

    def test_four_gb_plan_keeps_large_models_remote(self) -> None:
        hardware = Hardware("RTX 3050 Laptop GPU", 4.0, 3.0, 15.0, 100.0)
        plan = build_plan(self.registry, hardware)
        statuses = {model["id"]: model["status"] for model in plan["models"]}
        self.assertEqual(statuses["image2cad_visual_hull"], "local_ready")
        self.assertEqual(statuses["triposr"], "cpu_fallback_or_remote")
        self.assertEqual(statuses["trellis2"], "remote_gpu_required")
        self.assertEqual(plan["model_count"], len(self.registry["models"]))

    def test_connected_body_count_uses_compact_vertex_union(self) -> None:
        faces = np.asarray(
            [
                [0, 1, 2],
                [2, 1, 3],
                [4, 5, 6],
            ],
            dtype=np.int64,
        )
        self.assertEqual(_connected_body_count(faces, vertex_count=7), 2)

    def test_connected_body_count_ignores_unused_vertices(self) -> None:
        faces = np.asarray([[2, 3, 4]], dtype=np.int64)
        self.assertEqual(_connected_body_count(faces, vertex_count=20), 1)

    def test_validation_requires_an_approved_existing_preview_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "triangle.obj").write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
                encoding="utf-8",
            )
            preview = np.full((32, 32, 3), 180, np.uint8)
            cv2.imwrite(str(root / "preview.png"), preview)
            manifest = root / "manifest.json"
            record = {
                "model_id": "triposr",
                "mesh_path": "triangle.obj",
                "ground_truth_kind": "none",
                "target_masks": {},
                "rendered_masks": {},
            }
            manifest.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "preview_review is missing"):
                validate_manifest(manifest, self.registry_path, require_preview_approval=True)

            record["preview_review"] = {"status": "rejected", "preview_path": "preview.png"}
            manifest.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be approved"):
                validate_manifest(manifest, self.registry_path, require_preview_approval=True)

            record["preview_review"] = {"status": "approved", "preview_path": "preview.png"}
            manifest.write_text(json.dumps(record), encoding="utf-8")
            report = validate_manifest(manifest, self.registry_path, require_preview_approval=True)
            self.assertEqual(report["preview_review"]["status"], "approved")
            self.assertTrue(Path(report["preview_review"]["preview_path"]).is_file())

    def test_mesh_and_human_mask_bundle_is_validated_independently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "cube.obj"
            mesh.write_text(
                "\n".join(
                    [
                        "v 0 0 0", "v 1 0 0", "v 1 1 0", "v 0 1 0",
                        "v 0 0 1", "v 1 0 1", "v 1 1 1", "v 0 1 1",
                        "f 1 4 3", "f 1 3 2", "f 5 6 7", "f 5 7 8",
                        "f 1 2 6", "f 1 6 5", "f 2 3 7", "f 2 7 6",
                        "f 3 4 8", "f 3 8 7", "f 4 1 5", "f 4 5 8",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            mask = np.zeros((96, 96), np.uint8)
            cv2.rectangle(mask, (18, 18), (77, 77), 255, -1)
            target = root / "target.png"
            rendered = root / "rendered.png"
            cv2.imwrite(str(target), mask)
            cv2.imwrite(str(rendered), mask)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "model_id": "triposr",
                        "input_mode": "single_image",
                        "mesh_path": "cube.obj",
                        "target_height_mm": 431.8,
                        "metric_calibrated": True,
                        "ground_truth_kind": "human_reviewed",
                        "target_masks": {"front": "target.png"},
                        "rendered_masks": {"front": "rendered.png"},
                    }
                ),
                encoding="utf-8",
            )
            report = validate_manifest(manifest, self.registry_path)
            self.assertTrue(report["validation"]["mesh_import_passed"])
            self.assertTrue(report["validation"]["topology_passed"])
            self.assertTrue(report["validation"]["visual_accuracy_validated"])
            self.assertFalse(report["validation"]["manufacturing_accuracy_validated"])
            self.assertEqual(report["summary"]["mean_silhouette_iou"], 1.0)
            self.assertEqual(report["scale"]["uniform_scale_factor"], 431.8)

    def test_machine_masks_cannot_validate_visual_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "triangle.obj"
            mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
            mask = np.full((32, 32), 255, np.uint8)
            cv2.imwrite(str(root / "target.png"), mask)
            cv2.imwrite(str(root / "rendered.png"), mask)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "model_id": "shap_e",
                        "mesh_path": "triangle.obj",
                        "ground_truth_kind": "machine_pseudo_mask",
                        "target_masks": {"front": "target.png"},
                        "rendered_masks": {"front": "rendered.png"},
                    }
                ),
                encoding="utf-8",
            )
            report = validate_manifest(manifest, self.registry_path)
            self.assertTrue(report["summary"]["numerical_visual_gate_passed"])
            self.assertFalse(report["validation"]["visual_accuracy_validated"])

    def test_ranking_does_not_claim_manufacturing_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = []
            for model_id, score in (("first", 0.81), ("second", 0.92)):
                path = root / f"{model_id}.json"
                path.write_text(
                    json.dumps(
                        {
                            "model": {"id": model_id},
                            "summary": {
                                "mean_silhouette_iou": score,
                                "principal_view_floor": score - 0.05,
                                "mean_external_boundary_f1_2px": score - 0.02,
                            },
                            "mesh": {"watertight": True, "connected_bodies": 1},
                            "human_ground_truth": False,
                        }
                    ),
                    encoding="utf-8",
                )
                reports.append(path)
            ranking = rank_reports(reports)
            self.assertEqual(ranking["ranking"][0]["model_id"], "second")
            self.assertTrue(all(not row["manufacturing_accuracy_validated"] for row in ranking["ranking"]))


if __name__ == "__main__":
    unittest.main()
