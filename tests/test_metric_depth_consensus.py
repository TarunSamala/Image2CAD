"""Contract tests for the image-only metric-depth consensus experiment."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from pipeline.infer_metric_depth_backend import BACKEND_MODELS, sha256
from pipeline.run_metric_depth_consensus import (
    aggregate,
    build_input_manifest,
    colorize_depth,
    colorize_magnitude,
)


class MetricDepthConsensusTest(unittest.TestCase):
    root = Path.cwd()

    def test_depth_and_disagreement_legends_have_distinct_orientation(self) -> None:
        values = np.asarray([[0.0, 1.0]], dtype=np.float32)
        mask = np.ones((1, 2), dtype=bool)
        depth = colorize_depth(values, mask)
        disagreement = colorize_magnitude(values, mask)
        self.assertGreater(int(depth[0, 0, 2]), int(depth[0, 0, 0]))
        self.assertGreater(int(depth[0, 1, 0]), int(depth[0, 1, 2]))
        self.assertGreater(
            int(disagreement[0, 0, 0]), int(disagreement[0, 0, 2])
        )
        self.assertGreater(
            int(disagreement[0, 1, 2]), int(disagreement[0, 1, 0])
        )

    def test_real_input_manifest_selects_ring01_and_split_representatives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "experiment"
            manifest = build_input_manifest(
                self.root, output, ("ring_001", "ring_004", "ring_007")
            )
            self.assertEqual(manifest["sample_count"], 20)
            self.assertEqual(manifest["object_count"], 4)
            self.assertEqual(
                {sample["split"] for sample in manifest["samples"]},
                {"reference", "train", "val", "test"},
            )
            self.assertTrue(
                all(sample["physical_scale"] is None for sample in manifest["samples"])
            )
            for sample in manifest["samples"]:
                self.assertTrue((self.root / sample["image_path"]).is_file())
                self.assertTrue((self.root / sample["mask_path"]).is_file())

    def test_validation_manifest_can_exclude_ring01_and_test_objects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "experiment"
            manifest = build_input_manifest(
                self.root,
                output,
                ("ring_004", "ring_012", "ring_020"),
                include_ring01_reference=False,
            )
            self.assertEqual(manifest["sample_count"], 15)
            self.assertEqual(manifest["object_count"], 3)
            self.assertFalse(manifest["ring01_reference_included"])
            self.assertEqual(
                {sample["split"] for sample in manifest["samples"]}, {"val"}
            )
            self.assertEqual(
                {sample["object_id"] for sample in manifest["samples"]},
                {"ring_004", "ring_012", "ring_020"},
            )

    def test_two_backend_consensus_is_stability_not_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = root / "experiment"
            source_dir = root / "source"
            source_dir.mkdir(parents=True)
            experiment.mkdir(parents=True)
            source = np.full((32, 32, 3), 180, np.uint8)
            mask = np.zeros((32, 32), np.uint8)
            mask[5:27, 6:26] = 255
            cv2.imwrite(str(source_dir / "image.png"), source)
            cv2.imwrite(str(source_dir / "mask.png"), mask)
            manifest = {
                "schema_version": "test",
                "sample_count": 1,
                "samples": [
                    {
                        "sample_key": "object:front",
                        "object_id": "object",
                        "view": "front",
                        "image_path": "source/image.png",
                        "mask_path": "source/mask.png",
                        "source_sha256": sha256(source_dir / "image.png"),
                    }
                ],
            }
            (experiment / "input_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            manifest_sha256 = sha256(experiment / "input_manifest.json")
            y, x = np.mgrid[:32, :32]
            for index, backend in enumerate(
                ("depth_anything_v2_metric", "unidepth_v2")
            ):
                backend_dir = experiment / "backends" / backend
                depth_dir = backend_dir / "object"
                depth_dir.mkdir(parents=True)
                depth = (0.4 + y * 0.002 + x * 0.001 + index * 0.01).astype(np.float32)
                depth_path = depth_dir / "front_depth_m.npy"
                np.save(depth_path, depth)
                report = {
                    "backend": backend,
                    "sample_count": 1,
                    "input_manifest_sha256": manifest_sha256,
                    "records": [
                        {
                            "sample_key": "object:front",
                            "artifacts": {
                                "depth_m": str(depth_path.relative_to(root))
                            },
                            "depth_sha256": sha256(depth_path),
                        }
                    ],
                }
                (backend_dir / "backend_report.json").write_text(
                    json.dumps(report), encoding="utf-8"
                )

            result = aggregate(root, experiment, manifest)
            self.assertTrue(result["summary"]["consensus_ready"])
            self.assertFalse(
                result["summary"]["requested_four_model_ensemble_complete"]
            )
            self.assertTrue(result["claims"]["cross_model_stability_measured"])
            self.assertFalse(result["claims"]["depth_accuracy_measured"])
            self.assertFalse(result["claims"]["absolute_scale_validated"])
            self.assertTrue(result["same_view_alignment"]["complete"])
            self.assertEqual(
                result["same_view_alignment"]["normalization"],
                "per_model_masked_p02_p98",
            )
            self.assertEqual(
                result["samples"][0]["status"], "cross_model_stability_only"
            )
            self.assertEqual(
                result["visualization"]["depth_color_legend"],
                {"red": "near", "blue": "far"},
            )
            self.assertEqual(
                result["visualization"]["disagreement_color_legend"],
                {"red": "high_disagreement", "blue": "low_disagreement"},
            )
            self.assertFalse(
                result["visualization"]["numeric_depth_or_consensus_modified"]
            )
            self.assertTrue(
                (root / result["review_sheets"]["object"]).is_file()
            )

    def test_backend_licences_and_roles_are_explicit(self) -> None:
        self.assertEqual(set(BACKEND_MODELS), {
            "depth_anything_v2_metric",
            "unidepth_v2",
            "metric3d_v2",
            "depth_pro",
        })
        for backend in BACKEND_MODELS.values():
            self.assertTrue(backend["revision"])
            self.assertTrue(backend["license"])
            self.assertIn("hypothesis", backend["checkpoint_role"])


if __name__ == "__main__":
    unittest.main()
