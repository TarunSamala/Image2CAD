"""Tests for dataset-wide phase training contracts and metrics."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import torch

from train_dataset_phases import TinyJewelleryUNet, _binary_iou, _boundary_f1, _dice


class DatasetPhaseTrainingTest(unittest.TestCase):
    def test_tiny_model_preserves_spatial_shape(self) -> None:
        model = TinyJewelleryUNet()
        with torch.inference_mode():
            output = model(torch.zeros(2, 3, 64, 64))
        self.assertEqual(output.shape, (2, 1, 64, 64))
        self.assertLess(sum(parameter.numel() for parameter in model.parameters()), 1_000_000)

    def test_overlap_metrics_are_exact_for_identical_masks(self) -> None:
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:24, 10:22] = True
        self.assertEqual(_binary_iou(mask, mask), 1.0)
        self.assertEqual(_dice(mask, mask), 1.0)
        self.assertEqual(_boundary_f1(mask, mask), 1.0)

    def test_manifest_has_no_object_leakage(self) -> None:
        records = [
            json.loads(line)
            for line in Path("dataset/prepared_v1/manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        split_for_object = {}
        for record in records:
            previous = split_for_object.setdefault(record["object_id"], record["split"])
            self.assertEqual(previous, record["split"])
        self.assertEqual(len(split_for_object), 24)

    def test_missing_supervision_blocks_later_training_phases(self) -> None:
        report = json.loads(Path("dataset/prepared_v1/dataset_report.json").read_text(encoding="utf-8"))
        self.assertFalse(report["training_readiness"]["supervised_image_to_cad"])
        self.assertFalse(report["training_readiness"]["quantitative_3d_evaluation"])


if __name__ == "__main__":
    unittest.main()
