"""Validation for the prepared five-view jewellery dataset."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from jewellery_multiview_dataset import JewelleryMultiViewDataset, VIEW_ORDER


class JewelleryDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path("dataset/prepared_v1")
        cls.report = json.loads((cls.root / "dataset_report.json").read_text(encoding="utf-8"))
        cls.records = [
            json.loads(line)
            for line in (cls.root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_dataset_integrity_report_passes(self) -> None:
        self.assertTrue(self.report["dataset_valid"])
        self.assertTrue(all(self.report["checks"].values()))
        self.assertEqual(self.report["object_count"], 24)
        self.assertEqual(self.report["image_count"], 120)

    def test_splits_are_object_level_and_disjoint(self) -> None:
        groups = {
            split: {record["object_id"] for record in self.records if record["split"] == split}
            for split in ("train", "val", "test")
        }
        self.assertEqual({key: len(value) for key, value in groups.items()}, {"train": 18, "val": 3, "test": 3})
        self.assertFalse(groups["train"] & groups["val"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["val"] & groups["test"])

    def test_every_object_has_five_standardized_views(self) -> None:
        self.assertEqual(len(self.records), 24)
        for record in self.records:
            self.assertEqual(set(record["views"]), set(VIEW_ORDER))
            for metadata in record["views"].values():
                self.assertEqual(metadata["prepared_size_wh"], [768, 768])
                for key in ("image_path", "jewelry_mask_path", "edge_path"):
                    self.assertTrue(Path(metadata[key]).is_file(), metadata[key])

    def test_masks_are_binary_nonempty_and_preserve_negative_space(self) -> None:
        masks_with_holes = 0
        for record in self.records:
            for metadata in record["views"].values():
                mask = cv2.imread(metadata["jewelry_mask_path"], cv2.IMREAD_GRAYSCALE)
                self.assertIsNotNone(mask)
                self.assertTrue(set(int(value) for value in np.unique(mask)).issubset({0, 255}))
                occupancy = cv2.countNonZero(mask) / mask.size
                self.assertGreater(occupancy, 0.05)
                self.assertLess(occupancy, 0.50)
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                if hierarchy is not None and any(item[3] >= 0 for item in hierarchy[0]):
                    masks_with_holes += 1
        self.assertGreater(masks_with_holes, 0)

    def test_loader_returns_one_five_view_group_per_ring(self) -> None:
        expected = {"train": 18, "val": 3, "test": 3}
        for split, count in expected.items():
            dataset = JewelleryMultiViewDataset(self.root, split=split)
            self.assertEqual(len(dataset), count)
            sample = dataset[0]
            self.assertEqual(sample["images"].shape, (5, 768, 768, 3))
            self.assertEqual(sample["jewelry_masks"].shape, (5, 768, 768))
            self.assertEqual(sample["view_names"], VIEW_ORDER)

    def test_missing_cad_supervision_is_explicit_and_guarded(self) -> None:
        self.assertFalse(self.report["training_readiness"]["supervised_image_to_cad"])
        for record in self.records:
            self.assertIsNone(record["supervision"]["cad_target"])
            self.assertIsNone(record["supervision"]["mesh_target"])
            self.assertIsNone(record["supervision"]["physical_scale"])
        with self.assertRaisesRegex(ValueError, "no CAD targets"):
            JewelleryMultiViewDataset(self.root, split="train", require_cad_target=True)

    def test_source_dataset_was_preserved(self) -> None:
        sources = list(Path("dataset/STL-1").rglob("*.png"))
        self.assertEqual(len(sources), 120)


if __name__ == "__main__":
    unittest.main()
