"""Dataset loader for prepared five-view jewellery image groups."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from torch.utils.data import Dataset
except ImportError:  # The manifest remains usable without PyTorch installed.
    Dataset = object


VIEW_ORDER = ("front", "top", "iso", "lsv", "rsv")


class JewelleryMultiViewDataset(Dataset):
    """Load all five views of one jewellery object as a single sample.

    Splitting happens at object level. This prevents different views of the
    same ring from leaking between training and evaluation.
    """

    def __init__(
        self,
        root_dir: str | Path = "dataset/prepared_v1",
        split: str = "train",
        as_torch: bool = False,
        require_cad_target: bool = False,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"invalid split: {split}")
        self.root_dir = Path(root_dir)
        self.split = split
        self.as_torch = as_torch
        manifest_path = self.root_dir / "manifest.jsonl"
        self.records = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.records = [record for record in self.records if record["split"] == split]
        if require_cad_target and any(record["supervision"]["cad_target"] is None for record in self.records):
            raise ValueError(
                "This dataset has no CAD targets and cannot be used for supervised image-to-CAD training."
            )

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _read(path: str, grayscale: bool = False) -> np.ndarray:
        mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        image = cv2.imread(path, mode)
        if image is None:
            raise FileNotFoundError(path)
        if not grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        images, masks, edges = [], [], []
        for view in VIEW_ORDER:
            metadata = record["views"][view]
            images.append(self._read(metadata["image_path"]))
            masks.append(self._read(metadata["jewelry_mask_path"], grayscale=True))
            edges.append(self._read(metadata["edge_path"], grayscale=True))
        image_array = np.stack(images)
        mask_array = np.stack(masks)
        edge_array = np.stack(edges)
        sample: dict[str, Any] = {
            "object_id": record["object_id"],
            "category": record["category"],
            "split": record["split"],
            "view_names": VIEW_ORDER,
            "images": image_array,
            "jewelry_masks": mask_array,
            "edges": edge_array,
            "has_cad_target": record["supervision"]["cad_target"] is not None,
            "has_mesh_target": record["supervision"]["mesh_target"] is not None,
            "metadata": record,
        }
        if self.as_torch:
            import torch

            sample["images"] = torch.from_numpy(image_array.copy()).permute(0, 3, 1, 2).float() / 255.0
            sample["jewelry_masks"] = torch.from_numpy(mask_array.copy()).unsqueeze(1).float() / 255.0
            sample["edges"] = torch.from_numpy(edge_array.copy()).unsqueeze(1).float() / 255.0
        return sample
