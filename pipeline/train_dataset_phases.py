"""Train and evaluate the image-only jewellery dataset through reusable phase gates.

This experiment learns whole-jewellery foreground segmentation from the
prepared pseudo-masks. It deliberately does not synthesize missing CAD,
component, camera, or metric-scale supervision.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


VIEW_ORDER = ("front", "top", "iso", "lsv", "rsv")


def _read_manifest(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class JewelleryViewDataset(Dataset):
    """Flatten object groups into views while preserving object-level splits."""

    def __init__(self, root: Path, split: str, image_size: int, augment: bool = False) -> None:
        self.image_size = image_size
        self.augment = augment
        self.samples: list[dict[str, str]] = []
        for record in _read_manifest(root):
            if record["split"] != split:
                continue
            for view in VIEW_ORDER:
                metadata = record["views"][view]
                self.samples.append({
                    "object_id": record["object_id"],
                    "view": view,
                    "image": metadata["image_path"],
                    "mask": metadata["jewelry_mask_path"],
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = cv2.imread(sample["image"], cv2.IMREAD_COLOR)
        mask = cv2.imread(sample["mask"], cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(sample)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        if self.augment and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        image_tensor = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy((mask > 127).astype(np.float32))[None]
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "object_id": sample["object_id"],
            "view": sample["view"],
        }


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class TinyJewelleryUNet(nn.Module):
    """Small U-Net intended for a 4 GB laptop GPU."""

    def __init__(self, base_channels: int = 12) -> None:
        super().__init__()
        self.encoder1 = ConvBlock(3, base_channels)
        self.encoder2 = ConvBlock(base_channels, base_channels * 2)
        self.bridge = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.decoder2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.decoder1 = ConvBlock(base_channels * 2, base_channels)
        self.output = nn.Conv2d(base_channels, 1, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(value)
        enc2 = self.encoder2(self.pool(enc1))
        bridge = self.bridge(self.pool(enc2))
        dec2 = self.decoder2(torch.cat((self.up2(bridge), enc2), dim=1))
        dec1 = self.decoder1(torch.cat((self.up1(dec2), enc1), dim=1))
        return self.output(dec1)


def _dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * target).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def _binary_iou(target: np.ndarray, prediction: np.ndarray) -> float:
    intersection = np.count_nonzero(target & prediction)
    union = np.count_nonzero(target | prediction)
    return float(intersection / union) if union else 1.0


def _dice(target: np.ndarray, prediction: np.ndarray) -> float:
    intersection = np.count_nonzero(target & prediction)
    total = np.count_nonzero(target) + np.count_nonzero(prediction)
    return float(2 * intersection / total) if total else 1.0


def _boundary_f1(target: np.ndarray, prediction: np.ndarray, tolerance: int = 2) -> float:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    target_edge = cv2.morphologyEx(target.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    prediction_edge = cv2.morphologyEx(prediction.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    radius = 2 * tolerance + 1
    tolerance_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    target_near = cv2.dilate(target_edge.astype(np.uint8), tolerance_kernel) > 0
    prediction_near = cv2.dilate(prediction_edge.astype(np.uint8), tolerance_kernel) > 0
    precision = np.count_nonzero(prediction_edge & target_near) / max(1, np.count_nonzero(prediction_edge))
    recall = np.count_nonzero(target_edge & prediction_near) / max(1, np.count_nonzero(target_edge))
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def _refine(mask: np.ndarray) -> np.ndarray:
    value = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    value = cv2.morphologyEx(value, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(value, connectivity=8)
    minimum_area = max(8, round(value.size * 0.0002))
    retained = np.zeros_like(value)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_area:
            retained[labels == label] = 255
    return retained > 0


@torch.inference_mode()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> list[dict[str, Any]]:
    model.eval()
    records: list[dict[str, Any]] = []
    for batch in loader:
        probabilities = torch.sigmoid(model(batch["image"].to(device))).cpu().numpy()[:, 0]
        targets = batch["mask"].numpy()[:, 0] > 0.5
        for index in range(len(probabilities)):
            records.append({
                "object_id": batch["object_id"][index],
                "view": batch["view"][index],
                "probability": probabilities[index],
                "target": targets[index],
            })
    return records


def _score(records: list[dict[str, Any]], threshold: float, refine: bool) -> dict[str, Any]:
    samples = []
    by_view: dict[str, list[dict[str, float]]] = defaultdict(list)
    for record in records:
        prediction = record["probability"] >= threshold
        if refine:
            prediction = _refine(prediction)
        metrics = {
            "iou": _binary_iou(record["target"], prediction),
            "dice": _dice(record["target"], prediction),
            "boundary_f1_2px": _boundary_f1(record["target"], prediction),
        }
        samples.append({"object_id": record["object_id"], "view": record["view"], **metrics})
        by_view[record["view"]].append(metrics)

    def mean_metrics(values: list[dict[str, float]]) -> dict[str, float]:
        return {
            key: round(float(np.mean([item[key] for item in values])), 6)
            for key in ("iou", "dice", "boundary_f1_2px")
        }

    return {
        "sample_count": len(samples),
        "mean": mean_metrics(samples),
        "by_view": {view: mean_metrics(values) for view, values in sorted(by_view.items())},
        "samples": [{key: round(value, 6) if isinstance(value, float) else value for key, value in item.items()} for item in samples],
    }


def _phase1_report(root: Path) -> dict[str, Any]:
    records = _read_manifest(root)
    views = []
    for record in records:
        for view in VIEW_ORDER:
            metadata = record["views"][view]
            mask = cv2.imread(metadata["jewelry_mask_path"], cv2.IMREAD_GRAYSCALE)
            edge = cv2.imread(metadata["edge_path"], cv2.IMREAD_GRAYSCALE)
            if mask is None or edge is None:
                raise FileNotFoundError(metadata)
            binary = mask > 127
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            flipped = np.fliplr(binary)
            views.append({
                "object_id": record["object_id"],
                "split": record["split"],
                "view": view,
                "foreground_fraction": round(float(binary.mean()), 6),
                "edge_fraction": round(float(np.count_nonzero(edge) / edge.size), 6),
                "contour_count": len(contours),
                "hole_count": int(sum(item[3] >= 0 for item in hierarchy[0])) if hierarchy is not None else 0,
                "horizontal_symmetry_iou": round(_binary_iou(binary, flipped), 6),
            })
    return {
        "stage": "dataset_phase1_features",
        "object_count": len(records),
        "view_count": len(views),
        "measurements_are_metric": False,
        "checks": {
            "all_24_objects_processed": len(records) == 24,
            "all_120_views_processed": len(views) == 120,
            "all_masks_nonempty": all(item["foreground_fraction"] > 0 for item in views),
            "all_edges_nonempty": all(item["edge_fraction"] > 0 for item in views),
        },
        "views": views,
    }


def _save_previews(records: list[dict[str, Any]], threshold: float, refine: bool, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["object_id"]].append(record)
        prediction = record["probability"] >= threshold
        if refine:
            prediction = _refine(prediction)
        cv2.imwrite(str(output_dir / f'{record["object_id"]}_{record["view"]}.png'), prediction.astype(np.uint8) * 255)

    for object_id, items in grouped.items():
        cells = []
        for record in sorted(items, key=lambda item: VIEW_ORDER.index(item["view"])):
            target = record["target"]
            prediction = record["probability"] >= threshold
            if refine:
                prediction = _refine(prediction)
            canvas = np.full((*target.shape, 3), 245, np.uint8)
            canvas[target] = (190, 190, 190)
            canvas[prediction] = (210, 140, 80)
            canvas[target & prediction] = (90, 180, 110)
            cv2.putText(canvas, record["view"].upper(), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
            cells.append(canvas)
        cv2.imwrite(str(output_dir / f"{object_id}_review.png"), np.hstack(cells))


def run(root: Path, output_dir: Path, epochs: int, image_size: int, batch_size: int, device_name: str) -> dict[str, Any]:
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(7)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase1 = _phase1_report(root)
    (output_dir / "phase1_features.json").write_text(json.dumps(phase1, indent=2), encoding="utf-8")

    datasets = {
        split: JewelleryViewDataset(root, split, image_size, augment=split == "train")
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(data, batch_size=batch_size, shuffle=split == "train", num_workers=0)
        for split, data in datasets.items()
    }
    model = TinyJewelleryUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = 0.45 * bce(logits, targets) + 0.55 * _dice_loss(logits, targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 6)})

    validation_predictions = _predict(model, loaders["val"], device)
    candidates = []
    for threshold in np.arange(0.30, 0.71, 0.05):
        for refine in (False, True):
            score = _score(validation_predictions, float(threshold), refine)
            candidates.append({
                "threshold": round(float(threshold), 2),
                "refine": refine,
                "mean_iou": score["mean"]["iou"],
            })
    selected = max(candidates, key=lambda item: item["mean_iou"])
    validation = _score(validation_predictions, selected["threshold"], selected["refine"])
    test_predictions = _predict(model, loaders["test"], device)
    test = _score(test_predictions, selected["threshold"], selected["refine"])
    _save_previews(test_predictions, selected["threshold"], selected["refine"], output_dir / "test_predictions")

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "whole_jewellery_unet.pt"
    torch.save({
        "model": model.state_dict(),
        "architecture": "TinyJewelleryUNet",
        "base_channels": 12,
        "image_size": image_size,
        "threshold": selected["threshold"],
        "refine": selected["refine"],
        "training_target": "prepared_v1 pseudo-silhouette",
    }, checkpoint_path)

    training = {
        "stage": "dataset_phase2_training",
        "device": str(device),
        "epochs": epochs,
        "image_size": image_size,
        "batch_size": batch_size,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "split_view_counts": {split: len(data) for split, data in datasets.items()},
        "split_policy": "object-level; no ring appears in more than one split",
        "target_status": "automatically generated pseudo-silhouette; not human ground truth",
        "history": history,
        "validation_selection": {**selected, "candidates": candidates, "metrics": validation},
        "checkpoint": str(checkpoint_path),
    }
    (output_dir / "phase2_training.json").write_text(json.dumps(training, indent=2), encoding="utf-8")
    test_report = {
        "stage": "dataset_phase2_2_held_out_test",
        "selection_source": "validation split only",
        "threshold": selected["threshold"],
        "morphological_refinement": selected["refine"],
        "target_status": "pseudo-silhouette agreement, not independent pixel accuracy",
        "metrics": test,
    }
    (output_dir / "phase2_2_test.json").write_text(json.dumps(test_report, indent=2), encoding="utf-8")

    coverage = {
        "experiment": "prepared_v1_dataset_phase_coverage",
        "latest_project_checkpoint": "phase3_3_2",
        "executed": {
            "phase1": "features extracted for all 24 objects and 120 views",
            "phase2": "whole-jewellery foreground model trained on 18 objects and selected on 3 validation objects",
            "phase2_2": "fixed selected model tested on 3 unseen objects",
        },
        "not_trainable_from_this_dataset": {
            "phase2_components": "no individual stone, metal, shank, setting, or prong labels",
            "phase3": "no paired mesh/CAD target, camera calibration, or physical scale",
            "phase3_3": "no exact-solid target or human-reviewed multi-view masks",
            "phase3_3_1": "no individual prong instance ground truth",
            "phase3_3_2": "visibility correction requires an existing validated four-prong CAD model",
        },
        "important_scope": "The Ring01 Phase 3.3.2 CAD checkpoint remains valid only for Ring01 and was not copied onto unrelated rings.",
        "dataset_reaches_latest_phase_gate": False,
        "highest_executed_dataset_phase": "phase2_2",
        "manufacturing_accuracy_validated": False,
    }
    (output_dir / "phase_coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    summary = {
        "phase1_passed": all(phase1["checks"].values()),
        "validation": validation["mean"],
        "test": test["mean"],
        "selected_threshold": selected["threshold"],
        "selected_refinement": selected["refine"],
        "highest_executed_dataset_phase": coverage["highest_executed_dataset_phase"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/prepared_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/phase_runs/v1"))
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.output_dir, args.epochs, args.image_size, args.batch_size, args.device), indent=2))


if __name__ == "__main__":
    main()
