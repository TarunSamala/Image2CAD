"""Build Ring01-style per-view and all-phase audits for every dataset object."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from train_dataset_phases import TinyJewelleryUNet, _binary_iou, _boundary_f1, _dice


VIEWS = ("front", "top", "iso", "lsv", "rsv")
CELL_W, CELL_H, LABEL_H = 220, 165, 27
NAVY = (72, 42, 12)
SPLIT_COLORS = {"train": (70, 135, 40), "val": (25, 145, 210), "test": (75, 65, 200)}


def _read(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    value = cv2.imread(str(path), flags)
    if value is None:
        raise FileNotFoundError(path)
    return value


def _manifest(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _fit(value: np.ndarray, width: int = CELL_W, height: int = CELL_H) -> np.ndarray:
    if value.ndim == 2:
        value = cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)
    scale = min(width / value.shape[1], height / value.shape[0])
    resized = cv2.resize(value, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 248, np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _cell(value: np.ndarray, label: str, note: str = "") -> np.ndarray:
    canvas = np.full((CELL_H + LABEL_H, CELL_W, 3), 248, np.uint8)
    cv2.putText(canvas, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    if note:
        width = cv2.getTextSize(note, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0][0]
        cv2.putText(canvas, note, (CELL_W - width - 5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1, cv2.LINE_AA)
    canvas[LABEL_H:] = _fit(value)
    return canvas


def _unavailable(label: str) -> np.ndarray:
    canvas = np.full((CELL_H, CELL_W, 3), 242, np.uint8)
    cv2.rectangle(canvas, (2, 2), (CELL_W - 3, CELL_H - 3), (205, 205, 205), 1)
    cv2.putText(canvas, label, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.47, NAVY, 1, cv2.LINE_AA)
    cv2.putText(canvas, "NOT GENERATED", (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (55, 55, 190), 2, cv2.LINE_AA)
    cv2.putText(canvas, "NO CAD TARGET", (12, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (85, 85, 85), 1, cv2.LINE_AA)
    cv2.putText(canvas, "OR CALIBRATED SCALE", (12, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (85, 85, 85), 1, cv2.LINE_AA)
    return canvas


def _phase1_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (40, 210, 40), 2, cv2.LINE_AA)
    points = cv2.findNonZero(mask)
    if points is not None:
        x, y, width, height = cv2.boundingRect(points)
        cv2.rectangle(result, (x, y), (x + width - 1, y + height - 1), (30, 30, 220), 2)
    return result


def _comparison(image: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    result = image.astype(np.float32)
    categories = (
        (target & prediction, np.array((75, 180, 90), np.float32)),
        (~target & prediction, np.array((55, 65, 225), np.float32)),
        (target & ~prediction, np.array((225, 110, 45), np.float32)),
    )
    for active, color in categories:
        result[active] = result[active] * 0.30 + color * 0.70
    return result.astype(np.uint8)


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[TinyJewelleryUNet, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = TinyJewelleryUNet(base_channels=int(checkpoint["base_channels"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def _predict(model: TinyJewelleryUNet, image: np.ndarray, image_size: int, threshold: float, device: torch.device) -> np.ndarray:
    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(device) / 255.0
    probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    prediction = (probability >= threshold).astype(np.uint8) * 255
    return cv2.resize(prediction, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)


def _header(width: int, object_id: str, split: str, subtitle: str) -> np.ndarray:
    canvas = np.full((80, width, 3), 250, np.uint8)
    title = object_id.replace("_", " ").upper()
    cv2.putText(canvas, f"{title}  |  {split.upper()}", (18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.76, SPLIT_COLORS[split], 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (80, 80, 80), 1, cv2.LINE_AA)
    return canvas


def _write_view_audit(
    object_id: str,
    split: str,
    view: str,
    image: np.ndarray,
    mask: np.ndarray,
    edge: np.ndarray,
    prediction: np.ndarray,
    metrics: dict[str, float],
    output: Path,
) -> None:
    target = mask > 127
    predicted = prediction > 127
    first = np.hstack((
        _cell(image, "REFERENCE"),
        _cell(mask, "PHASE 1 SILHOUETTE"),
        _cell(edge, "PHASE 1 EDGES"),
        _cell(_phase1_overlay(image, mask), "PHASE 1 OVERLAY"),
        _cell(prediction, "PHASE 2 PREDICTION"),
    ))
    score = f'IoU {metrics["iou"]:.3f}'
    second = np.hstack((
        _cell(mask, "PSEUDO TARGET"),
        _cell(_comparison(image, target, predicted), "PHASE 2.2 COMPARE", score),
        _cell(_unavailable("PHASE 3 / 3.1"), "CAD RECONSTRUCTION"),
        _cell(_unavailable("PHASE 3.2 / 3.3"), "CAD REFINEMENT"),
        _cell(_unavailable("PHASE 3.3.1 / 3.3.2"), "PRONG / VISIBILITY"),
    ))
    width = first.shape[1]
    subtitle = f"{view.upper()} VIEW | green agreement, red prediction-only, blue target-only | pseudo-mask comparison"
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.vstack((_header(width, object_id, split, subtitle), first, second)))


def _write_object_summary(
    object_id: str,
    split: str,
    values: dict[str, dict[str, np.ndarray]],
    metrics: dict[str, dict[str, float]],
    output: Path,
) -> None:
    rows = []
    for view in VIEWS:
        item = values[view]
        target = item["mask"] > 127
        prediction = item["prediction"] > 127
        view_label = np.full((CELL_H + LABEL_H, 130, 3), 248, np.uint8)
        cv2.putText(view_label, view.upper(), (12, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.58, NAVY, 2, cv2.LINE_AA)
        score = f'IoU {metrics[view]["iou"]:.3f}'
        row = np.hstack((
            view_label,
            _cell(item["image"], "REFERENCE"),
            _cell(item["mask"], "PHASE 1 MASK"),
            _cell(item["edge"], "PHASE 1 EDGES"),
            _cell(item["prediction"], "PHASE 2"),
            _cell(_comparison(item["image"], target, prediction), "PHASE 2.2", score),
            _cell(_unavailable("PHASE 3 - 3.3.2"), "LATER PHASES"),
        ))
        rows.append(row)
    width = rows[0].shape[1]
    mean_iou = float(np.mean([metrics[view]["iou"] for view in VIEWS]))
    status = "HELD-OUT TEST" if split == "test" else "TRAINING FIT" if split == "train" else "VALIDATION"
    subtitle = f"ALL FIVE VIEWS | {status} | mean pseudo-silhouette IoU {mean_iou:.3f}"
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.vstack((_header(width, object_id, split, subtitle), *rows)))


def run(dataset_dir: Path, run_dir: Path, output_dir: Path, device_name: str) -> dict[str, Any]:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model, checkpoint = _load_model(run_dir / "checkpoints" / "whole_jewellery_unet.pt", device)
    records = _manifest(dataset_dir)
    view_reports = []
    summary_paths = {}
    audit_count = 0
    by_split: dict[str, list[dict[str, float]]] = defaultdict(list)

    for record in records:
        object_id, split = record["object_id"], record["split"]
        values = {}
        object_metrics = {}
        for view in VIEWS:
            metadata = record["views"][view]
            image = _read(metadata["image_path"])
            mask = _read(metadata["jewelry_mask_path"], cv2.IMREAD_GRAYSCALE)
            edge = _read(metadata["edge_path"], cv2.IMREAD_GRAYSCALE)
            prediction = _predict(model, image, int(checkpoint["image_size"]), float(checkpoint["threshold"]), device)
            target_binary, prediction_binary = mask > 127, prediction > 127
            metrics = {
                "iou": _binary_iou(target_binary, prediction_binary),
                "dice": _dice(target_binary, prediction_binary),
                "boundary_f1_2px": _boundary_f1(target_binary, prediction_binary),
            }
            values[view] = {"image": image, "mask": mask, "edge": edge, "prediction": prediction}
            object_metrics[view] = metrics
            by_split[split].append(metrics)
            view_reports.append({
                "object_id": object_id,
                "split": split,
                "view": view,
                **{key: round(value, 6) for key, value in metrics.items()},
            })
            audit_path = output_dir / object_id / "views" / f"{object_id}_{view}_audit.png"
            _write_view_audit(object_id, split, view, image, mask, edge, prediction, metrics, audit_path)
            audit_count += 1
        summary_path = output_dir / object_id / f"{object_id}_all_phases.png"
        _write_object_summary(object_id, split, values, object_metrics, summary_path)
        summary_paths[object_id] = str(summary_path)

    def means(items: list[dict[str, float]]) -> dict[str, float]:
        return {
            key: round(float(np.mean([item[key] for item in items])), 6)
            for key in ("iou", "dice", "boundary_f1_2px")
        }

    report = {
        "stage": "dataset_ring01_style_phase_audits",
        "device": str(device),
        "object_count": len(records),
        "view_audit_count": audit_count,
        "object_summary_count": len(summary_paths),
        "phase_columns": ["reference", "phase1", "phase2", "phase2_2", "phase3_to_phase3_3_2_status"],
        "metrics_by_split": {split: means(items) for split, items in sorted(by_split.items())},
        "evaluation_note": "Only test-split metrics are held out. Train and validation sheets are diagnostic.",
        "later_phase_status": "not generated: no paired CAD, component instances, calibrated cameras, or metric scale",
        "manufacturing_accuracy_validated": False,
        "object_summaries": summary_paths,
        "views": view_reports,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase_audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset/prepared_v1"))
    parser.add_argument("--run-dir", type=Path, default=Path("dataset/phase_runs/v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/phase_runs/v1/phase_audits"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    report = run(args.dataset_dir, args.run_dir, args.output_dir, args.device)
    print(json.dumps({
        "objects": report["object_count"],
        "view_audits": report["view_audit_count"],
        "object_summaries": report["object_summary_count"],
        "test_metrics": report["metrics_by_split"]["test"],
    }, indent=2))


if __name__ == "__main__":
    main()
