"""Build high-resolution comparison previews for prepared_v1 and its phase run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


VIEWS = ("front", "top", "iso", "lsv", "rsv")
NAVY = (75, 43, 9)
SPLIT_COLORS = {"train": (70, 135, 40), "val": (25, 145, 210), "test": (75, 65, 200)}


def _manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _image(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    value = cv2.imread(str(path), flags)
    if value is None:
        raise FileNotFoundError(path)
    return value


def _fit(value: np.ndarray, width: int, height: int, background: int = 247) -> np.ndarray:
    if value.ndim == 2:
        value = cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)
    scale = min(width / value.shape[1], height / value.shape[0])
    resized = cv2.resize(value, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), background, np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _label(image: np.ndarray, text: str, xy: tuple[int, int], scale: float = 0.55, color=NAVY, thickness: int = 1) -> None:
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _header(width: int, title: str, subtitle: str, height: int = 112) -> np.ndarray:
    canvas = np.full((height, width, 3), 250, np.uint8)
    _label(canvas, title, (32, 45), 1.02, NAVY, 2)
    _label(canvas, subtitle, (32, 80), 0.55, (95, 95, 95), 1)
    cv2.line(canvas, (32, height - 10), (width - 32, height - 10), (210, 210, 210), 1)
    return canvas


def build_overview(records: list[dict[str, Any]], output: Path) -> None:
    columns = 3
    card_width, card_height = 580, 190
    margin, gap = 26, 12
    rows = (len(records) + columns - 1) // columns
    width = margin * 2 + columns * card_width + (columns - 1) * gap
    body = np.full((margin + rows * card_height + (rows - 1) * gap, width, 3), 244, np.uint8)
    for index, record in enumerate(records):
        row, column = divmod(index, columns)
        x0 = margin + column * (card_width + gap)
        y0 = margin + row * (card_height + gap)
        card = np.full((card_height, card_width, 3), 255, np.uint8)
        split = record["split"]
        cv2.rectangle(card, (0, 0), (card_width - 1, card_height - 1), (215, 215, 215), 1)
        cv2.rectangle(card, (0, 0), (card_width - 1, 34), SPLIT_COLORS[split], -1)
        _label(card, f'{record["object_id"].replace("_", " ").upper()}  |  {split.upper()}', (12, 24), 0.56, (255, 255, 255), 1)
        thumb_width = (card_width - 20) // 5
        for view_index, view in enumerate(VIEWS):
            thumb = _fit(_image(record["views"][view]["image_path"]), thumb_width - 5, 122)
            x = 10 + view_index * thumb_width
            card[42:164, x : x + thumb_width - 5] = thumb
            _label(card, view.upper(), (x + 4, 181), 0.39, (70, 70, 70), 1)
        body[y0 : y0 + card_height, x0 : x0 + card_width] = card
    header = _header(width, "JEWELLERY DATASET - MULTI-VIEW OVERVIEW", "24 objects | 120 images | object-level train / validation / test split")
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.vstack((header, body)))


def build_processing(records: list[dict[str, Any]], output: Path) -> None:
    selected_ids = ("ring_001", "ring_004", "ring_007", "ring_012", "ring_015", "ring_023")
    selected = [next(record for record in records if record["object_id"] == object_id) for object_id in selected_ids]
    labels = ("SOURCE FRONT", "NORMALIZED", "PSEUDO-SILHOUETTE", "OPENCV EDGES")
    thumb = 330
    label_width, row_height = 190, 364
    width = label_width + thumb * len(labels) + 40
    body = np.full((row_height * len(selected), width, 3), 248, np.uint8)
    for column, name in enumerate(labels):
        _label(body, name, (label_width + column * thumb + 12, 26), 0.5, NAVY, 1)
    for row, record in enumerate(selected):
        y = row * row_height + 34
        split = record["split"]
        _label(body, record["object_id"].replace("_", " ").upper(), (18, y + 125), 0.63, NAVY, 2)
        _label(body, split.upper(), (18, y + 153), 0.48, SPLIT_COLORS[split], 2)
        metadata = record["views"]["front"]
        stages = (
            _image(metadata["source_path"]),
            _image(metadata["image_path"]),
            _image(metadata["jewelry_mask_path"], cv2.IMREAD_GRAYSCALE),
            _image(metadata["edge_path"], cv2.IMREAD_GRAYSCALE),
        )
        for column, stage in enumerate(stages):
            cell = _fit(stage, thumb - 16, thumb - 16)
            x = label_width + column * thumb + 8
            body[y : y + thumb - 16, x : x + thumb - 16] = cell
        cv2.line(body, (18, y + thumb + 2), (width - 18, y + thumb + 2), (222, 222, 222), 1)
    header = _header(width, "DATASET PREPARATION COMPARISON", "Representative train, validation, and test objects | source files remain unchanged")
    cv2.imwrite(str(output), np.vstack((header, body)))


def _prediction_overlay(image: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    canvas = image.astype(np.float32)
    true_positive = target & prediction
    false_positive = ~target & prediction
    false_negative = target & ~prediction
    colors = (
        (true_positive, np.array((75, 180, 90), np.float32)),
        (false_positive, np.array((55, 65, 225), np.float32)),
        (false_negative, np.array((225, 110, 45), np.float32)),
    )
    for active, color in colors:
        canvas[active] = canvas[active] * 0.30 + color * 0.70
    return canvas.astype(np.uint8)


def build_test_comparison(records: list[dict[str, Any]], run_dir: Path, output: Path) -> None:
    report = json.loads((run_dir / "phase2_2_test.json").read_text(encoding="utf-8"))
    metrics = {(item["object_id"], item["view"]): item for item in report["metrics"]["samples"]}
    test_records = [record for record in records if record["split"] == "test"]
    cell_size, label_width, row_height = 284, 190, 330
    width = label_width + cell_size * len(VIEWS) + 40
    body = np.full((row_height * len(test_records) + 48, width, 3), 248, np.uint8)
    for column, view in enumerate(VIEWS):
        _label(body, view.upper(), (label_width + column * cell_size + 12, 30), 0.56, NAVY, 2)
    for row, record in enumerate(test_records):
        y = 48 + row * row_height
        object_id = record["object_id"]
        object_scores = [metrics[(object_id, view)]["iou"] for view in VIEWS]
        _label(body, object_id.replace("_", " ").upper(), (18, y + 115), 0.62, NAVY, 2)
        _label(body, f'MEAN IoU {np.mean(object_scores):.3f}', (18, y + 145), 0.44, (70, 70, 70), 1)
        for column, view in enumerate(VIEWS):
            metadata = record["views"][view]
            image = _image(metadata["image_path"])
            target = _image(metadata["jewelry_mask_path"], cv2.IMREAD_GRAYSCALE) > 127
            prediction_small = _image(run_dir / "test_predictions" / f"{object_id}_{view}.png", cv2.IMREAD_GRAYSCALE)
            prediction = cv2.resize(prediction_small, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
            overlay = _prediction_overlay(image, target, prediction)
            cell = _fit(overlay, cell_size - 16, cell_size - 16)
            x = label_width + column * cell_size + 8
            body[y : y + cell_size - 16, x : x + cell_size - 16] = cell
            _label(body, f'IoU {metrics[(object_id, view)]["iou"]:.3f}', (x + 5, y + cell_size + 4), 0.42, (60, 60, 60), 1)
        cv2.line(body, (18, y + row_height - 8), (width - 18, y + row_height - 8), (220, 220, 220), 1)
    subtitle = "Green: agreement | Red: prediction only | Blue: target only | pseudo-mask evaluation"
    header = _header(width, "HELD-OUT TEST PREDICTION COMPARISON", subtitle)
    cv2.imwrite(str(output), np.vstack((header, body)))


def run(dataset_dir: Path, run_dir: Path, output_dir: Path) -> dict[str, Any]:
    records = _manifest(dataset_dir / "manifest.jsonl")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "dataset_overview": output_dir / "dataset_multiview_overview.png",
        "processing_comparison": output_dir / "dataset_processing_comparison.png",
        "held_out_predictions": output_dir / "held_out_prediction_comparison.png",
    }
    build_overview(records, paths["dataset_overview"])
    build_processing(records, paths["processing_comparison"])
    build_test_comparison(records, run_dir, paths["held_out_predictions"])
    report = {
        "stage": "dataset_comparison_previews",
        "object_count": len(records),
        "view_count": sum(len(record["views"]) for record in records),
        "previews": {name: str(path) for name, path in paths.items()},
        "prediction_color_key_bgr": {
            "target_and_prediction": [75, 180, 90],
            "prediction_only": [55, 65, 225],
            "target_only": [225, 110, 45],
        },
        "evaluation_target": "prepared_v1 pseudo-silhouettes",
    }
    (output_dir / "preview_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset/prepared_v1"))
    parser.add_argument("--run-dir", type=Path, default=Path("dataset/phase_runs/v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/phase_runs/v1/comparisons"))
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_dir, args.run_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
