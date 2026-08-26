"""Build per-view visual audit sheets covering Ring01 stages 0 through 2.2."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


VIEWS = ("front", "side", "top", "angled", "back")
CELL_W, CELL_H, LABEL_H = 220, 165, 25


def _cell(path: Path, label: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    canvas = np.full((CELL_H + LABEL_H, CELL_W, 3), 248, np.uint8)
    cv2.putText(canvas, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (15, 15, 15), 1, cv2.LINE_AA)
    if image is None:
        cv2.putText(canvas, "MISSING", (55, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 220), 2)
        return canvas
    scale = min(CELL_W / image.shape[1], CELL_H / image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_NEAREST)
    y = LABEL_H + (CELL_H - resized.shape[0]) // 2
    x = (CELL_W - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _row(cells: list[tuple[Path, str]], columns: int = 10) -> np.ndarray:
    images = [_cell(path, label) for path, label in cells]
    blank = np.full_like(images[0], 248)
    images.extend([blank] * (columns - len(images)))
    return np.hstack(images)


def build(data_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for view in VIEWS:
        source = data_dir / "ring01_reference_images" / f"ring01_{view}.png"
        phase1 = data_dir / "ring01_phase1"
        phase2 = data_dir / "ring01_phase2"
        refined = data_dir / "ring01_phase2_refined"
        refined_22 = data_dir / "ring01_phase2_2"
        rows = [
            _row([
                (source, "SOURCE"),
                (phase1 / "masks" / f"ring01_{view}_mask.png", "P1 silhouette"),
                (phase1 / "edges" / f"ring01_{view}_edges.png", "P1 edges"),
                (phase1 / "overlays" / f"ring01_{view}_overlay.png", "P1 overlay"),
            ]),
            _row([(source, "SOURCE")] + [
                (phase2 / "masks" / component / f"ring01_{view}_{component}.png", f"P2 {component}")
                for component in ("jewelry", "metal", "shank", "stone", "setting", "prongs", "shadow")
            ] + [(phase2 / "overlays" / f"ring01_{view}_components.png", "P2 overlay")]),
            _row([(source, "SOURCE")] + [
                (refined / "masks" / component / f"ring01_{view}_{component}.png", f"P2.1 {component}")
                for component in ("jewelry", "metal", "shank", "stone_visible", "stone_amodal", "setting", "prongs", "shadow")
            ] + [(refined / "overlays" / f"ring01_{view}_refined.png", "P2.1 overlay")]),
            _row([(source, "SOURCE")] + [
                (refined_22 / "masks" / component / f"ring01_{view}_{component}.png", f"P2.2 {component}")
                for component in ("jewelry", "metal", "shank", "stone_visible", "stone_amodal", "setting", "prongs", "shadow")
            ] + [(refined_22 / "review" / f"ring01_{view}_annotation_review.png", "P2.2 review")]),
        ]
        sheet = np.vstack(rows)
        cv2.imwrite(str(output_dir / f"ring01_{view}_audit.png"), sheet)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/ring01_validation"))
    args = parser.parse_args()
    build(args.data_dir, args.output_dir)
    print(f"Wrote {len(VIEWS)} audit sheets to {args.output_dir}")


if __name__ == "__main__":
    main()
