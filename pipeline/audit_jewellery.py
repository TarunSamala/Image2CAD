"""Run reusable phase audits for one image or a complete five-view jewellery set."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from prepare_jewellery_dataset import _background_bgr, _foreground_mask, _normalized
from reconstruct_dataset_phase3 import (
    _comparison as phase3_comparison,
    _fit_volume,
    _mesh_from_volume,
    _mesh_preview,
    _projections,
    _validate_3mf,
    _write_3mf,
)
from extract_phase2_3 import extract_phase2_3
from train_dataset_phases import TinyJewelleryUNet, _binary_iou, _boundary_f1


FIVE_VIEWS = ("front", "top", "iso", "lsv", "rsv")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower()
    return cleaned or "jewellery"


def _read(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def _write(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write image: {path}")
    return str(path)


def _dice(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.count_nonzero(first & second)
    total = np.count_nonzero(first) + np.count_nonzero(second)
    return float(2 * intersection / total) if total else 1.0


def _largest_components(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    minimum = max(40, round(mask.size * 0.00015))
    output = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum:
            output[labels == label] = 255
    return output


def _largest_seeded_component(candidate: np.ndarray, seed: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    best_label, best_overlap = 0, 0
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < 40:
            continue
        overlap = np.count_nonzero((labels == label) & (seed > 0))
        if overlap > best_overlap:
            best_label, best_overlap = label, overlap
    return np.where(labels == best_label, 255, 0).astype(np.uint8) if best_label else seed.copy()


def _grabcut_refine(image: np.ndarray, prior: np.ndarray) -> np.ndarray:
    inner = cv2.erode(prior, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    outer = cv2.dilate(prior, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)))
    labels = np.full(prior.shape, cv2.GC_BGD, np.uint8)
    labels[outer > 0] = cv2.GC_PR_BGD
    labels[prior > 0] = cv2.GC_PR_FGD
    labels[inner > 0] = cv2.GC_FGD
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image, labels, None, background_model, foreground_model, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return prior.copy()
    candidate = np.where((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return _largest_seeded_component(candidate, inner)


def _shadow_proposal(image: np.ndarray, jewelry: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    points = cv2.findNonZero(jewelry)
    if points is None:
        return np.zeros_like(jewelry), 0.0
    x, y, width, height = cv2.boundingRect(points)
    support = np.zeros_like(jewelry)
    support[min(image.shape[0], y + round(height * 0.68)):min(image.shape[0], y + height + round(height * 0.16)), max(0, x + round(width * 0.08)):min(image.shape[1], x + round(width * 0.92))] = 255
    support = cv2.bitwise_and(support, cv2.bitwise_not(jewelry))
    values = gray[support > 0]
    if not len(values):
        return np.zeros_like(jewelry), 0.0
    threshold = float(np.percentile(values, 34))
    shadow = np.where((gray <= threshold) & (support > 0), 255, 0).astype(np.uint8)
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    return cv2.bitwise_and(shadow, cv2.bitwise_not(jewelry)), threshold


def _opencv_mask(image: np.ndarray) -> np.ndarray:
    try:
        return _foreground_mask(image, _background_bgr(image))
    except ValueError:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = _largest_components(mask)
        if not np.any(mask):
            raise ValueError("No foreground object could be separated from the background")
        return mask


class Segmenter:
    def __init__(self, method: str, device_name: str, sam_checkpoint: Path, unet_checkpoint: Path) -> None:
        self.device_name = "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
        if self.device_name == "auto":
            self.device_name = "cpu"
        self.method = self._select(method, sam_checkpoint, unet_checkpoint)
        self.device = torch.device(self.device_name)
        self.model: Any = None
        self.checkpoint: dict[str, Any] | None = None
        self.predictor: Any = None
        if self.method == "unet":
            self.checkpoint = torch.load(unet_checkpoint, map_location=self.device, weights_only=True)
            self.model = TinyJewelleryUNet(int(self.checkpoint["base_channels"])).to(self.device)
            self.model.load_state_dict(self.checkpoint["model"])
            self.model.eval()
        elif self.method == "sam":
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            model = build_sam2(
                "configs/sam2.1/sam2.1_hiera_t.yaml",
                str(sam_checkpoint),
                device=self.device_name,
                apply_postprocessing=False,
            )
            self.predictor = SAM2ImagePredictor(model, max_hole_area=0, max_sprinkle_area=0)

    @staticmethod
    def _select(method: str, sam_checkpoint: Path, unet_checkpoint: Path) -> str:
        if method != "auto":
            if method == "sam" and not sam_checkpoint.is_file():
                raise FileNotFoundError(f"SAM checkpoint not found: {sam_checkpoint}")
            if method == "unet" and not unet_checkpoint.is_file():
                raise FileNotFoundError(f"U-Net checkpoint not found: {unet_checkpoint}")
            return method
        if sam_checkpoint.is_file():
            try:
                import sam2  # noqa: F401
                return "sam"
            except ImportError:
                pass
        if unet_checkpoint.is_file():
            return "unet"
        return "opencv"

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        if self.method == "opencv":
            return _opencv_mask(image), {"method": "OpenCV border-background difference", "confidence": None}
        if self.method == "unet":
            assert self.model is not None and self.checkpoint is not None
            size = int(self.checkpoint["image_size"])
            resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(self.device) / 255.0
            probability = torch.sigmoid(self.model(tensor))[0, 0].cpu().numpy()
            mask = probability >= float(self.checkpoint["threshold"])
            mask = cv2.resize(mask.astype(np.uint8) * 255, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            return _largest_components(mask), {
                "method": "trained Tiny Jewellery U-Net",
                "confidence": round(float(probability[probability >= float(self.checkpoint["threshold"])].mean()), 6) if np.any(probability >= float(self.checkpoint["threshold"])) else 0.0,
                "threshold": float(self.checkpoint["threshold"]),
            }

        proposal = _opencv_mask(image)
        points = cv2.findNonZero(proposal)
        if points is None:
            raise ValueError("SAM prompting failed because the OpenCV proposal is empty")
        x, y, width, height = cv2.boundingRect(points)
        h, w = image.shape[:2]
        if width >= round(w * 0.94) or height >= round(h * 0.94):
            x, y = round(w * 0.10), round(h * 0.10)
            width, height = round(w * 0.80), round(h * 0.80)
            proposal = np.zeros((h, w), np.uint8)
            proposal[y:y + height, x:x + width] = 255
        positives = [(x + width // 2, y + height // 2)]
        distance = cv2.distanceTransform((proposal > 0).astype(np.uint8), cv2.DIST_L2, 5)
        for x0, x1, y0, y1 in ((x, x + width // 2, y, y + height // 2), (x + width // 2, x + width, y, y + height // 2), (x, x + width // 2, y + height // 2, y + height), (x + width // 2, x + width, y + height // 2, y + height)):
            region = distance[y0:y1, x0:x1]
            if region.size and region.max() > 0:
                py, px = np.unravel_index(int(np.argmax(region)), region.shape)
                positives.append((x0 + int(px), y0 + int(py)))
        negatives = [(5, 5), (w - 6, 5), (5, h - 6), (w - 6, h - 6)]
        margin = round(max(width, height) * 0.08)
        box = np.asarray([max(0, x - margin), max(0, y - margin), min(w - 1, x + width + margin), min(h - 1, y + height + margin)], np.float32)
        self.predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        context = torch.autocast("cuda", dtype=torch.bfloat16) if self.device_name == "cuda" else nullcontext()
        with context:
            masks, scores, _ = self.predictor.predict(
                point_coords=np.asarray(positives + negatives, np.float32),
                point_labels=np.asarray([1] * len(positives) + [0] * len(negatives), np.int32),
                box=box,
                multimask_output=True,
            )
        index = int(np.argmax(scores))
        return masks[index].astype(np.uint8) * 255, {
            "method": "SAM 2.1 Tiny with OpenCV-derived prompts",
            "confidence": round(float(scores[index]), 6),
            "prompt_box_xyxy": [int(value) for value in box],
        }


def _transform_mask(mask: np.ndarray, transform: dict[str, Any], size: int = 768) -> np.ndarray:
    x0, y0, x1, y1 = transform["source_crop_xyxy"]
    px, py, width, height = transform["placement_xywh"]
    crop = mask[y0:y1, x0:x1]
    resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((size, size), np.uint8)
    canvas[py:py + height, px:px + width] = resized
    return canvas


def _detail_proposal(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    values = gray[mask > 0]
    output = np.zeros_like(mask)
    if not len(values):
        return output
    threshold = float(np.percentile(values, 72))
    output[(gray >= threshold) & (mask > 0)] = 255
    output = cv2.morphologyEx(output, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return _largest_components(output)


def _overlay(image: np.ndarray, mask: np.ndarray, details: np.ndarray, shadow: np.ndarray) -> np.ndarray:
    result = image.astype(np.float32).copy()
    result[details > 0] = result[details > 0] * 0.45 + np.asarray((30, 190, 235), np.float32) * 0.55
    result[shadow > 0] = result[shadow > 0] * 0.45 + np.asarray((190, 70, 170), np.float32) * 0.55
    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    result[boundary] = (45, 205, 45)
    return result.astype(np.uint8)


def _agreement(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    a, b = first > 0, second > 0
    canvas = np.full((*first.shape, 3), 245, np.uint8)
    canvas[a & b] = (80, 180, 90)
    canvas[a & ~b] = (225, 110, 45)
    canvas[~a & b] = (55, 65, 225)
    return canvas


def _cell(image: np.ndarray, label: str, subtitle: str = "", width: int = 300, height: int = 250) -> np.ndarray:
    if image.ndim == 2:
        gray = image.astype(np.uint8)
        if gray.max() <= 1:
            gray = gray * 255
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    canvas = np.full((height + 50, width, 3), 248, np.uint8)
    x, y = (width - resized.shape[1]) // 2, 50 + (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    cv2.putText(canvas, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (25, 25, 25), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (85, 85, 85), 1, cv2.LINE_AA)
    return canvas


def _audit_sheet(view: str, values: dict[str, np.ndarray], metrics: dict[str, float], phase3: np.ndarray | None) -> np.ndarray:
    cells = [
        _cell(values["source"], "REFERENCE"),
        _cell(values["normalized"], "PHASE 1 - NORMALIZED"),
        _cell(values["initial"], "PHASE 1 - SILHOUETTE"),
        _cell(values["edges"], "PHASE 1 - EDGES"),
        _cell(values["refined"], "PHASE 2 - REFINED"),
        _cell(values["overlay"], "PHASE 2 - PROPOSALS", "yellow detail / purple shadow"),
        _cell(values["agreement"], "PHASE 2.2 - COMPARE", f'IoU {metrics["phase_agreement_iou"]:.3f} (not accuracy)'),
        _cell(
            values["phase2_3"],
            "PHASE 2.3 - EVIDENCE",
            "generic instances / edges / voids" if metrics.get("phase2_3_input_mask_pass", True) else "INPUT MASK REVIEW REQUIRED",
        ),
    ]
    if phase3 is None:
        unavailable = np.full((250, 300, 3), 242, np.uint8)
        cv2.putText(unavailable, "NEEDS FIVE VIEWS", (48, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA)
        cells.append(_cell(unavailable, "PHASE 3 - NOT RUN"))
    else:
        cells.append(_cell(phase3, "PHASE 3 - REPROJECTION"))
    later = np.full((250, 300, 3), 242, np.uint8)
    cv2.putText(later, "NEEDS REVIEWED", (70, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(later, "COMPONENTS + SCALE", (42, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 80, 80), 1, cv2.LINE_AA)
    cells.append(_cell(later, "PHASE 3.1 - 3.3.2 - NOT RUN"))
    body = np.hstack(cells)
    header = np.full((78, body.shape[1], 3), 250, np.uint8)
    cv2.putText(header, f"{view.upper()} - COMPLETE PHASE AUDIT", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (72, 42, 12), 2, cv2.LINE_AA)
    cv2.putText(header, "Machine-generated evidence; component labels and dimensions require review", (18, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (75, 75, 75), 1, cv2.LINE_AA)
    return np.vstack((header, body))


def _phase_description(view_count: int, segmenter: str, phase3_ran: bool) -> str:
    phase3_text = (
        "Phase 3 intersected four orthographic silhouette hypotheses into a coarse visual hull, exported STL and 3MF, and reprojected the hull for comparison. It is non-metric and not manufacturing-ready."
        if phase3_ran else
        "Phase 3 was not run because a single image cannot constrain depth. Upload front, top, isometric, left-side and right-side images to enable the coarse visual hull."
    )
    return f"""# Jewellery phase audit

Input: {view_count} image{'s' if view_count != 1 else ''}. Segmentation backend: {segmenter}.

## Phase 1 - preparation and feature extraction

Each source image is decoded and hashed, then the jewellery foreground is segmented. The object is cropped, centred on a 768 x 768 canvas, and measured in pixels. Canny edges, contours, holes, occupancy and horizontal symmetry are recorded. These are image measurements, not millimetres.

## Phase 2 - silhouette refinement and proposals

GrabCut refines the Phase 1 silhouette in a narrow band. OpenCV then records a conservative bright-detail proposal and a local cast-shadow proposal. Bright regions may be stones, polished metal or reflections; they are deliberately not labelled as confirmed stones or prongs.

## Phase 2.2 - comparison

The initial and refined silhouettes are compared with IoU, Dice and boundary F1. These scores measure agreement between two machine stages. They are not accuracy scores because no human ground-truth mask was uploaded.

## Phase 2.3 - universal evidence extraction

Internal edges, multi-scale ridges and valleys, relief responses, negative spaces and likely highlight interference are preserved as separate evidence maps. Local detail regions receive stable observation IDs and conservative cross-view track hypotheses. They remain generic machine proposals until reviewed; bright regions are not automatically called gemstones.

## Phase 3 - multi-view geometry

{phase3_text}

## Later CAD phases

Phase 3.1 through Phase 3.3.2 are not automatically claimed for arbitrary uploads. Editable component CAD, individual stones, seats, cavities, prongs and manufacturing validation need reviewed component masks, calibrated cameras, one known physical measurement and topology-specific fitting.
"""


def run_audit(
    inputs: dict[str, Path],
    output_dir: Path,
    *,
    segmenter_name: str = "auto",
    device_name: str = "auto",
    sam_checkpoint: Path = Path("models/sam2.1_hiera_tiny.pt"),
    unet_checkpoint: Path = Path("dataset/phase_runs/v1/checkpoints/whole_jewellery_unet.pt"),
    resolution: int = 128,
    nominal_size_mm: float = 30.0,
) -> dict[str, Any]:
    if len(inputs) not in (1, 5):
        raise ValueError("Upload exactly one image or all five views: front, top, iso, lsv and rsv")
    if len(inputs) == 5 and set(inputs) != set(FIVE_VIEWS):
        raise ValueError(f"Five-view input must contain exactly: {', '.join(FIVE_VIEWS)}")
    for view, path in inputs.items():
        if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
            raise ValueError(f"Unsupported or missing {view} image: {path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    segmenter = Segmenter(segmenter_name, device_name, sam_checkpoint, unet_checkpoint)
    records: dict[str, Any] = {}
    masks: dict[str, np.ndarray] = {}

    for view, source_path in inputs.items():
        source = _read(source_path)
        initial_source, segmentation = segmenter.predict(source)
        refined_source = _grabcut_refine(source, initial_source)
        background = _background_bgr(source)
        normalized, initial, _, transform = _normalized(source, initial_source, background, 768)
        refined = _transform_mask(refined_source, transform)
        gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
        edges = cv2.bitwise_and(cv2.Canny(gray, 45, 120), cv2.dilate(refined, np.ones((3, 3), np.uint8)))
        shadow_source, shadow_threshold = _shadow_proposal(source, refined_source)
        details_source = _detail_proposal(source, refined_source)
        shadow = cv2.bitwise_and(_transform_mask(shadow_source, transform), cv2.bitwise_not(refined))
        details = cv2.bitwise_and(_transform_mask(details_source, transform), refined)
        overlay = _overlay(normalized, refined, details, shadow)
        agreement = _agreement(initial, refined)
        initial_binary, refined_binary = initial > 0, refined > 0
        metrics = {
            "phase_agreement_iou": round(_binary_iou(initial_binary, refined_binary), 6),
            "phase_agreement_dice": round(_dice(initial_binary, refined_binary), 6),
            "phase_agreement_boundary_f1_2px": round(_boundary_f1(initial_binary, refined_binary), 6),
        }
        contours, hierarchy = cv2.findContours(refined, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        view_dir = output_dir / "views" / view
        copied = output_dir / "inputs" / f"{view}{source_path.suffix.lower()}"
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, copied)
        artifacts = {
            "input_copy": str(copied),
            "normalized": _write(view_dir / "phase1_normalized.png", normalized),
            "phase1_silhouette": _write(view_dir / "phase1_silhouette.png", initial),
            "phase1_edges": _write(view_dir / "phase1_edges.png", edges),
            "phase2_refined_silhouette": _write(view_dir / "phase2_refined_silhouette.png", refined),
            "phase2_detail_proposal": _write(view_dir / "phase2_detail_proposal.png", details),
            "phase2_shadow_proposal": _write(view_dir / "phase2_shadow_proposal.png", shadow),
            "phase2_overlay": _write(view_dir / "phase2_overlay.png", overlay),
            "phase2_2_agreement": _write(view_dir / "phase2_2_agreement.png", agreement),
        }
        masks[view] = refined > 0
        records[view] = {
            "source": str(source_path),
            "source_sha256": _sha256(source_path),
            "copied_sha256": _sha256(copied),
            "segmentation": segmentation,
            "normalization": transform,
            "measurements": {
                "foreground_fraction": round(float(refined_binary.mean()), 6),
                "edge_fraction": round(float(np.count_nonzero(edges) / edges.size), 6),
                "contour_count": len(contours),
                "hole_count": int(sum(item[3] >= 0 for item in hierarchy[0])) if hierarchy is not None else 0,
                "horizontal_symmetry_iou": round(_binary_iou(refined_binary, np.fliplr(refined_binary)), 6),
                "shadow_gray_threshold": round(float(shadow_threshold), 3),
            },
            "phase2_2": metrics,
            "artifacts": artifacts,
            "images": {"source": source, "normalized": normalized, "initial": initial, "edges": edges, "refined": refined, "overlay": overlay, "agreement": agreement},
        }

    phase3_report: dict[str, Any] = {"status": "not_run_single_image", "reason": "Depth is underconstrained from one image."}
    projections: dict[str, np.ndarray] = {}
    if len(inputs) == 5:
        volume, side_mode, fit, calibrated = _fit_volume(masks, resolution)
        mesh = _mesh_from_volume(volume, nominal_size_mm)
        projections = _projections(volume)
        geometry_dir = output_dir / "phase3"
        geometry_dir.mkdir(parents=True, exist_ok=True)
        stl = geometry_dir / "jewellery_phase3_visual_hull_non_metric.stl"
        three_mf = geometry_dir / "jewellery_phase3_visual_hull_non_metric.3mf"
        mesh.export(stl)
        _write_3mf(mesh, three_mf, "Uploaded jewellery Phase 3 visual hull", nominal_size_mm)
        reprojection = {
            view: {
                "iou": round(_binary_iou(calibrated[view], projections[view]), 6),
                "boundary_f1_2px": round(_boundary_f1(calibrated[view], projections[view]), 6),
            }
            for view in projections
        }
        manifest_record = {"object_id": "uploaded_jewellery", "views": {view: {"image_path": records[view]["artifacts"]["normalized"]} for view in FIVE_VIEWS}}
        comparison_path = geometry_dir / "phase3_comparison.png"
        _write(comparison_path, phase3_comparison(manifest_record, calibrated, projections, reprojection, mesh))
        phase3_report = {
            "status": "experimental_non_metric_pass" if mesh.is_watertight else "failed_mesh_validation",
            "selected_side_consensus": side_mode,
            "fit": fit,
            "reprojection": reprojection,
            "mean_reprojection_iou": round(float(np.mean([item["iou"] for item in reprojection.values()])), 6),
            "mesh": {"vertices": len(mesh.vertices), "faces": len(mesh.faces), "watertight": bool(mesh.is_watertight), "body_count": int(mesh.body_count)},
            "exports": {"stl": str(stl), "3mf": str(three_mf), "comparison": str(comparison_path)},
            "checks": {"stl_readable": bool(_read_mesh(stl)), "3mf_package_valid": _validate_3mf(three_mf)},
            "camera_calibrated": False,
            "physical_scale_calibrated": False,
            "manufacturing_accuracy_validated": False,
        }
        projections["iso"] = _mesh_preview(mesh)

    phase2_3_inputs = {
        view: {
            "image": Path(record["artifacts"]["normalized"]),
            "foreground": Path(record["artifacts"]["phase2_refined_silhouette"]),
        }
        for view, record in records.items()
    }
    phase2_3_report = extract_phase2_3(phase2_3_inputs, output_dir / "phase2_3")
    for view, record in records.items():
        record["images"]["phase2_3"] = _read(
            Path(phase2_3_report["views"][view]["artifacts"]["review_overlay"])
        )
        record["phase2_2"]["phase2_3_input_mask_pass"] = phase2_3_report["views"][view]["input_mask_quality"]["passed"]

    audit_paths = []
    for view, record in records.items():
        projection = projections.get(view)
        audit = _audit_sheet(view, record.pop("images"), record["phase2_2"], projection)
        path = output_dir / "audits" / f"{view}_all_phases.png"
        audit_paths.append(_write(path, audit))
        record["artifacts"]["phase_audit"] = str(path)
    summary = cv2.vconcat([cv2.resize(_read(Path(path)), (1600, 252), interpolation=cv2.INTER_AREA) for path in audit_paths])
    summary_path = _write(output_dir / "all_phases_summary.png", summary)
    explanation = _phase_description(len(inputs), segmenter.method, len(inputs) == 5)
    explanation_path = output_dir / "PROCESS.md"
    explanation_path.write_text(explanation, encoding="utf-8")
    report = {
        "schema_version": "uploaded_jewellery_audit_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "input_mode": "five_view" if len(inputs) == 5 else "single_image",
        "view_count": len(inputs),
        "views": records,
        "phase2_3": {
            "status": (
                "machine_proposals_pending_review"
                if phase2_3_report["validation"]["ready_for_human_instance_review"]
                else "input_mask_review_required"
            ),
            "evidence_graph": str(output_dir / "phase2_3" / "evidence_graph.json"),
            "review_manifest": phase2_3_report["review_manifest"],
            "observation_count": sum(
                view["counts"]["machine_detail_proposals"]
                for view in phase2_3_report["views"].values()
            ),
            "cross_view_track_count": len(phase2_3_report["cross_view_tracks"]),
            "human_review_complete": False,
        },
        "phase3": phase3_report,
        "later_phases": {"status": "not_run", "reason": "Need reviewed semantic components, calibrated cameras, physical scale and topology-specific CAD fitting."},
        "accuracy_scope": {"human_ground_truth_masks": False, "metric_scale": False, "manufacturing_accuracy_validated": False},
        "artifacts": {"summary": summary_path, "process_explanation": str(explanation_path), "view_audits": audit_paths},
    }
    report_path = output_dir / "audit_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _read_mesh(path: Path) -> bool:
    import trimesh
    return bool(trimesh.load(path, force="mesh").vertices.size)


def _input_map(args: argparse.Namespace) -> dict[str, Path]:
    multi = {name: getattr(args, name) for name in FIVE_VIEWS if getattr(args, name) is not None}
    if args.image and multi:
        raise ValueError("Use --image for one image, or the five named view options, not both")
    if args.image:
        return {"single": args.image}
    return multi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="One image; Phase 1 through Phase 2.2")
    parser.add_argument("--front", type=Path)
    parser.add_argument("--top", type=Path)
    parser.add_argument("--iso", type=Path)
    parser.add_argument("--lsv", type=Path, help="Left-side view")
    parser.add_argument("--rsv", type=Path, help="Right-side view")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--name", default="jewellery")
    parser.add_argument("--segmenter", choices=("auto", "sam", "unet", "opencv"), default="auto")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--sam-checkpoint", type=Path, default=Path("models/sam2.1_hiera_tiny.pt"))
    parser.add_argument("--unet-checkpoint", type=Path, default=Path("dataset/phase_runs/v1/checkpoints/whole_jewellery_unet.pt"))
    parser.add_argument("--resolution", type=int, default=128)
    args = parser.parse_args()
    inputs = _input_map(args)
    if not inputs:
        parser.error("provide --image or all five named views")
    output = args.output_dir or Path("runs/uploads") / f'{datetime.now():%Y%m%d-%H%M%S-%f}-{_safe_name(args.name)}'
    report = run_audit(inputs, output, segmenter_name=args.segmenter, device_name=args.device, sam_checkpoint=args.sam_checkpoint, unet_checkpoint=args.unet_checkpoint, resolution=args.resolution)
    print(json.dumps({"output_dir": str(output), "mode": report["input_mode"], "summary": report["artifacts"]["summary"], "phase3": report["phase3"]["status"]}, indent=2))


if __name__ == "__main__":
    main()
