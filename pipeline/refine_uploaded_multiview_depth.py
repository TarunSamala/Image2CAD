"""Apply confidence-gated metric-depth hypotheses to an uploaded visual hull."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from pipeline.audit_saraswati_blind_depth import _edge_agreement
from pipeline.reconstruct_dataset_phase3 import _fit_volume, _mesh_from_volume
from pipeline.refine_saraswati_blind_relief import _normalized_depth, _warp_centered_values
from pipeline.refine_saraswati_multiview_relief import carve_axis_relief
from pipeline.run_saraswati_blind_benchmark import _foreground_mask, _sha256, _write_json


DepthInfer = Callable[[Path], dict[str, Any]]


def _side_correlation(
    left: np.ndarray,
    left_mask: np.ndarray,
    right: np.ndarray,
    right_mask: np.ndarray,
) -> float:
    width = min(left.shape[1], right.shape[1])
    height = min(left.shape[0], right.shape[0])
    size = (width, height)
    left = cv2.resize(left, size, interpolation=cv2.INTER_LINEAR)
    right = cv2.resize(np.fliplr(right), size, interpolation=cv2.INTER_LINEAR)
    left_mask = cv2.resize(left_mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    right_mask = cv2.resize(
        np.fliplr(right_mask).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
    ) > 0
    common = left_mask & right_mask
    if np.count_nonzero(common) <= 20:
        return 0.0
    return float(np.corrcoef(left[common], right[common])[0, 1])


def refine_uploaded_views(
    images: dict[str, Path],
    output_dir: Path,
    infer_depth: DepthInfer,
    *,
    resolution: int = 256,
    strength: float = 0.06,
) -> tuple[Path | None, Path]:
    """Return a refined OBJ when depth gates pass, otherwise a rejection record."""
    required = {"front", "top", "lsv", "rsv"}
    if set(images) != required:
        raise ValueError(f"exactly these views are required: {sorted(required)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    masks = {view: _foreground_mask(path)[0] for view, path in images.items()}
    volume, side_mode, fit, _ = _fit_volume(masks, resolution)
    normalized: dict[str, np.ndarray] = {}
    depth_records = {}
    for view, path in images.items():
        prediction = infer_depth(path)
        depth = np.asarray(prediction["depth_m"], dtype=np.float32).squeeze()
        if depth.shape != masks[view].shape:
            depth = cv2.resize(
                depth, (masks[view].shape[1], masks[view].shape[0]), interpolation=cv2.INTER_CUBIC
            )
        if not np.isfinite(depth).all() or np.any(depth <= 0):
            raise ValueError(f"invalid depth prediction for {view}")
        values, depth_range = _normalized_depth(depth, masks[view])
        normalized[view] = values
        depth_path = output_dir / f"{view}_depth_m.npy"
        np.save(depth_path, depth)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        agreement = _edge_agreement(image, values, masks[view])
        depth_records[view] = {
            "path": str(depth_path),
            "sha256": _sha256(depth_path),
            "edge_agreement": round(agreement, 6),
            "eligible": agreement >= 0.35,
            **depth_range,
        }

    correlation = _side_correlation(
        normalized["lsv"], masks["lsv"], normalized["rsv"], masks["rsv"]
    )
    selected = [view for view in ("front", "top") if depth_records[view]["eligible"]]
    if correlation >= 0.25:
        selected.extend(
            view for view in ("lsv", "rsv") if depth_records[view]["eligible"]
        )
    excluded = [view for view in ("front", "top", "lsv", "rsv") if view not in selected]
    operations = []
    refined = volume.copy()
    if {"front", "top"}.issubset(selected):
        for view in selected:
            scale = float(fit["axis_alignment"]["warp_scales_input_px_to_voxel"][view])
            warped, valid = _warp_centered_values(
                normalized[view], masks[view], scale, resolution
            )
            if view == "rsv":
                warped, valid = np.fliplr(warped), np.fliplr(valid)
            refined, operation = carve_axis_relief(
                refined, warped, valid, view, strength, "high"
            )
            operations.append(operation)

    accepted = bool(operations) and bool(np.any(refined))
    mesh_path: Path | None = None
    proposal = None
    if accepted:
        mesh = _mesh_from_volume(refined, nominal_size_mm=50.0)
        mesh_path = output_dir / "image_depth_relief_proposal.obj"
        mesh.export(mesh_path)
        proposal = {
            "path": str(mesh_path),
            "sha256": _sha256(mesh_path),
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
        }

    record = {
        "schema_version": "image2cad_uploaded_multiview_depth_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "depth_anything_v2_confidence_gated_multi_axis_relief",
        "resolution": resolution,
        "strength": strength,
        "side_consensus": side_mode,
        "depth_views": depth_records,
        "cross_view": {
            "left_right_mirrored_correlation": round(correlation, 6),
            "selected_views": selected,
            "excluded_views": excluded,
        },
        "operations": operations,
        "accepted_as_research_hypothesis": accepted,
        "proposal": proposal,
        "claims": {
            "model_outputs_metric_units": True,
            "physical_scale_recovered": False,
            "fine_relief_recovered": False,
            "manufacturing_accuracy_validated": False,
        },
    }
    record_path = output_dir / "depth_refinement_record.json"
    _write_json(record_path, record)
    return mesh_path, record_path
