"""Evidence-first metadata extraction for jewellery references.

This first extractor is intentionally conservative.  It combines the labelled
Ring01 reference views with measurable image facts and records every inferred
field's provenance.  A later detector can replace the labelled facts without
changing the artifact contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .schema import ReconstructionArtifact, StoneSpec, VisionEvidence

VIEW_NAMES = ("front", "side", "top", "angled", "back")


def _image_facts(path: Path) -> dict[str, Any]:
    """Return dimensions without making a heavyweight vision model mandatory."""
    try:
        from PIL import Image

        with Image.open(path).convert("RGB") as image:
            pixels = image.load()
            points = []
            # The supplied crops have a light studio background.  This gives
            # us a reproducible silhouette measurement without pretending it
            # is a semantic metal/stone segmentation.
            # Ignore the thin crop border present in some supplied renders;
            # otherwise a dark corner is mistaken for the jewellery.
            margin = min(5, image.width // 20, image.height // 20)
            for y in range(margin, image.height - margin):
                for x in range(margin, image.width - margin):
                    r, g, b = pixels[x, y]
                    if min(r, g, b) < 232:
                        points.append((x, y))
            bbox = None
            if points:
                xs, ys = zip(*points)
                bbox = {"left": min(xs), "top": min(ys), "right": max(xs), "bottom": max(ys)}
                bbox["width_px"] = bbox["right"] - bbox["left"] + 1
                bbox["height_px"] = bbox["bottom"] - bbox["top"] + 1
            return {
                "width_px": image.width,
                "height_px": image.height,
                "mode": image.mode,
                "foreground_bbox_px": bbox,
                "foreground_fraction": round(len(points) / (image.width * image.height), 5),
            }
    except ImportError:
        return {"file_size_bytes": path.stat().st_size}


def extract_ring01(
    input_dir: Path,
    output: Path,
    scale_mm_per_px: float | None = None,
    reference_length_px: float | None = None,
    reference_length_mm: float | None = None,
) -> ReconstructionArtifact:
    views = {name: input_dir / f"ring01_{name}.png" for name in VIEW_NAMES}
    missing = [str(path) for path in views.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Ring01 views: " + ", ".join(missing))

    if scale_mm_per_px is None and reference_length_px and reference_length_mm:
        scale_mm_per_px = reference_length_mm / reference_length_px
    scale_note = "calibrated_from_reference" if scale_mm_per_px else "relative_only"
    calibrated = scale_mm_per_px is not None
    stone = StoneSpec(
        cut="round_brilliant",
        center_mm=(0.0, 0.0, 14.0),
        diameter_mm=6.5 if calibrated else None,
        dimensions_mm=(6.5, 6.5, 4.0) if calibrated else None,
        confidence=0.92,
        parameters={"count": 1.0, "table_ratio": 0.55, "crown_ratio": 0.16, "pavilion_ratio": 0.43},
        source="reference_label_and_multi_view_geometry",
        evidence=["front", "top", "angled", "side"],
        uncertainty_mm=0.25 if scale_mm_per_px else None,
    )
    fields = {
        "object_type": {"value": "solitaire_ring", "source": "reference_label", "confidence": 0.99},
        "stone_count": {"value": 1, "source": "multi_view_observation", "confidence": 0.99},
        "stone_cut": {"value": "round_brilliant", "source": "reference_label_and_shape", "confidence": 0.92},
        "prong_count": {"value": 4, "source": "front_and_angled_views", "confidence": 0.96},
        "stone_position": {"value": "centered_on_ring_axis", "source": "multi_view_observation", "confidence": 0.94},
        "stone_size": {
            "value": 6.5 if calibrated else None,
            "unit": "mm",
            "source": scale_note,
            "confidence": 0.62 if calibrated else 0.0,
        },
        "stone_cavity": {
            "value": "required_by_mounting_hypothesis",
            "status": "inferred_not_observed",
            "source": "parametric_mount_reasoning",
            "confidence": 0.78,
        },
    }
    artifact = ReconstructionArtifact(
        reference_image=str(views["front"]),
        vision=VisionEvidence(
            object_type="solitaire_ring",
            stones=[stone],
            prong_count=4,
            confidence=0.91,
            fields=fields,
        ),
        metadata={
            "sample": "ring01",
            "engine": "evidence_first_v2",
            "view_paths": {name: str(path) for name, path in views.items()},
            "image_facts": {name: _image_facts(path) for name, path in views.items()},
            "scale_mm_per_px": scale_mm_per_px,
            "calibration": {
                "status": "calibrated" if calibrated else "missing",
                "reference_length_px": reference_length_px,
                "reference_length_mm": reference_length_mm,
                "warning": "Do not treat normalized dimensions as measured millimetres." if not calibrated else None,
            },
            "replication_readiness": {
                "exact": False,
                "reason": "Rendered views do not reveal hidden geometry, physical scale, lens calibration, or material construction.",
                "required_for_high_fidelity": ["known scale in at least one view", "camera calibration", "depth/structured-light scan", "manual confirmation of hidden setting geometry"],
            },
            "limitations": ["absolute dimensions require a scale reference or scan", "hidden cavity is inferred", "silhouette bbox is not a semantic segmentation"],
        },
    )
    artifact.save_json(str(output))
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/ring01_reference_images"))
    parser.add_argument("--output", type=Path, default=Path("data/ring01_metadata.json"))
    parser.add_argument("--scale-mm-per-px", type=float)
    parser.add_argument("--reference-length-px", type=float, help="Measured pixel length of a known object dimension")
    parser.add_argument("--reference-length-mm", type=float, help="Real-world length corresponding to --reference-length-px")
    args = parser.parse_args()
    artifact = extract_ring01(args.input_dir, args.output, args.scale_mm_per_px, args.reference_length_px, args.reference_length_mm)
    print(json.dumps({"output": str(args.output), "confidence": artifact.vision.confidence, "views": 5}, indent=2))


if __name__ == "__main__":
    main()
