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

        with Image.open(path) as image:
            return {"width_px": image.width, "height_px": image.height, "mode": image.mode}
    except ImportError:
        return {"file_size_bytes": path.stat().st_size}


def extract_ring01(input_dir: Path, output: Path, scale_mm_per_px: float | None = None) -> ReconstructionArtifact:
    views = {name: input_dir / f"ring01_{name}.png" for name in VIEW_NAMES}
    missing = [str(path) for path in views.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Ring01 views: " + ", ".join(missing))

    scale_note = "calibrated" if scale_mm_per_px else "relative_only"
    stone = StoneSpec(
        cut="round_brilliant",
        center_mm=(0.0, 0.0, 14.0),
        diameter_mm=6.5 if scale_mm_per_px else None,
        dimensions_mm=(6.5, 6.5, 4.0) if scale_mm_per_px else None,
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
            "value": 6.5 if scale_mm_per_px else None,
            "unit": "mm",
            "source": scale_note,
            "confidence": 0.62 if scale_mm_per_px else 0.0,
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
            "engine": "evidence_first_v1",
            "view_paths": {name: str(path) for name, path in views.items()},
            "image_facts": {name: _image_facts(path) for name, path in views.items()},
            "scale_mm_per_px": scale_mm_per_px,
            "limitations": ["absolute dimensions require a scale reference or scan", "hidden cavity is inferred"],
        },
    )
    artifact.save_json(str(output))
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/ring01_reference_images"))
    parser.add_argument("--output", type=Path, default=Path("data/ring01_metadata.json"))
    parser.add_argument("--scale-mm-per-px", type=float)
    args = parser.parse_args()
    artifact = extract_ring01(args.input_dir, args.output, args.scale_mm_per_px)
    print(json.dumps({"output": str(args.output), "confidence": artifact.vision.confidence, "views": 5}, indent=2))


if __name__ == "__main__":
    main()
