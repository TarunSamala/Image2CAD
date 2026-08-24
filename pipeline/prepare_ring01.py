"""Prepare the Ring 01 multi-view reference as a reconstruction artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from .schema import ReconstructionArtifact, VisionEvidence, StoneSpec


VIEW_NAMES = ("front", "side", "top", "angled", "back")


def prepare(input_dir: Path, output: Path) -> ReconstructionArtifact:
    views = {name: input_dir / f"ring01_{name}.png" for name in VIEW_NAMES}
    missing = [str(path) for path in views.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Ring 01 views: " + ", ".join(missing))

    artifact = ReconstructionArtifact(
        reference_image=str(views["front"]),
        vision=VisionEvidence(
            object_type="solitaire_ring",
            stones=[
                StoneSpec(
                    cut="round_brilliant",
                    center_mm=(0.0, 0.0, 0.0),
                    confidence=1.0,
                    parameters={"count": 1.0},
                )
            ],
            prong_count=4,
            confidence=1.0,
        ),
        metadata={
            "sample": "ring01",
            "view_paths": {name: str(path) for name, path in views.items()},
            "view_count": len(views),
            "notes": "Known reference labels; dimensions remain to be estimated.",
        },
    )
    artifact.save_json(str(output))
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/ring01_reference_images"))
    parser.add_argument("--output", type=Path, default=Path("data/ring01_artifact.json"))
    args = parser.parse_args()
    prepare(args.input_dir, args.output)
    print(f"Prepared Ring 01 artifact: {args.output}")


if __name__ == "__main__":
    main()
