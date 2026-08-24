"""Dependency-light contracts shared by vision, 3D, CAD, and validation stages.

The model stages are deliberately represented as evidence and proposals.  This
keeps measurements from the reference image available to the geometric
reasoner instead of compressing everything into a generated mesh.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal
import json


@dataclass
class StoneSpec:
    cut: str
    center_mm: tuple[float, float, float]
    diameter_mm: float | None = None
    dimensions_mm: tuple[float, float, float] | None = None
    confidence: float = 0.0
    parameters: dict[str, float] = field(default_factory=dict)


@dataclass
class VisionEvidence:
    object_type: str = "unknown"
    camera_pose: dict[str, float] = field(default_factory=dict)
    metal_mask_path: str | None = None
    depth_path: str | None = None
    normal_path: str | None = None
    stones: list[StoneSpec] = field(default_factory=list)
    prong_count: int | None = None
    measurements_mm: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class ComponentProposal:
    component_id: str
    kind: Literal["metal", "stone", "unknown"]
    source: Literal["vision", "foundation_model", "parametric"]
    mesh_path: str | None = None
    cad_parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ReconstructionArtifact:
    """The authoritative state passed between reconstruction stages."""

    reference_image: str
    vision: VisionEvidence = field(default_factory=VisionEvidence)
    components: list[ComponentProposal] = field(default_factory=list)
    master_geometry_path: str | None = None
    exports: dict[str, str] = field(default_factory=dict)
    validation: ValidationReport | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

