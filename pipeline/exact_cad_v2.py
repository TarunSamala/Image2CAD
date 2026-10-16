"""Reusable exact-CAD V2 generators for jewellery structural components.

The generators consume explicit designed parameters.  They are deliberately
separate from image inference so machine-proposed measurements cannot silently
be promoted to physical truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


@dataclass(frozen=True)
class StoneSpec:
    component_id: str
    kind: str = "round"
    x_mm: float = 0.0
    y_mm: float = 0.0
    length_mm: float = 5.8
    width_mm: float = 5.8
    depth_mm: float = 3.2
    setting_style: str = "prong"
    prong_count: int = 4


@dataclass(frozen=True)
class RingSpec:
    object_id: str
    family: str
    inner_radius_mm: float = 8.3
    shank_radius_mm: float = 1.15
    head_depth_mm: float = 3.0
    stones: tuple[StoneSpec, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RingBodySpec:
    """Scale-normalized exact shank used before head/detail reconstruction."""

    object_id: str
    inner_width: float
    inner_height: float
    radial_thickness: float
    axial_depth: float
    scale_status: str = "nominal_display_scale_not_metric"


@dataclass(frozen=True)
class HeadSupportSpec:
    component_id: str
    height: float
    base_z: float
    shoulder_radius: float
    scale_status: str = "nominal_display_scale_not_metric"


def build_ring_body(cq, spec: RingBodySpec) -> dict[str, Any]:
    """Build one exact elliptical annular shank in the XZ ring plane."""
    if min(
        spec.inner_width,
        spec.inner_height,
        spec.radial_thickness,
        spec.axial_depth,
    ) <= 0.0:
        raise ValueError("Ring-body dimensions must be positive")
    outer_width = spec.inner_width + 2.0 * spec.radial_thickness
    outer_height = spec.inner_height + 2.0 * spec.radial_thickness
    body = (
        cq.Workplane("XZ")
        .ellipse(outer_width / 2.0, outer_height / 2.0)
        .ellipse(spec.inner_width / 2.0, spec.inner_height / 2.0)
        .extrude(spec.axial_depth / 2.0, both=True)
        .val()
        .clean()
    )
    return {
        "spec": spec,
        "shank": body,
        "assembly": body,
        "dimensions": {
            "inner_width": spec.inner_width,
            "inner_height": spec.inner_height,
            "outer_width": outer_width,
            "outer_height": outer_height,
            "radial_thickness": spec.radial_thickness,
            "axial_depth": spec.axial_depth,
        },
    }


def build_profiled_ring_body(
    cq,
    spec: RingBodySpec,
    inner_profile_xz: tuple[tuple[float, float], ...],
    outer_profile_xz: tuple[tuple[float, float], ...],
) -> dict[str, Any]:
    """Build an exact shank from periodic image-derived profile controls."""
    if len(inner_profile_xz) < 12 or len(outer_profile_xz) < 12:
        raise ValueError("Profiled ring bodies require at least 12 controls")
    if len(inner_profile_xz) != len(outer_profile_xz):
        raise ValueError("Inner and outer profiles must have equal control counts")
    base_y = -spec.axial_depth / 2.0
    inner_vectors = [cq.Vector(x, base_y, z) for x, z in inner_profile_xz]
    outer_vectors = [cq.Vector(x, base_y, z) for x, z in outer_profile_xz]
    inner_wire = cq.Wire.assembleEdges(
        [cq.Edge.makeSpline(inner_vectors, periodic=True)]
    )
    outer_wire = cq.Wire.assembleEdges(
        [cq.Edge.makeSpline(outer_vectors, periodic=True)]
    )
    body = cq.Solid.extrudeLinear(
        outer_wire,
        [inner_wire],
        cq.Vector(0.0, spec.axial_depth, 0.0),
    ).clean()
    return {
        "spec": spec,
        "shank": body,
        "assembly": body,
        "inner_wire": inner_wire,
        "outer_wire": outer_wire,
        "profile_control_count": len(inner_profile_xz),
        "dimensions": {
            "inner_width": spec.inner_width,
            "inner_height": spec.inner_height,
            "radial_thickness": spec.radial_thickness,
            "axial_depth": spec.axial_depth,
        },
    }


def build_head_support_envelope(
    cq,
    ring_body: dict[str, Any],
    spec: HeadSupportSpec,
    front_profile_xy: tuple[tuple[float, float], ...],
    top_profile_xz: tuple[tuple[float, float], ...],
    front_opening_profiles_xy: tuple[
        tuple[tuple[float, float], ...], ...
    ] = (),
) -> dict[str, Any]:
    """Attach an exact two-view head visual hull and shoulder supports."""
    if len(front_profile_xy) < 12 or len(top_profile_xz) < 12:
        raise ValueError("Head support requires at least 12 controls per profile")
    if spec.height <= 0.0 or spec.shoulder_radius <= 0.0:
        raise ValueError("Head-support dimensions must be positive")
    vectors = [cq.Vector(x, y, spec.base_z) for x, y in front_profile_xy]
    profile_wire = cq.Wire.makePolygon([*vectors, vectors[0]], close=True)
    front_prism = cq.Solid.extrudeLinear(
        profile_wire,
        [],
        cq.Vector(0.0, 0.0, spec.height),
    ).clean()
    profile_depth = max(y for _, y in front_profile_xy) - min(
        y for _, y in front_profile_xy
    )
    top_vectors = [
        cq.Vector(x, -profile_depth / 2.0, z) for x, z in top_profile_xz
    ]
    top_wire = cq.Wire.makePolygon([*top_vectors, top_vectors[0]], close=True)
    top_prism = cq.Solid.extrudeLinear(
        top_wire,
        [],
        cq.Vector(0.0, profile_depth, 0.0),
    ).clean()
    head = front_prism.intersect(top_prism).clean()
    opening_trials: list[dict[str, Any]] = []
    accepted_opening_cutters: list[Any] = []
    cutter_padding = max(spec.height * 0.01, 0.02)
    for opening_index, opening_profile in enumerate(
        front_opening_profiles_xy, start=1
    ):
        trial: dict[str, Any] = {
            "profile_index": opening_index - 1,
            "accepted": False,
            "reason": "invalid_profile",
            "removed_volume": 0.0,
        }
        if len(opening_profile) < 8:
            opening_trials.append(trial)
            continue
        opening_vectors = [
            cq.Vector(x, y, spec.base_z - cutter_padding)
            for x, y in opening_profile
        ]
        try:
            opening_wire = cq.Wire.makePolygon(
                [*opening_vectors, opening_vectors[0]], close=True
            )
            cutter = cq.Solid.extrudeLinear(
                opening_wire,
                [],
                cq.Vector(0.0, 0.0, spec.height + 2.0 * cutter_padding),
            ).clean()
            previous_volume = float(head.Volume())
            candidate = head.cut(cutter).clean()
            removed_volume = previous_volume - float(candidate.Volume())
            trial["removed_volume"] = removed_volume
            if removed_volume <= 1e-7:
                trial["reason"] = "no_head_intersection"
            elif not candidate.isValid():
                trial["reason"] = "invalid_brep_after_cut"
            elif len(candidate.Solids()) != 1:
                trial["reason"] = "head_disconnected_after_cut"
            else:
                head = candidate
                accepted_opening_cutters.append(cutter)
                trial["accepted"] = True
                trial["reason"] = "valid_connected_exact_cut"
        except Exception as error:  # OCCT rejects degenerate image contours.
            trial["reason"] = f"boolean_error:{type(error).__name__}"
        opening_trials.append(trial)

    body_spec = ring_body["spec"]
    top_z = spec.base_z + spec.height * 0.22
    start_z = body_spec.inner_height * 0.38
    start_x = body_spec.inner_width * 0.31
    profile_width = max(x for x, _ in front_profile_xy) - min(
        x for x, _ in front_profile_xy
    )
    end_x = max(start_x * 0.72, profile_width * 0.30)

    shoulders: list[Any] = []
    for sign in (-1.0, 1.0):
        start = cq.Vector(sign * start_x, 0.0, start_z)
        end = cq.Vector(sign * end_x, 0.0, top_z)
        direction = end - start
        length = direction.Length
        shoulder = cq.Solid.makeCone(
            spec.shoulder_radius,
            spec.shoulder_radius * 0.82,
            length,
            start,
            direction.normalized(),
        )
        shoulders.append(shoulder)

    positives = [ring_body["shank"], *shoulders, head]
    metal = positives[0].copy()
    for positive in positives[1:]:
        metal = metal.fuse(positive.copy()).clean()
    return {
        "spec": spec,
        "shank": ring_body["shank"],
        "head_support": head,
        "front_prism": front_prism,
        "top_prism": top_prism,
        "shoulders": shoulders,
        "metal": metal,
        "assembly": metal,
        "profile_wire": profile_wire,
        "top_profile_wire": top_wire,
        "opening_trials": opening_trials,
        "accepted_opening_cutters": accepted_opening_cutters,
    }


def _ellipse_prism(cq, length: float, width: float, height: float):
    return cq.Workplane("XY").ellipse(length / 2.0, width / 2.0).extrude(height).val()


def _stone_shape(cq, stone: StoneSpec, bottom_z: float):
    return (
        cq.Workplane("XY")
        .ellipse(stone.length_mm * 0.34, stone.width_mm * 0.34)
        .workplane(offset=stone.depth_mm * 0.42)
        .ellipse(stone.length_mm * 0.50, stone.width_mm * 0.50)
        .workplane(offset=stone.depth_mm * 0.58)
        .ellipse(stone.length_mm * 0.30, stone.width_mm * 0.30)
        .loft(combine=True)
        .val()
        .translate((stone.x_mm, stone.y_mm, bottom_z))
    )


def _setting_components(cq, stone: StoneSpec, head_z: float) -> dict[str, Any]:
    clearance = 0.10
    wall = 0.32
    seat_height = 0.54
    seat_z = head_z + 0.48
    stone_z = head_z + 0.58
    outer = _ellipse_prism(
        cq,
        stone.length_mm + 2.0 * (clearance + wall),
        stone.width_mm + 2.0 * (clearance + wall),
        seat_height,
    )
    inner = _ellipse_prism(
        cq,
        stone.length_mm + 2.0 * clearance,
        stone.width_mm + 2.0 * clearance,
        seat_height + 0.20,
    ).translate((0.0, 0.0, -0.10))
    seat = outer.cut(inner).translate((stone.x_mm, stone.y_mm, seat_z))
    cavity = _ellipse_prism(
        cq,
        stone.length_mm + 2.0 * clearance,
        stone.width_mm + 2.0 * clearance,
        1.45,
    ).translate((stone.x_mm, stone.y_mm, head_z + 0.12))
    components: dict[str, Any] = {
        stone.component_id: _stone_shape(cq, stone, stone_z),
        stone.component_id.replace("stone_", "seat_"): seat,
        stone.component_id.replace("stone_", "cavity_"): cavity,
    }
    if stone.setting_style == "bezel":
        bezel_outer = _ellipse_prism(
            cq,
            stone.length_mm + 2.0 * (clearance + wall * 0.72),
            stone.width_mm + 2.0 * (clearance + wall * 0.72),
            stone.depth_mm * 0.52,
        )
        bezel_inner = _ellipse_prism(
            cq,
            stone.length_mm + 2.0 * clearance,
            stone.width_mm + 2.0 * clearance,
            stone.depth_mm * 0.52 + 0.20,
        ).translate((0.0, 0.0, -0.10))
        components[stone.component_id.replace("stone_", "bezel_")] = (
            bezel_outer.cut(bezel_inner).translate(
                (stone.x_mm, stone.y_mm, seat_z)
            )
        )
        return components

    radius_x = stone.length_mm / 2.0 + clearance + wall * 0.42
    radius_y = stone.width_mm / 2.0 + clearance + wall * 0.42
    prong_radius = max(0.15, min(0.27, min(stone.length_mm, stone.width_mm) * 0.075))
    prong_height = stone.depth_mm + 0.74
    prong_z = head_z + 0.35
    for index in range(stone.prong_count):
        angle = math.radians(45.0 + index * 360.0 / stone.prong_count)
        x = stone.x_mm + radius_x * math.cos(angle)
        y = stone.y_mm + radius_y * math.sin(angle)
        prong = (
            cq.Workplane("XY")
            .circle(prong_radius)
            .extrude(prong_height)
            .faces(">Z")
            .workplane()
            .sphere(prong_radius * 1.16, combine=True)
            .val()
            .translate((x, y, prong_z))
        )
        components[f"prong_{stone.component_id}_{index + 1:02d}"] = prong
    return components


def build_ring(cq, spec: RingSpec) -> dict[str, Any]:
    """Build exact source components plus a fused metal and separate stones."""
    if not spec.stones:
        raise ValueError("A ring specification requires at least one stone")
    major_radius = spec.inner_radius_mm + spec.shank_radius_mm
    shank = cq.Solid.makeTorus(major_radius, spec.shank_radius_mm).rotate(
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 90.0
    )
    head_z = major_radius + spec.shank_radius_mm - 0.38
    left = min(
        stone.x_mm - stone.length_mm / 2.0 - 0.8 for stone in spec.stones
    )
    right = max(
        stone.x_mm + stone.length_mm / 2.0 + 0.8 for stone in spec.stones
    )
    head_base = (
        cq.Workplane("XY")
        .box(right - left, spec.head_depth_mm, 0.82, centered=(True, True, False))
        .translate(((left + right) / 2.0, 0.0, head_z))
        .val()
    )
    components: dict[str, Any] = {
        "shank_001": shank,
        "head_base_001": head_base,
    }
    for stone in spec.stones:
        components.update(_setting_components(cq, stone, head_z))

    positives = [
        (component_id, shape)
        for component_id, shape in components.items()
        if not component_id.startswith("stone_")
        and not component_id.startswith("cavity_")
    ]
    cavities = [
        shape for component_id, shape in components.items()
        if component_id.startswith("cavity_")
    ]
    stones = [
        shape for component_id, shape in components.items()
        if component_id.startswith("stone_")
    ]
    # Sequential Booleans are slower than a many-shape fuse but avoid the
    # invalid coincident-face result OCCT can produce for dense halo layouts.
    # OCCT Boolean operations can mutate wrapper state; work on copies so the
    # source component solids remain independently inspectable and exportable.
    metal = positives[0][1].copy()
    union_trace = [{"component_id": positives[0][0], "solid_count": 1}]
    for component_id, positive in positives[1:]:
        metal = metal.fuse(positive.copy()).clean()
        union_trace.append({
            "component_id": component_id,
            "solid_count": len(metal.Solids()),
        })
    for cavity in cavities:
        metal = metal.cut(cavity.copy()).clean()
    assembly = cq.Compound.makeCompound([metal, *stones])
    return {
        "spec": spec,
        "components": components,
        "metal": metal,
        "stones": stones,
        "assembly": assembly,
        "head_z": head_z,
        "union_trace": union_trace,
    }


def default_family_specs() -> tuple[RingSpec, ...]:
    halo_stones = [
        StoneSpec("stone_001", length_mm=5.4, width_mm=5.4, depth_mm=3.0)
    ]
    for index in range(8):
        angle = math.radians(index * 45.0)
        halo_stones.append(
            StoneSpec(
                f"stone_{index + 2:03d}",
                x_mm=4.0 * math.cos(angle),
                y_mm=3.8 * math.sin(angle),
                length_mm=1.35,
                width_mm=1.35,
                depth_mm=0.95,
                setting_style="bezel",
                prong_count=0,
            )
        )
    return (
        RingSpec(
            "cad_v2_round_solitaire", "round_solitaire",
            stones=(StoneSpec("stone_001"),),
        ),
        RingSpec(
            "cad_v2_oval_solitaire", "oval_solitaire",
            stones=(StoneSpec(
                "stone_001", kind="oval", length_mm=7.2, width_mm=5.2,
                depth_mm=3.4,
            ),),
        ),
        RingSpec(
            "cad_v2_three_stone", "three_stone",
            stones=(
                StoneSpec("stone_001", length_mm=4.8, width_mm=4.8, depth_mm=2.8),
                StoneSpec("stone_002", x_mm=-4.2, length_mm=3.3, width_mm=3.3, depth_mm=2.0),
                StoneSpec("stone_003", x_mm=4.2, length_mm=3.3, width_mm=3.3, depth_mm=2.0),
            ),
        ),
        RingSpec(
            "cad_v2_halo", "halo", head_depth_mm=10.2,
            stones=tuple(halo_stones),
        ),
    )
