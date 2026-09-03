"""Shared geometry definitions for individually addressable claw prongs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeVar

import numpy as np


PRONG_ANGLES = (45, 135, 225, 315)
PRONG_IDS = {45: "prong_ne", 135: "prong_nw", 225: "prong_sw", 315: "prong_se"}

VectorT = TypeVar("VectorT")


@dataclass(frozen=True)
class ClawControl:
    prong_id: str
    angle_degrees: int
    start: tuple[float, float, float]
    control_1: tuple[float, float, float]
    control_2: tuple[float, float, float]
    tip: tuple[float, float, float]
    radii: tuple[float, float, float, float]


def _point(radius: float, angle_degrees: float, z: float) -> tuple[float, float, float]:
    angle = np.deg2rad(angle_degrees)
    return float(radius * np.cos(angle)), float(radius * np.sin(angle)), float(z)


def claw_control(parameters: dict[str, float], angle: int) -> ClawControl:
    """Resolve one prong's cubic controls from shared and per-instance values."""
    prong_id = PRONG_IDS[angle]
    resolved_angle = angle + parameters.get(f"{prong_id}_angle_offset", 0.0)
    radial_offset = parameters.get(f"{prong_id}_radial_offset", 0.0)
    height_offset = parameters.get(f"{prong_id}_height_offset", 0.0)
    stone_radius = parameters["stone_radius"]
    lower_z = parameters["lower_gallery_z"]
    tip_z = parameters["prong_tip_z"] + height_offset
    crest_z = tip_z + parameters["prong_hook_drop"]
    start_z = lower_z + 0.08
    rise = max(0.2, crest_z - start_z)
    return ClawControl(
        prong_id=prong_id,
        angle_degrees=angle,
        start=_point(stone_radius + parameters["prong_base_extra"] + radial_offset, resolved_angle, start_z),
        control_1=_point(
            stone_radius + parameters["prong_c1_extra"] + radial_offset,
            resolved_angle,
            start_z + rise * parameters["prong_c1_height_fraction"],
        ),
        control_2=_point(stone_radius - parameters["prong_c2_inset"] + radial_offset, resolved_angle, crest_z),
        tip=_point(stone_radius - parameters["prong_tip_inset"] + radial_offset, resolved_angle, tip_z),
        radii=(
            parameters["prong_base_radius"], parameters["prong_mid_radius"],
            parameters["prong_neck_radius"], parameters["prong_tip_radius"],
        ),
    )


def cubic_points(
    control: ClawControl,
    vector: Callable[[tuple[float, float, float]], VectorT] = np.asarray,
    count: int = 11,
) -> tuple[list[VectorT], list[float]]:
    """Sample a cubic claw and a smooth four-key radius profile."""
    p0, p1, p2, p3 = (np.asarray(point, dtype=float) for point in (
        control.start, control.control_1, control.control_2, control.tip,
    ))
    points: list[VectorT] = []
    radii: list[float] = []
    for t in np.linspace(0.0, 1.0, count):
        point = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
        points.append(vector(tuple(float(value) for value in point)))
        radii.append(float(np.interp(t, (0.0, 0.42, 0.78, 1.0), control.radii)))
    return points, radii


def default_claw_parameters() -> dict[str, float | str]:
    result: dict[str, float | str] = {
        "prong_curve_version": "cubic_claw_v1",
        "prong_c1_extra": 0.48745,
        "prong_c1_height_fraction": 0.33333,
        "prong_c2_inset": -0.10943,
        "prong_tip_inset": 0.36636,
        "prong_tip_z": 14.507,
        "prong_hook_drop": 0.24,
        "prong_mid_radius": 0.30667,
        "prong_neck_radius": 0.235,
        "prong_tip_radius": 0.20,
        "prong_cap_radius": 0.5296,
    }
    for prong_id in PRONG_IDS.values():
        result[f"{prong_id}_angle_offset"] = 0.0
        result[f"{prong_id}_radial_offset"] = 0.0
        result[f"{prong_id}_height_offset"] = 0.0
    return result

