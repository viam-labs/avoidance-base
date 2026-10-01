"""Resolve camera poses and the base footprint from the Viam frame system.

Poses stay in Viam's base frame: millimetres, +Y forward, +X right, +Z up.
A camera is usable only when walking its parents reaches the configured base
before ``world``.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional, Sequence, Tuple

import numpy as np

WORLD_FRAME = "world"
FRAME_SYSTEM_TIMEOUT_S = 5.0


def _pose_to_matrix(pose) -> np.ndarray:
    """4×4 pose of a child in its parent, translation in millimetres."""
    from viam.proto.common import Orientation
    from viam.spatialmath import OrientationVector

    ov = OrientationVector.from_proto(
        Orientation(
            o_x=float(pose.o_x),
            o_y=float(pose.o_y),
            o_z=float(pose.o_z),
            theta=float(pose.theta),
        )
    )
    # The spatialmath buffer is column-major. Reading it row-major sends an
    # orientation-vector +Z of +Y (o_y=1) to base -Y, so a forward depth cloud
    # is painted behind the robot.
    rotation = np.asarray(ov.to_quaternion().to_rotation_matrix().elements, dtype=float)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation.reshape((3, 3), order="F")
    transform[0, 3] = float(pose.x)
    transform[1, 3] = float(pose.y)
    transform[2, 3] = float(pose.z)
    return transform


def _frame_entries(configs: Sequence[Any]) -> dict[str, Any]:
    """Map frame name → frame entry (``Transform`` or the same shape)."""
    out: dict[str, Any] = {}
    for cfg in configs:
        frame = getattr(cfg, "frame", None) or cfg
        name = str(getattr(frame, "reference_frame", "") or "")
        if name:
            out[name] = frame
    return out


def pose_matrix_in_destination(
    configs: Sequence[Any],
    frame_name: str,
    destination: str,
) -> Optional[np.ndarray]:
    """``destination_T_frame`` (millimetres), or ``None`` when the chain misses.

    The walk follows each frame's parent. If ``world`` appears before
    ``destination``, the camera is not mounted on the base and the result is
    ``None``.
    """
    frames = _frame_entries(configs)
    if frame_name == destination:
        return np.eye(4, dtype=float)
    if frame_name not in frames:
        return None

    transform = np.eye(4, dtype=float)
    current = frame_name
    seen: set[str] = set()
    while current != destination:
        if current in seen or len(seen) > 64:
            return None
        seen.add(current)
        entry = frames.get(current)
        if entry is None:
            return None
        pose_in_parent = entry.pose_in_observer_frame
        parent = str(getattr(pose_in_parent, "reference_frame", "") or "")
        if not parent:
            return None
        if parent == WORLD_FRAME and destination != WORLD_FRAME:
            return None
        transform = _pose_to_matrix(pose_in_parent.pose) @ transform
        current = parent
    return transform


def base_box_forward_lateral(
    configs: Sequence[Any],
    base_name: str,
) -> Optional[Tuple[float, float]]:
    """``(length_m, width_m)`` of the base box.

    Viam base axes: +Y forward is length, +X right is width.
    """
    frames = _frame_entries(configs)
    entry = frames.get(base_name)
    if entry is None:
        return None
    geom = getattr(entry, "physical_object", None)
    if geom is None or not geom.ByteSize():
        return None
    if geom.WhichOneof("geometry_type") != "box":
        return None
    dims = geom.box.dims_mm
    lateral = abs(float(dims.x)) / 1000.0
    forward = abs(float(dims.y)) / 1000.0
    if lateral < 1e-4 or forward < 1e-4:
        return None
    return forward, lateral


def resolve_footprint(
    box: Optional[Tuple[float, float]],
    *,
    length_m: Optional[float],
    width_m: Optional[float],
    robot_radius_m: float,
) -> Tuple[float, float]:
    """Length (forward) and width (lateral) in metres.

    An explicit side overrides the base box. A side that is still unset uses
    ``2 * robot_radius_m``. With neither box nor overrides, both sides do.
    """
    fallback = 2.0 * float(robot_radius_m)
    length = length_m if length_m is not None and length_m > 0 else None
    width = width_m if width_m is not None and width_m > 0 else None
    if box is not None:
        if length is None:
            length = float(box[0])
        if width is None:
            width = float(box[1])
    if length is None:
        length = fallback
    if width is None:
        width = fallback
    return length, width


async def fetch_frame_system_config(
    robot, *, timeout_s: float = FRAME_SYSTEM_TIMEOUT_S
) -> list[Any]:
    """``robot.get_frame_system_config()`` on the caller's event loop."""
    return list(await asyncio.wait_for(robot.get_frame_system_config(), timeout=timeout_s))
