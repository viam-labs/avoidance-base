"""Parse camera point clouds and express them in the base frame."""

from __future__ import annotations

import math
from typing import List

import numpy as np


def parse_pcd(raw: bytes) -> np.ndarray:
    """Parse a Viam PCD into ``(N, 3)`` XYZ in the file's units (millimetres).

    Supports ASCII and uncompressed little-endian binary clouds with x/y/z.
    Extra fields are ignored.
    """
    if not raw:
        return np.empty((0, 3))
    marker = raw.find(b"DATA ")
    if marker < 0:
        return np.empty((0, 3))
    newline = raw.find(b"\n", marker)
    if newline < 0:
        return np.empty((0, 3))
    header_text = raw[:newline].decode("ascii", errors="replace")
    data_fmt = raw[marker + 5 : newline].decode("ascii").strip().lower()
    body = raw[newline + 1 :]

    fields: List[str] = []
    sizes: List[int] = []
    types: List[str] = []
    counts: List[int] = []
    npoints = 0
    for line in header_text.splitlines():
        parts = line.split()
        if not parts:
            continue
        key = parts[0].upper()
        if key == "FIELDS":
            fields = parts[1:]
        elif key == "SIZE":
            sizes = [int(part) for part in parts[1:]]
        elif key == "TYPE":
            types = parts[1:]
        elif key == "COUNT":
            counts = [int(part) for part in parts[1:]]
        elif key == "POINTS":
            npoints = int(parts[1])
        elif key == "WIDTH" and npoints == 0:
            npoints = int(parts[1])
    if not counts:
        counts = [1] * len(fields)
    if not fields or not {"x", "y", "z"} <= set(fields):
        return np.empty((0, 3))

    if data_fmt == "ascii":
        rows = [row.split() for row in body.decode("ascii", errors="replace").splitlines() if row.strip()]
        if not rows:
            return np.empty((0, 3))
        array = np.array(rows, dtype=float)
        columns = {name: index for index, name in enumerate(fields)}
        return array[:, [columns["x"], columns["y"], columns["z"]]].astype(float)

    type_map = {
        ("F", 4): "f4",
        ("F", 8): "f8",
        ("U", 1): "u1",
        ("U", 2): "u2",
        ("U", 4): "u4",
        ("I", 1): "i1",
        ("I", 2): "i2",
        ("I", 4): "i4",
    }
    dtype_fields = []
    for name, size, kind, count in zip(fields, sizes, types, counts):
        numpy_type = type_map.get((kind.upper(), size), f"V{size}")
        for index in range(count):
            field_name = name if count == 1 else f"{name}_{index}"
            dtype_fields.append((field_name, np.dtype(numpy_type)))
    if not dtype_fields:
        return np.empty((0, 3))
    record = np.dtype(dtype_fields)
    if record.itemsize == 0:
        return np.empty((0, 3))
    if npoints == 0:
        npoints = len(body) // record.itemsize
    structured = np.frombuffer(body[: npoints * record.itemsize], dtype=record)
    names = set(structured.dtype.names or ())
    if not {"x", "y", "z"} <= names:
        return np.empty((0, 3))
    return np.stack([structured["x"], structured["y"], structured["z"]], axis=1).astype(float)


def _downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    step = int(math.ceil(len(points) / float(max_points)))
    return points[:: max(1, step)]


def prepare_base_points(
    points_mm: np.ndarray,
    base_t_camera_mm: np.ndarray,
    *,
    z_min_m: float,
    z_max_m: float,
    footprint_length_m: float,
    footprint_width_m: float,
    max_range_m: float,
    max_points: int = 8000,
) -> np.ndarray:
    """Return ``(N, 2)`` obstacle XY in the base frame, metres.

    ``base_t_camera_mm`` is the camera pose in the base (+Y forward, +X right).
    Invalid and ``(0, 0, 0)`` pixels are dropped. Points inside the unpadded
    footprint are dropped so the chassis does not count as an obstacle.
    """
    points = np.asarray(points_mm, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3 or points.size == 0:
        return np.empty((0, 2))
    points = points[:, :3]
    keep = np.isfinite(points).all(axis=1) & np.any(points != 0.0, axis=1)
    points = points[keep]
    if points.size == 0:
        return np.empty((0, 2))

    rotation = np.asarray(base_t_camera_mm, dtype=float)[:3, :3]
    translation = np.asarray(base_t_camera_mm, dtype=float)[:3, 3]
    transformed_m = (points @ rotation.T + translation) / 1000.0
    band = (transformed_m[:, 2] >= z_min_m) & (transformed_m[:, 2] <= z_max_m)
    transformed_m = transformed_m[band]
    if transformed_m.size == 0:
        return np.empty((0, 2))

    xy = transformed_m[:, :2]
    half_length = float(footprint_length_m) / 2.0
    half_width = float(footprint_width_m) / 2.0
    if half_length > 0.0 and half_width > 0.0:
        inside = (np.abs(xy[:, 0]) <= half_width) & (np.abs(xy[:, 1]) <= half_length)
        xy = xy[~inside]
    if xy.size == 0:
        return np.empty((0, 2))
    if max_range_m > 0.0:
        radial = np.hypot(xy[:, 0], xy[:, 1])
        xy = xy[radial <= max_range_m]
    return _downsample(xy, max_points)
