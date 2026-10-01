"""Point cloud parse and base-frame transform tests."""

from __future__ import annotations

import numpy as np
from viam.proto.common import Pose
from viam.spatialmath.rotation_matrix import RotationMatrix

from src.frames import pose_matrix_in_destination
from src.obstacles import parse_pcd, prepare_base_points
from tests.test_frames import _identity, _part


def _ascii(points: list[tuple[float, float, float]]) -> bytes:
    lines = [
        "VERSION .7",
        "FIELDS x y z",
        "SIZE 4 4 4",
        "TYPE F F F",
        "COUNT 1 1 1",
        f"WIDTH {len(points)}",
        "HEIGHT 1",
        f"POINTS {len(points)}",
        "DATA ascii",
    ]
    lines.extend(f"{x} {y} {z}" for x, y, z in points)
    return ("\n".join(lines) + "\n").encode()


def _binary(points: list[tuple[float, float, float]]) -> bytes:
    header = (
        "VERSION .7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {len(points)}\n"
        "HEIGHT 1\n"
        f"POINTS {len(points)}\n"
        "DATA binary\n"
    )
    body = np.asarray(points, dtype="<f4").tobytes()
    return header.encode() + body


def _camera_on_base(pose: Pose):
    return [
        _part("cart", "world", _identity(0, 0, 0)),
        _part("cam", "cart", pose),
    ]


def test_ascii_and_binary_pcd_match():
    points = [(0.0, 1000.0, 0.0), (10.0, 20.0, 30.0)]
    ascii_points = parse_pcd(_ascii(points))
    binary_points = parse_pcd(_binary(points))
    assert np.allclose(ascii_points, points)
    assert np.allclose(binary_points, points)


def test_body_and_optical_points_land_in_front_of_the_base():
    optical = RotationMatrix([1, 0, 0, 0, 0, 1, 0, -1, 0]).to_quaternion().to_pose(0, 200, 300)
    body = _identity(0, 200, 300)
    cases = (
        (body, (0.0, 1000.0, 0.0)),
        (optical, (0.0, 0.0, 1000.0)),
    )
    for pose, raw_point in cases:
        transform = pose_matrix_in_destination(_camera_on_base(pose), "cam", "cart")
        assert transform is not None
        cloud = np.array([raw_point, (0.0, 0.0, 0.0), (0.0, 10.0, 0.0)], dtype=float)
        xy = prepare_base_points(
            cloud,
            transform,
            z_min_m=0.05,
            z_max_m=2.0,
            footprint_length_m=0.6,
            footprint_width_m=0.4,
            max_range_m=10.0,
        )
        assert xy.shape == (1, 2)
        assert abs(xy[0, 0]) < 1e-6
        assert abs(xy[0, 1] - 1.2) < 1e-6
