"""Frame-system chain and footprint tests."""

from __future__ import annotations

import math

from viam.proto.common import Geometry, Pose, PoseInFrame, RectangularPrism, Transform, Vector3
from viam.proto.robot import FrameSystemConfig
from viam.spatialmath.rotation_matrix import RotationMatrix

from src.frames import (
    base_box_forward_lateral,
    pose_matrix_in_destination,
    resolve_footprint,
)


def _part(name: str, parent: str, pose: Pose, box: tuple[float, float, float] | None = None):
    geometry = None
    if box is not None:
        geometry = Geometry(
            box=RectangularPrism(dims_mm=Vector3(x=box[0], y=box[1], z=box[2]))
        )
    return FrameSystemConfig(
        frame=Transform(
            reference_frame=name,
            pose_in_observer_frame=PoseInFrame(reference_frame=parent, pose=pose),
            physical_object=geometry,
        )
    )


def _identity(x: float, y: float, z: float) -> Pose:
    return Pose(x=x, y=y, z=z, o_x=0, o_y=0, o_z=1, theta=0)


def test_camera_parented_to_world_is_rejected():
    configs = [
        _part("cart", "world", _identity(0, 0, 0)),
        _part("cam", "world", _identity(0, 200, 300)),
    ]
    assert pose_matrix_in_destination(configs, "cam", "cart") is None


def test_camera_chain_must_reach_the_base():
    configs = [
        _part("cart", "world", _identity(0, 0, 0)),
        _part("mast", "world", _identity(0, 100, 0)),
        _part("cam", "mast", _identity(0, 50, 0)),
    ]
    assert pose_matrix_in_destination(configs, "cam", "cart") is None


def test_camera_mounted_through_a_link_on_the_base():
    configs = [
        _part("cart", "world", _identity(0, 0, 90), box=(400, 800, 200)),
        _part("mast", "cart", _identity(0, 100, 0)),
        _part("cam", "mast", _identity(10, 50, 200)),
    ]
    transform = pose_matrix_in_destination(configs, "cam", "cart")
    assert transform is not None
    assert transform[0, 3] == 10
    assert transform[1, 3] == 150
    assert transform[2, 3] == 200
    box = base_box_forward_lateral(configs, "cart")
    assert box == (0.8, 0.4)
    length, width = resolve_footprint(box, length_m=None, width_m=0.5, robot_radius_m=0.25)
    assert length == 0.8
    assert width == 0.5
    length, width = resolve_footprint(None, length_m=None, width_m=None, robot_radius_m=0.25)
    assert (length, width) == (0.5, 0.5)


def test_optical_pose_round_trip_matrix():
    # Camera +Z (forward) → base +Y, camera +X (right) → base +X, camera +Y → base -Z.
    elements = [1, 0, 0, 0, 0, 1, 0, -1, 0]
    pose = RotationMatrix(elements).to_quaternion().to_pose(0, 200, 300)
    configs = [
        _part("cart", "world", _identity(0, 0, 0)),
        _part("cam", "cart", pose),
    ]
    transform = pose_matrix_in_destination(configs, "cam", "cart")
    assert transform is not None
    point = transform[:3, :3] @ [0.0, 0.0, 1000.0] + transform[:3, 3]
    assert abs(point[0] - 0.0) < 1e-6
    assert abs(point[1] - 1200.0) < 1e-6
    assert abs(point[2] - 300.0) < 1e-6
    assert abs(math.hypot(point[0], point[1]) - 1200.0) < 1e-6
