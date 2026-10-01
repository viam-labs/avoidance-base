"""Avoidance base command gating."""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest
from viam.components.base import Vector3
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from src.avoidance_base import AvoidanceBase, ObstacleStop
from src.guard import GuardConfig


class _Child:
    def __init__(self) -> None:
        self.stopped = 0
        self.velocities: list[tuple[float, float]] = []

    async def stop(self, **kwargs):
        del kwargs
        self.stopped += 1

    async def set_velocity(self, linear, angular, **kwargs):
        del kwargs
        self.velocities.append((float(linear.y), float(angular.z)))

    async def is_moving(self) -> bool:
        return False

    async def do_command(self, command, **kwargs):
        del kwargs
        return {"forwarded": command.get("command")}


def _ready(points: np.ndarray) -> tuple[AvoidanceBase, _Child]:
    base = AvoidanceBase("avoid")
    child = _Child()
    base._child = child  # type: ignore[assignment]
    base._frames_ok = True
    base._camera_names = ["cam"]
    base._guard = GuardConfig(
        length_m=0.6,
        width_m=0.4,
        padding_m=0.05,
        stop_gap_m=0.04,
        time_to_collision_s=1.2,
        min_points=1,
    )
    base._clouds = {"cam": (time.monotonic(), points)}
    base._source_timeout_s = 0.5
    return base, child


def test_validate_config_requires_base_and_cameras():
    config = ComponentConfig(
        attributes=dict_to_struct({"base": "cart", "cameras": ["front", "rear"]})
    )
    required, optional = AvoidanceBase.validate_config(config)
    assert list(required) == ["cart", "front", "rear"]
    assert list(optional) == []
    with pytest.raises(ValueError):
        AvoidanceBase.validate_config(ComponentConfig(attributes=dict_to_struct({})))


def test_stale_cloud_and_near_obstacle_stop_the_child():
    async def _run() -> None:
        stale, child = _ready(np.empty((0, 2)))
        stale._clouds["cam"] = (time.monotonic() - 5.0, np.empty((0, 2)))
        await stale.set_velocity(Vector3(x=0, y=400, z=0), Vector3(x=0, y=0, z=0))
        assert child.velocities == []
        assert child.stopped == 1

        near, near_child = _ready(np.array([[0.0, 0.32]]))
        await near.set_velocity(Vector3(x=0, y=1000, z=0), Vector3(x=0, y=0, z=0))
        assert near_child.velocities == []
        assert near_child.stopped == 1

        far, far_child = _ready(np.array([[0.0, 0.85]]))
        await far.set_velocity(Vector3(x=0, y=1000, z=0), Vector3(x=0, y=0, z=0))
        assert far_child.stopped == 0
        sent_y, sent_w = far_child.velocities[-1]
        assert 300 < sent_y < 500
        assert sent_w == 0

        powered, power_child = _ready(np.empty((0, 2)))
        await powered.set_power(Vector3(x=0, y=1, z=0), Vector3(x=0, y=0, z=0))
        assert power_child.velocities[-1][0] == pytest.approx(500)

    asyncio.run(_run())


def test_move_straight_raises_when_the_obstacle_is_inside_the_stop_zone():
    async def _run() -> None:
        base, child = _ready(np.array([[0.0, 0.32]]))
        with pytest.raises(ObstacleStop):
            await base.move_straight(1000, 200)
        assert child.stopped >= 1

    asyncio.run(_run())


def test_avoidance_status_is_not_forwarded():
    async def _run() -> None:
        base, child = _ready(np.empty((0, 2)))
        status = await base.do_command({"command": "avoidance_status"})
        assert status["action"] == "clear"
        assert status["frames_ok"] is True
        assert status["cameras"]["cam"]["age_s"] < 1.0
        forwarded = await base.do_command({"command": "other"})
        assert forwarded == {"forwarded": "other"}
        assert child.stopped == 0

    asyncio.run(_run())
