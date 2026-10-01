"""Stop versus slow decisions."""

from __future__ import annotations

import math

import numpy as np

from src.guard import GuardConfig, Twist, decide, regulate


def _cfg(**overrides) -> GuardConfig:
    values = dict(
        length_m=0.6,
        width_m=0.4,
        padding_m=0.05,
        stop_gap_m=0.04,
        time_to_collision_s=1.2,
        min_points=1,
    )
    values.update(overrides)
    return GuardConfig(**values)


def test_near_point_stops_and_farther_point_only_slows():
    cfg = _cfg()
    command = Twist(vy_mps=1.0)
    stopped = regulate(command, np.array([[0.0, 0.32]]), cfg)
    assert stopped.action == "stop"
    assert stopped.twist.vy_mps == 0.0

    slowed = regulate(command, np.array([[0.0, 0.85]]), cfg)
    assert slowed.action == "slow"
    assert 0.3 < slowed.twist.vy_mps < 0.5
    assert slowed.twist.wz_rad_s == 0.0


def test_speckle_below_min_points_is_ignored():
    cfg = _cfg(min_points=3)
    command = Twist(vy_mps=0.4)
    result = regulate(command, np.array([[0.0, 0.32], [0.01, 0.32]]), cfg)
    assert result.action == "clear"
    assert result.twist.vy_mps == 0.4


def test_stale_cloud_stops():
    result = decide(
        Twist(vy_mps=0.5),
        np.empty((0, 2)),
        _cfg(),
        frames_ok=True,
        stale=True,
    )
    assert result.action == "stop"
    assert result.twist.planar_speed == 0.0


def test_unresolved_frames_stop():
    result = decide(
        Twist(vy_mps=0.5),
        np.array([[0.0, 5.0]]),
        _cfg(),
        frames_ok=False,
        stale=False,
    )
    assert result.action == "stop"


def test_clear_path_keeps_the_command_and_curvature():
    result = regulate(Twist(vy_mps=0.3, wz_rad_s=0.2), np.array([[2.0, 0.0]]), _cfg())
    assert result.action == "clear"
    assert math.isclose(result.twist.vy_mps, 0.3)
    assert math.isclose(result.twist.wz_rad_s, 0.2)
