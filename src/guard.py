"""Time-to-collision speed limit in the Viam base frame.

The body sits at the origin: +Y is forward, +X is right, +yaw is CCW.
Commanded planar speed is scaled so the padded footprint stays
``time_to_collision_s`` from the next hit, and is zeroed when the remaining
distance is inside ``stop_gap_m``. Curvature is preserved by scaling yaw with
the same ratio.

This follows Nav2's collision monitor. The approach model only slows the
commanded velocity, so a wall ahead does not apply to a reverse command.
Points already inside the padding, the same case as a velocity polygon that
has already tripped, may be moved away from or passed alongside: a step is a
hit only when it brings the body closer than that point's current clearance
(Nav2 does not keep a surround zone active for every direction at once). It
does not steer around obstacles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Twist:
    """Body twist. Linear components are m/s, yaw is rad/s."""

    vx_mps: float = 0.0
    vy_mps: float = 0.0
    wz_rad_s: float = 0.0

    def scaled(self, scale: float) -> "Twist":
        return Twist(self.vx_mps * scale, self.vy_mps * scale, self.wz_rad_s * scale)

    @property
    def planar_speed(self) -> float:
        return math.hypot(self.vx_mps, self.vy_mps)


@dataclass(frozen=True)
class GuardConfig:
    length_m: float
    width_m: float
    padding_m: float = 0.05
    # Obstacles already inside the padding may be left or passed alongside,
    # but a step may not come within ``min_gap_m`` or close by more than
    # ``near_slack_m``. Same rule as the Nav2 footprint approach check.
    min_gap_m: float = 0.02
    near_slack_m: float = 0.01
    stop_gap_m: float = 0.04
    time_to_collision_s: float = 2.0
    min_points: int = 3
    step_m: float = 0.025
    rot_step_rad: float = 0.04


@dataclass(frozen=True)
class GuardResult:
    twist: Twist
    action: str  # clear | slow | stop
    free_m: float


def _poses_along(twist: Twist, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Origin pose in the current base frame after ``times`` seconds."""
    vx, vy, wz = twist.vx_mps, twist.vy_mps, twist.wz_rad_s
    if abs(wz) < 1e-6:
        return vx * times, vy * times, np.zeros_like(times)
    theta = wz * times
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    # +Y forward, +X right, +wz CCW from the +Y heading.
    x = (vx / wz) * sin_t + (vy / wz) * (cos_t - 1.0)
    y = (vx / wz) * (1.0 - cos_t) + (vy / wz) * sin_t
    return x, y, theta


def _counts_inside(
    xs: np.ndarray,
    ys: np.ndarray,
    thetas: np.ndarray,
    points: np.ndarray,
    half_width: float,
    half_length: float,
) -> np.ndarray:
    """How many points lie in the rectangle at each pose."""
    if points.size == 0:
        return np.zeros(len(xs), dtype=int)
    dx = points[None, :, 0] - xs[:, None]
    dy = points[None, :, 1] - ys[:, None]
    cos_t = np.cos(thetas)[:, None]
    sin_t = np.sin(thetas)[:, None]
    right = cos_t * dx + sin_t * dy
    forward = -sin_t * dx + cos_t * dy
    inside = (np.abs(right) <= half_width) & (np.abs(forward) <= half_length)
    return inside.sum(axis=1)


def _action_for(scale: float) -> str:
    if scale <= 1e-6:
        return "stop"
    if scale >= 0.999:
        return "clear"
    return "slow"


def _result(twist: Twist, scale: float, free_m: float) -> GuardResult:
    scale = float(min(1.0, max(0.0, scale)))
    return GuardResult(twist.scaled(scale), _action_for(scale), free_m)


def _body_clearance(
    points: np.ndarray, half_width: float, half_length: float
) -> np.ndarray:
    """Distance outside the body rectangle. Zero when a point is inside it."""
    if points.size == 0:
        return np.empty(0)
    lateral = np.maximum(np.abs(points[:, 0]) - half_width, 0.0)
    forward = np.maximum(np.abs(points[:, 1]) - half_length, 0.0)
    return np.hypot(lateral, forward)


def _clearance_along(
    xs: np.ndarray,
    ys: np.ndarray,
    thetas: np.ndarray,
    points: np.ndarray,
    half_width: float,
    half_length: float,
) -> np.ndarray:
    """``(K, N)`` body-rectangle clearance of each point at each pose."""
    if points.size == 0:
        return np.zeros((len(xs), 0))
    dx = points[None, :, 0] - xs[:, None]
    dy = points[None, :, 1] - ys[:, None]
    cos_t = np.cos(thetas)[:, None]
    sin_t = np.sin(thetas)[:, None]
    right = cos_t * dx + sin_t * dy
    forward = -sin_t * dx + cos_t * dy
    lateral = np.maximum(np.abs(right) - half_width, 0.0)
    ahead = np.maximum(np.abs(forward) - half_length, 0.0)
    return np.hypot(lateral, ahead)


def _hit_counts(
    xs: np.ndarray,
    ys: np.ndarray,
    thetas: np.ndarray,
    points: np.ndarray,
    cfg: GuardConfig,
    half_width: float,
    half_length: float,
) -> np.ndarray:
    """Points that enter the padding, or near points the step moves closer to.

    ``half_width`` and ``half_length`` are the unpadded body. Points already
    inside the padding keep a clearance floor and only count when a pose drops
    below it, so backing away from a wall is not itself a collision.
    """
    counts = np.zeros(len(xs), dtype=int)
    if points.size == 0:
        return counts
    clearance = _body_clearance(points, half_width, half_length)
    near = clearance <= cfg.padding_m
    far = points[~near]
    near_points = points[near]
    if far.size:
        counts += _counts_inside(
            xs,
            ys,
            thetas,
            far,
            half_width + cfg.padding_m,
            half_length + cfg.padding_m,
        )
    if near_points.size:
        floor = np.maximum(cfg.min_gap_m, clearance[near] - cfg.near_slack_m)
        distance = _clearance_along(xs, ys, thetas, near_points, half_width, half_length)
        counts += (distance < floor[None, :]).sum(axis=1)
    return counts


def regulate(twist: Twist, points: np.ndarray, cfg: GuardConfig) -> GuardResult:
    """Scale ``twist`` from the obstacles in the base frame."""
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        points = np.empty((0, 2))
    else:
        points = points.reshape(-1, 2)

    half_length = cfg.length_m / 2.0
    half_width = cfg.width_m / 2.0
    min_points = max(1, int(cfg.min_points))
    speed = twist.planar_speed
    yaw = abs(twist.wz_rad_s)
    if speed < 1e-4 and yaw < 1e-4:
        return GuardResult(Twist(), "clear", math.inf)

    if speed < 1e-4:
        return _regulate_spin(twist, points, cfg, half_width, half_length, min_points)
    return _regulate_arc(twist, points, cfg, half_width, half_length, min_points, speed)


def _regulate_arc(
    twist: Twist,
    points: np.ndarray,
    cfg: GuardConfig,
    half_width: float,
    half_length: float,
    min_points: int,
    speed: float,
) -> GuardResult:
    horizon = max(0.5, speed * cfg.time_to_collision_s + cfg.stop_gap_m + 0.5)
    dt = cfg.step_m / speed
    if abs(twist.wz_rad_s) > 1e-6:
        dt = min(dt, cfg.rot_step_rad / abs(twist.wz_rad_s))
    steps = int(math.ceil(horizon / cfg.step_m))
    steps = max(1, min(steps, 400))
    times = (np.arange(steps) + 1) * dt
    xs, ys, thetas = _poses_along(twist, times)
    counts = _hit_counts(xs, ys, thetas, points, cfg, half_width, half_length)
    hits = np.flatnonzero(counts >= min_points)
    if hits.size == 0:
        return _result(twist, 1.0, math.inf)
    free_m = float(speed * times[int(hits[0])])
    if free_m <= cfg.stop_gap_m:
        return _result(twist, 0.0, free_m)
    allowed = (free_m - cfg.stop_gap_m) / cfg.time_to_collision_s
    return _result(twist, allowed / speed, free_m)


def _regulate_spin(
    twist: Twist,
    points: np.ndarray,
    cfg: GuardConfig,
    half_width: float,
    half_length: float,
    min_points: int,
) -> GuardResult:
    reach = max(math.hypot(half_length, half_width), 1e-3)
    stop_gap_rad = cfg.stop_gap_m / reach
    yaw = abs(twist.wz_rad_s)
    horizon = max(0.5, yaw * cfg.time_to_collision_s + stop_gap_rad)
    step = cfg.rot_step_rad
    steps = max(1, min(int(math.ceil(horizon / step)), 400))
    angles = (np.arange(steps) + 1) * step
    signed = math.copysign(1.0, twist.wz_rad_s) * angles
    counts = _hit_counts(
        np.zeros(steps), np.zeros(steps), signed, points, cfg, half_width, half_length
    )
    hits = np.flatnonzero(counts >= min_points)
    if hits.size == 0:
        return _result(twist, 1.0, math.inf)
    free_rad = float(angles[int(hits[0])])
    free_m = free_rad * reach
    if free_rad <= stop_gap_rad:
        return _result(twist, 0.0, free_m)
    allowed = (free_rad - stop_gap_rad) / cfg.time_to_collision_s
    return _result(twist, allowed / yaw, free_m)


def decide(
    twist: Twist,
    points: np.ndarray,
    cfg: GuardConfig,
    *,
    frames_ok: bool,
    stale: bool,
) -> GuardResult:
    """``regulate``, or an immediate stop when frames or clouds are unusable."""
    if not frames_ok or stale:
        return GuardResult(Twist(), "stop", 0.0)
    return regulate(twist, points, cfg)
