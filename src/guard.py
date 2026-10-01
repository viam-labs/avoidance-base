"""Time-to-collision speed limit in the Viam base frame.

The body sits at the origin: +Y is forward, +X is right, +yaw is CCW.
Commanded planar speed is scaled so the padded footprint stays
``time_to_collision_s`` from the next hit, and is zeroed when the remaining
distance is inside ``stop_gap_m``. A command that is already in contact is
zeroed only when it does not drive out of that contact, so backing away from
a wall ahead still works. Curvature is preserved by scaling yaw with the same
ratio. This matches Nav2's collision-monitor approach model and nav-stack's
``v = (free - stop_gap) / time_to_collision`` law. It does not steer around
obstacles.
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
    stop_gap_m: float = 0.04
    time_to_collision_s: float = 1.2
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


def _without_departing_contact(
    twist: Twist,
    points: np.ndarray,
    half_width: float,
    half_length: float,
    min_points: int,
) -> tuple[np.ndarray, bool]:
    """Drop padding points this command moves outward, or refuse the command.

    The second value is true when points already inside the padded footprint
    are not being left, so the command would keep pressing into them.
    """
    stopped = twist.planar_speed < 1e-4 and abs(twist.wz_rad_s) < 1e-4
    if points.size == 0 or stopped:
        inside = (
            (np.abs(points[:, 0]) <= half_width) & (np.abs(points[:, 1]) <= half_length)
            if points.size
            else np.zeros(0, dtype=bool)
        )
        return points, int(inside.sum()) >= min_points

    px = points[:, 0]
    py = points[:, 1]
    inside = (np.abs(px) <= half_width) & (np.abs(py) <= half_length)
    if int(inside.sum()) < min_points:
        return points, False

    dists = np.stack(
        [
            half_width - px,
            half_width + px,
            half_length - py,
            half_length + py,
        ],
        axis=1,
    )
    nearest = np.argmin(dists, axis=1)
    # Right, left, front, back. Outward is away from the interior.
    outward = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])[nearest]
    # World points move opposite the body twist.
    v_rel_x = -twist.vx_mps + twist.wz_rad_s * py
    v_rel_y = -twist.vy_mps - twist.wz_rad_s * px
    leaving = inside & (outward[:, 0] * v_rel_x + outward[:, 1] * v_rel_y > 1e-3)
    if int((inside & ~leaving).sum()) >= min_points:
        return points, True
    return points[~leaving], False


def regulate(twist: Twist, points: np.ndarray, cfg: GuardConfig) -> GuardResult:
    """Scale ``twist`` from the obstacles in the base frame."""
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        points = np.empty((0, 2))
    else:
        points = points.reshape(-1, 2)

    half_length = cfg.length_m / 2.0 + cfg.padding_m
    half_width = cfg.width_m / 2.0 + cfg.padding_m
    min_points = max(1, int(cfg.min_points))
    points, pressing = _without_departing_contact(
        twist, points, half_width, half_length, min_points
    )
    if pressing:
        return _result(twist, 0.0, 0.0)

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
    counts = _counts_inside(xs, ys, thetas, points, half_width, half_length)
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
    counts = _counts_inside(
        np.zeros(steps), np.zeros(steps), signed, points, half_width, half_length
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
