"""``viam-labs:base:avoidance``: a base that slows or stops for obstacles.

Commands are forwarded to an underlying base. ``SetVelocity`` and ``SetPower``
are latched and rechecked at ``control_rate_hz`` against point clouds refreshed
in the background at ``image_rate_hz``. ``MoveStraight`` and ``Spin`` run as
velocity loops so the same limit applies mid-motion.
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any, ClassVar, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from typing_extensions import Self
from viam.components.base import Base, Vector3
from viam.components.camera import Camera
from viam.logging import getLogger
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes, struct_to_dict

from .frames import (
    base_box_forward_lateral,
    fetch_frame_system_config,
    pose_matrix_in_destination,
    resolve_footprint,
)
from .guard import GuardConfig, GuardResult, Twist, decide
from .obstacles import parse_pcd, prepare_base_points
from .runtime import get_parent_robot

LOGGER = getLogger(__name__)

DEFAULT_TIME_TO_COLLISION_S = 2.0
DEFAULT_STOP_GAP_M = 0.04
DEFAULT_PADDING_M = 0.05
DEFAULT_MIN_POINTS = 3
DEFAULT_SOURCE_TIMEOUT_S = 0.5
DEFAULT_CONTROL_RATE_HZ = 10.0
DEFAULT_IMAGE_RATE_HZ = 10.0
DEFAULT_Z_MIN_M = 0.05
DEFAULT_Z_MAX_M = 2.0
DEFAULT_ROBOT_RADIUS_M = 0.25
DEFAULT_ASSUMED_MAX_LINEAR_MPS = 0.5
DEFAULT_ASSUMED_MAX_ANGULAR_DPS = 90.0
MAX_CLOUD_POINTS = 8000


class ObstacleStop(RuntimeError):
    """A straight move or spin was cut short because an obstacle was too close."""


class AvoidanceBase(Base):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "base"), "avoidance")

    def __init__(self, name: str):
        super().__init__(name)
        self._child: Optional[Base] = None
        self._cameras: Dict[str, Camera] = {}
        self._camera_names: list[str] = []
        self._base_name = ""
        self._guard = GuardConfig(length_m=0.5, width_m=0.5)
        self._source_timeout_s = DEFAULT_SOURCE_TIMEOUT_S
        self._control_rate_hz = DEFAULT_CONTROL_RATE_HZ
        self._image_rate_hz = DEFAULT_IMAGE_RATE_HZ
        self._z_min_m = DEFAULT_Z_MIN_M
        self._z_max_m = DEFAULT_Z_MAX_M
        self._max_range_m = 0.0
        self._robot_radius_m = DEFAULT_ROBOT_RADIUS_M
        self._length_override: Optional[float] = None
        self._width_override: Optional[float] = None
        self._assumed_max_linear_mps = DEFAULT_ASSUMED_MAX_LINEAR_MPS
        self._assumed_max_angular_dps = DEFAULT_ASSUMED_MAX_ANGULAR_DPS
        self._poses_mm: Dict[str, np.ndarray] = {}
        self._frames_ok = False
        self._frame_error = "frame system not loaded"
        self._frames_checked_at = 0.0
        self._clouds: Dict[str, Tuple[float, np.ndarray]] = {}
        self._latch = Twist()
        self._mode = "idle"
        self._motion_token: Optional[object] = None
        self._cancel_motion = asyncio.Event()
        self._gen = 0
        self._tasks: list[asyncio.Task] = []
        self._write_lock: Optional[asyncio.Lock] = None
        self._last_key: Optional[Tuple[float, float, float]] = None
        self._last_result = GuardResult(Twist(), "stop", 0.0)
        self._closed = False

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        attrs = struct_to_dict(config.attributes)
        base_name = attrs.get("base")
        cameras = attrs.get("cameras")
        if not isinstance(base_name, str) or not base_name:
            raise ValueError("base is required")
        if (
            not isinstance(cameras, list)
            or not cameras
            or not all(isinstance(name, str) and name for name in cameras)
        ):
            raise ValueError("cameras must be a non-empty list of camera names")
        return [base_name, *cameras], []

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        base = cls(config.name)
        base.reconfigure(config, dependencies)
        return base

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs = struct_to_dict(config.attributes)
        base_name = str(attrs["base"])
        camera_names = [str(name) for name in attrs["cameras"]]
        child = dependencies.get(Base.get_resource_name(base_name))
        if child is None:
            raise ValueError(f"base {base_name!r} is not a dependency")
        cameras: Dict[str, Camera] = {}
        for name in camera_names:
            camera = dependencies.get(Camera.get_resource_name(name))
            if camera is None:
                raise ValueError(f"camera {name!r} is not a dependency")
            cameras[name] = camera  # type: ignore[assignment]

        self._gen += 1
        for task in self._tasks:
            task.cancel()
        self._tasks = []
        self._cancel_motion.set()
        self._motion_token = None
        self._mode = "idle"
        self._latch = Twist()
        self._last_key = None
        self._clouds = {}
        self._poses_mm = {}
        self._frames_ok = False
        self._frame_error = "frame system not loaded"
        self._frames_checked_at = 0.0

        self._child = child  # type: ignore[assignment]
        self._cameras = cameras
        self._camera_names = camera_names
        self._base_name = base_name
        self._source_timeout_s = _positive(attrs, "source_timeout_s", DEFAULT_SOURCE_TIMEOUT_S)
        self._control_rate_hz = _positive(attrs, "control_rate_hz", DEFAULT_CONTROL_RATE_HZ)
        self._image_rate_hz = _positive(attrs, "image_rate_hz", DEFAULT_IMAGE_RATE_HZ)
        self._z_min_m = _float(attrs, "z_min", DEFAULT_Z_MIN_M)
        self._z_max_m = _float(attrs, "z_max", DEFAULT_Z_MAX_M)
        self._max_range_m = _float(attrs, "max_range_m", 0.0)
        self._robot_radius_m = _positive(attrs, "robot_radius", DEFAULT_ROBOT_RADIUS_M)
        self._length_override = _optional_positive(attrs, "footprint_length_m")
        self._width_override = _optional_positive(attrs, "footprint_width_m")
        self._assumed_max_linear_mps = _positive(
            attrs, "assumed_max_linear_mps", DEFAULT_ASSUMED_MAX_LINEAR_MPS
        )
        self._assumed_max_angular_dps = _positive(
            attrs, "assumed_max_angular_dps", DEFAULT_ASSUMED_MAX_ANGULAR_DPS
        )
        length, width = resolve_footprint(
            None,
            length_m=self._length_override,
            width_m=self._width_override,
            robot_radius_m=self._robot_radius_m,
        )
        self._set_guard(length, width, attrs)
        self._spawn()

    def _set_guard(self, length_m: float, width_m: float, attrs: Mapping[str, Any]) -> None:
        self._guard = GuardConfig(
            length_m=length_m,
            width_m=width_m,
            padding_m=_positive(attrs, "padding_m", DEFAULT_PADDING_M),
            stop_gap_m=_nonnegative(attrs, "stop_gap_m", DEFAULT_STOP_GAP_M),
            time_to_collision_s=_positive(
                attrs, "time_to_collision_s", DEFAULT_TIME_TO_COLLISION_S
            ),
            min_points=max(1, int(_positive(attrs, "min_points", DEFAULT_MIN_POINTS))),
        )

    def _spawn(self) -> None:
        if self._closed:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        gen = self._gen
        self._tasks = [
            loop.create_task(self._image_loop(gen), name=f"{self.name}-images"),
            loop.create_task(self._control_loop(gen), name=f"{self.name}-control"),
        ]

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._gen += 1
        self._cancel_motion.set()
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks = []
        if self._child is not None:
            try:
                await self._child.stop()
            except Exception:  # noqa: BLE001
                LOGGER.warning("avoidance %r failed to stop child on close", self.name)

    def _lock(self) -> asyncio.Lock:
        if self._write_lock is None:
            self._write_lock = asyncio.Lock()
        return self._write_lock

    async def _image_loop(self, gen: int) -> None:
        period = 1.0 / self._image_rate_hz
        while gen == self._gen and not self._closed:
            started = time.monotonic()
            try:
                await self._refresh_frames()
                await self._refresh_clouds()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("avoidance %r image refresh failed: %s", self.name, exc)
            await asyncio.sleep(max(0.0, period - (time.monotonic() - started)))

    def _note_frame_error(self, message: str) -> None:
        self._frames_ok = False
        if message != self._frame_error:
            LOGGER.warning("avoidance %r %s", self.name, message)
        self._frame_error = message

    async def _refresh_frames(self) -> None:
        now = time.monotonic()
        if self._frames_ok and now - self._frames_checked_at < 2.0:
            return
        robot = get_parent_robot()
        if robot is None:
            self._note_frame_error("module parent robot is unavailable")
            return
        try:
            configs = await fetch_frame_system_config(robot)
        except Exception as exc:  # noqa: BLE001
            self._note_frame_error(f"frame system read failed: {exc}")
            return
        self._frames_checked_at = time.monotonic()
        poses: Dict[str, np.ndarray] = {}
        missing: list[str] = []
        for name in self._camera_names:
            transform = pose_matrix_in_destination(configs, name, self._base_name)
            if transform is None:
                missing.append(name)
            else:
                poses[name] = transform
        box = base_box_forward_lateral(configs, self._base_name)
        length, width = resolve_footprint(
            box,
            length_m=self._length_override,
            width_m=self._width_override,
            robot_radius_m=self._robot_radius_m,
        )
        self._guard = GuardConfig(
            length_m=length,
            width_m=width,
            padding_m=self._guard.padding_m,
            stop_gap_m=self._guard.stop_gap_m,
            time_to_collision_s=self._guard.time_to_collision_s,
            min_points=self._guard.min_points,
            step_m=self._guard.step_m,
            rot_step_rad=self._guard.rot_step_rad,
        )
        self._poses_mm = poses
        if missing:
            self._note_frame_error(
                f"cameras {missing} are not framed on base {self._base_name!r}; "
                "parent each camera to that base (or a link under it), not world"
            )
            return
        self._frames_ok = True
        self._frame_error = ""

    async def _refresh_clouds(self) -> None:
        if not self._frames_ok:
            return

        async def _one(name: str) -> None:
            camera = self._cameras.get(name)
            pose = self._poses_mm.get(name)
            if camera is None or pose is None:
                return
            data = await camera.get_point_cloud(timeout=2.0)
            raw = data[0] if isinstance(data, tuple) else data
            loop = asyncio.get_running_loop()
            length = self._guard.length_m
            width = self._guard.width_m
            points = await loop.run_in_executor(
                None,
                lambda: prepare_base_points(
                    parse_pcd(raw),
                    pose,
                    z_min_m=self._z_min_m,
                    z_max_m=self._z_max_m,
                    footprint_length_m=length,
                    footprint_width_m=width,
                    max_range_m=self._max_range_m,
                    max_points=MAX_CLOUD_POINTS,
                ),
            )
            self._clouds[name] = (time.monotonic(), points)

        results = await asyncio.gather(
            *(_one(name) for name in self._camera_names), return_exceptions=True
        )
        for name, result in zip(self._camera_names, results):
            if isinstance(result, Exception):
                LOGGER.warning("avoidance %r camera %r failed: %s", self.name, name, result)

    async def _control_loop(self, gen: int) -> None:
        period = 1.0 / self._control_rate_hz
        while gen == self._gen and not self._closed:
            started = time.monotonic()
            try:
                if self._mode == "velocity":
                    result = self._evaluate(self._latch)
                    await self._publish(result, mode_expected="velocity")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("avoidance %r control tick failed: %s", self.name, exc)
            await asyncio.sleep(max(0.0, period - (time.monotonic() - started)))

    def _observation(self) -> tuple[np.ndarray, bool]:
        """Merged base-frame points, and whether any camera is missing or stale."""
        if not self._frames_ok:
            return np.empty((0, 2)), True
        now = time.monotonic()
        chunks: list[np.ndarray] = []
        for name in self._camera_names:
            cached = self._clouds.get(name)
            if cached is None or now - cached[0] > self._source_timeout_s:
                return np.empty((0, 2)), True
            if cached[1].size:
                chunks.append(cached[1])
        if not chunks:
            return np.empty((0, 2)), False
        return np.concatenate(chunks, axis=0), False

    def _evaluate(self, twist: Twist) -> GuardResult:
        points, stale = self._observation()
        result = decide(twist, points, self._guard, frames_ok=self._frames_ok, stale=stale)
        self._last_result = result
        return result

    async def _publish(self, result: GuardResult, *, mode_expected: str) -> None:
        async with self._lock():
            if self._mode != mode_expected or self._child is None:
                return
            await self._write_child(result)

    async def _write_child(self, result: GuardResult) -> None:
        key = (
            round(result.twist.vx_mps, 4),
            round(result.twist.vy_mps, 4),
            round(result.twist.wz_rad_s, 4),
        )
        if key == self._last_key:
            return
        self._last_key = key
        assert self._child is not None
        if key == (0.0, 0.0, 0.0):
            await self._child.stop()
            return
        await self._child.set_velocity(
            Vector3(x=result.twist.vx_mps * 1000.0, y=result.twist.vy_mps * 1000.0, z=0.0),
            Vector3(x=0.0, y=0.0, z=math.degrees(result.twist.wz_rad_s)),
        )

    def _status(self) -> Dict[str, Any]:
        now = time.monotonic()
        cameras: Dict[str, Any] = {}
        for name in self._camera_names:
            cached = self._clouds.get(name)
            cameras[name] = {
                "age_s": None if cached is None else now - cached[0],
                "points": 0 if cached is None else int(len(cached[1])),
            }
        free = self._last_result.free_m
        return {
            "action": self._last_result.action,
            "free_m": None if not math.isfinite(free) else free,
            "frames_ok": self._frames_ok,
            "frame_error": self._frame_error,
            "cameras": cameras,
            "linear_mm_s": {
                "x": self._last_result.twist.vx_mps * 1000.0,
                "y": self._last_result.twist.vy_mps * 1000.0,
            },
            "angular_deg_s": {"z": math.degrees(self._last_result.twist.wz_rad_s)},
        }

    async def set_velocity(
        self,
        linear: Vector3,
        angular: Vector3,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> None:
        del extra, timeout, kwargs
        self._begin_velocity(
            Twist(
                vx_mps=float(linear.x) / 1000.0,
                vy_mps=float(linear.y) / 1000.0,
                wz_rad_s=math.radians(float(angular.z)),
            )
        )
        await self._publish(self._evaluate(self._latch), mode_expected="velocity")

    async def set_power(
        self,
        linear: Vector3,
        angular: Vector3,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> None:
        del extra, timeout, kwargs
        self._begin_velocity(
            Twist(
                vx_mps=float(linear.x) * self._assumed_max_linear_mps,
                vy_mps=float(linear.y) * self._assumed_max_linear_mps,
                wz_rad_s=math.radians(float(angular.z) * self._assumed_max_angular_dps),
            )
        )
        await self._publish(self._evaluate(self._latch), mode_expected="velocity")

    def _begin_velocity(self, twist: Twist) -> None:
        self._cancel_motion.set()
        self._motion_token = None
        self._mode = "velocity"
        self._latch = twist

    async def stop(
        self,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> None:
        del extra, timeout, kwargs
        self._cancel_motion.set()
        self._motion_token = None
        self._latch = Twist()
        self._mode = "idle"
        self._last_result = GuardResult(Twist(), "stop", self._last_result.free_m)
        async with self._lock():
            self._last_key = (0.0, 0.0, 0.0)
            if self._child is not None:
                await self._child.stop()

    async def move_straight(
        self,
        distance: int,
        velocity: float,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> None:
        del extra, timeout
        if distance == 0 or velocity == 0:
            await self.stop()
            return
        signed_mm = float(distance if velocity >= 0 else -distance)
        speed_mps = abs(float(velocity)) / 1000.0
        direction = math.copysign(1.0, signed_mm)
        await self._run_motion(
            remaining=abs(signed_mm) / 1000.0,
            twist=Twist(vy_mps=direction * speed_mps),
            speed=speed_mps,
            operation=self.get_operation(kwargs),
            what="move_straight",
            angular=False,
        )

    async def spin(
        self,
        angle: float,
        velocity: float,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> None:
        del extra, timeout
        if angle == 0 or velocity == 0:
            await self.stop()
            return
        signed_deg = float(angle if velocity >= 0 else -angle)
        speed_rad_s = math.radians(abs(float(velocity)))
        direction = math.copysign(1.0, signed_deg)
        await self._run_motion(
            remaining=math.radians(abs(signed_deg)),
            twist=Twist(wz_rad_s=direction * speed_rad_s),
            speed=speed_rad_s,
            operation=self.get_operation(kwargs),
            what="spin",
            angular=True,
        )

    async def _run_motion(
        self,
        *,
        remaining: float,
        twist: Twist,
        speed: float,
        operation,
        what: str,
        angular: bool,
    ) -> None:
        token = object()
        self._motion_token = token
        self._cancel_motion = asyncio.Event()
        self._mode = "trajectory"
        self._latch = Twist()
        period = 1.0 / self._control_rate_hz
        try:
            while remaining > 1e-4:
                if self._motion_token is not token or self._cancel_motion.is_set():
                    return
                if await operation.is_cancelled():
                    return
                result = self._evaluate(twist)
                if result.action == "stop" or result.twist.planar_speed + abs(result.twist.wz_rad_s) < 1e-4:
                    raise ObstacleStop(
                        f"{what} stopped with {remaining:.3f} remaining: obstacle inside stopping distance"
                    )
                async with self._lock():
                    if self._motion_token is not token or self._mode != "trajectory":
                        return
                    await self._write_child(result)
                step = min(period, remaining / max(speed, 1e-6))
                await asyncio.sleep(step)
                traveled = (
                    abs(result.twist.wz_rad_s) if angular else result.twist.planar_speed
                ) * step
                remaining -= traveled
        finally:
            if self._motion_token is token:
                self._motion_token = None
                self._mode = "idle"
                self._latch = Twist()
                async with self._lock():
                    self._last_key = (0.0, 0.0, 0.0)
                    if self._child is not None:
                        await self._child.stop()

    async def is_moving(self) -> bool:
        if self._mode == "trajectory":
            return True
        if self._child is None:
            return False
        return await self._child.is_moving()

    async def get_properties(self, *, timeout: Optional[float] = None, **kwargs) -> Base.Properties:
        del kwargs
        if self._child is None:
            raise RuntimeError(f"avoidance base {self.name!r} has no underlying base")
        return await self._child.get_properties(timeout=timeout)

    async def get_geometries(self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None):
        if self._child is None:
            raise RuntimeError(f"avoidance base {self.name!r} has no underlying base")
        return await self._child.get_geometries(extra=extra, timeout=timeout)

    async def get_status(self, *, timeout: Optional[float] = None, **kwargs) -> Mapping[str, ValueTypes]:
        del kwargs
        if self._child is None:
            return self._status()
        try:
            child_status = await self._child.get_status(timeout=timeout)
        except Exception:  # noqa: BLE001
            child_status = {}
        return {"avoidance": self._status(), "base": dict(child_status)}

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        if command.get("command") == "avoidance_status":
            commanded = self._latch if self._mode == "velocity" else Twist()
            self._evaluate(commanded)
            return self._status()
        if self._child is None:
            raise RuntimeError(f"avoidance base {self.name!r} has no underlying base")
        return await self._child.do_command(command, timeout=timeout, **kwargs)


def _float(attrs: Mapping[str, Any], key: str, default: float) -> float:
    value = attrs.get(key, default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _positive(attrs: Mapping[str, Any], key: str, default: float) -> float:
    value = _float(attrs, key, default)
    return value if value > 0 else default


def _nonnegative(attrs: Mapping[str, Any], key: str, default: float) -> float:
    value = _float(attrs, key, default)
    return value if value >= 0 else default


def _optional_positive(attrs: Mapping[str, Any], key: str) -> Optional[float]:
    if key not in attrs or attrs[key] is None:
        return None
    value = _float(attrs, key, 0.0)
    if value <= 0:
        return None
    return value


Registry.register_resource_creator(
    Base.API,
    AvoidanceBase.MODEL,
    ResourceCreatorRegistration(AvoidanceBase.new, AvoidanceBase.validate_config),
)
