# avoidance-base

Wraps a Viam base and slows or stops it when camera point clouds show an obstacle.

The module provides one model, `viam-labs:base:avoidance` (`rdk:component:base`). It forwards base commands to an underlying base. `SetVelocity` and `SetPower` are latched and rechecked against the latest clouds. `MoveStraight` and `Spin` run as velocity loops so the same limit applies while they are in progress. The module does not steer around obstacles: it keeps the commanded curvature and reduces speed so the robot stays about `time_to_collision_s` from a hit. The check follows the commanded direction, the same way Nav2's collision monitor uses a velocity polygon and an approach time. A wall already inside the padding blocks motion that closes on it, and does not block backing up or sliding along it.

Point clouds come from any camera that implements `GetPointCloud` (lidar, depth camera, ultrasonic models that publish a cloud, and so on). Each cloud is transformed into the underlying base frame with the machine frame system. A camera is ignored, and driving stays stopped, until that camera's frame parents to the configured base. A camera parented to `world` does not count.

`GetPointCloud` runs in the background at `image_rate_hz`. Drive commands use the cached cloud and do not wait on the camera. If any camera's cloud is older than `source_timeout_s`, or the frame chain is unresolved, the safe command is zero. A camera or base call that never returns is abandoned, so the next tick can drive again instead of staying stopped until the module is restarted.

## Configuration

```json
{
  "name": "safe-base",
  "api": "rdk:component:base",
  "model": "viam-labs:base:avoidance",
  "attributes": {
    "base": "cart",
    "cameras": ["front-depth", "lidar"]
  }
}
```

Give `cart` a frame. Parent each camera to `cart` (or to a link whose parent chain reaches `cart`) and set the pose to the frame `GetPointCloud` is expressed in, in millimetres. Viam base axes are +Y forward, +X right, +Z up. The base frame's box geometry is the footprint: box Y is length, box X is width.

```json
{
  "name": "front-depth",
  "frame": {
    "parent": "cart",
    "translation": { "x": 0, "y": 200, "z": 300 },
    "orientation": { "type": "ov_degrees", "value": { "x": 0, "y": -1, "z": 0, "th": 90 } }
  }
}
```

The orientation above aims a camera-optical cloud (+Z forward, +X right, +Y down) along the base's +Y axis. A cloud that is already in base axes (lidar +Y forward) uses an identity orientation instead. The frame you configure has to be the frame the points are in.

Optional attributes:

| Attribute | Default | Meaning |
| --- | --- | --- |
| `time_to_collision_s` | `2.0` | Slow so this many seconds remain before the padded body hits |
| `stop_gap_m` | `0.04` | Zero the command when the free distance is inside this gap |
| `padding_m` | `0.05` | Extra keep-out around the footprint |
| `min_points` | `3` | Ignore clusters smaller than this |
| `source_timeout_s` | `0.5` | Stop when a camera cloud is older than this |
| `control_rate_hz` | `10` | How often the latched command is rechecked |
| `image_rate_hz` | `10` | How often each camera point cloud is refreshed |
| `z_min`, `z_max` | `0.05`, `2.0` | Height band in the base frame, metres |
| `max_range_m` | `0` | Drop points farther than this in XY; `0` keeps them |
| `robot_radius` | `0.25` | Footprint fallback when a side has no base box |
| `footprint_length_m`, `footprint_width_m` | unset | Override one side of the base box (Y forward, X right) |
| `assumed_max_linear_mps` | `0.5` | `SetPower` linear scale of `1` maps to this speed |
| `assumed_max_angular_dps` | `90` | `SetPower` angular scale of `1` maps to this yaw rate |

`DoCommand` with `{"command": "avoidance_status"}` returns the last action (`clear`, `slow`, or `stop`), free distance, and per-camera age. Other commands are forwarded to the underlying base.
