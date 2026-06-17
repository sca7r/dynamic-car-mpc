# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/); this project
uses semantic-ish versioning.

## [1.0.0] — Production package

### Added
- **Installable Python package** (`src/dcmpc/` layout, `pyproject.toml`).
  Install with `pip install -e .`; console entry points `dcmpc-sim`,
  `dcmpc-drive`, `dcmpc-3d`, `dcmpc-top`, `dcmpc-carla`, `dcmpc-manual`,
  `dcmpc-plot`.
- **Virtual-LiDAR perception** in the CARLA bridge (`carla_bridge.py`):
  attaches CARLA's ray-cast LiDAR, filters by sensor-frame height band,
  forward FOV arc, ego-exclusion radius and max range, then grid-clusters
  returns (flood-fill, no SciPy) into circular keep-outs. Rejects oversized
  clusters (buildings/walls) and far-lateral scenery (sidewalks).
- **Multi-obstacle MPC**: the QP now carries `MAX_OBS` simultaneous soft
  half-plane keep-outs (one slot per nearby obstacle) instead of a single one.
  `solve()` accepts `obstacles=[...]` (list) as well as the legacy
  `obstacle=` (single) for backward compatibility.
- **Global-frame pass-side commitment**: `solve(global_pose=...)` keys the
  left/right pass lock on world position, so the chosen side stays stable as
  the car moves (fixes mid-manoeuvre flip / spin).
- **Velocity-profile reference**: `get_ref_trajectory` now generates an
  acceleration/braking-limited speed profile instead of a constant target,
  producing dynamically feasible references.
- **Recovery mode**: when cross-track error exceeds 1.5 m the reference speed
  is throttled, giving the solver room to steer back gently instead of
  diverging.

### Changed
- **Tire model smoothing**: slip-angle saturation switched from a hard clamp to
  `tanh` soft saturation, and the low-speed `vx` floor is now smooth. Both give
  the solver continuous derivatives and more stable solves.
- **Cost weights** retuned: lateral tracking weight 80→150, steering-rate
  penalty 60→250 (smoother lane re-entry, no violent snap-back).
- **`pass_zone`** default 6→18 m, so the keep-out spans the whole overtake and
  the car doesn't cut back in early.
- **`AdaptiveSpeed.a_lat_comfort`** default kept at 4.5 m/s² for lively pygame
  cornering; the CARLA bridge passes 1.5 m/s² explicitly for cautious town
  driving.
- **`OBSTACLE_SAFETY_MARGIN`** set to 1.0 m (down from earlier 1.5) so the
  keep-out doesn't collapse a narrow CARLA lane to a STOP.
- Files renamed for the package: `dynamic_bicycle_mpc.py`→`controller.py`,
  `carla_mpc.py`→`carla_bridge.py`, `drive_pygame.py`→`viz/top_down.py`,
  `drive_pygame_3d.py`→`viz/chase_3d.py`, `drive.py`→`viz/drive.py`.

### Fixed
- CARLA emergency-braking-on-every-tick caused by a fragile obstacle unpack
  that raised inside `solve()` and was swallowed by the solver-fallback
  `except`, inflating solve time to ~1900 ms. Unpack now tolerates 3- or
  5-tuples; warm solves are ~60–70 ms again.
- LiDAR false obstacles from walls/ground: tightened height band catches
  vehicle bodies, not building façades.

## [0.4.0] — Human-like speed + robustness

### Added
- `AdaptiveSpeed`: corner slowdown from path curvature
  (`v_max = sqrt(a_lat / kappa)`), proximity-proportional obstacle braking, and
  a first-order low-pass for smooth transitions. HUD shows the active reason
  (FREE / CORNER / OBS).

### Fixed
- Spinning at obstacles: raised `v_min` so the MPC horizon always covers the
  obstacle, and added pass-side commitment so the avoidance side can't flip.

## [0.3.0] — Visualizers

### Added
- Unified pygame viewer (`viz/drive.py`): split-screen 3D + top-down, switchable
  with `V`; AUTO/MANUAL with `TAB`.
- Software-rendered 3D chase camera (no OpenGL): layered painter
  (ground→road→markings→props), per-face depth sorting, distance fog, kerbs,
  dashed centreline, full telemetry HUD with G-force circle.

### Fixed
- 3D render artifacts: grass painting over distant road and floating kerbs,
  resolved by layer-based drawing instead of a single merged depth sort.
- Flat/glitchy car and obstacle: each polygon face now computes its own
  centroid distance; obstacle cylinder gets backface culling.

## [0.2.0] — Obstacle avoidance

### Added
- Soft half-plane keep-out constraints with slack penalty.
- Overtake/stop decision: lane-change when a side has room, controlled stop
  behind the obstacle when neither side does.

## [0.1.0] — Core controller

### Added
- Dynamic single-track (bicycle) vehicle model with a linear tire model.
- Iterative MPC: numerical Jacobian linearization, exact matrix-exponential
  (ZOH) discretization, DPP-compliant QP (~70 ms at 10 Hz).
- Robust solver fallback chain OSQP→CLARABEL→OSQP with emergency brake.
- Headless closed-loop sim (`simulate.py`), open-loop manual drive
  (`manual_drive.py`), CARLA bridge with CSV trace + `plot_trace.py`.
