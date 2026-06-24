# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/); this project
uses semantic-ish versioning.

## [1.2.0] — Return-to-lane tuning: lateral corridor, sideslip damping, A/B tooling

This release attacks the post-pass overshoot logged as a Known Issue in 1.1.0.
Rather than crank the cross-track weight (which would make the controller fight
*harder* to recentre and saturate steering even more), it gives the controller
permission to *not* fight a small, harmless off-centre error, and adds direct
damping on the motion that produces the overshoot.

### Added
- **Lateral corridor — soft deadband on cross-track** (`CROSS_BAND`, default
  `0.7` m; `controller.py`). Cross-track error inside ±`CROSS_BAND` is now
  penalty-free; only the excess beyond the band is costed. Implemented as a
  **hinge slack**: the cross weight is pulled out of the quadratic state cost and
  re-applied to a non-negative `cross_excess` variable constrained by
  `cross_excess_k ≥ |err_cross_k| − band`. This keeps the program convex and
  DPP-compliant (no `abs()` in the objective), so canonicalisation is unchanged
  and solve time is unaffected. Set `CROSS_BAND = 0.0` to restore strict
  centreline tracking (exact pre-1.2 behaviour).
- **A/B trace comparison tool** (`ab_compare.py`). Head-to-head of two CARLA
  traces: worst |cross-track|, peak |heading error|, max |steering|, steering
  saturation %, max speed, plus a per-obstacle-pass breakdown (peak cross /
  heading in a pass + 4 s window) and a one-line verdict pointing at the next
  knob. Run with `python ab_compare.py baseline.csv treatment.csv`.

### Changed
- **Sideslip damping raised**: stage `STATE_COST` lateral-velocity (`vy`) weight
  `2 → 6`. The return overshoot is a `vy`/yaw transient; damping `vy` in the cost
  calms the snap directly, where raising cross-track weight only made it worse.
- **`OBSTACLE_SAFETY_MARGIN` `0.8 → 0.2` m** for tighter, more natural town
  passes. The decoupled planner already guarantees a feasible path, so the extra
  geometric buffer was mostly widening the swerve.
- The combined effect (corridor + `vy` damping + tighter margin) shrinks worst
  cross-track from ~2.85 m to ~1.96 m on the standard three-car route; see a
  before/after with `ab_compare.py`.


### Known issues
- **Steering still saturates intermittently** (~3% of ticks at `MAX_STEER`) on
  the return, and the corridor buys some of its smoothness by *allowing* more
  cross-track error — so a smoother trace is not automatically a tighter-tracking
  one. Confirm any further tuning with `ab_compare.py` and watch that saturation
  duty falls **without** worst cross-track creeping back up. A steering-rate
  (Δδ) penalty bump targets the saturation more directly than a yaw-rate weight.
- **Package version metadata** is bumped to `1.2.0` in `pyproject.toml` and
  `__init__.py` to match this entry.

## [1.1.0] — Decoupled planner, Kalman tracking, synchronous CARLA

### Added
- **Decoupled two-layer avoidance** (`planner.py`, `USE_PATH_PLANNER`, default
  `True` on CARLA). A purely kinematic Layer-1 planner bends the reference path
  around obstacles with a smooth raised-cosine ramp (ramp in → hold for
  `PASS_ZONE` → ramp out), and the dynamic MPC tracks that already-safe path with
  **no obstacle constraint in the QP**. Splits avoidance (geometry) from tracking
  (dynamics) so each stays well-conditioned; fixes the multi-second solver
  freezes and swerves that came from solving both in one QP at low speed.
- **Multi-object Kalman tracker** (`tracker.py`, `CARLA_TRACKER_ENABLE`).
  Constant-velocity filter per obstacle with Hungarian assignment for stable
  identities, a confirmation count, and coasting through dropouts
  (`CARLA_TRACKER_MAX_MISS`). Drop-in: same `(x, y, r, vx, vy)` tuples downstream.
- **Synchronous-mode instrumentation** in the CARLA bridge: a `frame_lag`
  diagnostic (`world_frame − lidar_frame`) plus per-stage wall-clock timing
  (`tick/lidar/track/filter/speed/plan/ref/solve/apply/loop` ms) and the LiDAR
  frame/timestamp, all written to the trace CSV and optionally printed per tick
  (`CARLA_LOG_TIMING`). Confirmed the loop is fully synchronous (`frame_lag = 0`
  on every tick) and entirely MPC-solve-bound.
- **Hold-last-command recovery** (`CARLA_SOLVE_HOLD_TICKS`). On a failed solve
  the bridge reuses the last good command for a few ticks instead of
  emergency-braking — braking removed the speed the controller needed to keep
  steering and could stall the car; the hold recovers cleanly.
- **Configurable trace output folder** (`CARLA_OUTPUT_DIR`); traces collect under
  `output/` next to the launch directory instead of the working directory.

### Changed
- CARLA obstacle avoidance now runs through the Layer-1 planner; the MPC carries
  **no half-plane keep-out** on CARLA (the planner guarantees a safe path). The
  single-QP soft-half-plane mode remains for the pygame sim and
  `USE_PATH_PLANNER = False`.
- Planner pass-side and shift magnitude are **committed in world coordinates** and
  updated **in place** each tick, so the chosen side and bend can't wobble while
  the obstacle is visible.
- `OBSTACLE_SAFETY_MARGIN` 1.0 → 0.8 m and `PASS_ZONE` → 1.5 m for tighter,
  more natural town passes; `CARLA_TRACKER_MAX_MISS` raised to 15 (~1.5 s coast).
- README documents the decoupled architecture, the Kalman perception layer, the
  synchronous-mode guarantee, and the pygame-vs-CARLA config split; `config.py`
  gains a first-run orientation guide and a section index.

### Fixed
- **Planner bend wiped every tick**: the obstacle-memory dict was being replaced
  on each refresh, erasing the committed pass-side displacement and letting the
  bend wobble. Memory is now updated in place; the committed `needed` shift
  survives until the obstacle is passed.
- **Bend collapsing on a missed detection while the obstacle was still ahead**,
  which caused a late, violent swerve when it reappeared at close range. The bend
  is now held at full strength while the obstacle is ahead (with a phantom guard)
  and only ramps out once the obstacle is behind the car.
- **Planner steering toward already-clear obstacles**: bending an obstacle the
  path already cleared pulled the path *toward* it. Obstacles outside the
  corridor are now skipped (no shift, no lock).
- Investigated and **ruled out a perception/sync lag** as the cause of the
  post-pass path overshoot: with the new `frame_lag` instrumentation it reads 0
  on every tick, so the overshoot is a control-tuning characteristic (steering
  saturates on the return), not stale perception.

### Known issues
- **Post-pass return overshoot** (worst cross-track ~2.85 m, within the road):
  on the return to lane the MPC saturates steering (`MAX_STEER`) and the heading
  overshoots, because the cost weights cross-track tracking (150) far above
  heading/yaw damping. Bounded and safe; a future pass will retune the cost and
  enable the centreline-referenced `AVOID` speed cap.
- **Intermittent CARLA server crash** in `UTaggedComponent::CreateSceneProxy()`
  during Unreal garbage collection — an engine-side race in CARLA 0.9.15, not in
  this code. Workaround: launch the server with `-quality-level=Low`, or run
  without `--view-lidar-3d`.

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