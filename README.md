# Dynamic-Bicycle Car MPC

A real-time **iterative Model Predictive Controller** for an autonomous car,
built on the **dynamic single-track (bicycle) model with a linear tire model** —
the same class of model used in professional autonomous-driving and motorsport
stacks. The car tracks a 600 m circuit at cruise speed, avoids stalled obstacles,
and adapts its speed to corners and hazards the way a human driver would.

![demo](demo.gif)

---

## Quick start

```bash
pip install -r requirements.txt   # cvxpy, numpy, scipy, matplotlib, osqp
python simulate.py                # run the sim, save result.png + demo.gif
python simulate.py --no-gif       # skip the gif (much faster)
```

For a live interactive window:

```bash
pip install pygame
python drive.py                   # recommended entry point (3D + top-down)
```

---

## Project structure

```
├── config.py              ← Edit this to change the road, car, obstacles, speed
├── dynamic_bicycle_mpc.py ← Controller, vehicle model, path helpers, AdaptiveSpeed
├── simulate.py            ← Headless closed-loop sim → result.png, demo.gif
├── drive.py               ← Live pygame viewer: 3D chase + top-down (V to switch)
├── drive_pygame.py        ← Standalone top-down viewer
├── drive_pygame_3d.py     ← Standalone 3D chase-camera viewer
├── manual_drive.py        ← Open-loop: give your own throttle/steer inputs
├── carla_mpc.py           ← CARLA bridge (real physics engine, Town10)
├── plot_trace.py          ← Plot a carla_trace_*.csv produced by carla_mpc.py
├── requirements.txt
└── README.md
```

**The only file you need to edit is `config.py`.** Everything else reads from it.

---

## Live viewer (`drive.py`)

```bash
python drive.py
```

Press **V** to cycle layouts:

| Layout | What you see |
|--------|-------------|
| SPLIT  | 3D chase camera (left) + top-down map (right) |
| 3D     | Full-window 3D chase camera |
| TOP    | Full-window top-down with full telemetry HUD |

Press **TAB** to toggle driving mode:

| Mode   | Who drives |
|--------|-----------|
| AUTO   | The MPC drives, adapts speed, avoids obstacles |
| MANUAL | You drive — arrow keys (feel the dynamic tire model) |

Other keys: `+` / `-` adjust cruise speed · `R` reset · `ESC` quit.

The HUD shows speed (with a target-speed marker on the bar), steering angle,
yaw rate, sideslip, lateral-g, a G-force circle, lap times, and — in AUTO mode
— an `ADAPT` row showing the adaptive speed and reason (`FREE` / `CORNER` / `OBS`).

---

## Configuring the scenario (`config.py`)

```python
# Road centreline waypoints (metres) — add/move/remove to reshape the track
TRACK_X = [0, 60, 120, ...]
TRACK_Y = [0,  0,  10, ...]

# Cruise speed the controller tries to maintain
TARGET_SPEED = 11.0          # m/s  (~40 km/h)

# Obstacles — place as many as you like
OBSTACLES = [
    {"along": 0.13, "offset":  1.8, "radius": 2.5},  # fraction of lap, metres left
    {"along": 0.40, "offset": -2.0, "radius": 2.0},
    {"x": 77.0, "y": 3.0, "radius": 2.5},            # or exact world coords
]

# Car physics (Tesla Model 3 defaults)
CAR = dict(m=1845.0, Iz=2600.0, lf=1.44, lr=1.44, Cf=140000.0, Cr=160000.0)

# Obstacle avoidance tuning
OBSTACLE_SAFETY_MARGIN = 1.5  # metres clearance from obstacle surface
ROAD_HALFWIDTH = 5.0          # drivable width for overtake/stop decision
PASS_ZONE = 6.0               # longitudinal extent of the keep-out zone [m]
```

---

## Vehicle model

**State** `X = [x, y, ψ, vx, vy, r]` — position, yaw, longitudinal velocity,
lateral velocity, yaw rate. **Control** `u = [a, δ]` — acceleration and front
steering angle.

```
ẋ   = vx·cos ψ − vy·sin ψ
ẏ   = vx·sin ψ + vy·cos ψ
ψ̇   = r
v̇x  = a + vy·r
v̇y  = (Fyf + Fyr)/m − vx·r
ṙ   = (lf·Fyf − lr·Fyr)/Iz
```

Tire forces from slip angles (linear model with saturation and low-speed fade):

```
αf = δ − (vy + lf·r)/vx        Fyf = Cf·αf
αr =   − (vy − lr·r)/vx        Fyr = Cr·αr
```

The model is validated before use: at 13 m/s the lateral eigenvalues are
stable (`−15.5 ± 1.2j`), steady-state cornering shows mild understeer, and
the discretized model (matrix-exponential ZOH) is numerically stable.

**Two physical guards keep the model valid at the limits:**
- **Slip-angle saturation** (`α_max = 0.12 rad`) — tires have a grip ceiling.
- **Low-speed fade** — lateral forces and yaw accel fade to zero below 2 m/s,
  preventing the singularity at standstill from causing divergence.

---

## Controller architecture

The MPC works in the **ego frame** each tick: the reference trajectory is
expressed relative to the car's current pose, so `solve()` always receives
`[0, 0, 0, vx, vy, r]` as the initial state. The optimal plan is converted
back to world coordinates with `ego_to_global`.

The QP is **DPP-compliant** — parameters enter as `Parameter @ Variable` so
CVXPY canonicalises once and reuses the structure, giving ~70–80 ms solves at
10 Hz (real-time). Solver fallback chain: OSQP (warm, polished) → CLARABEL
(cold) → OSQP (cold). Emergency brake on total failure; warm start resets.

---

## Adaptive speed (`AdaptiveSpeed`)

The car does not maintain a fixed cruise speed. `AdaptiveSpeed` modulates the
speed reference each tick based on three behaviours:

1. **Obstacle braking** — smooth deceleration proportional to proximity,
   starting 40 m from the obstacle surface and reaching minimum speed at 12 m.
2. **Corner slowdown** — looks 25 m ahead, finds the tightest curvature `κ`,
   and caps speed at `v_max = √(a_lat / κ)` (comfortable lateral-g limit).
   Speed rises automatically on straights.
3. **Smooth transitions** — first-order low-pass (τ = 1.2 s) so speed changes
   feel gradual, not stepped.

The HUD `ADAPT` row shows the active speed and the reason in colour:
green = FREE, amber = CORNER, red = OBS.

---

## Obstacle avoidance

Obstacles are enforced as **soft half-plane constraints** with a slack penalty.
The controller makes one of three decisions per tick:

| Situation | Decision |
|-----------|----------|
| Obstacle offset; room to pass | **LANE-CHANGE** — lateral keep-out on the far side, gated to the length alongside the obstacle |
| Obstacle centred; room either side | **LANE-CHANGE** — picks the side with more margin |
| No room to pass (narrow road) | **STOP** — longitudinal keep-out + zero speed reference |

The minimum clearance between the car body and obstacle surface is controlled
by `OBSTACLE_SAFETY_MARGIN` in `config.py` (default 1.5 m → measured buffer
~2.4 m from obstacle centre).

---

## CARLA integration (`carla_mpc.py`)

```bash
./CarlaUE4.sh                 # start CARLA server (Town10HD)
pip install carla
python carla_mpc.py           # INFO logging + CSV trace
python carla_mpc.py --debug   # per-tick details
python carla_mpc.py --quiet   # warnings/errors only
python carla_mpc.py --no-csv  # skip the CSV trace
python carla_mpc.py --no-obstacle  # don't spawn stalled cars
```

Key things the bridge does:
- Builds the reference path from the **actual CARLA lane** (not a hand-coded
  spline), following waypoints for up to 800 m.
- **Spawns the car on `path[0]` facing along the path** — eliminates the
  heading-mismatch bug that causes the car to steer off-road at startup.
- Reads the vehicle's **real max steering angle** from `get_physics_control()`.
- **Three stalled obstacle cars** spawned along the route; all destroyed on exit.
- Adaptive speed active (slower defaults for town driving).
- Writes a per-tick **CSV trace** (`carla_trace_<timestamp>.csv`); plot it with
  `python plot_trace.py` to see cross-track error, heading error, speed, and
  steering over time, with red lines marking emergency-brake ticks.

Default vehicle: `vehicle.audi.tt`. Change `VEHICLE_FILTER` at the top of
`carla_mpc.py`. Run this to list what's available in your build:

```bash
python -c "import carla; w=carla.Client('localhost',2000).get_world(); [print(b.id) for b in w.get_blueprint_library().filter('vehicle.*')]"
```

---

## Manual drive (`manual_drive.py`)

Edit the `INPUTS` list — a schedule of `(duration, acceleration, steer)` — and
run `python manual_drive.py`. No controller in the loop; this is the best way to
feel the dynamic model: a step of steering doesn't instantly become a turn rate,
the car's mass and tires take a moment to respond (yaw-rate lag, sideslip build-up).

---

## Possible extensions

| Extension | What it adds |
|-----------|-------------|
| Pacejka "magic formula" tire model | Grip saturation, realistic limit-handling |
| Multi-obstacle simultaneous constraints | Handle clusters, not just nearest-one |
| ROS 2 bridge | Publish to a real vehicle or Gazebo |
| PyOpenGL / Ursina 3D view | Photoreal rendering without CARLA |
| System identification | Fit `Cf`, `Cr`, `Iz` to CARLA telemetry for tighter tracking |
