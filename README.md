# dcmpc - Dynamic-Bicycle Car MPC

A real-time **iterative Model Predictive Controller (MPC)** for an autonomous
car, built on the **dynamic single-track ("bicycle") model with a linear tire
model** - the same class of model used in professional autonomous-driving and
motorsport stacks. It tracks a reference path, slows for corners and obstacles
the way a human driver does, avoids multiple obstacles via soft half-plane
constraints, and runs both in a custom pygame visualizer and on CARLA with a
virtual-LiDAR perception front-end.

![demo](demo.gif)

---

## Table of contents

1. [Quick start (step by step)](#quick-start-step-by-step)
2. [What each command does](#what-each-command-does)
3. [Debugging the LiDAR](#debugging-the-lidar)
4. [Package layout](#package-layout)
5. [The maths, and why the car behaves the way it does](#the-maths-and-why-the-car-behaves-the-way-it-does)
6. [Configuring the scenario](#configuring-the-scenario)
7. [Extensions](#extensions)

---

## Quick start (step by step)

**1. Get the code and a clean Python environment.**

```bash
git clone <your-repo> dcmpc
cd dcmpc
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

**2. Install the package (editable, so edits take effect immediately).**

```bash
pip install -e ".[viz]"            # core solver + pygame viewers
```

That single command pulls in NumPy, SciPy, CVXPY, OSQP, Matplotlib and pygame,
and registers the `dcmpc-*` commands on your PATH.

**3. Run the headless simulation first - no GPU, no CARLA, ~10 seconds.**

```bash
dcmpc-sim --no-gif
```

You'll get `result.png` (the driven line around the track) and `telemetry.png`
(speed / steering / sideslip / lateral-g). This is the fastest way to confirm
the controller works on your machine.

**4. Watch it drive live.**

```bash
dcmpc-drive
```

Controls: **V** switches view (split / 3D / top-down), **TAB** toggles
AUTO ↔ MANUAL, **+ / −** change the cruise speed, **R** resets, **ESC** quits.
In MANUAL mode you steer with the arrow keys - useful for *feeling* the tire
dynamics (the yaw rate and sideslip lag behind your steering input, exactly as
in a real car).

**5. (Optional) Run on CARLA.**

```bash
# install CARLA's Python API to match your server build
pip install carla==<your CARLA server version>

# start the simulator (separate terminal), wait for the map to load
./CarlaUE4.sh                      # Windows: CarlaUE4.exe

# then, in the venv:
dcmpc-carla --debug
```

If CARLA refuses to connect, it's almost always (a) the server isn't finished
loading, or (b) a port clash on 2000 - see [Debugging the LiDAR](#debugging-the-lidar)
and the troubleshooting note at the end of this section.

**6. Inspect a CARLA run.** Every CARLA session writes `carla_trace_*.csv`:

```bash
dcmpc-plot                         # plots the newest trace
```

> **CARLA connection troubleshooting.** If you see a 10 s timeout: confirm the
> server window is fully loaded; check the port with `netstat -ano | findstr 2000`
> (Windows). If two processes are listening on 2000, kill both and start CARLA
> once. To use a non-default port, launch with `-carla-rpc-port=N` and set
> `CARLA_PORT = N` in `src/dcmpc/carla_bridge.py`.

---

## What each command does

| Command | Description |
|---------|-------------|
| `dcmpc-sim`    | Headless closed-loop sim → `result.png`, `telemetry.png`, `demo.gif` |
| `dcmpc-drive`  | Live pygame viewer: 3D chase + top-down split (recommended) |
| `dcmpc-3d`     | Live 3D chase camera only |
| `dcmpc-top`    | Live top-down only |
| `dcmpc-manual` | Open-loop: drive with your own scripted inputs (feel the dynamics) |
| `dcmpc-carla`  | CARLA bridge with virtual-LiDAR perception |
| `dcmpc-plot`   | Plot a `carla_trace_*.csv` from a CARLA run |

Useful flags: `dcmpc-sim --no-gif` (skip the slow animation), `dcmpc-carla
--debug` (per-tick logging), `dcmpc-carla --no-lidar` (use ground-truth actor
positions instead of LiDAR), `dcmpc-carla --view-lidar` (live 2-D filter-debug
plot), `dcmpc-carla --view-lidar-3d` (live 3-D semantic point cloud).

As a library:

```python
from dcmpc import DynamicBicycleMPC, CarParams, AdaptiveSpeed, get_ref_trajectory
```

---

## Debugging the LiDAR

In CARLA the obstacles fed to the MPC come from a **virtual LiDAR**: the raw
point cloud is filtered (height band, forward field-of-view, range, ego
exclusion), grid-clustered, and each surviving cluster becomes a circular
keep-out. When the car stops with apparent free space, or swerves at nothing,
you need to *see what the perception layer sees*:

```bash
dcmpc-carla --view-lidar
```

A live top-down Matplotlib window opens, colour-coded by filter stage:

| Colour | Meaning |
|--------|---------|
| **Grey** | returns dropped by the height filter (ground, overhead) |
| **Orange** | dropped by FOV / range / ego-exclusion |
| **Green** | survived all filters → handed to clustering |
| **Red circle** | a detected obstacle the MPC actually receives (centre + keep-out radius) |

Reading it: if a wall shows up as a **red circle**, raise `LIDAR_MIN_PTS` or
lower `LIDAR_MAX_RADIUS` in `config.py`. If a real car never turns **green**,
loosen the height band (`LIDAR_Z_MIN`, `LIDAR_Z_MAX`) or widen `LIDAR_FOV_DEG`.
All of these live in the config (see *Configuring* below).

### 3D semantic view (like a real AV stack)

For a richer view - a 3D perspective cloud where **each detected cluster gets
its own colour** (cars, poles, scenery), the ground dimmed, and the ego car at
the origin - use:

```bash
dcmpc-carla --view-lidar-3d
```

This uses **Open3D** if installed (`pip install open3d`, included in the `gui`
extra) for a dense, mouse-orbitable GPU window; if Open3D isn't present it falls
back automatically to a lower-fidelity matplotlib-3D window. The clustering used
for colouring is computed independently for the display and never touches the
controller, so the view can't affect driving. You can run `--view-lidar` (2-D
filter debug) and `--view-lidar-3d` (semantic 3-D) together or separately.

> Both live views redraw every tick and **slow the loop down** - use them for
> debugging, not timed runs.

---

## Package layout

```
src/dcmpc/
├── __init__.py        # public API: DynamicBicycleMPC, AdaptiveSpeed, helpers
├── controller.py      # vehicle model + iMPC + AdaptiveSpeed + path helpers
├── config.py          # EDIT THIS: road, obstacles, speed, car physics, CARLA
├── simulate.py        # headless closed-loop sim
├── manual_drive.py    # open-loop scripted inputs
├── carla_bridge.py    # CARLA bridge + virtual-LiDAR perception + 2-D viewer
├── lidar_view3d.py    # 3-D semantic point-cloud viewer (Open3D / matplotlib)
├── plot_trace.py      # CARLA CSV trace plots
└── viz/
    ├── drive.py       # unified split-screen viewer
    ├── top_down.py    # standalone top-down
    └── chase_3d.py    # standalone 3D chase camera
```

Because the install is editable, editing `src/dcmpc/config.py` takes effect
immediately - no reinstall.

---

## The maths, and why the car behaves the way it does

### 1. State and the dynamic bicycle model

The car is modelled as a single track (front and rear axles each collapsed to
one wheel). The state and control are

```
X = [x, y, ψ, vx, vy, r]          u = [a, δ]
```

`x, y` position, `ψ` heading, `vx` longitudinal and `vy` lateral velocity (both
in the car's own frame), `r` yaw rate; `a` longitudinal acceleration, `δ` front
steering angle. The continuous dynamics `Ẋ = f(X, u)` are

```
ẋ   = vx·cos ψ − vy·sin ψ          (body velocity rotated into the world)
ẏ   = vx·sin ψ + vy·cos ψ
ψ̇   = r
v̇x  = a + vy·r
v̇y  = (Fyf + Fyr)/m − vx·r
ṙ   = (lf·Fyf − lr·Fyr)/Iz
```

The two terms that make this a *dynamic* model rather than a *kinematic* one are
the `vy·r` / `vx·r` cross-couplings (centripetal effects) and, crucially, the
tire forces `Fyf`, `Fyr`. A kinematic bicycle assumes the car goes exactly where
the wheels point; this model lets the tires **slip**, which is what actually
happens above walking pace and is the whole point of using it.

### 2. Tire forces and slip angles - the source of the behaviour

Each tire generates lateral force in proportion to its **slip angle** - the
angle between where the tire points and where it's actually travelling:

```
αf = δ − (vy + lf·r)/vx        Fyf = Cf · αf
αr =   − (vy − lr·r)/vx        Fyr = Cr · αr
```

`Cf`, `Cr` are the cornering stiffnesses. This linear law is why **the
front-to-rear stiffness balance sets the handling character**:

- `lf·Cf > lr·Cr`-ish balance → the rear grips relatively harder → mild
  **understeer** (the safe, stable default; the car gently pushes wide and
  self-corrects). This project's default Tesla-like parameters are tuned here.
- Lower `Cr` relative to `Cf` → the rear lets go first → **oversteer** (the tail
  steps out). Try it in `config.py` and watch the sideslip trace grow.

Two guards keep this valid at the limits, and both shape the behaviour you see:

- **`tanh` slip saturation:** `α ← α_max · tanh(α/α_max)`. A real tire has a grip
  ceiling - past a few degrees of slip the force stops climbing. The hard clamp
  we used originally has a *zero* derivative past the limit, which makes the
  linearization (below) singular and the solver stall. `tanh` saturates smoothly,
  so the Jacobian stays well-conditioned. This is why solves are stable even in
  hard avoidance manoeuvres.
- **Low-speed fade:** `fade = min(1, |vx|/v_blend)` multiplies the lateral and
  yaw accelerations. The slip-angle formulas divide by `vx`, which blows up as
  the car approaches a stop. Fading the lateral dynamics to zero below a few m/s
  removes that singularity - the model degrades gracefully to "can't generate
  cornering force at standstill," which is physically correct.

**Why steering feels laggy in MANUAL mode:** a step of `δ` doesn't instantly
become yaw rate. It first creates a front slip angle, which builds `Fyf`, which
produces `ṙ`, which integrates into `r`, which then changes the slip angles
again. That chain of integrations is the real first-order-ish lag you see in the
yaw-rate and sideslip telemetry - the signature of a dynamic model.

### 3. From nonlinear model to a QP: linearize, then discretize exactly

MPC needs a *linear* model at each step. Around the current operating point
`(x̄, ū)` we take a first-order expansion `Ẋ ≈ A·X + B·u + c`, where `A`, `B`
are Jacobians and `c` is the affine residual. We compute `A`, `B` by **finite
differences** (`controller._model_matrices`): it keeps the tire algebra in one
obviously-correct place and is robust to the `tanh` nonlinearity. (Analytic
Jacobians would be a pure speed optimization.)

We then need the **discrete** step `X_{k+1} = A_d·X_k + B_d·u_k + C_d`. Rather
than forward Euler (`A_d ≈ I + A·dt`), which can turn the stiff lateral modes
*unstable* for the step sizes we use, we discretize **exactly** with a single
matrix exponential (zero-order hold). Stacking `A`, `B`, `c` into one augmented
matrix `Φ` and exponentiating gives `A_d`, `B_d`, `C_d` in one shot:

```
        ⎡ A  B  c ⎤
Φ·dt =  ⎢ 0  0  0 ⎥ · dt        exp(Φ·dt) → top block = [A_d  B_d  C_d]
        ⎣ 0  0  0 ⎦
```

This is why the prediction stays stable even though the lateral dynamics are
stiff.

### 4. The optimization (QP)

Over a horizon of `N` steps we solve, each control cycle,

```
min  Σ_k [ ‖X_k − X_ref,k‖²_Q + ‖u_k‖²_R + ‖u_k − u_{k−1}‖²_Rd ]
       + ‖X_N − X_ref,N‖²_Qf  +  ρ·Σ_k Σ_j s_{j,k}

s.t. X_{k+1} = A_d,k X_k + B_d,k u_k + C_d,k      (dynamics)
     X_0 = current state
     control & rate limits
     n_{j,k} · X_{k+1} ≥ b_{j,k} − s_{j,k},  s ≥ 0  (obstacle half-planes)
```

- `Q = diag(5, 150, 8, 40, 2, 2)` - the large weight on `y` (150) is lateral
  tracking; that's what keeps the car pinned to the path.
- `Rd = (1, 250)` heavily penalises **steering-rate** changes - this is what
  stops the violent snap-back during a lane change and gives the smooth re-entry.
- The constraints are **soft**: a slack `s` with a big penalty `ρ = 1e5` lets a
  geometrically impossible request degrade gracefully (squeeze the margin a
  little) instead of returning "infeasible" and triggering an emergency brake.

The whole problem is built to be **DPP-compliant** (every parameter enters as
`Parameter @ Variable`), so CVXPY canonicalises the structure once and only
swaps in numbers each cycle. That's what gets solve times to ~60-80 ms at 10 Hz.
The solver is wrapped in an OSQP→CLARABEL→OSQP fallback chain, with an emergency
brake only if all of them fail on the same cycle.

### 5. Obstacle avoidance as half-planes

Each obstacle becomes one or more **half-plane** keep-outs. With the obstacle at
`o`, a keep-out radius `R = r_obs + buffer`, and a chosen normal `n` (pointing
from the obstacle toward the side we pass on), the constraint

```
n · (X_xy − o) ≥ R
```

forbids the car from entering the disc on that side. When the car is still
approaching, `n` points longitudinally (stay behind / slow); as it comes
alongside, `n` rotates lateral (push to the open side) - a smooth bulge around
the obstacle. The MPC carries up to `MAX_OBS` of these simultaneously, so it can
thread between several obstacles at once. Which side it commits to is **locked in
world coordinates** so the choice can't flip frame-to-frame as the car moves
(the cause of the earlier spinning).

If neither side has room (`max(room_left, room_right) < 0`), the decision flips
to **stop**: the position reference is clamped to a stop line behind the obstacle
and the speed reference is zeroed, so the controller brakes to a halt instead of
fighting a forward-pulling reference.

### 6. Human-like speed (`AdaptiveSpeed`)

The car does not hold a fixed speed. Three effects shape the speed reference:

- **Corner slowdown.** Looking ahead along the path, we find the tightest
  curvature `κ` and cap speed so lateral acceleration stays within a comfortable
  limit. This is just circular-motion physics, `a_lat = v²·κ`, solved for `v`:

  ```
  v_max = √(a_lat_comfort / κ)
  ```

  Higher `a_lat_comfort` → faster, more aggressive cornering; the pygame default
  is 4.5 m/s², while the CARLA bridge passes a cautious 1.5 m/s² for town driving.

- **Obstacle braking.** Speed is reduced in proportion to proximity, scaling from
  cruise down to a floor as the obstacle nears.

- **Smoothing.** The result is passed through a first-order low-pass (time
  constant τ), so the car eases on and off the throttle rather than stepping -
  the gradual feel a human gives.

The HUD's `ADAPT` row shows which effect is active (green = FREE, amber =
CORNER, red = OBS).

### 7. A dynamically feasible reference

The reference itself respects acceleration limits: instead of placing reference
points at a constant `target_v`, `get_ref_trajectory` integrates a speed profile
that ramps up/down within ±a few m/s², so the MPC is never asked to track a
step it physically can't. A **recovery mode** throttles that reference speed when
the car is pushed far off-line (cross-track > 1.5 m), giving the solver room to
steer gently back instead of diverging.

---

## Configuring the scenario

**Every tunable in the project lives in one place - `src/dcmpc/config.py`** -
grouped by subsystem, each with a comment explaining what it does and which way
to turn it. Because the install is editable, edits take effect with no reinstall.
The groups:

- **Road** - `TRACK_X` / `TRACK_Y` centreline waypoints (a smooth spline is fit
  through them).
- **Speed** - `TARGET_SPEED`, `START_SPEED`.
- **Obstacles** - the `OBSTACLES` list: place by world coords `{"x","y","radius"}`
  or relative to the road `{"along": 0-1, "offset": ±m, "radius"}`.
- **Car physics** - `CAR = dict(m, Iz, lf, lr, Cf, Cr)`. Lower `Cr` for
  oversteer; raise it for more understeer.
- **Controller limits** - `MAX_SPEED`, `MAX_ACC`, `MAX_D_ACC`, `MAX_STEER`,
  `MAX_D_STEER`: the physical envelope the MPC may command.
- **Cost weights** - `STATE_COST`, `FINAL_STATE_COST`, `INPUT_COST`,
  `INPUT_RATE_COST`: what the MPC optimises. Raise the cross-track term for
  tighter path-following; raise the steering-rate term for smoother steering.
- **Avoidance** - `OBSTACLE_SAFETY_MARGIN` (clearance), `ROAD_HALFWIDTH`
  (overtake-vs-stop), `PASS_ZONE` (keep-out length past the obstacle),
  `SLACK_PENALTY`.
- **Adaptive speed** - `ADAPT_A_LAT_COMFORT` (cornering aggressiveness),
  `ADAPT_LOOKAHEAD`, `ADAPT_OBS_BRAKE_START/END`, `ADAPT_V_MIN`, `ADAPT_TAU`.
- **Timing** - `DT`, `HORIZON_TIME`.
- **CARLA** - `CARLA_TARGET_SPEED`, throttle/brake mapping, detected-obstacle
  radius, `CARLA_A_LAT_COMFORT`.
- **Virtual LiDAR** - `LIDAR_Z_MIN/MAX` (height band), `LIDAR_FOV_DEG`,
  `LIDAR_EGO_EXCLUSION`, `LIDAR_MAX_RANGE`, `LIDAR_CLUSTER_CELL`, `LIDAR_MIN_PTS`,
  `LIDAR_MAX_RADIUS` (wall rejection). These decide what the perception layer
  treats as an obstacle versus ground/wall/scenery.

Quick tuning cheatsheet (also at the top of the file):

| Symptom | Knob |
|---------|------|
| Passes obstacles too close | raise `OBSTACLE_SAFETY_MARGIN` |
| Swerves too wide | lower `OBSTACLE_SAFETY_MARGIN` |
| Too timid in corners | raise `ADAPT_A_LAT_COMFORT` |
| Cuts back in too early after a pass | raise `PASS_ZONE` |
| Drifts off the path | raise `STATE_COST` cross-track term (index 1) |
| Steering oscillates / snaps | raise `INPUT_RATE_COST` steering term (index 1) |
| Walls detected as obstacles (CARLA) | raise `LIDAR_MIN_PTS` or lower `LIDAR_MAX_RADIUS` |

---

## Performance note (CARLA frame rate)

CARLA renders on the GPU; the MPC solves on the CPU. The QP solve is the
per-tick bottleneck, so on a slower machine the control loop runs slower than
CARLA's render rate. This is expected - it's a CPU optimisation problem, not a
graphics one, and a GPU would not help a QP this small (the host↔device transfer
overhead exceeds the solve). If you need more headroom, the CPU levers are:
shorten `HORIZON_TIME` (solve time scales with horizon length) and keep the
scene's obstacle count modest. These trade foresight for speed, so change them
deliberately and re-test.

---

## Extensions

Pacejka nonlinear tire model (a real friction ceiling instead of the linear-plus-
`tanh` approximation), analytic Jacobians (faster solves), a ROS 2 bridge, and a
learned-dynamics variant with conformal-prediction safety tubes for
out-of-distribution robustness.

See `CHANGELOG.md` for the full development history.
