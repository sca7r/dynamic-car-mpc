# Dynamic-Bicycle Car MPC

A real-time **iterative MPC** trajectory-tracking controller for a **car**,
built on the **dynamic bicycle (single-track) model with a linear tire model** —
the same class of model used in real autonomous-driving and racing stacks. It
tracks a ~600 m circuit at cruise speed while avoiding a stalled car intruding
into the lane, and reports full vehicle telemetry (speed, steering, sideslip,
lateral-g).

![demo](demo.gif)

## Why the dynamic model

A *kinematic* bicycle model assumes the tires never slip — the car goes exactly
where the wheels point. That is fine at low speed, but it becomes wrong in
exactly the situations that matter: hard, fast cornering and quick avoidance
manoeuvres, where the tires actually develop slip and the car's lateral and yaw
dynamics dominate. The **dynamic** model carries lateral velocity and yaw rate
as states and generates cornering force from tire **slip angles**, so it
captures understeer/oversteer, sideslip, and the real limits of grip.

State `X = [x, y, ψ, vx, vy, r]` — position, yaw, longitudinal velocity, lateral
velocity, yaw rate. Control `u = [a, δ]` — longitudinal acceleration and front
steering angle.

```
x_dot   = vx·cos ψ − vy·sin ψ
y_dot   = vx·sin ψ + vy·cos ψ
 psi_dot = r
vx_dot  = a + vy·r
vy_dot  = (Fyf + Fyr)/m − vx·r
r_dot   = (lf·Fyf − lr·Fyr)/Iz
```

with linear tire forces from slip angles

```
alpha_f = δ − (vy + lf·r)/vx ,   Fyf = Cf·alpha_f
alpha_r =   − (vy − lr·r)/vx ,   Fyr = Cr·alpha_r
```

The model was sanity-checked before use: at 15 m/s the lateral dynamics are
stable and well-damped (eigenvalues ≈ −12.4 ± 3.9j), steady-state cornering
shows mild, safe understeer, and the discretized model is stable.

## How it relates to the other vehicle models

This shares its entire architecture with the kinematic controllers (the iMPC
loop, the track-relative cost, the soft obstacle constraints, the warm-started
DPP problem). The differences are localized:

- The model function (`_dynamics`) and the state/control dimensions (6 / 2).
- Because the tire dynamics are stiffer, the model is **linearized numerically**
  (finite-difference Jacobians, which keep the tire math in one obviously-correct
  place) and **discretized exactly** with a matrix exponential (ZOH) rather than
  forward Euler, which can go unstable on the lateral modes.

## Real-time performance

The QP is **DPP-compliant** — every parameter enters as a `Parameter-matrix @
Variable` term, so cvxpy canonicalizes once and reuses it. Steady-state solve
time is ~70–80 ms (the controller runs at 10 Hz, dt = 0.1 s), which is real-time
for this rate. The dominant remaining cost is the matrix-exponential
discretization at each horizon node; an analytic Jacobian would speed it up.

The solver is wrapped for robustness: it tries OSQP (warm-started, polished),
then falls back to CLARABEL, and only if every solver fails on a given cycle does
it command an **emergency brake** and reset the warm start. In a full lap with
the avoidance manoeuvre this fallback fires at most about once (typically the
first cold start).

## Low-speed handling

The slip angles divide by `vx`, which is singular at standstill. `vx` is floored
at `v_min` inside the tire model — the standard pragmatic fix. A more rigorous
option is to blend to the kinematic model below a few m/s; the demo simply starts
the car already moving, as a car on a road would be.

## Obstacle avoidance

Obstacles are enforced as **soft half-plane constraints** (with a slack penalty,
so tight situations degrade gracefully instead of going infeasible). The
constraint plane is tangent to the keep-out disc and oriented from the obstacle
toward the reference path, so it points longitudinally while the car approaches
and rotates to lateral as the car comes alongside — a smooth bulge around the
obstacle. In the demo the car holds a minimum clearance of 1.40 m (exactly the
configured safety buffer) around a stalled car protruding into the lane.

This formulation assumes the obstacle is **offset** from the centreline. An
obstacle sitting dead-centre on the path is the degenerate case (the
obstacle→path direction collapses) and needs a lane-change variant: a half-plane
perpendicular to the road on a chosen pass side, gated by longitudinal distance.
That is a natural next feature (see below).

## Files

| File                      | Purpose                                                       |
|---------------------------|---------------------------------------------------------------|
| `config.py`               | **Edit this** to change the road, obstacles, speed, car physics. |
| `simulate.py`             | Closed-loop MPC sim; writes `result.png`, `telemetry.png`, `demo.gif`. |
| `drive.py`                | **Live: 3D + top-down together** (split-screen, switchable). Start here. |
| `drive_pygame.py`         | Live top-down only (also standalone).                              |
| `drive_pygame_3d.py`      | Live 3D chase-camera only (also standalone).                       |
| `manual_drive.py`         | Scripted open-loop inputs -> a path + response plot (no window).   |
| `dynamic_bicycle_mpc.py`  | The `DynamicBicycleMPC` controller, model, and path helpers.  |

## Quick start

```bash
pip install -r requirements.txt
python simulate.py            # run the closed-loop sim, save the three outputs
```

This produces `result.png` (the road + driven line + obstacles), `telemetry.png`
(speed / steering / sideslip / lateral-g over the run), and `demo.gif` (animation
with the live MPC plan). Flags:

```bash
python simulate.py --no-gif   # skip the gif (much faster)
python simulate.py --live     # also open an interactive window
```

## Watch it live (pygame)

For an interactive window (no Gazebo/CARLA needed — pygame is a tiny install):

```bash
pip install pygame
python drive.py
```

`drive.py` shows **both views at once** and lets you switch layout with **V**:

- **SPLIT** — 3D chase camera (left) + top-down (right), in sync
- **3D** — full-window 3D chase camera
- **TOP** — full-window top-down

Two driving modes, toggled with **TAB**:

- **AUTO** — the MPC drives, avoiding obstacles, drawing its live planned path
- **MANUAL** — *you* drive (no controller): `UP` accelerate, `DOWN` brake,
  `LEFT`/`RIGHT` steer (self-centres). Watch the yaw rate and sideslip in the HUD
  respond with the lag a real car has.

Keys: `V` layout · `TAB` mode · `R` reset · `+`/`-` target speed · `ESC` quit.

(If you prefer a single view, `drive_pygame.py` is top-down-only and
`drive_pygame_3d.py` is 3D-only — both run standalone.) All three read the same
`config.py` and run the same controller and dynamics; only the drawing differs.
The 3D view is software-rendered (3D maths + depth-sorted shaded polygons through
pygame), so it needs no OpenGL; a GPU engine (Ursina/Panda3D) or PyOpenGL would
be the route to photoreal graphics later.

## Changing the road, obstacles, speed, and car (config.py)

Everything you'd want to tweak lives in `config.py`:

- **Road** — `TRACK_X` / `TRACK_Y` are the centreline waypoints; a smooth spline
  is fit through them. Add, move, or delete points to reshape the road.
- **Obstacles** — `OBSTACLES` is a list. Place one explicitly with
  `{"x": .., "y": .., "radius": ..}`, or relative to the road with
  `{"along": 0.13, "offset": 3.0, "radius": 2.5}` where `along` is the fraction
  around the lap (0–1) and `offset` is metres left (+) / right (−) of the
  centreline. Add as many as you like.
- **Speed** — `TARGET_SPEED` and `START_SPEED`.
- **Car physics** — `CAR` holds mass, yaw inertia, axle distances, and front/rear
  cornering stiffness. For example, lowering `Cr` relative to `Cf` makes the car
  more oversteery; you'll see the difference in the telemetry and the driven line.

Two caveats baked into the comments: the controller avoids **one** obstacle at a
time (the nearest one in view), so keep obstacles reasonably spaced; and keep
obstacles **offset** from the centreline (an obstacle dead-centre on the path is
the degenerate case — see below).

## Giving your own inputs (manual_drive.py)

To *feel* the dynamics directly, open `manual_drive.py` and edit the `INPUTS`
list — a schedule of `(duration_seconds, acceleration, steering_angle)` commands.
There's no controller in the loop, so you're driving the open-loop car:

```python
INPUTS = [
    (2.0, 0.0,  0.00),   # cruise straight
    (1.0, 0.0,  0.10),   # steer left ~5.7 deg
    (1.0, 0.0, -0.10),   # steer right
    (2.0, 2.0,  0.05),   # accelerate through a gentle left
    (2.0, -3.0, 0.00),   # brake in a straight line
]
```

```bash
python manual_drive.py        # writes manual_result.png (path + response)
```

The response plot shows the **yaw-rate lag and sideslip** that distinguish the
dynamic model from a kinematic one — a step of steering doesn't instantly become
a turn rate; the car's mass and tires take a moment to respond.


## Extension ideas

- **Nonlinear tire model (Pacejka "magic formula").** The linear tire model has
  no force ceiling, so it can't represent grip saturation or sliding. A Pacejka
  or brush model adds the friction limit — the single biggest step toward
  realistic limit-handling and racing behaviour.
- **Lane-change obstacle avoidance.** Add the road-perpendicular, pass-side,
  longitudinally-gated half-plane so the car can handle an obstacle squarely in
  its lane, and support multiple simultaneous obstacles.
- **Curvature-based speed profile.** Slow down before tight corners instead of
  holding a fixed target speed — closer to how a real driver and a real planner
  behave.
- **Analytic Jacobians** to cut solve time, and **ROS 2 integration** to publish
  steering/throttle to a simulator (CARLA) or vehicle.

## A note on the controller's internal frame

The controller works in the **ego frame** each cycle: the reference is
transformed so the car sits at the origin facing forward, and the initial state
passed to `solve()` is `[0, 0, 0, vx, vy, r]` — only the body-frame velocities
and yaw rate carry over, since position and heading are zero by construction. The
optimal plan is mapped back to global coordinates with `ego_to_global`; the
controls `(a, δ)` are frame-independent.
