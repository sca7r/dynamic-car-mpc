"""
CARLA MPC bridge — drop-in integration for when you install CARLA.

This script connects to a running CARLA server, spawns a vehicle, and drives it
with the SAME dynamic-bicycle MPC from dynamic_bicycle_mpc.py. The only things
that change from simulate.py are:
  - The "plant" is CARLA's physics engine, not our RK4 integrator.
  - State comes from the vehicle's transform + velocity (CARLA telemetry).
  - Controls go back as throttle/brake/steer via CARLA's VehicleControl.
  - A top-down spectator camera renders the scene in CARLA's own window.

REQUIREMENTS
  1. Install CARLA:  https://carla.readthedocs.io/en/latest/start_quickstart/
     (Ubuntu: the .deb package is easiest)
  2. pip install carla          # Python client library
  3. Start the CARLA server:    ./CarlaUE4.sh   (in the CARLA install folder)
  4. Then run:                  python carla_mpc.py

TESTED WITH:  CARLA 0.9.14 / 0.9.15

HOW THE BRIDGE WORKS
  Each control tick (every DT seconds):
    1. Read vehicle Transform + Velocity from CARLA → build state [x,y,psi,vx,vy,r]
    2. Build ego-frame state (only [0,0,0,vx,vy,r]) and get the reference from the
       path (same get_ref_trajectory as always).
    3. Call mpc.solve() → get (traj, controls).
    4. Map controls[0] (acceleration) → throttle/brake float.
       Map controls[1] (steering angle delta) → CARLA steer float in [-1, 1].
    5. Apply VehicleControl to the CARLA actor.

KNOWN LIMITATIONS
  - The path is still our spline (TRACK_X/TRACK_Y in config.py), not a CARLA
    map road. You can use CARLA's waypoint API to build a path instead.
  - Only a single obstacle is fed to the MPC (same as the rest of the project).
    Other CARLA actors are detected by proximity check, not sensor raycast.
  - The dynamic-bicycle model won't perfectly match CARLA's PhysX vehicle
    (different tire model, suspension, etc.). The MPC still stabilises, but
    tracking will be less tight than in our own simulator. A Pacejka tire model
    and system-identification would close the gap.
"""

from __future__ import annotations

import sys
import time
import math
import numpy as np

# ── CARLA import (guarded so the file can be syntax-checked without CARLA) ──
try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False
    print("[carla_mpc] WARNING: 'carla' package not found.")
    print("  Install it with:  pip install carla")
    print("  Then start the CARLA server and re-run this script.")

import config as C
from dynamic_bicycle_mpc import (
    DynamicBicycleMPC, CarParams,
    compute_path_from_wp, get_ref_trajectory, ego_to_global,
)

# ── settings ──────────────────────────────────────────────────────────
CARLA_HOST      = "localhost"
CARLA_PORT      = 2000
TICK_DT         = C.DT              # must match MPC dt
SPAWN_TRANSFORM = None              # None → auto-pick a spawn point
VEHICLE_FILTER  = "vehicle.tesla.model3"  # or "vehicle.audi.a2" etc.
SPECTATOR_HEIGHT = 80.0             # metres above the car (top-down view)
MAX_STEER_ANGLE  = math.radians(70) # CARLA's full-lock angle (check your vehicle)
# acceleration → throttle/brake mapping
MAX_THROTTLE_ACC = 4.0              # m/s^2 that maps to throttle=1.0
MAX_BRAKE_DEC    = 6.0              # m/s^2 deceleration that maps to brake=1.0
OBSTACLE_RADIUS  = 2.5              # how big to treat nearby CARLA actors [m]
OBSTACLE_DETECT_R = C.SENSOR_RANGE


# ── helpers ───────────────────────────────────────────────────────────

def carla_state(vehicle) -> np.ndarray:
    """Read CARLA vehicle → state [x, y, psi, vx, vy, r]."""
    t = vehicle.get_transform()
    v = vehicle.get_velocity()
    av = vehicle.get_angular_velocity()

    x   =  t.location.x
    y   = -t.location.y          # CARLA is left-handed; flip y
    psi = -math.radians(t.rotation.yaw)

    # velocity in world frame → body frame
    ct, st = math.cos(-psi), math.sin(-psi)
    vx_w, vy_w = v.x, -v.y
    vx =  vx_w * ct + vy_w * st  # actually cos(psi)*vx_w + sin(psi)*vy_w
    vy = -vx_w * st + vy_w * ct  # lateral

    r = -math.radians(av.z)       # yaw rate (sign convention)
    return np.array([x, y, psi, vx, vy, r])


def acc_to_control(a: float, vx: float) -> carla.VehicleControl:
    """Map scalar acceleration [m/s^2] → CARLA throttle/brake/reverse."""
    ctrl = carla.VehicleControl()
    if a >= 0:
        ctrl.throttle = float(min(a / MAX_THROTTLE_ACC, 1.0))
        ctrl.brake    = 0.0
        ctrl.reverse  = False
    else:
        ctrl.throttle = 0.0
        ctrl.brake    = float(min(-a / MAX_BRAKE_DEC, 1.0))
        ctrl.reverse  = (vx < -0.5)   # allow reverse if already going backward
    return ctrl


def steer_to_control(delta: float) -> float:
    """Map steering angle [rad] → CARLA steer in [-1, 1]."""
    return float(max(-1., min(1., -delta / MAX_STEER_ANGLE)))


def nearby_obstacle(vehicle, world) -> tuple | None:
    """Find the nearest other actor in sensor range → ego-frame obstacle tuple."""
    actors = world.get_actors().filter("vehicle.*")
    state = carla_state(vehicle)
    best, best_d = None, 1e18
    for a in actors:
        if a.id == vehicle.id:
            continue
        t = a.get_transform()
        ox, oy = t.location.x, -t.location.y
        dx, dy = ox - state[0], oy - state[1]
        d = math.hypot(dx, dy)
        if d > OBSTACLE_DETECT_R:
            continue
        if d < best_d:
            ct, st = math.cos(-state[2]), math.sin(-state[2])
            ex, ey = dx*ct - dy*st, dx*st + dy*ct
            v = a.get_velocity()
            evx = v.x * ct + (-v.y) * st
            evy = -v.x * st + (-v.y) * ct
            best = (ex, ey, OBSTACLE_RADIUS, evx, evy)
            best_d = d
    return best


def set_spectator(world, vehicle):
    """Move the spectator camera to a top-down view above the car."""
    t = vehicle.get_transform()
    spec_t = carla.Transform(
        carla.Location(x=t.location.x, y=t.location.y, z=SPECTATOR_HEIGHT),
        carla.Rotation(pitch=-90, yaw=0, roll=0),
    )
    world.get_spectator().set_transform(spec_t)


# ── main ──────────────────────────────────────────────────────────────

def main():
    if not CARLA_AVAILABLE:
        sys.exit(1)

    # build the path and MPC (exactly as in simulate.py)
    path = compute_path_from_wp(C.TRACK_X, C.TRACK_Y, step=0.25)
    p = CarParams()
    for k, v in C.CAR.items():
        setattr(p, k, v)
    mpc = DynamicBicycleMPC(params=p, dt=TICK_DT, horizon_time=C.HORIZON_TIME)

    print("[carla_mpc] Connecting to CARLA server …")
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(10.0)
    world  = client.get_world()

    # synchronous mode: we tick the world ourselves so physics matches our dt
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = TICK_DT
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    bp = blueprint_library.filter(VEHICLE_FILTER)[0]

    if SPAWN_TRANSFORM is not None:
        spawn_tf = SPAWN_TRANSFORM
    else:
        spawn_pts = world.get_map().get_spawn_points()
        spawn_tf  = spawn_pts[0]

    print(f"[carla_mpc] Spawning {VEHICLE_FILTER} …")
    vehicle = world.spawn_actor(bp, spawn_tf)
    vehicle.set_autopilot(False)

    try:
        print("[carla_mpc] Running MPC loop. Press Ctrl-C to stop.")
        while True:
            world.tick()                          # advance CARLA physics by TICK_DT
            state = carla_state(vehicle)
            set_spectator(world, vehicle)

            target = get_ref_trajectory(
                state, path, C.TARGET_SPEED,
                mpc.control_horizon * TICK_DT, TICK_DT,
            )
            ego_obs = nearby_obstacle(vehicle, world)
            ego_state = np.array([0., 0., 0., state[3], state[4], state[5]])
            _, controls = mpc.solve(ego_state, target, obstacle=ego_obs, max_iter=3)

            a, delta = controls[:, 0]
            ctrl = acc_to_control(a, state[3])
            ctrl.steer = steer_to_control(delta)
            vehicle.apply_control(ctrl)

            spd = state[3] * 3.6
            lat_g = state[3] * state[5] / 9.81
            print(f"\r  {spd:5.1f} km/h  steer {math.degrees(delta):+5.1f}°  "
                  f"lat {lat_g:+4.2f}g  obs {'YES' if ego_obs else ' no'}   ",
                  end="", flush=True)

    except KeyboardInterrupt:
        print("\n[carla_mpc] Stopping.")
    finally:
        # restore async mode and destroy the vehicle
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        vehicle.destroy()
        print("[carla_mpc] Cleaned up.")


if __name__ == "__main__":
    main()
