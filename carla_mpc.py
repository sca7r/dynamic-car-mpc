"""
CARLA MPC bridge — drives a CARLA vehicle with the dynamic-bicycle MPC.
Includes structured logging, a per-tick CSV trace, and an optional stalled
obstacle car spawned on the route (offset from the lane centre) so you can see
the same avoidance behaviour as in the pygame sim.

Verbosity:
  python carla_mpc.py            # INFO
  python carla_mpc.py --debug    # per-tick details
  python carla_mpc.py --quiet    # warnings/errors only
  python carla_mpc.py --no-csv   # skip the CSV trace
  python carla_mpc.py --no-obstacle   # don't spawn the obstacle

REQUIREMENTS:  pip install carla ; start ./CarlaUE4.sh ; then run this.
"""

from __future__ import annotations

import sys
import csv
import math
import time
import logging
import argparse
import numpy as np

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False

import config as C
from dynamic_bicycle_mpc import (
    DynamicBicycleMPC, CarParams,
    compute_path_from_wp, get_ref_trajectory, _nn_idx,
    AdaptiveSpeed,
)

# ── settings ──────────────────────────────────────────────────────────
# Connection
CARLA_HOST       = "localhost"
CARLA_PORT       = 2000
TICK_DT          = C.DT
ROUTE_LENGTH_M   = 800.0       # how far down the lane to build the path
SPECTATOR_HEIGHT = 45.0        # metres above the car for the top-down view

# Vehicle + speed — read from config so one file controls everything
VEHICLE_FILTER     = "vehicle.audi.tt"
OBSTACLE_FILTER    = "vehicle.jeep.wrangler_rubicon"
CARLA_TARGET_SPEED = getattr(C, "CARLA_TARGET_SPEED",     7.0)
MAX_THROTTLE_ACC   = getattr(C, "CARLA_MAX_THROTTLE_ACC", 4.0)
MAX_BRAKE_DEC      = getattr(C, "CARLA_MAX_BRAKE_DEC",    6.0)

# Obstacle detection — CARLA vehicles get this radius in the MPC keep-out.
# Real cars are ~1 m half-width; 1.5 m adds a buffer without over-inflating
# the keep-out zone on real narrow town lanes.
OBSTACLE_RADIUS  = getattr(C, "CARLA_OBSTACLE_RADIUS", 1.5)
OBSTACLE_DETECT_R= C.SENSOR_RANGE

# Where to place the three stalled obstacle cars along the route
OBS_ALONG  = 60.0    # metres along the path for the first obstacle
OBS_OFFSET = 1.8     # metres left (+) / right (-) of lane centre

log = logging.getLogger("carla_mpc")


# ── logging / trace setup ─────────────────────────────────────────────

def setup_logging(level):
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(h)
    log.setLevel(level)
    log.propagate = False


class TraceWriter:
    FIELDS = ["t", "x", "y", "psi_deg", "vx", "vy", "yaw_rate_deg",
              "cross_track_m", "heading_err_deg", "mpc_a", "mpc_delta_deg",
              "throttle", "brake", "steer", "solve_ms", "braked", "obstacle"]

    def __init__(self, path):
        self.enabled = path is not None
        if not self.enabled:
            return
        self.f = open(path, "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=self.FIELDS)
        self.w.writeheader()
        log.info("Trace -> %s", path)

    def row(self, **kw):
        if self.enabled:
            self.w.writerow({k: kw.get(k, "") for k in self.FIELDS})

    def close(self):
        if self.enabled:
            self.f.close()


# ── state / control conversions ───────────────────────────────────────

def carla_state(vehicle) -> np.ndarray:
    """CARLA vehicle -> state [x, y, psi, vx, vy, r] (our convention: y flipped)."""
    t  = vehicle.get_transform()
    v  = vehicle.get_velocity()
    av = vehicle.get_angular_velocity()
    x   =  t.location.x
    y   = -t.location.y
    psi = -math.radians(t.rotation.yaw)
    vx_w, vy_w = v.x, -v.y
    c, s = math.cos(psi), math.sin(psi)
    vx =  c * vx_w + s * vy_w
    vy = -s * vx_w + c * vy_w
    r = -math.radians(av.z)
    return np.array([x, y, psi, vx, vy, r])


def acc_to_control(a: float, vx: float):
    ctrl = carla.VehicleControl()
    a = float(a); vx = float(vx)
    if vx < 0.5 and a >= 0.0:
        ctrl.throttle = 0.4
        ctrl.brake = 0.0
        return ctrl
    if a >= 0.0:
        ctrl.throttle = float(min(a / MAX_THROTTLE_ACC, 1.0))
        ctrl.brake = 0.0
    else:
        ctrl.throttle = 0.0
        ctrl.brake = float(min(-a / MAX_BRAKE_DEC, 1.0))
    return ctrl


def steer_to_control(delta: float, max_steer: float) -> float:
    return float(max(-1.0, min(1.0, -delta / max_steer)))


def nearby_obstacle(vehicle, world):
    """Nearest other vehicle in range -> ego-frame obstacle (x, y, r, vx, vy)."""
    state = carla_state(vehicle)
    best, best_d = None, 1e18
    for a in world.get_actors().filter("vehicle.*"):
        if a.id == vehicle.id:
            continue
        t = a.get_transform()
        ox, oy = t.location.x, -t.location.y
        dx, dy = ox - state[0], oy - state[1]
        d = math.hypot(dx, dy)
        if d > OBSTACLE_DETECT_R or d >= best_d:
            continue
        c, s = math.cos(-state[2]), math.sin(-state[2])
        ex, ey = dx * c - dy * s, dx * s + dy * c
        v = a.get_velocity()
        evx =  c * v.x + s * (-v.y)
        evy = -s * v.x + c * (-v.y)
        best, best_d = (ex, ey, OBSTACLE_RADIUS, evx, evy), d
    return best


def cross_track_and_heading(state, path):
    i = _nn_idx(np.array([state[0], state[1], state[2]]), path)
    px, py, pth = path[0, i], path[1, i], path[2, i]
    dx, dy = state[0] - px, state[1] - py
    cross = -math.sin(pth) * dx + math.cos(pth) * dy
    herr = (state[2] - pth + math.pi) % (2 * math.pi) - math.pi
    return cross, herr


def set_spectator(world, vehicle):
    t = vehicle.get_transform()
    world.get_spectator().set_transform(carla.Transform(
        carla.Location(x=t.location.x, y=t.location.y, z=SPECTATOR_HEIGHT),
        carla.Rotation(pitch=-90, yaw=t.rotation.yaw, roll=0),
    ))


def build_carla_path(world, start_location, length_m=ROUTE_LENGTH_M, step_m=2.0):
    carla_map = world.get_map()
    wp = carla_map.get_waypoint(start_location, project_to_road=True)
    xs, ys, travelled = [], [], 0.0
    while travelled < length_m:
        loc = wp.transform.location
        xs.append(loc.x)
        ys.append(-loc.y)
        nxt = wp.next(step_m)
        if not nxt:
            log.warning("Lane ended after %.0f m.", travelled)
            break
        wp = nxt[0]
        travelled += step_m
    log.info("Route built: %d lane points, ~%.0f m.", len(xs), travelled)
    return compute_path_from_wp(xs, ys, step=0.25)


def spawn_obstacle(world, path, base_z, along=OBS_ALONG, offset=OBS_OFFSET):
    """Spawn a stalled car on the route, offset from the centreline."""
    cdist = np.append([0.], np.cumsum(np.hypot(np.diff(path[0]), np.diff(path[1]))))
    oi = min(int(np.searchsorted(cdist, along)), path.shape[1] - 1)
    oth = path[2, oi]
    ox = path[0, oi] - math.sin(oth) * offset
    oy = path[1, oi] + math.cos(oth) * offset
    bp = world.get_blueprint_library().filter(OBSTACLE_FILTER)[0]
    tf = carla.Transform(
        carla.Location(x=float(ox), y=float(-oy), z=base_z),
        carla.Rotation(yaw=math.degrees(-oth)),
    )
    actor = world.try_spawn_actor(bp, tf)
    if actor:
        actor.set_simulate_physics(False)
        log.info("Obstacle spawned at along=%.0f m, offset=%+.1f m.", along, offset)
    else:
        log.warning("Obstacle spawn blocked (along=%.0f offset=%.1f) — try different values.",
                    along, offset)
    return actor


# ── main ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-csv", action="store_true")
    ap.add_argument("--no-obstacle", action="store_true")
    args = ap.parse_args()
    level = logging.DEBUG if args.debug else logging.WARNING if args.quiet else logging.INFO
    setup_logging(level)

    if not CARLA_AVAILABLE:
        log.error("'carla' package not found.  pip install carla")
        sys.exit(1)

    trace = TraceWriter(None if args.no_csv
                        else f"carla_trace_{time.strftime('%Y%m%d_%H%M%S')}.csv")

    p = CarParams()
    for k, v in C.CAR.items():
        setattr(p, k, v)
    log.info("Car params: %s", C.CAR)
    log.info("Target speed %.1f m/s (%.0f km/h), dt %.3f s, horizon %.1f s.",
             CARLA_TARGET_SPEED, CARLA_TARGET_SPEED * 3.6, TICK_DT, C.HORIZON_TIME)
    mpc = DynamicBicycleMPC(params=p, dt=TICK_DT, horizon_time=C.HORIZON_TIME,
                            road_halfwidth=getattr(C, "ROAD_HALFWIDTH", 5.0),
                            pass_zone=getattr(C, "PASS_ZONE", 6.0),
                            safety_margin=getattr(C, "OBSTACLE_SAFETY_MARGIN", 0.8))
    speed_ctrl = AdaptiveSpeed(base_speed=CARLA_TARGET_SPEED,
                               obs_brake_start=25.0, obs_brake_end=5.0, v_min=3.0)

    log.info("Connecting to CARLA at %s:%d ...", CARLA_HOST, CARLA_PORT)
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(10.0)
    world = client.get_world()
    log.info("Connected. Map: %s", world.get_map().name)

    settings = world.get_settings()
    old_sync, old_dt = settings.synchronous_mode, settings.fixed_delta_seconds
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = TICK_DT
    world.apply_settings(settings)

    bp = world.get_blueprint_library().filter(VEHICLE_FILTER)[0]
    seed = world.get_map().get_spawn_points()[0]
    path = build_carla_path(world, seed.location)
    log.info("Route start (%.1f, %.1f) -> end (%.1f, %.1f).",
             path[0, 0], path[1, 0], path[0, -1], path[1, -1])

    path_heading0 = math.atan2(path[1, 1] - path[1, 0], path[0, 1] - path[0, 0])
    spawn_tf = carla.Transform(
        carla.Location(x=float(path[0, 0]), y=float(-path[1, 0]),
                       z=seed.location.z + 0.3),
        carla.Rotation(yaw=math.degrees(-path_heading0)),
    )

    vehicle = None
    for attempt in range(5):
        vehicle = world.try_spawn_actor(bp, spawn_tf)
        if vehicle:
            break
        log.warning("Spawn attempt %d blocked; nudging up.", attempt + 1)
        spawn_tf.location.z += 0.5
    if vehicle is None:
        log.error("Could not spawn vehicle.")
        settings.synchronous_mode, settings.fixed_delta_seconds = old_sync, old_dt
        world.apply_settings(settings)
        sys.exit(1)
    vehicle.set_autopilot(False)
    world.tick()
    log.info("Spawned %s (id %d).", VEHICLE_FILTER, vehicle.id)

    obstacle_actors = []
    if not args.no_obstacle:
        z = spawn_tf.location.z
        obstacle_actors = [
            spawn_obstacle(world, path, z, along=OBS_ALONG,        offset= OBS_OFFSET),
            spawn_obstacle(world, path, z, along=OBS_ALONG + 180,  offset=-2.0),
            spawn_obstacle(world, path, z, along=OBS_ALONG + 380,  offset= 2.5),
        ]
        world.tick()

    st0 = carla_state(vehicle)
    herr = (path_heading0 - st0[2] + math.pi) % (2 * math.pi) - math.pi
    if abs(herr) < math.radians(20):
        log.info("Heading error at start: %+.1f deg  (OK)", math.degrees(herr))
    else:
        log.warning("Heading error at start: %+.1f deg  (CHECK!)", math.degrees(herr))

    phys = vehicle.get_physics_control()
    max_steer = math.radians(max(w.max_steer_angle for w in phys.wheels))
    log.info("Vehicle max steer = %.1f deg.", math.degrees(max_steer))

    t0 = time.time()
    tick = brakes = 0
    worst_cross = solve_ms_sum = 0.0
    min_obs_gap = 1e18

    try:
        log.info("Running MPC loop. Ctrl-C to stop.")
        while True:
            world.tick()
            tick += 1
            state = carla_state(vehicle)
            set_spectator(world, vehicle)

            if np.hypot(state[0] - path[0, -1], state[1] - path[1, -1]) < 5.0:
                log.info("Reached end of route after %d ticks.", tick)
                break

            cross, herr = cross_track_and_heading(state, path)
            worst_cross = max(worst_cross, abs(cross))

            ego_obs = nearby_obstacle(vehicle, world)
            if ego_obs is not None:
                gap = math.hypot(ego_obs[0], ego_obs[1]) - ego_obs[2]
                min_obs_gap = min(min_obs_gap, gap)
            adaptive_v = speed_ctrl.update(state, path, ego_obs, TICK_DT)
            target = get_ref_trajectory(
                state, path, adaptive_v,
                mpc.control_horizon * TICK_DT, TICK_DT,
            )
            ego_state = np.array([0., 0., 0., state[3], state[4], state[5]])

            t_solve = time.time()
            traj, controls = mpc.solve(ego_state, target, obstacle=ego_obs, max_iter=3)
            solve_ms = (time.time() - t_solve) * 1000.0
            solve_ms_sum += solve_ms
            braked = traj is None
            if braked:
                brakes += 1
                log.warning("tick %d: EMERGENCY BRAKE (cross=%.2f, vx=%.1f).",
                            tick, cross, state[3])

            a, delta = controls[:, 0]
            ctrl = acc_to_control(a, state[3])
            ctrl.steer = steer_to_control(delta, max_steer)
            vehicle.apply_control(ctrl)

            if abs(cross) > 4.0:
                log.warning("tick %d: large cross-track %.2f m.", tick, cross)

            log.debug("t=%.2f cross=%+.2f herr=%+.1f a=%+.2f d=%+.1f "
                      "thr=%.2f brk=%.2f steer=%+.2f obs=%s solve=%.0fms",
                      tick * TICK_DT, cross, math.degrees(herr), a,
                      math.degrees(delta), ctrl.throttle, ctrl.brake,
                      ctrl.steer, "Y" if ego_obs else "-", solve_ms)

            trace.row(
                t=round(tick * TICK_DT, 3), x=round(state[0], 2), y=round(state[1], 2),
                psi_deg=round(math.degrees(state[2]), 1), vx=round(state[3], 2),
                vy=round(state[4], 3), yaw_rate_deg=round(math.degrees(state[5]), 2),
                cross_track_m=round(cross, 3), heading_err_deg=round(math.degrees(herr), 2),
                mpc_a=round(float(a), 3), mpc_delta_deg=round(math.degrees(delta), 2),
                throttle=round(ctrl.throttle, 3), brake=round(ctrl.brake, 3),
                steer=round(ctrl.steer, 3), solve_ms=round(solve_ms, 1),
                braked=int(braked), obstacle=int(ego_obs is not None),
            )

            if level <= logging.INFO and tick % 5 == 0:
                act = mpc._last_obstacle_action
                log.info("t=%5.1fs | %5.1f km/h | cross %+.2f m | steer %+5.1f deg "
                         "| %s | solve %3.0f ms",
                         tick * TICK_DT, state[3] * 3.6, cross,
                         math.degrees(delta),
                         act.upper() if act != "none" else "   ", solve_ms)

    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        dur = time.time() - t0
        gap_str = f"{min_obs_gap:.2f} m" if min_obs_gap < 1e17 else "n/a"
        log.info("Summary: %d ticks, %.1f s, %d brake(s), worst cross-track %.2f m, "
                 "min obstacle gap %s, avg solve %.0f ms.",
                 tick, dur, brakes, worst_cross, gap_str, solve_ms_sum / max(tick, 1))
        trace.close()
        settings.synchronous_mode, settings.fixed_delta_seconds = old_sync, old_dt
        world.apply_settings(settings)
        for obs in obstacle_actors:
            if obs is not None:
                obs.destroy()
        if vehicle is not None:
            vehicle.destroy()
        log.info("Cleaned up. Bye.")


if __name__ == "__main__":
    main()