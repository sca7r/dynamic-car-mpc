"""
CARLA MPC bridge - drives a CARLA vehicle with the dynamic-bicycle MPC.
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

from dcmpc import config as C
from dcmpc.controller import (
    DynamicBicycleMPC, CarParams,
    compute_path_from_wp, get_ref_trajectory, _nn_idx,
    AdaptiveSpeed, build_mpc, build_adaptive_speed,
)

# ── settings ──────────────────────────────────────────────────────────
# Connection
CARLA_HOST       = getattr(C, "CARLA_HOST", "localhost")
CARLA_PORT       = getattr(C, "CARLA_PORT", 2000)
TICK_DT          = C.DT
ROUTE_LENGTH_M   = getattr(C, "CARLA_ROUTE_LENGTH_M", 800.0)   # path length down the lane

# Vehicle + speed - read from config so one file controls everything
VEHICLE_FILTER     = getattr(C, "CARLA_VEHICLE_FILTER", "vehicle.audi.tt")
OBSTACLE_FILTER    = getattr(C, "CARLA_OBSTACLE_FILTER", "vehicle.jeep.wrangler_rubicon")
CARLA_TARGET_SPEED = getattr(C, "CARLA_TARGET_SPEED",     7.0)
MAX_THROTTLE_ACC   = getattr(C, "CARLA_MAX_THROTTLE_ACC", 4.0)
MAX_BRAKE_DEC      = getattr(C, "CARLA_MAX_BRAKE_DEC",    6.0)

# Obstacle detection - CARLA vehicles get this radius in the MPC keep-out.
OBSTACLE_RADIUS  = getattr(C, "CARLA_OBSTACLE_RADIUS", 1.5)
OBSTACLE_DETECT_R= C.SENSOR_RANGE

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
        ctrl.throttle = getattr(C, "CARLA_CREEP_THROTTLE", 0.4)
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


# ── virtual LiDAR: real CARLA ray-cast sensor -> clustered obstacle list ──
def attach_lidar(world, vehicle):
    """Attach a ray-cast LiDAR; returns (sensor, buffer) where buffer['pts']
    holds the latest scan as Nx3 points in the SENSOR frame."""
    bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
    bp.set_attribute("range", str(float(OBSTACLE_DETECT_R)))
    bp.set_attribute("rotation_frequency", str(int(1.0 / TICK_DT)))
    bp.set_attribute("channels", str(int(getattr(C, "CARLA_LIDAR_CHANNELS", 32))))
    bp.set_attribute("points_per_second",
                     str(int(getattr(C, "CARLA_LIDAR_POINTS_PER_SEC", 120000))))
    bp.set_attribute("upper_fov", str(float(getattr(C, "CARLA_LIDAR_UPPER_FOV", 2.0))))
    bp.set_attribute("lower_fov", str(float(getattr(C, "CARLA_LIDAR_LOWER_FOV", -8.0))))
    mount_z = float(getattr(C, "CARLA_LIDAR_MOUNT_Z", 2.4))
    tf = carla.Transform(carla.Location(x=0.0, z=mount_z))
    sensor = world.spawn_actor(bp, tf, attach_to=vehicle)
    buf = {"pts": np.empty((0, 3), dtype=np.float32)}

    def _on_scan(scan):
        raw = np.frombuffer(scan.raw_data, dtype=np.float32).reshape(-1, 4)
        buf["pts"] = raw[:, :3].copy()
    sensor.listen(_on_scan)
    return sensor, buf


class LidarViewer:
    """Live top-down scatter of the LiDAR point cloud, colour-coded by the
    perception filter stage, with detected obstacle clusters circled.

    Pure matplotlib (no extra deps). Opt-in via `--view-lidar`. Updates in the
    main loop; close the window or pass no flag to disable.

    Colours:
      grey   = raw returns dropped by the height (z) filter  (ground / overhead)
      orange = survived z-filter but dropped by FOV / range / ego-exclusion
      green  = kept points fed to the clustering
      red O  = a detected obstacle (centre + keep-out radius)
    """

    def __init__(self, max_range=45.0, fov_deg=90.0):
        import matplotlib.pyplot as plt
        self.plt = plt
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.max_range = max_range
        self.fov_deg = fov_deg
        self._setup_axes()
        self.fig.canvas.manager.set_window_title("LiDAR - perception debug")

    def _setup_axes(self):
        r = self.max_range
        self.ax.clear()
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-r * 0.4, r)          # mostly forward
        self.ax.set_ylim(-r * 0.7, r * 0.7)
        self.ax.set_xlabel("x forward (m)")
        self.ax.set_ylabel("y left (m)")
        self.ax.grid(alpha=0.25)
        # car marker at origin + heading arrow
        self.ax.plot(0, 0, "ks", ms=8)
        self.ax.arrow(0, 0, 3, 0, head_width=0.8, color="k", alpha=0.6)
        # FOV wedge
        half = np.radians(self.fov_deg / 2)
        for s in (+1, -1):
            self.ax.plot([0, self.max_range * np.cos(s * half)],
                         [0, self.max_range * np.sin(s * half)],
                         "b--", lw=0.8, alpha=0.5)

    def update(self, raw_pts, obstacles, z_min, z_max,
               ego_exclusion, max_range, fov_deg):
        """Re-draw the cloud. raw_pts = Nx3 sensor-frame; obstacles = ego-frame
        (x,y,r,...) list already produced by lidar_obstacles()."""
        self._setup_axes()
        if raw_pts is not None and raw_pts.shape[0] > 0:
            # replicate the filter stages purely for colouring
            zmask = (raw_pts[:, 2] > z_min) & (raw_pts[:, 2] < z_max)
            dropped_z = raw_pts[~zmask]
            kept_z = raw_pts[zmask].copy()
            kept_z[:, 1] = -kept_z[:, 1]                  # y flip, our frame
            dist = np.hypot(kept_z[:, 0], kept_z[:, 1])
            bearing = np.arctan2(kept_z[:, 1], kept_z[:, 0])
            fwd = ((kept_z[:, 0] > 0) & (dist <= max_range) &
                   (dist >= ego_exclusion) &
                   (np.abs(bearing) <= np.radians(fov_deg / 2)))
            kept = kept_z[fwd]
            dropped_f = kept_z[~fwd]
            # ground/overhead (grey) - flip y for display consistency
            if dropped_z.shape[0]:
                self.ax.scatter(dropped_z[:, 0], -dropped_z[:, 1], s=1,
                                c="0.7", alpha=0.3)
            if dropped_f.shape[0]:
                self.ax.scatter(dropped_f[:, 0], dropped_f[:, 1], s=2,
                                c="orange", alpha=0.4)
            if kept.shape[0]:
                self.ax.scatter(kept[:, 0], kept[:, 1], s=4,
                                c="green", alpha=0.7)
        for ob in obstacles:
            ox, oy, r = ob[0], ob[1], ob[2]
            self.ax.add_patch(self.plt.Circle((ox, oy), r, color="red",
                                              fill=False, lw=2))
            self.ax.plot(ox, oy, "rx", ms=8)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        self.plt.close(self.fig)


def lidar_obstacles(
    points,
    max_obs=4,
    z_min=-1.2,
    z_max=-0.1,
    cell=1.0,
    min_pts=8,
    fov_deg=90.0,
    ego_exclusion=4.5,
    max_range=45.0,
    max_radius=3.5,
    debug=False,
):
    """
    Sensor-frame LiDAR points -> up to max_obs ego-frame obstacles
    (x, y, r, vx, vy).

    Pipeline:
      1. Remove ground / overhead returns with a tighter sensor-frame z band.
      2. Flip CARLA LiDAR y into our convention.
      3. Keep only forward points inside a configurable FOV arc.
      4. Remove returns too close to the ego vehicle.
      5. Grid-cluster occupied cells with flood fill.
      6. Convert clusters into circular keep-outs.

    Returns obstacles sorted nearest-first.
    """
    if points is None or points.shape[0] == 0:
        if debug:
            print("[lidar_obstacles] no points")
        return []

    total0 = points.shape[0]

    # 1) Vertical filtering in the LiDAR sensor frame
    zmask = (points[:, 2] > z_min) & (points[:, 2] < z_max)
    hits = points[zmask][:, :2]
    if hits.shape[0] == 0:
        if debug:
            print(f"[lidar_obstacles] total={total0} after_z=0")
        return []

    # 2) CARLA LiDAR y-axis flip
    hits = hits.copy()
    hits[:, 1] = -hits[:, 1]

    # 3) Range and forward-arc filtering
    dist = np.hypot(hits[:, 0], hits[:, 1])
    bearing = np.arctan2(hits[:, 1], hits[:, 0])  # radians, x forward
    half_fov = np.radians(fov_deg * 0.5)

    fmask = (
        (hits[:, 0] > 0.0) &
        (dist <= max_range) &
        (dist >= ego_exclusion) &
        (np.abs(bearing) <= half_fov)
    )
    hits = hits[fmask]

    if debug:
        after_z = int(np.count_nonzero(zmask))
        after_f = int(hits.shape[0])
        print(
            f"[lidar_obstacles] total={total0} after_z={after_z} "
            f"after_fov_range_ego={after_f} "
            f"z=[{z_min:.2f},{z_max:.2f}] "
            f"fov={fov_deg:.1f}deg ego_excl={ego_exclusion:.1f}m "
            f"max_range={max_range:.1f}m"
        )

    if hits.shape[0] == 0:
        return []

    # 4) Grid clustering: bucket points into square cells
    keys = np.floor(hits / cell).astype(np.int64)
    buckets = {}
    for (cx, cy), (hx, hy) in zip(map(tuple, keys), hits):
        buckets.setdefault((cx, cy), []).append((hx, hy))

    # 5) Flood fill connected occupied cells
    seen, clusters = set(), []
    cluster_sizes = []

    for key in buckets:
        if key in seen:
            continue

        stack, comp = [key], []
        while stack:
            c = stack.pop()
            if c in seen or c not in buckets:
                continue
            seen.add(c)
            comp.extend(buckets[c])

            cx, cy = c
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (cx + dx, cy + dy)
                    if nb in buckets and nb not in seen:
                        stack.append(nb)

        if len(comp) >= min_pts:
            arr = np.asarray(comp, dtype=np.float32)
            clusters.append(arr)
            cluster_sizes.append(arr.shape[0])

    if debug:
        print(
            f"[lidar_obstacles] buckets={len(buckets)} "
            f"clusters={len(clusters)} "
            f"sizes={cluster_sizes[:10]}"
        )

    # 6) Cluster -> circular obstacle
    obs = []
    for comp in clusters:
        cx = float(np.mean(comp[:, 0]))
        cy = float(np.mean(comp[:, 1]))

        spread = float(np.max(np.hypot(comp[:, 0] - cx, comp[:, 1] - cy)))
        r = max(spread, 0.3)

        # --- FIX: PERCEPTION SANITY CHECKS ---
        # 1. Reject massive objects (buildings, long hedges, walls).
        # Passenger cars are ~1.5m-2.0m in radius. 
        if r > max_radius:
            continue
            
        # 2. Reject objects far off the lateral center of the road.
        # This prevents sidewalks and parked scenery from triggering avoidance.
        if abs(cy) > 8.0:
            continue
        # -------------------------------------

        obs.append((cx, cy, r, 0.0, 0.0))

    obs.sort(key=lambda o: math.hypot(o[0], o[1]))
    obs = obs[:max_obs]

    if debug:
        print(f"[lidar_obstacles] returned={len(obs)}")
        for i, o in enumerate(obs):
            print(
                f"  obs[{i}] x={o[0]:7.2f} y={o[1]:7.2f} "
                f"r={o[2]:5.2f} vx={o[3]:5.2f} vy={o[4]:5.2f} "
                f"dist={math.hypot(o[0], o[1]):6.2f}"
            )

    return obs


def cross_track_and_heading(state, path):
    i = _nn_idx(np.array([state[0], state[1], state[2]]), path)
    px, py, pth = path[0, i], path[1, i], path[2, i]
    dx, dy = state[0] - px, state[1] - py
    cross = -math.sin(pth) * dx + math.cos(pth) * dy
    herr = (state[2] - pth + math.pi) % (2 * math.pi) - math.pi
    return cross, herr


def set_spectator(world, vehicle):
    """Position the free spectator camera each tick per the config preset
    (CARLA_CAMERA_MODE: 'topdown' | 'chase' | 'front')."""
    t = vehicle.get_transform()
    yaw = t.rotation.yaw
    yaw_rad = math.radians(yaw)
    fx, fy = math.cos(yaw_rad), math.sin(yaw_rad)   # car forward unit vector
    mode = getattr(C, "CARLA_CAMERA_MODE", "topdown")

    if mode == "chase":
        back = getattr(C, "CARLA_CAM_CHASE_BACK", 8.0)
        h    = getattr(C, "CARLA_CAM_CHASE_HEIGHT", 4.0)
        pit  = getattr(C, "CARLA_CAM_CHASE_PITCH", -12.0)
        loc = carla.Location(x=t.location.x - fx * back,
                             y=t.location.y - fy * back,
                             z=t.location.z + h)
        rot = carla.Rotation(pitch=pit, yaw=yaw, roll=0)
    elif mode == "front":
        ahead = getattr(C, "CARLA_CAM_FRONT_AHEAD", 8.0)
        h     = getattr(C, "CARLA_CAM_FRONT_HEIGHT", 3.0)
        pit   = getattr(C, "CARLA_CAM_FRONT_PITCH", -8.0)
        loc = carla.Location(x=t.location.x + fx * ahead,
                             y=t.location.y + fy * ahead,
                             z=t.location.z + h)
        rot = carla.Rotation(pitch=pit, yaw=yaw + 180.0, roll=0)  # look back
    else:  # "topdown"
        h = getattr(C, "CARLA_CAM_TOPDOWN_HEIGHT",
                    getattr(C, "CARLA_SPECTATOR_HEIGHT", 40.0))
        loc = carla.Location(x=t.location.x, y=t.location.y, z=h)
        rot = carla.Rotation(pitch=-90, yaw=yaw, roll=0)

    world.get_spectator().set_transform(carla.Transform(loc, rot))


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


def spawn_obstacle(world, path, base_z, along=60.0, offset=1.8):
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
        log.warning("Obstacle spawn blocked (along=%.0f offset=%.1f) - try different values.",
                    along, offset)
    return actor


# ── main ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-csv", action="store_true")
    ap.add_argument("--no-obstacle", action="store_true")
    ap.add_argument("--no-lidar", action="store_true",
                    help="use ground-truth actor positions instead of virtual LiDAR")
    ap.add_argument("--view-lidar", action="store_true",
                    help="open a live top-down plot of the LiDAR cloud (debug)")
    ap.add_argument("--view-lidar-3d", action="store_true",
                    help="open a live 3D semantic point-cloud window "
                         "(each cluster its own colour; uses Open3D if installed)")
    args = ap.parse_args()
    level = logging.DEBUG if args.debug else logging.WARNING if args.quiet else logging.INFO
    setup_logging(level)

    if not CARLA_AVAILABLE:
        log.error("'carla' package not found.  pip install carla")
        sys.exit(1)

    trace = TraceWriter(None if args.no_csv
                        else f"carla_trace_{time.strftime('%Y%m%d_%H%M%S')}.csv")

    log.info("Car params: %s", C.CAR)
    log.info("Target speed %.1f m/s (%.0f km/h), dt %.3f s, horizon %.1f s.",
             CARLA_TARGET_SPEED, CARLA_TARGET_SPEED * 3.6, TICK_DT, C.HORIZON_TIME)
    mpc = build_mpc(C, dt=TICK_DT)
    speed_ctrl = build_adaptive_speed(C, base_speed=CARLA_TARGET_SPEED, carla=True)

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

    lidar = None
    lidar_buf = None
    lidar_view = None
    lidar_view3d = None
    if not args.no_lidar:
        lidar, lidar_buf = attach_lidar(world, vehicle)
        world.tick()
        log.info("Virtual LiDAR attached (range %.0f m, multi-obstacle MPC).",
                 OBSTACLE_DETECT_R)
        if args.view_lidar:
            lidar_view = LidarViewer(max_range=OBSTACLE_DETECT_R)
            log.info("LiDAR debug view open (grey=ground, orange=out-of-FOV, "
                     "green=clustered, red=obstacle).")
        if args.view_lidar_3d:
            from dcmpc.lidar_view3d import LidarView3D
            lidar_view3d = LidarView3D(
                max_range=getattr(C, "LIDAR_MAX_RANGE", 45.0),
                fov_deg=getattr(C, "LIDAR_FOV_DEG", 90.0),
                cfg=dict(cell=getattr(C, "LIDAR_CLUSTER_CELL", 1.0),
                         min_pts=getattr(C, "LIDAR_MIN_PTS", 8),
                         max_radius=getattr(C, "LIDAR_MAX_RADIUS", 3.5)))
            log.info("3D semantic LiDAR view open (backend: %s).",
                     lidar_view3d.backend_name)
            if lidar_view3d.backend_name.startswith("none"):
                log.warning("No 3D backend available. For the best look install "
                            "Open3D:  pip install open3d  (or a matplotlib GUI).")

    obstacle_actors = []
    if not args.no_obstacle:
        z = spawn_tf.location.z
        obs_specs = getattr(C, "CARLA_OBSTACLES",
                            [{"along": 60.0, "offset": 1.8}])
        for spec in obs_specs:
            actor = spawn_obstacle(world, path, z,
                                   along=spec.get("along", 60.0),
                                   offset=spec.get("offset", 0.0))
            if actor is not None:
                obstacle_actors.append(actor)
        world.tick()
        log.info("Spawned %d obstacle(s) from config.", len(obstacle_actors))

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

            if lidar_buf is not None:
                obs_list = lidar_obstacles(
                    lidar_buf["pts"], max_obs=mpc.MAX_OBS,
                    z_min=getattr(C, "LIDAR_Z_MIN", -1.2),
                    z_max=getattr(C, "LIDAR_Z_MAX", -0.1),
                    cell=getattr(C, "LIDAR_CLUSTER_CELL", 1.0),
                    min_pts=getattr(C, "LIDAR_MIN_PTS", 8),
                    fov_deg=getattr(C, "LIDAR_FOV_DEG", 90.0),
                    ego_exclusion=getattr(C, "LIDAR_EGO_EXCLUSION", 4.5),
                    max_range=getattr(C, "LIDAR_MAX_RANGE", 45.0),
                    max_radius=getattr(C, "LIDAR_MAX_RADIUS", 3.5))
                ego_obs = obs_list[0] if obs_list else None
                if lidar_view is not None:
                    lidar_view.update(
                        lidar_buf["pts"], obs_list,
                        z_min=getattr(C, "LIDAR_Z_MIN", -1.2),
                        z_max=getattr(C, "LIDAR_Z_MAX", -0.1),
                        ego_exclusion=getattr(C, "LIDAR_EGO_EXCLUSION", 4.5),
                        max_range=getattr(C, "LIDAR_MAX_RANGE", 45.0),
                        fov_deg=getattr(C, "LIDAR_FOV_DEG", 90.0))
                if lidar_view3d is not None:
                    lidar_view3d.update(
                        lidar_buf["pts"], obs_list,
                        z_min=getattr(C, "LIDAR_Z_MIN", -1.2),
                        z_max=getattr(C, "LIDAR_Z_MAX", -0.1),
                        ego_exclusion=getattr(C, "LIDAR_EGO_EXCLUSION", 4.5),
                        max_range=getattr(C, "LIDAR_MAX_RANGE", 45.0),
                        fov_deg=getattr(C, "LIDAR_FOV_DEG", 90.0))
            else:
                ego_obs = nearby_obstacle(vehicle, world)
                obs_list = [ego_obs] if ego_obs is not None else []
            if ego_obs is not None:
                gap = math.hypot(ego_obs[0], ego_obs[1]) - ego_obs[2]
                min_obs_gap = min(min_obs_gap, gap)
            adaptive_v = speed_ctrl.update(state, path, ego_obs, TICK_DT)
            # 1. Generate a rough path prediction using CURRENT speed to filter scenery
            rough_target = get_ref_trajectory(state, path, state[3], C.HORIZON_TIME, TICK_DT, ego_frame=True)

            # 2. Path-based scenery filter & bumper guard (all tunable in config)
            _cap_r   = getattr(C, "CARLA_OBSTACLE_RADIUS_CAP", 1.5)
            _bg_back = getattr(C, "CARLA_BUMPER_GUARD_BACK", -8.0)
            _bg_front= getattr(C, "CARLA_BUMPER_GUARD_FRONT", 2.0)
            _bg_lat  = getattr(C, "CARLA_BUMPER_GUARD_LATERAL", 6.0)
            _path_margin = getattr(C, "CARLA_PATH_FILTER_MARGIN", 2.5)
            filtered_obs = []
            if obs_list:
                for ob in obs_list:
                    ox, oy, r = ob[0], ob[1], ob[2]
                    r = min(r, _cap_r)            # cap massive scenery objects
                    ob = (ox, oy, r, ob[3], ob[4])

                    # Bumper guard: keep an obstacle in memory while it is
                    # alongside/just behind, so the car doesn't forget it
                    # mid-swerve and clip it with the rear bumper.
                    if _bg_back < ox < _bg_front and abs(oy) < _bg_lat:
                        filtered_obs.append(ob)
                        continue

                    # Standard path check for obstacles ahead on the route
                    dists = np.hypot(rough_target[0, :] - ox, rough_target[1, :] - oy)
                    if np.min(dists) < (r + _path_margin):
                        filtered_obs.append(ob)

            obs_list = filtered_obs

            # 3. NOW calculate adaptive speed using the CLEAN obstacle list
            clean_ego_obs = obs_list[0] if obs_list else None
            adaptive_v = speed_ctrl.update(state, path, clean_ego_obs, TICK_DT)

            # 4. Generate the actual target trajectory for the MPC solver
            # Notice we completely removed the manual Y-bending!
            target = get_ref_trajectory(state, path, adaptive_v, C.HORIZON_TIME, TICK_DT, ego_frame=True)
            
            ego_state = np.array([0., 0., 0., state[3], state[4], state[5]])
            
            
           

            # ── DEBUG: WHAT THE CAR SEES ──────────────────────────────────────
            # Print every 5 ticks (~0.5s) to keep the terminal readable, 
            # or force print if the solver just failed on the previous tick.
            if tick % 5 == 0 or brakes > 0: 
                obs_msg = "None in range"
                if obs_list and obs_list[0] is not None:
                    o = obs_list[0]
                    odist = math.hypot(o[0], o[1])
                    obs_msg = f"x={o[0]:.1f}, y={o[1]:.1f} (dist={odist:.1f}m, r={o[2]:.1f}m)"
                
                log.info(
                    "\n"
                    "  [VISION @ tick %d]\n"
                    "  | Ego Vx    : %.2f m/s\n"
                    "  | Target V  : %.2f m/s (Reason: %s)\n"
                    "  | Path End  : x=%.1f, y=%.1f [Ego Frame]\n"
                    "  | Obstacle  : %s",
                    tick, state[3], adaptive_v, getattr(speed_ctrl, 'reason', 'N/A'),
                    target[0, -1], target[1, -1], obs_msg
                )
            # ──────────────────────────────────────────────────────────────────

           
            
            t_solve = time.time()
            # Pass the global 'state' into global_pose
            traj, controls = mpc.solve(ego_state, target, obstacles=obs_list, max_iter=getattr(C, "MPC_MAX_ITER", 3), global_pose=state)
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
        if lidar is not None:
            lidar.destroy()
        if lidar_view is not None:
            lidar_view.close()
        if lidar_view3d is not None:
            lidar_view3d.close()
        if vehicle is not None:
            vehicle.destroy()
        log.info("Cleaned up. Bye.")


if __name__ == "__main__":
    main()