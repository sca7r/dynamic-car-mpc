"""
================================================================================
 dcmpc - CENTRAL CONFIGURATION  (the single remote control for the whole stack)
================================================================================
Every tunable in the project lives here. Nothing is hardcoded in the code: the
controller, the simulator, the pygame viewers and the CARLA bridge all read
their numbers from this file, so this is the one place to tune and debug.

Each setting documents: what it does, its units, a sensible RANGE, and which way
to turn it. Values left alone keep the car driving exactly as shipped.

PYGAME vs CARLA: the pygame sim and the CARLA bridge are tuned differently
(CARLA drives slower and more cautiously in town traffic). Where they differ,
there are SEPARATE settings: the ADAPT_* group drives the pygame sim, and the
CARLA_* group drives the CARLA bridge.

Quick cheatsheet:
    Passes obstacles too close        -> raise OBSTACLE_SAFETY_MARGIN
    Swerves too wide                  -> lower OBSTACLE_SAFETY_MARGIN
    Too timid in corners (pygame)     -> raise ADAPT_A_LAT_COMFORT
    Too timid in corners (CARLA)      -> raise CARLA_A_LAT_COMFORT
    Cuts back in too early            -> raise PASS_ZONE
    Drifts off the path               -> raise STATE_COST index 1 (cross-track)
    Steering oscillates / snaps       -> raise INPUT_RATE_COST index 1 (steer)
    Walls seen as obstacles (CARLA)   -> raise LIDAR_MIN_PTS or lower LIDAR_MAX_RADIUS
    Low FPS / slow solve              -> lower MPC_MAX_ITER or HORIZON_TIME

Run:  dcmpc-sim   .   dcmpc-drive   .   dcmpc-carla
================================================================================
"""

# ============================================================================ #
#  ROAD  (pygame sim)
#  Centreline waypoints in metres; a smooth spline is fit through them.
#  RANGE: any list of (x, y); keep points spaced so the spline stays smooth.
# ============================================================================ #
TRACK_X = [0, 60, 120, 160, 175, 160, 120, 70, 20, -20, -35, -25, 10]
TRACK_Y = [0,  0,  10,  45,  90, 130, 150, 145, 120, 85,  40,  10, -5]

# ============================================================================ #
#  DRIVING SPEED  (pygame sim)
# ============================================================================ #
TARGET_SPEED = 11.0    # cruise speed the controller aims to hold.
                       # units m/s. RANGE 3..25. Higher = faster lap, harder
                       # to track tight corners.
START_SPEED  = 11.0    # car's speed at sim start.
                       # units m/s. RANGE 0..TARGET_SPEED.

# ============================================================================ #
#  OBSTACLES  (pygame sim; list, add as many as you like)
#    explicit:        {"x": 77.0, "y": 3.0, "radius": 2.5}
#    along the road:  {"along": 0.13, "offset": 1.8, "radius": 2.5}
#       along  = fraction around the lap, RANGE 0..1
#       offset = metres LEFT (+) / RIGHT (-) of centreline, RANGE about -4..4
#       radius = obstacle size in metres, RANGE 0.5..3.0
# ============================================================================ #
OBSTACLES = [
    {"along": 0.13, "offset":  1.8, "radius": 2.5},   # early straight, left
    {"along": 0.40, "offset": -2.0, "radius": 2.0},   # mid circuit, right
    {"along": 0.70, "offset":  2.5, "radius": 2.0},   # late circuit, left
]

# ============================================================================ #
#  SENSOR  (pygame sim forward-facing detector)
# ============================================================================ #
SENSOR_RANGE   = 45.0    # how far ahead obstacles are detected.
                         # units m. RANGE 20..80. Larger = reacts earlier.
SENSOR_FOV_DEG = 100.0   # total detection cone width.
                         # units deg. RANGE 60..180. Wider = sees more to the side.

# ============================================================================ #
#  CAR PHYSICAL PARAMETERS  (dynamic bicycle model; shared by sim and MPC)
#    m  : mass.                       units kg.       RANGE 800..3000
#    Iz : yaw inertia.                units kg m^2.   RANGE 1000..4000
#    lf : CG to front axle.           units m.        RANGE 1.0..1.8
#    lr : CG to rear axle.            units m.        RANGE 1.0..1.8
#    Cf : front cornering stiffness.  units N/rad.    RANGE 80k..200k
#    Cr : rear cornering stiffness.   units N/rad.    RANGE 80k..200k
#  Handling: lower Cr (vs Cf) gives more OVERSTEER (tail slides); higher Cr
#  gives more UNDERSTEER (pushes wide, stable). The MPC model AND the simulator
#  read this same dict, so they always stay matched.
# ============================================================================ #
CAR = dict(m=1845.0, Iz=2600.0, lf=1.44, lr=1.44, Cf=140000.0, Cr=160000.0)

# ============================================================================ #
#  CONTROLLER  -  actuation limits (the physical envelope the MPC may command)
# ============================================================================ #
MAX_SPEED   = 30.0     # speed ceiling the MPC plans up to.
                       # units m/s. RANGE 10..40.
MAX_ACC     = 4.0      # max longitudinal acceleration (throttle authority).
                       # units m/s^2. RANGE 2..8. Higher = harder acceleration/braking.
MAX_D_ACC   = 6.0      # max longitudinal jerk (rate of accel change).
                       # units m/s^3. RANGE 3..12. Lower = smoother throttle.
MAX_STEER   = 0.5      # max front steering angle (about 29 deg).
                       # units rad. RANGE 0.3..0.7. Higher = sharper turns possible.
MAX_D_STEER = 0.6      # max steering rate.
                       # units rad/s. RANGE 0.3..1.0. Lower = smoother, lazier steering.

# ============================================================================ #
#  CONTROLLER  -  cost weights (what the MPC optimises)
#  STATE_COST / FINAL_STATE_COST order:
#      [along-track, CROSS-track, vx, psi(heading), vy, yaw-rate]
#    index 1 (cross-track): raise for TIGHTER path following. RANGE 50..400.
#    index 3 (heading):     raise to point straighter down the path. RANGE 10..80.
#  INPUT_COST order [accel, steer]: penalises large commands. RANGE each 0.5..10.
#  INPUT_RATE_COST order [accel-rate, steer-rate]: penalises CHANGES (smoothness)
#    index 1 (steer-rate): raise to stop snappy/oscillating steering. RANGE 50..400.
#  Bigger weight = the controller works harder to drive that term to zero.
# ============================================================================ #
STATE_COST       = (5, 150, 8, 40, 2, 2)
FINAL_STATE_COST = (5,  80, 8, 40, 2, 2)
INPUT_COST       = (1, 5)
INPUT_RATE_COST  = (1, 250)

# MPC iterative refinement: how many linearise-and-resolve passes per control
# step (Sequential Quadratic Programming iterations).
# units count. RANGE 1..5. Lower = faster solve, slightly less accurate; higher
# = more accurate near sharp manoeuvres but slower. Affects BOTH sim and CARLA.
MPC_MAX_ITER = 3

# ============================================================================ #
#  OBSTACLE AVOIDANCE  (shared by sim and CARLA)
# ============================================================================ #
ROAD_HALFWIDTH         = 5.0   # drivable half-width used for the overtake-vs-stop
                               # decision. units m. RANGE 3..7. Room on a side ->
                               # LANE-CHANGE; no room either side -> STOP behind.
OBSTACLE_SAFETY_MARGIN = 1.0   # clearance kept from the obstacle surface.
                               # units m. RANGE 0.3..2.0. Raise -> passes wider /
                               # safer (may STOP if the lane is tight); lower ->
                               # passes closer.
PASS_ZONE              = 6.0   # longitudinal keep-out distance past the obstacle,
                               # so the car does not cut back in too early.
                               # units m. RANGE 4..20. Raise if it clips the rear.
SLACK_PENALTY          = 1e5   # how hard the soft keep-out constraint pushes back.
                               # units cost. RANGE 1e3..1e6. Advanced; rarely changed.

# ============================================================================ #
#  ADAPTIVE SPEED  -  PYGAME SIM tuning (human-like speed modulation)
#  (CARLA uses the separate CARLA_* values further below.)
# ============================================================================ #
ADAPT_A_LAT_COMFORT   = 4.5    # comfortable lateral acceleration in corners.
                               # units m/s^2. RANGE 1.0..6.0. Raise = faster, more
                               # aggressive cornering; lower = gentler.
ADAPT_LOOKAHEAD       = 25.0   # distance scanned ahead for the tightest corner.
                               # units m. RANGE 10..40. Larger = brakes earlier for
                               # corners.
ADAPT_OBS_BRAKE_START = 35.0   # distance at which braking for an obstacle begins.
                               # units m. RANGE 15..50.
ADAPT_OBS_BRAKE_END   = 5.0    # distance at which the speed floor is reached.
                               # units m. RANGE 2..10. Must be < ADAPT_OBS_BRAKE_START.
ADAPT_V_MIN           = 4.0    # slowest creep speed near obstacles/corners.
                               # units m/s. RANGE 1..6. Keep high enough that the
                               # MPC horizon still covers a close obstacle
                               # (about ADAPT_V_MIN * HORIZON_TIME metres).
ADAPT_TAU             = 1.2    # speed smoothing time constant.
                               # units s. RANGE 0.5..3.0. Bigger = smoother / lazier.

# ============================================================================ #
#  CONTROLLER / SIM TIMING  (shared)
# ============================================================================ #
DT           = 0.1     # control + sim timestep (controller runs at 1/DT Hz).
                       # units s. RANGE 0.05..0.2. Smaller = finer control, more CPU.
HORIZON_TIME = 3.0     # MPC look-ahead window.
                       # units s. RANGE 1.5..4.0. Shorter = faster solves, less
                       # foresight; longer = smoother planning, slower solves.

# ============================================================================ #
#  CARLA BRIDGE  -  connection + scene setup (dcmpc-carla only)
# ============================================================================ #
CARLA_HOST            = "localhost"  # CARLA server host. Use the server's IP if
                                     # it runs on another machine.
CARLA_PORT            = 2000         # CARLA RPC port. RANGE 1024..65535. Change
                                     # if you launched the server with -carla-rpc-port.
CARLA_VEHICLE_FILTER  = "vehicle.audi.tt"             # blueprint for the ego car.
CARLA_OBSTACLE_FILTER = "vehicle.jeep.wrangler_rubicon" # blueprint for spawned obstacles.
CARLA_ROUTE_LENGTH_M  = 800.0   # how far down the lane the driving path is built.
                                # units m. RANGE 100..2000.
CARLA_SPECTATOR_HEIGHT= 40.0    # camera height above the car for the top-down
                                # spectator view. units m. RANGE 15..80.
                                # (kept for the "topdown" camera preset below)
CARLA_CREEP_THROTTLE  = 0.4     # throttle applied to pull away from a near-stop
                                # (below 0.5 m/s). units 0..1. RANGE 0.2..0.6.
                                # Higher = pulls away faster from rest.

# ============================================================================ #
#  CARLA BRIDGE  -  OBSTACLES  (the stalled cars placed along the route)
#  --------------------------------------------------------------------------
#  Each entry is one obstacle car placed along the driving path.
#  FORMAT (one dict per obstacle):
#      {"along": <metres>, "offset": <metres>}
#         along  = how far along the route to place it, metres from the start.
#                  RANGE 10 .. CARLA_ROUTE_LENGTH_M.
#         offset = sideways shift from the lane centre, in metres.
#                  POSITIVE = LEFT, NEGATIVE = RIGHT. RANGE about -4 .. 4.
#  Add / remove dicts to change how many obstacles spawn and where.
#  Use CARLA_OBSTACLES = []  (or run with --no-obstacle) for a clear road.
#  EXAMPLE - a three-car slalom:
#      CARLA_OBSTACLES = [
#          {"along":  60, "offset":  1.8},   # first,  to the left
#          {"along": 240, "offset": -2.0},   # second, to the right
#          {"along": 440, "offset":  2.5},   # third,  to the left
#      ]
# ============================================================================ #
CARLA_OBSTACLES = [
    {"along":  60.0, "offset":  1.8},   # first obstacle,  left of centre
    {"along": 240.0, "offset": -2.0},   # second obstacle, right of centre
    {"along": 440.0, "offset":  2.5},   # third obstacle,  left of centre
]

# ============================================================================ #
#  CARLA BRIDGE  -  CAMERA / SPECTATOR VIEW
#  --------------------------------------------------------------------------
#  The free spectator camera follows the ego car every tick. Pick a preset
#  with CARLA_CAMERA_MODE, then fine-tune that preset's values below.
#    "topdown" : straight down from above (best for path / avoidance view).
#    "chase"   : behind-and-above, looking forward (driver-ish view).
#    "front"   : ahead of the car, looking back at it (cinematic).
# ============================================================================ #
CARLA_CAMERA_MODE = "topdown"     # "topdown" | "chase" | "front"

# "topdown" preset
CARLA_CAM_TOPDOWN_HEIGHT = 40.0   # height above the car. units m. RANGE 15..80.
                                  # Lower = more zoomed in.
# "chase" preset (behind and above, looking forward)
CARLA_CAM_CHASE_BACK     = 8.0    # distance behind the car. units m. RANGE 4..15.
CARLA_CAM_CHASE_HEIGHT   = 4.0    # height above the car. units m. RANGE 2..10.
CARLA_CAM_CHASE_PITCH    = -12.0  # downward tilt. units deg. RANGE -30..0.
# "front" preset (ahead of the car, looking back)
CARLA_CAM_FRONT_AHEAD    = 8.0    # distance ahead of the car. units m. RANGE 4..15.
CARLA_CAM_FRONT_HEIGHT   = 3.0    # height above the car. units m. RANGE 1..8.
CARLA_CAM_FRONT_PITCH    = -8.0   # downward tilt. units deg. RANGE -30..0.

CARLA_TARGET_SPEED     = 7.0   # town driving cruise speed (about 25 km/h).
                               # units m/s. RANGE 3..15.
CARLA_MAX_THROTTLE_ACC = 4.0   # acceleration that maps to throttle = 1.0.
                               # units m/s^2. RANGE 2..6. Lower = gentler throttle.
CARLA_MAX_BRAKE_DEC    = 6.0   # deceleration that maps to brake = 1.0.
                               # units m/s^2. RANGE 3..10. Lower = gentler braking.
CARLA_OBSTACLE_RADIUS  = 1.5   # radius assigned to ground-truth detected vehicles
                               # (used only with --no-lidar).
                               # units m. RANGE 1.0..2.5.

# ============================================================================ #
#  CARLA BRIDGE  -  adaptive speed (SEPARATE from the pygame ADAPT_* values,
#  because town driving is slower and more cautious)
# ============================================================================ #
CARLA_A_LAT_COMFORT   = 1.5    # cautious cornering lateral accel for town.
                               # units m/s^2. RANGE 1.0..3.0. Raise = quicker corners.
CARLA_OBS_BRAKE_START = 25.0   # distance to start braking for an obstacle.
                               # units m. RANGE 15..40.
CARLA_OBS_BRAKE_END   = 5.0    # distance at which the speed floor is reached.
                               # units m. RANGE 2..10.
CARLA_V_MIN           = 3.0    # slowest creep speed in CARLA.
                               # units m/s. RANGE 1..5.

# ============================================================================ #
#  CARLA BRIDGE  -  virtual LiDAR sensor hardware (the simulated sensor itself)
# ============================================================================ #
CARLA_LIDAR_CHANNELS        = 32      # number of vertical laser channels.
                                      # RANGE 16..64. More = denser cloud, slower.
CARLA_LIDAR_POINTS_PER_SEC  = 120000  # total points emitted per second.
                                      # RANGE 50000..300000. More = denser, slower.
CARLA_LIDAR_UPPER_FOV       = 2.0     # highest beam angle above horizontal.
                                      # units deg. RANGE 0..15.
CARLA_LIDAR_LOWER_FOV       = -8.0    # lowest beam angle below horizontal.
                                      # units deg. RANGE -30..-2. More negative =
                                      # sees the road closer to the car.
CARLA_LIDAR_MOUNT_Z         = 2.4     # sensor mount height above the car origin.
                                      # units m. RANGE 1.6..3.0.

# ============================================================================ #
#  CARLA BRIDGE  -  LiDAR perception (turns the point cloud into obstacles).
#  These decide what counts as an obstacle vs ground / wall / scenery.
# ============================================================================ #
LIDAR_Z_MIN         = -1.2   # bottom of the kept height band (sensor frame).
                             # units m. RANGE -2.0..-0.5. Below this is treated as
                             # ground and dropped. Lower if cars are missed.
LIDAR_Z_MAX         = -0.1   # top of the kept height band (sensor frame).
                             # units m. RANGE -0.5..1.5. Above this is overhead and
                             # dropped. Raise if tall vehicles are missed.
LIDAR_FOV_DEG       = 90.0   # forward arc considered for obstacles.
                             # units deg. RANGE 60..150.
LIDAR_EGO_EXCLUSION = 4.5    # ignore returns closer than this (the ego car body).
                             # units m. RANGE 2.5..6.0.
LIDAR_MAX_RANGE     = 45.0   # ignore returns beyond this.
                             # units m. RANGE 20..80.
LIDAR_CLUSTER_CELL  = 1.0    # grid cell size used for clustering.
                             # units m. RANGE 0.5..2.0. Smaller = finer separation.
LIDAR_MIN_PTS       = 8      # minimum points for a valid cluster.
                             # units count. RANGE 4..20. Raise to reject noise and
                             # thin scenery; lower to detect smaller objects.
LIDAR_MAX_RADIUS    = 3.5    # clusters bigger than this are treated as walls /
                             # buildings and rejected.
                             # units m. RANGE 2.0..6.0. Raise if large vehicles get
                             # wrongly rejected.

# ============================================================================ #
#  CARLA BRIDGE  -  obstacle memory / scenery filter (the "bumper guard")
#  Keeps an obstacle in memory while it is alongside or just behind, so the car
#  does not forget it mid-swerve and clip it with the rear bumper.
# ============================================================================ #
CARLA_OBSTACLE_RADIUS_CAP = 1.5   # cap applied to detected cluster radius so a
                                  # large scenery cluster cannot inflate the
                                  # keep-out. units m. RANGE 1.0..2.5.
CARLA_BUMPER_GUARD_BACK   = -8.0  # how far BEHIND the car (negative x) an obstacle
                                  # is still remembered. units m. RANGE -12..-4.
CARLA_BUMPER_GUARD_FRONT  = 2.0   # how far AHEAD of the CG the guard zone extends.
                                  # units m. RANGE 1..4.
CARLA_BUMPER_GUARD_LATERAL= 6.0   # lateral half-width of the memory zone.
                                  # units m. RANGE 3..8. Wider = holds an obstacle
                                  # longer through a swerve.
CARLA_PATH_FILTER_MARGIN  = 2.5   # an obstacle is kept if it comes within
                                  # (its radius + this) of the planned path.
                                  # units m. RANGE 1.0..4.0. Larger = more cautious.
