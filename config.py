"""
Scenario configuration -- edit this file to change the road, the obstacles, the
speed, and the car's physical parameters. Nothing else needs to be touched.

Run a closed-loop MPC simulation with:   python simulate.py
Try giving your own throttle/steer inputs with:   python manual_drive.py
"""

# ====================================================================== #
#  ROAD  (centreline waypoints, in metres). The path is a smooth spline
#  through these points. Add/move/remove points to reshape the road.
# ====================================================================== #
TRACK_X = [0, 60, 120, 160, 175, 160, 120, 70, 20, -20, -35, -25, 10]
TRACK_Y = [0,  0,  10,  45,  90, 130, 150, 145, 120, 85,  40,  10, -5]

# ====================================================================== #
#  DRIVING
# ====================================================================== #
TARGET_SPEED = 11.0    # cruise speed the controller tries to hold [m/s]
START_SPEED = 11.0     # car's initial speed [m/s]

# ====================================================================== #
#  OBSTACLES  (a list -- add as many as you like)
#  Each obstacle is a dict. Two ways to place one:
#    explicit:        {"x": 77.0, "y": 3.0, "radius": 2.5}
#    along the road:  {"along": 0.13, "offset": 3.0, "radius": 2.5}
#       along  = fraction around the lap (0..1)
#       offset = metres to the LEFT (+) or RIGHT (-) of the centreline
#  NOTE: the controller handles ONE obstacle at a time (the nearest one in
#  view), so keep obstacles reasonably spaced for clean behaviour. Also keep
#  them OFFSET from the centreline -- an obstacle dead-centre on the path is the
#  degenerate case (see README).
# ====================================================================== #
OBSTACLES = [
    {"along": 0.13, "offset":  1.8, "radius": 2.5},   # obstacle 1 – early straight, left
    {"along": 0.40, "offset": -2.0, "radius": 2.0},   # obstacle 2 – mid circuit, right
    {"along": 0.70, "offset":  2.5, "radius": 2.0},   # obstacle 3 – late circuit, left
]

# ====================================================================== #
#  SENSOR (a forward-facing detector: range + field of view)
# ====================================================================== #
SENSOR_RANGE = 45.0      # metres
SENSOR_FOV_DEG = 100.0   # degrees (total cone)

# ====================================================================== #
#  CAR PHYSICAL PARAMETERS  (try changing these and re-running!)
#    m  : mass [kg]              Iz : yaw inertia [kg m^2]
#    lf : CG->front axle [m]     lr : CG->rear axle [m]
#    Cf : front cornering stiffness [N/rad]   Cr : rear [N/rad]
#  e.g. lowering Cr relative to Cf makes the car more oversteery.
# ====================================================================== #
# Tesla Model 3 (approx): ~1845 kg, wheelbase 2.875 m, near 50/50 split, stiff tires.
# Both the MPC's prediction model AND our own simulator read this one dict, so
# they always stay matched (a mismatch is what makes the car drift off-road).
CAR = dict(m=1845.0, Iz=2600.0, lf=1.44, lr=1.44, Cf=140000.0, Cr=160000.0)

# ====================================================================== #
#  OBSTACLE AVOIDANCE ROBUSTNESS
#  ROAD_HALFWIDTH = drivable half-width [m] used to decide whether there is
#  room to overtake an obstacle sitting in the lane:
#    * a side has room  -> car LANE-CHANGES around the obstacle
#    * no room either side -> car STOPS behind it (brakes to a halt)
#  Bigger = more willing to swerve wide; smaller = more likely to stop.
#  (A CARLA town lane is ~3.5 m; ~5 m lets the car borrow the next lane.)
#  PASS_ZONE = how far along the road the keep-out extends past the obstacle [m].
# ====================================================================== #
ROAD_HALFWIDTH = 5.0
PASS_ZONE = 6.0

# Minimum clearance the MPC keeps between the car body and any obstacle surface.
# Increase this if the car passes too close; decrease if it swerves too wide.
OBSTACLE_SAFETY_MARGIN = 0.8   # metres — clearance from obstacle surface
                                # (was 1.5 but that made keep-out too large for
                                #  CARLA town lanes; 0.8 gives ~1.8 m real clearance)

# ====================================================================== #
#  CARLA BRIDGE SETTINGS  (used by carla_mpc.py only)
# ====================================================================== #
CARLA_TARGET_SPEED     = 7.0   # m/s  — town driving speed (~25 km/h)
CARLA_MAX_THROTTLE_ACC = 4.0   # m/s² that maps to throttle = 1.0
CARLA_MAX_BRAKE_DEC    = 6.0   # m/s² that maps to brake = 1.0
CARLA_OBSTACLE_RADIUS  = 1.5   # m    — radius assigned to detected CARLA vehicles
                                # Real cars are ~1 m half-width; 1.5 adds a small buffer
                                # without over-inflating the keep-out zone

# ====================================================================== #
#  CONTROLLER / SIM timing
# ====================================================================== #
DT = 0.1               # control + sim timestep [s]  (controller runs at 1/DT Hz)
HORIZON_TIME = 3.0     # MPC look-ahead [s]