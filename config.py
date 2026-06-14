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
    {"along": 0.13, "offset": 3.0, "radius": 2.5},   # stalled car in the lane
    # {"along": 0.55, "offset": -3.0, "radius": 2.0},  # add a second one
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
CAR = dict(m=1500.0, Iz=2250.0, lf=1.2, lr=1.6, Cf=120000.0, Cr=120000.0)

# ====================================================================== #
#  CONTROLLER / SIM timing
# ====================================================================== #
DT = 0.1               # control + sim timestep [s]  (controller runs at 1/DT Hz)
HORIZON_TIME = 3.0     # MPC look-ahead [s]
