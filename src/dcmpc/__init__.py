"""
dcmpc - Dynamic-Bicycle Car MPC.

A real-time iterative Model Predictive Controller for an autonomous car built on
the dynamic single-track (bicycle) model with a linear tire model. Tracks a
reference path, adapts speed to corners and obstacles, and avoids multiple
obstacles via soft half-plane constraints. Includes a pygame visualizer and a
CARLA bridge with a virtual-LiDAR perception front-end.
"""

from dcmpc.controller import (
    DynamicBicycleMPC,
    CarParams,
    AdaptiveSpeed,
    vehicle_dynamics,
    rk4_step,
    compute_path_from_wp,
    get_ref_trajectory,
    ego_to_global,
    build_mpc,
    build_adaptive_speed,
)

__version__ = "1.0.0"

__all__ = [
    "DynamicBicycleMPC",
    "CarParams",
    "AdaptiveSpeed",
    "vehicle_dynamics",
    "rk4_step",
    "compute_path_from_wp",
    "get_ref_trajectory",
    "ego_to_global",
    "build_mpc",
    "build_adaptive_speed",
    "__version__",
]
