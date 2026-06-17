"""
Manual drive -- give the car your OWN inputs and watch how it responds, with no
controller in the loop. This is the best way to *feel* the dynamic-bicycle model:
a step of steering, a quick swerve, trail-braking, etc.

You provide a schedule of commands. Each entry is (duration_seconds, accel, steer):
    accel  -- longitudinal acceleration in m/s^2  (+ speeds up, - brakes)
    steer  -- front steering angle in radians      (+ left, - right)

Edit INPUTS below and run:   python manual_drive.py
Outputs manual_result.png (path + telemetry).
"""

from __future__ import annotations

import sys
import numpy as np
import matplotlib

if "--live" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dcmpc import config as C
from dcmpc.controller import CarParams, rk4_step

# ====================================================================== #
#  YOUR INPUTS  -- (duration [s], accel [m/s^2], steer [rad])
#  This example: cruise, a left-right swerve (lane change feel), then settle.
# ====================================================================== #
INPUTS = [
    (2.0, 0.0,  0.00),   # hold speed, straight
    (1.0, 0.0,  0.10),   # steer left
    (1.0, 0.0, -0.10),   # steer right
    (2.0, 0.0,  0.00),   # straighten up
    (2.0, 2.0,  0.05),   # accelerate through a gentle left
    (2.0, -3.0, 0.00),   # brake in a straight line
]

START_SPEED = 12.0   # initial vx [m/s]


def main():
    p = CarParams()
    for k, v in C.CAR.items():
        setattr(p, k, v)

    state = np.array([0.0, 0.0, 0.0, START_SPEED, 0.0, 0.0])  # x,y,psi,vx,vy,r
    dt = 0.02  # fine integration for an accurate open-loop response

    H, tele = [state.copy()], []
    t = 0.0
    for dur, a, delta in INPUTS:
        for _ in range(int(dur / dt)):
            u = np.array([a, delta])
            tele.append((t, state[3], delta, state[4], state[5],
                         state[3] * state[5] / 9.81))
            state = rk4_step(state, u, dt, p)
            H.append(state.copy())
            t += dt
    H = np.array(H); tele = np.array(tele)
    print(f"Simulated {t:.1f} s. Final speed {state[3]*3.6:.1f} km/h, "
          f"heading {np.degrees(state[2]):.0f} deg.")

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].plot(H[:, 0], H[:, 1], "C0", lw=2)
    axs[0].plot(H[0, 0], H[0, 1], "go", ms=10, label="start")
    axs[0].plot(H[-1, 0], H[-1, 1], "rs", ms=10, label="end")
    axs[0].set_aspect("equal"); axs[0].grid(alpha=0.3); axs[0].legend()
    axs[0].set_title("Path (open-loop, your inputs)")
    axs[0].set_xlabel("x (m)"); axs[0].set_ylabel("y (m)")

    ax = axs[1]
    ax.plot(tele[:, 0], tele[:, 1] * 3.6, label="speed (km/h)")
    ax.plot(tele[:, 0], np.degrees(tele[:, 2]) , label="steer (deg)")
    ax.plot(tele[:, 0], tele[:, 3], label="sideslip vy (m/s)")
    ax.plot(tele[:, 0], np.degrees(tele[:, 4]), label="yaw rate (deg/s)")
    ax.plot(tele[:, 0], tele[:, 5], label="lateral (g)")
    ax.grid(alpha=0.3); ax.legend(); ax.set_xlabel("time (s)")
    ax.set_title("Response to your inputs")

    fig.tight_layout(); fig.savefig("manual_result.png", dpi=110, bbox_inches="tight")
    print("Saved manual_result.png")
    if "--live" in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()
