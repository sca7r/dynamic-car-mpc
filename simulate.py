"""
Closed-loop MPC simulation. Reads the scenario from config.py, drives the car
around the road while avoiding obstacles, and saves:

    result.png     - the road, the driven line, and the obstacles
    telemetry.png  - speed / steering / sideslip / lateral-g over the run
    demo.gif       - animation with the live MPC plan

Usage:
    python simulate.py            # run + save all three outputs
    python simulate.py --no-gif   # skip the (slow) gif
    python simulate.py --live     # also pop up an interactive window
"""

from __future__ import annotations

import sys
import numpy as np
import matplotlib

if "--live" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import config as C
from dynamic_bicycle_mpc import (
    DynamicBicycleMPC, CarParams, rk4_step,
    compute_path_from_wp, get_ref_trajectory, ego_to_global, _nn_idx,
)


def resolve_obstacles(path):
    """Turn config obstacle specs into {x, y, radius} dicts."""
    cdist = np.append([0.0], np.cumsum(np.hypot(np.diff(path[0]), np.diff(path[1]))))
    out = []
    for o in C.OBSTACLES:
        if "x" in o and "y" in o:
            out.append({"x": float(o["x"]), "y": float(o["y"]),
                        "radius": float(o["radius"])})
        else:  # placed "along" the path with a lateral "offset"
            s = float(o["along"]) * cdist[-1]
            i = int(np.searchsorted(cdist, s))
            i = min(max(i, 0), path.shape[1] - 1)
            th = path[2, i]
            perp = np.array([-np.sin(th), np.cos(th)])
            off = float(o.get("offset", 0.0))
            out.append({"x": path[0, i] + perp[0] * off,
                        "y": path[1, i] + perp[1] * off,
                        "radius": float(o["radius"])})
    return out


def sense(state, obstacles):
    """Return the most relevant in-view obstacle in EGO coords, or None.

    The controller takes a single obstacle, so we pick the closest one that is
    within range and inside the forward field of view."""
    best, best_d = None, 1e18
    for o in obstacles:
        dx, dy = o["x"] - state[0], o["y"] - state[1]
        d = np.hypot(dx, dy)
        if d - o["radius"] > C.SENSOR_RANGE:
            continue
        rel = (np.arctan2(dy, dx) - state[2] + np.pi) % (2 * np.pi) - np.pi
        half_fov = np.radians(C.SENSOR_FOV_DEG) / 2
        if abs(rel) - np.arcsin(min(1.0, o["radius"] / max(d, 1e-6))) > half_fov:
            continue
        if d < best_d:
            ct, st = np.cos(-state[2]), np.sin(-state[2])
            ego = (dx * ct - dy * st, dx * st + dy * ct, o["radius"], 0.0, 0.0)
            best, best_d = ego, d
    return best


def main():
    path = compute_path_from_wp(C.TRACK_X, C.TRACK_Y, step=0.25)
    obstacles = resolve_obstacles(path)
    p = CarParams()
    for k, v in C.CAR.items():
        setattr(p, k, v)
    mpc = DynamicBicycleMPC(params=p, dt=C.DT, horizon_time=C.HORIZON_TIME)

    state = np.array([path[0, 0], path[1, 0], path[2, 0], C.START_SPEED, 0.0, 0.0])
    hist, plans, tele = [state.copy()], [None], []
    brakes = 0

    for i in range(2000):
        if not np.all(np.isfinite(state)):
            print("Simulation diverged."); break
        if i > 20 and np.hypot(state[0] - path[0, -1], state[1] - path[1, -1]) < 3.0:
            break
        target = get_ref_trajectory(state, path, C.TARGET_SPEED,
                                    mpc.control_horizon * C.DT, C.DT)
        ego_obs = sense(state, obstacles)
        ego_state = np.array([0.0, 0.0, 0.0, state[3], state[4], state[5]])
        traj_ego, u = mpc.solve(ego_state, target, obstacle=ego_obs, max_iter=3)
        if traj_ego is None:
            brakes += 1
        plans.append(ego_to_global(state, traj_ego) if traj_ego is not None else None)
        tele.append((i * C.DT, state[3], u[1, 0], state[4], state[3] * state[5] / 9.81))
        state = rk4_step(state, u[:, 0], C.DT, p)
        hist.append(state.copy())

    H = np.array(hist); tele = np.array(tele)
    print(f"Finished in {len(H)} steps ({len(H)*C.DT:.0f} s of driving), "
          f"{brakes} emergency-brake event(s).")
    for j, o in enumerate(obstacles):
        cl = np.min(np.hypot(H[:, 0] - o["x"], H[:, 1] - o["y"]) - o["radius"])
        print(f"  obstacle {j+1} at ({o['x']:.0f},{o['y']:.0f}): "
              f"min clearance {cl:.2f} m (buffer {mpc._vehicle_buffer:.2f})")

    _plots(path, H, tele, obstacles, plans, live=("--live" in sys.argv),
           make_gif=("--no-gif" not in sys.argv))


def _plots(path, H, tele, obstacles, plans, live, make_gif):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(path[0], path[1], "--", color="0.6", label="road centreline")
    ax.plot(H[:, 0], H[:, 1], "-", color="C0", lw=2, label="driven line")
    ax.plot(H[0, 0], H[0, 1], "o", color="green", ms=11, label="start")
    ax.plot(path[0, -1], path[1, -1], "*", color="red", ms=16, label="finish")
    for o in obstacles:
        ax.add_patch(plt.Circle((o["x"], o["y"]), o["radius"], color="C3", alpha=0.6))
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(loc="best")
    ax.set_title("Dynamic-bicycle car MPC: tracking + obstacle avoidance")
    fig.savefig("result.png", dpi=110, bbox_inches="tight")
    print("Saved result.png")

    fig2, axs = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    axs[0].plot(tele[:, 0], tele[:, 1] * 3.6, "C0"); axs[0].set_ylabel("speed\n(km/h)")
    axs[1].plot(tele[:, 0], np.degrees(tele[:, 2]), "C1"); axs[1].set_ylabel("steer\n(deg)")
    axs[2].plot(tele[:, 0], tele[:, 3], "C2"); axs[2].set_ylabel("sideslip\nvy (m/s)")
    axs[3].plot(tele[:, 0], tele[:, 4], "C3"); axs[3].set_ylabel("lateral\n(g)")
    axs[3].set_xlabel("time (s)")
    for a in axs: a.grid(alpha=0.3)
    axs[0].set_title("Vehicle telemetry")
    fig2.tight_layout(); fig2.savefig("telemetry.png", dpi=110, bbox_inches="tight")
    print("Saved telemetry.png")

    if make_gif:
        step = max(1, len(H) // 150)
        idx = list(range(0, len(H), step))
        fig3, ax3 = plt.subplots(figsize=(9, 9))
        ax3.plot(path[0], path[1], "--", color="0.6")
        for o in obstacles:
            ax3.add_patch(plt.Circle((o["x"], o["y"]), o["radius"], color="C3", alpha=0.6))
        ax3.set_aspect("equal"); ax3.grid(alpha=0.3)
        ax3.set_xlim(min(path[0]) - 15, max(path[0]) + 15)
        ax3.set_ylim(min(path[1]) - 15, max(path[1]) + 15)
        ax3.set_title("Dynamic-bicycle car MPC")
        (trace,) = ax3.plot([], [], "-", color="C0", lw=2)
        (plan_line,) = ax3.plot([], [], "-", color="C1", lw=2, alpha=0.9)
        (car,) = ax3.plot([], [], "o", color="C0", ms=10)

        def upd(f):
            i = idx[f]
            trace.set_data(H[: i + 1, 0], H[: i + 1, 1])
            car.set_data([H[i, 0]], [H[i, 1]])
            pl = plans[i]
            plan_line.set_data(pl[0], pl[1]) if pl is not None else plan_line.set_data([], [])
            return trace, car, plan_line

        anim = FuncAnimation(fig3, upd, frames=len(idx), interval=70, blit=True)
        anim.save("demo.gif", writer=PillowWriter(fps=14))
        print("Saved demo.gif")

    if live:
        plt.show()


if __name__ == "__main__":
    main()
