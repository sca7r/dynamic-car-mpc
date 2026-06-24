"""Kinematic obstacle-avoidance path planner (Layer 1 of the decoupled stack).

Why this exists
---------------
The dynamic-bicycle MPC tracker (controller.py) tracks a reference well, but
folding obstacle keep-out INTO that same QP made the problem infeasible at low
speed off-path (the multi-second solver freezes and wild swerves). The fix used
in industry is to split the job: a purely *kinematic* planner bends the path
around obstacles (always feasible - it is just geometry), and the dynamic
tracker then follows that already-safe path with NO obstacle constraint.

This module is that planner. It is deliberately simple and side-effect free:

    safe_path = plan_safe_path(path, obstacles_ego, state, C)

- path           : (3, M) global centerline, rows [x, y, heading]
- obstacles_ego  : list of (x, y, r, vx, vy) in the EGO frame (as the bridge has)
- state          : global ego state [x, y, psi, vx, vy, r]
- C              : config module (reads vehicle width, margin, road halfwidth)

Returns a NEW (3, M) global path, rows [x, y, heading], with the lateral
position bent away from obstacles by exactly the required clearance and the
bend smoothed so the tracker can follow it. If there is nothing to avoid the
ORIGINAL path is returned unchanged (bit-identical), so behaviour when clear is
exactly as before.

The planner never moves a point beyond +/- road_halfwidth, and if an obstacle
cannot be cleared within the road it leaves the path alone and reports it via
the returned `blocked` flag (the speed governor then handles the stop, exactly
as today).
"""

from __future__ import annotations
import math
import numpy as np


def _ego_obstacles_to_global(obstacles_ego, state):
    """Convert (x,y,r,...) ego-frame obstacles to global (ox, oy, r)."""
    px, py, psi = float(state[0]), float(state[1]), float(state[2])
    c, s = math.cos(psi), math.sin(psi)
    out = []
    for ob in obstacles_ego:
        ex, ey, r = float(ob[0]), float(ob[1]), float(ob[2])
        gx = px + ex * c - ey * s
        gy = py + ex * s + ey * c
        out.append((gx, gy, r))
    return out


# --- module-level obstacle memory ----------------------------------------
# When an obstacle drops out of the live list (passed, or briefly lost), we keep
# applying its path bend for a few ticks with a decaying weight, so the
# reference path ramps back to centre SMOOTHLY instead of snapping. A sudden
# snap was leaving the car off the (newly-centred) path at low speed, which
# tripped the recovery speed-cap and stalled it. Keyed by rounded global pos.
_obstacle_memory = {}   # key -> dict(ox, oy, r, ttl, w)


def reset_planner_memory():
    """Clear remembered obstacles (call at the start of a fresh run)."""
    _obstacle_memory.clear()



def plan_safe_path(path, obstacles_ego, state, C):
    """Bend `path` laterally around obstacles. Returns (safe_path, blocked).

    safe_path : (3, M) global path rows [x, y, heading]
    blocked   : True if some obstacle could not be cleared within the road
                (caller should fall back to stopping via the speed governor).
    """
    if path is None or path.shape[1] < 3:
        return path, False

    # --- vehicle / road geometry (mirror controller._decide_one) ---
    width = float(getattr(C, "CAR", {}).get("width", 2.0)) \
        if isinstance(getattr(C, "CAR", {}), dict) else 2.0
    # controller uses width/2 + safety_margin as the buffer
    veh_buffer = width / 2.0 + float(getattr(C, "OBSTACLE_SAFETY_MARGIN", 0.9))
    road_half = float(getattr(C, "ROAD_HALFWIDTH", 4.0))
    # how far along the path (in points) to ramp the bend in/out; 0.1 m steps
    step = 0.1
    # ramp length scales with the lateral shift so the bend angle stays gentle
    # (a 3.4 m shift over ~12 m of travel is ~16 deg, a comfortable lane change).
    ramp_gain = float(getattr(C, "PLANNER_RAMP_GAIN", 3.5))  # metres of ramp per metre of shift
    min_ramp_m = float(getattr(C, "PLANNER_RAMP_MIN", 6.0))
    # how many ticks a passed obstacle's bend lingers + decays (smooth ramp-out)
    decay_ticks = int(getattr(C, "PLANNER_DECAY_TICKS", 12))

    live = _ego_obstacles_to_global(obstacles_ego, state) if obstacles_ego else []
    psi = float(state[2])
    cpsi, spsi = math.cos(psi), math.sin(psi)
    # Hold budget: how many ticks a still-ahead but momentarily-undetected
    # obstacle keeps its bend at full strength before we give up on it. Generous,
    # because a real obstacle is seen continuously until passed; this only guards
    # against a phantom one-frame detection pinning the path forever.
    hold_budget = max(decay_ticks, 40)

    # Refresh memory with live obstacles. IMPORTANT: update IN PLACE so the
    # committed pass-side displacement ("needed") survives. Replacing the dict
    # (the old behaviour) wiped the commitment every tick, letting the pass side
    # and magnitude wobble while the obstacle was visible.
    refreshed = set()
    for (ox, oy, orad) in live:
        key = (round(ox / 2.0), round(oy / 2.0))
        refreshed.add(key)
        m = _obstacle_memory.get(key)
        if m is None:
            _obstacle_memory[key] = dict(ox=ox, oy=oy, r=orad,
                                         ttl=decay_ticks, hold=hold_budget, w=1.0)
        else:
            m["ox"], m["oy"], m["r"] = ox, oy, orad
            m["ttl"], m["hold"], m["w"] = decay_ticks, hold_budget, 1.0

    # Age out obstacles not seen this tick.
    dead = []
    for key, m in _obstacle_memory.items():
        if key in refreshed:
            continue
        # Longitudinal position of the obstacle in the current ego frame: > 0 is
        # ahead of the car, < 0 is behind.
        ex = (m["ox"] - state[0]) * cpsi + (m["oy"] - state[1]) * spsi
        if ex > -m["r"]:
            # STILL AHEAD: hold the bend at FULL strength through the dropout, so
            # the path does not straighten and then snap back when the obstacle
            # reappears at close range (the cause of the late, violent swerve).
            # The spatial raised-cosine ramp below still gives the smooth shape;
            # we just refuse to collapse the bend on a missed detection.
            m["w"] = 1.0
            m["hold"] = m.get("hold", hold_budget) - 1
            if m["hold"] <= 0:
                dead.append(key)           # never returned -> drop the phantom
            continue
        # PASSED (now behind the car): ramp the bend out smoothly, then drop.
        m["ttl"] -= 1
        m["w"] = max(0.0, m["ttl"] / float(decay_ticks))
        if m["ttl"] <= 0:
            dead.append(key)
    for key in dead:
        _obstacle_memory.pop(key, None)

    # obstacles to bend around = memory (live ones have w=1.0, faded ones < 1.0)
    obstacles = [(m["ox"], m["oy"], m["r"], m["w"]) for m in _obstacle_memory.values()]
    if not obstacles:
        return path, False

    M = path.shape[1]
    px, py, pth = path[0].copy(), path[1].copy(), path[2].copy()
    # lateral displacement to apply at each path point (signed, +left)
    disp = np.zeros(M)
    blocked = False

    # unit lateral (left) vector at each point, from heading
    lx = -np.sin(pth)
    ly = np.cos(pth)

    for (ox, oy, orad, w) in obstacles:
        keep = orad + veh_buffer
        # nearest path point to this obstacle
        d2 = (px - ox) ** 2 + (py - oy) ** 2
        i = int(np.argmin(d2))
        key = (round(ox / 2.0), round(oy / 2.0))
        mem = _obstacle_memory.get(key, {})
        # if we already committed a displacement for this obstacle, reuse it so
        # the pass side cannot flip as the obstacle fades from memory.
        if "needed" in mem:
            needed = mem["needed"]
        else:
            # signed lateral offset of obstacle from the path at i (+left)
            lat = (ox - px[i]) * lx[i] + (oy - py[i]) * ly[i]
            # If the path already clears the obstacle by the full keep-out, do
            # NOT bend. Bending here would set the path to sit EXACTLY `keep`
            # from the obstacle, which for an already-clear obstacle pulls the
            # path TOWARD it - the "steers for an obstacle that isn't in the
            # path" behaviour. Only intrusions into the corridor get a bend.
            if abs(lat) >= keep:
                continue                       # already clear: no shift, no lock
            # how much room either side of the obstacle within the road
            room_left = road_half - (lat + keep)
            room_right = road_half + (lat - keep)
            if max(room_left, room_right) < 0.0:
                # cannot clear within the road -> let caller stop
                blocked = True
                continue
            # pass on the side with more room (same rule as the controller)
            pass_left = room_left >= room_right
            if pass_left:
                target_lat = lat + keep        # push path to the LEFT of obstacle
            else:
                target_lat = lat - keep        # push path to the RIGHT of obstacle
            needed = max(-road_half, min(road_half, target_lat))
            mem["needed"] = needed             # lock the committed displacement
        # apply the decay weight so the bend ramps out smoothly as the obstacle
        # fades from memory (prevents the path snapping back to centre)
        needed = needed * w
        # ramp length scales with the magnitude of the shift -> gentle bend angle
        ramp_m = max(min_ramp_m, ramp_gain * abs(needed))
        ramp_pts = max(1, int(ramp_m / step))
        # hold the offset for a "pass zone" PAST the obstacle before ramping back,
        # so the tracker does not cut back across the obstacle's tail (which was
        # grazing it). Mirrors the old controller PASS_ZONE.
        hold_pts = max(0, int(float(getattr(C, "PASS_ZONE", 2.5)) / step))
        loidx = max(0, i - ramp_pts)
        hiidx = min(M, i + hold_pts + ramp_pts + 1)
        for k in range(loidx, hiidx):
            if k <= i:
                # ramp UP into the bend (raised cosine, 0 at loidx edge -> 1 at i)
                rw = 0.5 * (1.0 + math.cos(math.pi * (k - i) / ramp_pts))
            elif k <= i + hold_pts:
                # HOLD full offset while passing and just past the obstacle
                rw = 1.0
            else:
                # ramp DOWN back to centre after the hold
                rw = 0.5 * (1.0 + math.cos(math.pi * (k - i - hold_pts) / ramp_pts))
            rw = max(0.0, rw)
            # take the largest-magnitude displacement if obstacles overlap
            cand = needed * rw
            if abs(cand) > abs(disp[k]):
                disp[k] = cand

    if not np.any(disp):
        return path, blocked

    # apply displacement along the lateral direction
    nx = px + disp * lx
    ny = py + disp * ly
    # recompute heading along the bent path so row [2] stays consistent
    nth = pth.copy()
    if M >= 2:
        nth[:-1] = np.arctan2(np.diff(ny), np.diff(nx))
        nth[-1] = nth[-2]
    safe = np.vstack((nx, ny, nth))
    return safe, blocked