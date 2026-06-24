"""
Multi-object Kalman tracker for the CARLA virtual-LiDAR obstacles.

Raw LiDAR clustering gives a fresh, memoryless snapshot every tick: obstacle
identities can swap between ticks, a detection can blink out for a frame, and
there is no velocity estimate (the raw vx, vy are 0). This tracker sits between
lidar_obstacles() and the rest of the bridge and gives each obstacle a STABLE
identity, a SMOOTHED position, an estimated VELOCITY, and short COASTING through
brief dropouts.

It is perception-only: it outputs the same (x, y, r, vx, vy) tuples that the
bridge already consumes, so the controller and the existing obstacle filtering
are untouched. Toggle with CARLA_TRACKER_ENABLE in config.

Each track is a constant-velocity Kalman filter:
    state  X = [x, y, vx, vy]
    meas   Z = [x, y]            (cluster centroid, ego frame)
The constant-velocity model assumes obstacles hold their velocity between ticks
(true over a 0.1 s step); the filter corrects that with each new measurement via
the Kalman gain.
"""

from __future__ import annotations
import math
import numpy as np
from scipy.optimize import linear_sum_assignment


class _Track:
    """One Kalman-filtered obstacle with constant-velocity dynamics."""
    _next_id = 0

    def __init__(self, x, y, r, dt, q_accel, meas_var):
        self.id = _Track._next_id
        _Track._next_id += 1
        self.dt = dt
        self.X = np.array([x, y, 0.0, 0.0], dtype=float)        # [x,y,vx,vy]
        self.P = np.diag([meas_var, meas_var, 4.0, 4.0]).astype(float)
        self.r = r
        self.hits = 1          # measurements associated so far
        self.misses = 0        # consecutive ticks without a measurement
        self.age = 1

        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)
        dt2, dt3, dt4 = dt*dt, dt*dt*dt, dt*dt*dt*dt
        q = q_accel
        self.Q = q * np.array([[dt4/4, 0,     dt3/2, 0    ],
                               [0,     dt4/4, 0,     dt3/2],
                               [dt3/2, 0,     dt2,   0    ],
                               [0,     dt3/2, 0,     dt2  ]], dtype=float)
        self.R = np.diag([meas_var, meas_var]).astype(float)

    def predict(self):
        self.X = self.F @ self.X
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1

    def update(self, z, r):
        y = z - self.H @ self.X                      # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)     # Kalman gain
        self.X = self.X + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.r = 0.6 * self.r + 0.4 * r              # smooth radius
        self.hits += 1
        self.misses = 0

    @property
    def pos(self):
        return self.X[0], self.X[1]

    @property
    def vel(self):
        return self.X[2], self.X[3]


class ObstacleTracker:
    """
    Nearest-neighbour multi-object tracker.

    Parameters (all surfaced in config):
      dt           : tick timestep [s]
      gate         : max distance to associate a detection to a track [m]
      q_accel      : process-noise accel variance (higher = trusts measurements
                     more, more agile but noisier)
      meas_var     : measurement-noise variance (higher = smoother but laggier)
      confirm_hits : a track must be seen this many times before it is output
      max_misses   : a track coasts this many ticks of no detection before being
                     dropped (survives brief dropouts)
    """
    def __init__(self, dt=0.1, gate=3.0, q_accel=4.0, meas_var=0.25,
                 confirm_hits=2, max_misses=5):
        self.dt = dt
        self.gate = gate
        self.q_accel = q_accel
        self.meas_var = meas_var
        self.confirm_hits = confirm_hits
        self.max_misses = max_misses
        self.tracks = []

    def update(self, detections):
        """
        detections: list of (x, y, r, vx, vy) from lidar_obstacles (raw vx, vy
                    are ignored - the tracker estimates velocity itself).
        returns:    list of (x, y, r, vx, vy) for confirmed tracks, nearest
                    first, with smoothed position and estimated velocity.
        """
        # 1) predict every existing track forward one tick
        for t in self.tracks:
            t.predict()

        # 2) associate detections to tracks (optimal / Hungarian assignment)
        #    Greedy nearest-neighbour can lock in a locally-cheap pairing that
        #    forces a worse one elsewhere; Hungarian minimises TOTAL distance
        #    across all pairs at once, so two obstacles passing close keep their
        #    correct identities. Pairs further apart than 'gate' are rejected.
        dets = [(d[0], d[1], d[2]) for d in detections]
        unmatched = set(range(len(dets)))
        updated = set()
        if self.tracks and dets:
            D = np.empty((len(self.tracks), len(dets)))
            for i, t in enumerate(self.tracks):
                tx, ty = t.pos
                for j, (dx, dy, _) in enumerate(dets):
                    D[i, j] = math.hypot(tx - dx, ty - dy)
            row_idx, col_idx = linear_sum_assignment(D)
            for i, j in zip(row_idx, col_idx):
                if D[i, j] > self.gate:
                    continue                # too far apart - not the same object
                dx, dy, dr = dets[j]
                self.tracks[i].update(np.array([dx, dy]), dr)
                updated.add(i)
                unmatched.discard(j)

        # 3) tracks not updated this tick coast (count a miss)
        for i, t in enumerate(self.tracks):
            if i not in updated:
                t.misses += 1

        # 4) spawn new tracks for unmatched detections
        for j in unmatched:
            dx, dy, dr = dets[j]
            self.tracks.append(
                _Track(dx, dy, dr, self.dt, self.q_accel, self.meas_var))

        # 5) cull tracks missing too long
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        # 6) output confirmed tracks, nearest first
        out = []
        for t in self.tracks:
            if t.hits >= self.confirm_hits:
                vx, vy = t.vel
                out.append((t.pos[0], t.pos[1], t.r, vx, vy))
        out.sort(key=lambda o: math.hypot(o[0], o[1]))
        return out