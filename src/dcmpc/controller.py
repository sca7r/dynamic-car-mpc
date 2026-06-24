"""
Iterative MPC (iMPC) trajectory-tracking controller for a CAR modeled with the
DYNAMIC BICYCLE (single-track) model and a linear tire model.

Unlike the kinematic bicycle (which assumes tires never slip), this model carries
lateral velocity and yaw rate as states and generates cornering force from tire
SLIP ANGLES. That is what real autonomous-driving / racing controllers use,
because the kinematic model becomes wrong precisely when it matters: hard, fast
cornering where the tires actually slip.

State   X = [x, y, psi, vx, vy, r]
            position (x,y), yaw psi, longitudinal vel vx, lateral vel vy, yaw rate r
Control u = [a, delta]
            longitudinal acceleration, front steering angle

Continuous dynamics (body-frame velocities, global position):
    x_dot   = vx*cos(psi) - vy*sin(psi)
    y_dot   = vx*sin(psi) + vy*cos(psi)
    psi_dot = r
    vx_dot  = a + vy*r
    vy_dot  = (Fyf + Fyr)/m - vx*r
    r_dot   = (lf*Fyf - lr*Fyr)/Iz
with linear tire forces from slip angles:
    alpha_f = delta - (vy + lf*r)/vx ,   Fyf = Cf*alpha_f
    alpha_r =       - (vy - lr*r)/vx ,   Fyr = Cr*alpha_r

The 1/vx in the slip angles is singular at standstill, so vx is floored at v_min
inside the tire model (a standard pragmatic fix; a more rigorous option is to
blend to the kinematic model at low speed).

Compared to the kinematic controllers, the ONLY parts that change are the model
function and the state/control dimensions. The iMPC loop, the track-relative
cost, the soft half-plane obstacle constraints, the warm-starting, the emergency
brake, and the DPP problem structure are all identical.
"""

from __future__ import annotations

import math
import numpy as np
import numpy.typing as npt
import cvxpy as opt
from scipy.linalg import expm


def vehicle_dynamics(X, u, p):
    """Continuous dynamic-bicycle dynamics f(X, u) for car parameters `p`."""
    _, _, psi, vx, vy, r = X
    a, delta = u
    
    # Smooth floor prevents step-function explosions near 0 m/s
    vxs = max(vx, p.v_min)
    
    alpha_f = delta - (vy + p.lf * r) / vxs
    alpha_r = -(vy - p.lr * r) / vxs
    
    am = getattr(p, "alpha_max", 0.12)        
    
    # Soft saturation guarantees a continuous, non-zero derivative for the solver
    alpha_f = am * np.tanh(alpha_f / am)
    alpha_r = am * np.tanh(alpha_r / am)
    
    Fyf = p.Cf * alpha_f
    Fyr = p.Cr * alpha_r
    
    vb = getattr(p, "v_blend", 2.0)           
    fade = min(1.0, abs(vx) / vb)             
    
    return np.array([
        vx * np.cos(psi) - vy * np.sin(psi),
        vx * np.sin(psi) + vy * np.cos(psi),
        r,
        a + vy * r,
        fade * ((Fyf + Fyr) / p.m - vx * r),
        fade * ((p.lf * Fyf - p.lr * Fyr) / p.Iz),
    ])


def rk4_step(X, u, dt, p, substeps=1):
    """Integrate the true plant one control step with RK4."""
    h = dt / substeps
    for _ in range(substeps):
        k1 = vehicle_dynamics(X, u, p)
        k2 = vehicle_dynamics(X + h / 2 * k1, u, p)
        k3 = vehicle_dynamics(X + h / 2 * k2, u, p)
        k4 = vehicle_dynamics(X + h * k3, u, p)
        X = X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return X


class CarParams:
    """Passenger-car parameters (roughly a mid-size sedan)."""

    def __init__(self):
        self.m = 1500.0      # mass [kg]
        self.Iz = 2250.0     # yaw inertia [kg m^2]
        self.lf = 1.2        # CG -> front axle [m]
        self.lr = 1.6        # CG -> rear axle [m]
        self.Cf = 120000.0   # front axle cornering stiffness [N/rad]
        self.Cr = 120000.0   # rear axle cornering stiffness [N/rad]
        self.v_min = 1.0     # floor on vx used in the tire model [m/s]
        self.alpha_max = 0.12  # slip-angle saturation [rad] (~7 deg grip limit)
        self.v_blend = 2.0     # below this speed, lateral dynamics fade out [m/s]

    @property
    def wheelbase(self):
        return self.lf + self.lr


class DynamicBicycleMPC:
    STATE_DIM = 6     # [x, y, psi, vx, vy, r]
    CONTROL_DIM = 2   # [a, delta]
    MAX_OBS = 4       # max simultaneous obstacle keep-out constraints

    def __init__(
        self,
        params: CarParams | None = None,
        dt: float = 0.1,
        horizon_time: float = 3.0,
        width: float = 1.8,
        max_speed: float = 30.0,      # ~108 km/h
        max_acc: float = 4.0,         # m/s^2
        max_d_acc: float = 6.0,       # m/s^3 (long. jerk)
        max_steer: float = 0.5,       # rad (~29 deg)
        max_d_steer: float = 0.6,     # rad/s (steering rate)
        # cost weights: state = [along, cross, vx, psi, vy, r], ctrl = [a, delta]
        state_cost: tuple[float, ...] = (5, 150, 8, 40, 2, 2),
        final_state_cost: tuple[float, ...] = (5, 80, 8, 40, 2, 2),
        # ... existing code ...
        input_cost: tuple[float, ...] = (1, 5),
        
        # FIX: Increase steering rate penalty from 60 to 250. 
        # This prevents violent snap-backs and forces a smooth lane re-entry.
        input_rate_cost: tuple[float, ...] = (1, 250),
        
        safety_margin: float = 0.5,
        slack_penalty: float = 1e5,
        road_halfwidth: float = 5.0,
        # Lateral corridor: cross-track error up to this many metres is penalty-
        # FREE; only the excess beyond it is penalised (a soft deadband on the
        # cross term). 0.0 = strict centreline tracking (exact old behaviour).
        # Keep it well under road_halfwidth - obstacle clearance so the free band
        # cannot let the car drift into an obstacle uncosted.
        cross_band: float = 0.0,

        # FIX: Increase pass_zone from 6.0 to 18.0.
        # This prevents the MPC from looking past the obstacle and cutting in early.
        pass_zone: float = 18.0, 
    ) -> None:
        self.p = params or CarParams()
        self.dt = dt
        self.control_horizon = int(horizon_time / dt)

        self.width = width
        self.max_speed = max_speed
        self.max_acc = max_acc
        self.max_d_acc = max_d_acc
        self.max_steer = max_steer
        self.max_d_steer = max_d_steer
        self._road_halfwidth = road_halfwidth
        self._pass_zone = pass_zone
        self._last_obstacle_action = "none"
        self._obs_side_lock = None    # committed pass side; held until obstacle clears
        self._obs_locks = {}          # per-obstacle committed side, keyed by rounded pos

        self.q = np.diag(state_cost)
        self.qf = np.diag(final_state_cost)
        self.r = np.diag(input_cost)
        self._rr_sqrt = np.sqrt(np.asarray(input_rate_cost, dtype=float))

        # Lateral corridor (soft deadband on cross-track). When cross_band > 0 we
        # pull the cross weight OUT of the quadratic q and re-apply it only to the
        # part of the cross error that exceeds the band, via a hinge slack. This
        # keeps the problem convex and DPP. cross is state index 1.
        self._cross_band = float(cross_band)
        self._cross_w = float(state_cost[1])
        self._cross_wf = float(final_state_cost[1])
        if self._cross_band > 0.0:
            # zero the cross term in the quadratic blocks; it is handled by hinge
            self.q[1, 1] = 0.0
            self.qf[1, 1] = 0.0

        self._slack_penalty = slack_penalty
        self._vehicle_buffer = self.width / 2.0 + safety_margin

        N = self.control_horizon
        n, m = self.STATE_DIM, self.CONTROL_DIM
        J = self.MAX_OBS

        # --- decision variables ---
        self._states = opt.Variable((n, N + 1), name="states")
        self._controls = opt.Variable((m, N), name="controls")
        # one slack vector per obstacle slot
        self._slack = [opt.Variable(N, nonneg=True, name=f"slack{j}") for j in range(J)]
        self._err = opt.Variable((n, N + 1), name="track_error")
        # hinge slack for the lateral corridor: cross_excess_k >= |err_cross_k| - band
        self._cross_excess = opt.Variable(N + 1, nonneg=True, name="cross_excess")

        # --- parameters (all in Parameter-matrix @ Variable form for DPP) ---
        self._x0 = opt.Parameter(n, name="x0")
        self._last_u = opt.Parameter(m, name="last_u")
        self._A = [opt.Parameter((n, n)) for _ in range(N)]
        self._B = [opt.Parameter((n, m)) for _ in range(N)]
        self._C = [opt.Parameter(n) for _ in range(N)]
        self._M = [opt.Parameter((n, n)) for _ in range(N + 1)]
        self._ref = [opt.Parameter(n) for _ in range(N + 1)]
        # one half-plane (normal + offset) per slot per step
        self._obs_n = [[opt.Parameter(n) for _ in range(N)] for _ in range(J)]
        self._obs_safe = [opt.Parameter(N) for _ in range(J)]

        self._prev_traj: npt.NDArray | None = None
        self._prev_u: npt.NDArray | None = None

        self._problem = self._build_problem()

    # ------------------------------------------------------------------ #
    # The model layer
    # ------------------------------------------------------------------ #
    def _dynamics(self, X, u):
        """Continuous dynamic-bicycle dynamics (delegates to the shared fn)."""
        return vehicle_dynamics(X, u, self.p)

    def _model_matrices(self, x_bar, u_bar, eps=1e-6):
        """Numerically linearize, then EXACTLY discretize (ZOH) via expm.

        Numerical Jacobians keep the tire math in one place and obviously
        correct; exact discretization avoids the instability forward-Euler can
        cause on the stiff lateral dynamics. (Analytic Jacobians would be a
        speed optimization.)
        """
        n, m = self.STATE_DIM, self.CONTROL_DIM
        f0 = self._dynamics(x_bar, u_bar)
        A = np.zeros((n, n))
        B = np.zeros((n, m))
        for i in range(n):
            dx = x_bar.copy(); dx[i] += eps
            A[:, i] = (self._dynamics(dx, u_bar) - f0) / eps
        for j in range(m):
            du = u_bar.copy(); du[j] += eps
            B[:, j] = (self._dynamics(x_bar, du) - f0) / eps

        # affine residual so that x_dot ~= A x + B u + c around (x_bar, u_bar)
        c = f0 - A @ x_bar - B @ u_bar

        # exact ZOH discretization of the affine system via one matrix exponential
        aug = np.zeros((n + m + 1, n + m + 1))
        aug[:n, :n] = A
        aug[:n, n:n + m] = B
        aug[:n, n + m] = c
        Mexp = expm(aug * self.dt)
        A_d = Mexp[:n, :n]
        B_d = Mexp[:n, n:n + m]
        C_d = Mexp[:n, n + m]
        return A_d, B_d, C_d

    # ------------------------------------------------------------------ #
    def _build_problem(self) -> opt.Problem:
        cost = 0
        constraints = []
        N = self.control_horizon
        use_band = self._cross_band > 0.0

        # err_k == M_k @ state_k - ref_k  (DPP-blessed parameter @ variable)
        for k in range(N + 1):
            constraints += [self._err[:, k] == self._M[k] @ self._states[:, k]
                            - self._ref[k]]
            if use_band:
                # cross_excess_k >= |err_cross_k| - band, cross_excess_k >= 0
                # (so error within +/-band is free; only the excess is penalised)
                constraints += [
                    self._cross_excess[k] >= self._err[1, k] - self._cross_band,
                    self._cross_excess[k] >= -self._err[1, k] - self._cross_band,
                ]

        for k in range(N):
            constraints += [
                self._states[:, k + 1]
                == self._A[k] @ self._states[:, k]
                + self._B[k] @ self._controls[:, k]
                + self._C[k]
            ]
            cost += opt.quad_form(self._err[:, k], self.q)
            if use_band:
                cost += self._cross_w * opt.square(self._cross_excess[k])

            # one soft half-plane keep-out per obstacle slot
            for j in range(self.MAX_OBS):
                constraints += [
                    self._obs_n[j][k] @ self._states[:, k + 1]
                    >= self._obs_safe[j][k] - self._slack[j][k]
                ]
                cost += self._slack_penalty * self._slack[j][k]

            cost += opt.quad_form(self._controls[:, k], self.r)
            if k == 0:
                cost += opt.sum_squares(
                    opt.multiply(self._rr_sqrt, self._controls[:, 0] - self._last_u)
                )
            else:
                cost += opt.sum_squares(
                    opt.multiply(
                        self._rr_sqrt, self._controls[:, k] - self._controls[:, k - 1]
                    )
                )

        cost += opt.quad_form(self._err[:, -1], self.qf)
        if use_band:
            cost += self._cross_wf * opt.square(self._cross_excess[-1])

        constraints += [self._states[:, 0] == self._x0]
        constraints += [opt.abs(self._states[3, :]) <= self.max_speed]   # |vx|
        constraints += [opt.abs(self._controls[0, :]) <= self.max_acc]   # |a|
        constraints += [opt.abs(self._controls[1, :]) <= self.max_steer]  # |delta|
        constraints += [
            opt.abs(self._controls[0, 0] - self._last_u[0]) / self.dt <= self.max_d_acc
        ]
        constraints += [
            opt.abs(self._controls[1, 0] - self._last_u[1]) / self.dt
            <= self.max_d_steer
        ]
        for k in range(1, N):
            constraints += [
                opt.abs(self._controls[0, k] - self._controls[0, k - 1]) / self.dt
                <= self.max_d_acc
            ]
            constraints += [
                opt.abs(self._controls[1, k] - self._controls[1, k - 1]) / self.dt
                <= self.max_d_steer
            ]

        return opt.Problem(opt.Minimize(cost), constraints)

    # ------------------------------------------------------------------ #
    def _try_solve(self) -> bool:
        """Solve robustly: OSQP warm -> CLARABEL cold -> OSQP cold. OSQP warm-
        started is the fast path (~70 ms); it can occasionally return an
        inaccurate-but-usable solution, which we accept. A cold CLARABEL / OSQP
        retry recovers the rare hard case. Returns True if a usable solution was
        found. (Genuine failures are handled gracefully by the bridge, which
        holds the last good command rather than emergency-braking.)"""
        attempts = [
            dict(solver=opt.OSQP, warm_start=True, enforce_dpp=True, polish=True,
                 eps_abs=1e-5, eps_rel=1e-5, max_iter=8000),
            dict(solver=opt.CLARABEL, warm_start=False, enforce_dpp=True),
            dict(solver=opt.OSQP, warm_start=False, enforce_dpp=True, max_iter=8000),
        ]
        for kw in attempts:
            try:
                self._problem.solve(**kw)
                v = self._states.value
                # accept only a finite, non-runaway solution
                if v is not None and np.all(np.isfinite(v)) and np.max(np.abs(v)) < 1e5:
                    return True
            except (opt.error.SolverError, Exception):
                continue
        return False

    # ------------------------------------------------------------------ #
    def _decide_one(self, obstacle, x_ref, y_ref, psi_ref, lock_key):
        """Decide pass-left / pass-right / stop for ONE obstacle.

        Returns a decision dict. Uses self._obs_locks[lock_key] to commit to a
        pass side so the choice can't flip mid-manoeuvre.
        """
        ox, oy, orad = obstacle[0], obstacle[1], obstacle[2]
        ovx = obstacle[3] if len(obstacle) > 3 else 0.0
        ovy = obstacle[4] if len(obstacle) > 4 else 0.0
        keep = orad + self._vehicle_buffer
        dref = np.hypot(x_ref - ox, y_ref - oy)
        kstar = int(np.argmin(dref))
        ths = psi_ref[kstar]
        txs, tys = np.cos(ths), np.sin(ths)
        lxs, lys = -np.sin(ths), np.cos(ths)
        lateral_obs = (ox - x_ref[kstar]) * lxs + (oy - y_ref[kstar]) * lys
        room_left = self._road_halfwidth - (lateral_obs + keep)
        room_right = self._road_halfwidth + (lateral_obs - keep)
        can_pass = max(room_left, room_right) >= 0.0

        if not can_pass:
            self._obs_locks.pop(lock_key, None)
            pass_left = False
        else:
            if lock_key not in self._obs_locks:
                self._obs_locks[lock_key] = (
                    "pass_left" if room_left >= room_right else "pass_right")
            pass_left = (self._obs_locks[lock_key] == "pass_left")

        return dict(ox=ox, oy=oy, orad=orad, ovx=ovx, ovy=ovy, keep=keep,
                    txs=txs, tys=tys, can_pass=can_pass, pass_left=pass_left)

    def _fill_halfplane(self, decision, x_ref, y_ref, psi_ref):
        """Build (obs_n[N,n], obs_safe[N]) keep-out half-planes for one decision.
        A None decision -> trivially-satisfied (inactive) constraint."""
        N, n = self.control_horizon, self.STATE_DIM
        obs_n = np.zeros((N, n))
        obs_safe = np.full(N, -1e6)
        if decision is None:
            obs_n[:, 0] = 1.0
            return obs_n, obs_safe

        ox, oy = decision["ox"], decision["oy"]
        ovx, ovy, keep = decision["ovx"], decision["ovy"], decision["keep"]
        can_pass, pass_left = decision["can_pass"], decision["pass_left"]
        activation = decision["orad"] + self._pass_zone
        for k in range(N):
            oxk = ox + ovx * k * self.dt
            oyk = oy + ovy * k * self.dt
            thk = psi_ref[k + 1]
            txk, tyk = np.cos(thk), np.sin(thk)
            lxk, lyk = -np.sin(thk), np.cos(thk)
            if not can_pass:
                obs_n[k, 0], obs_n[k, 1] = -txk, -tyk
                obs_safe[k] = -(txk * oxk + tyk * oyk) + keep
                continue
            s_gap = (x_ref[k + 1] - oxk) * txk + (y_ref[k + 1] - oyk) * tyk
            if abs(s_gap) < activation:
                if pass_left:
                    obs_n[k, 0], obs_n[k, 1] = lxk, lyk
                    obs_safe[k] = lxk * oxk + lyk * oyk + keep
                else:
                    obs_n[k, 0], obs_n[k, 1] = -lxk, -lyk
                    obs_safe[k] = -(lxk * oxk + lyk * oyk) + keep
            else:
                obs_n[k, 0] = 1.0
                obs_safe[k] = -1e6
        return obs_n, obs_safe

    # ------------------------------------------------------------------ #
    def solve(self, initial_state, target, obstacle=None, obstacles=None,
              max_iter=3, tolerance=1e-2, global_pose=None):
        """iMPC solve. target rows = [x_ref, y_ref, vx_ref, psi_ref] (ego frame).

        obstacle, if given, is (x, y, radius, vx, vy) in the EGO frame.
        Returns (trajectory, controls); on solver failure (None, brake).
        """
        N = self.control_horizon
        n = self.STATE_DIM
        self._x0.value = np.asarray(initial_state, dtype=float)
        self._last_u.value = (
            self._prev_u[:, 0] if self._prev_u is not None
            else np.zeros(self.CONTROL_DIM)
        )

        x_ref, y_ref = np.array(target[0], float), np.array(target[1], float)
        vx_ref, psi_ref = np.array(target[2], float), target[3]

        # ---- normalize obstacles into a list (nearest first) ----
        obs_list = []
        if obstacles:
            obs_list = list(obstacles)
        elif obstacle is not None:
            obs_list = [obstacle]
        # sort by ego-frame distance (closest first), keep up to MAX_OBS
        obs_list.sort(key=lambda o: float(np.hypot(o[0], o[1])))
        obs_list = obs_list[:self.MAX_OBS]

        self._last_obstacle_action = "none"

        # ---- decision per obstacle; nearest one drives stop / speed clamp ----
        decisions = []
        active_keys = set()
        for j, ob in enumerate(obs_list):
            
            # FIX 1: Convert obstacle to global coordinates so the lock is stable.
            if global_pose is not None:
                c_psi, s_psi = math.cos(global_pose[2]), math.sin(global_pose[2])
                ox_g = global_pose[0] + float(ob[0]) * c_psi - float(ob[1]) * s_psi
                oy_g = global_pose[1] + float(ob[0]) * s_psi + float(ob[1]) * c_psi
                key = (round(ox_g / 3.0), round(oy_g / 3.0))
            else:
                key = (round(float(ob[0]) / 3.0), round(float(ob[1]) / 3.0))
                
            active_keys.add(key)
            dec = self._decide_one(ob, x_ref, y_ref, psi_ref, key)
            decisions.append(dec)
        # release locks for obstacles no longer present
        for k_ in list(self._obs_locks.keys()):
            if k_ not in active_keys:
                self._obs_locks.pop(k_, None)
        if not obs_list:
            self._obs_locks.clear()
        self._obs_side_lock = None  # legacy field kept for callers; unused here

        if decisions:
            nearest = decisions[0]
            self._last_obstacle_action = (
                ("pass_left" if nearest["pass_left"] else "pass_right")
                if nearest["can_pass"] else "stop")
            if not nearest["can_pass"]:
                # Stop: clamp position reference to a stop line behind the
                # nearest obstacle and zero the speed target.
                ox, oy = nearest["ox"], nearest["oy"]
                txs, tys, keep = nearest["txs"], nearest["tys"], nearest["keep"]
                stop_x = ox - txs * keep
                stop_y = oy - tys * keep
                for k in range(len(x_ref)):
                    s_k = (x_ref[k] - ox) * txs + (y_ref[k] - oy) * tys
                    if s_k > -keep:
                        x_ref[k], y_ref[k] = stop_x, stop_y
                vx_ref = np.zeros_like(vx_ref)

        cos_v, sin_v = np.cos(psi_ref), np.sin(psi_ref)
        along_ref = cos_v * x_ref + sin_v * y_ref
        cross_ref = -sin_v * x_ref + cos_v * y_ref

        # reference params -> set once
        for k in range(N + 1):
            c, s = cos_v[k], sin_v[k]
            M = np.zeros((n, n))
            M[0, 0], M[0, 1] = c, s          # along-track
            M[1, 0], M[1, 1] = -s, c         # cross-track
            M[2, 3] = 1.0                    # vx
            M[3, 2] = 1.0                    # psi
            M[4, 4] = 1.0                    # vy (regularized to 0)
            M[5, 5] = 1.0                    # r  (regularized to 0)
            self._M[k].value = M
            self._ref[k].value = np.array(
                [along_ref[k], cross_ref[k], vx_ref[k], psi_ref[k], 0.0, 0.0]
            )

        # fill each obstacle slot's half-plane; unused slots are inactive
        for j in range(self.MAX_OBS):
            dec = decisions[j] if j < len(decisions) else None
            obs_n_j, obs_safe_j = self._fill_halfplane(dec, x_ref, y_ref, psi_ref)
            for k in range(N):
                self._obs_n[j][k].value = obs_n_j[k]
            self._obs_safe[j].value = obs_safe_j

        # warm-start guess for the iMPC loop
        if self._prev_traj is not None and self._prev_u is not None:
            x_guess = np.roll(self._prev_traj, -1, axis=1)
            x_guess[:, -1] = self._prev_traj[:, -1]
            u_guess = np.roll(self._prev_u, -1, axis=1)
            u_guess[:, -1] = self._prev_u[:, -1]
        else:
            # FIX 2: Shift the geometric path so it starts exactly at the car's 
            # current position (0,0 in ego frame). This stops the solver from 
            # crashing if the car is pushed far off the reference line.
            x_guess = np.zeros((n, N + 1))
            x_guess[0, :] = x_ref - x_ref[0]
            x_guess[1, :] = y_ref - y_ref[0]
            x_guess[2, :] = psi_ref
            x_guess[3, :] = vx_ref
            u_guess = np.zeros((self.CONTROL_DIM, N))

        new_x = new_u = None
        for _ in range(max_iter):
            for k in range(N):
                A_k, B_k, C_k = self._model_matrices(x_guess[:, k], u_guess[:, k])
                self._A[k].value = A_k
                self._B[k].value = B_k
                self._C[k].value = C_k

            ok = self._try_solve()
            if not ok:  # solver failed even after fallbacks -> emergency brake
                brake = np.zeros((self.CONTROL_DIM, N))
                vx = initial_state[3]
                for k in range(N):
                    acc = -self.max_acc if vx > 0 else 0.0
                    brake[0, k] = acc
                    vx = max(0.0, vx + acc * self.dt)
                # clear the warm start so a bad solve can't poison the next cycle
                self._prev_traj = None
                self._prev_u = brake.copy()
                return None, brake

            new_x = np.asarray(self._states.value)
            new_u = np.asarray(self._controls.value)
            if np.max(np.abs(new_x - x_guess)) < tolerance:
                break
            x_guess, u_guess = new_x, new_u

        self._prev_traj = new_x.copy()
        self._prev_u = new_u.copy()
        return new_x, new_u


# ====================================================================== #
#  Path / reference helpers
# ====================================================================== #
from scipy.interpolate import splev, splprep


def compute_path_from_wp(xs, ys, step: float = 0.1) -> npt.NDArray:
    """Smooth path through waypoints. Returns (3, M): rows [x, y, heading]."""
    tck, _ = splprep([xs, ys], s=0.0)
    u_fine = np.linspace(0, 1, 4000)
    xf, yf = splev(u_fine, tck)
    arc = np.zeros(len(u_fine))
    arc[1:] = np.cumsum(np.hypot(np.diff(xf), np.diff(yf)))
    n = int(arc[-1] / step)
    u_uni = np.interp(np.linspace(0, arc[-1], n), arc, u_fine)
    fx, fy = splev(u_uni, tck)
    theta = np.arctan2(np.gradient(fy), np.gradient(fx))
    return np.vstack((fx, fy, theta))


def _nn_idx(state, path) -> int:
    d = np.hypot(state[0] - path[0], state[1] - path[1])
    i = int(np.argmin(d))
    if i + 1 < path.shape[1]:
        seg = np.array([path[0, i + 1] - path[0, i], path[1, i + 1] - path[1, i]])
        if np.linalg.norm(seg) > 0:
            seg /= np.linalg.norm(seg)
            to_pt = np.array([path[0, i] - state[0], path[1, i] - state[1]])
            return i if np.dot(to_pt, seg) > 0 else i + 1
    return i


def get_ref_trajectory(state, path, target_v, T, DT, ego_frame: bool = True):
    """Sample a (4, K+1) reference [x, y, vx, psi] over the horizon."""
    K = int(T / DT)
    xref = np.zeros((4, K + 1))
    pose = np.array([state[0], state[1], state[2]])  # x, y, psi
    ind = _nn_idx(pose, path)
    
    # --- FIX: RECOVERY MODE ---
    # Calculate how far off the road the car currently is.
    dx = pose[0] - path[0, ind]
    dy = pose[1] - path[1, ind]
    cross_err = math.hypot(dx, dy)
    
    # If pushed far off the path, throttle the ghost car's target speed.
    # This prevents the solver from dying by giving it time to steer 
    # back to the road gently.
    if cross_err > 1.5:
        target_v = min(target_v, 2.5)
    # --------------------------

    cdist = np.append([0.0], np.cumsum(np.hypot(np.diff(path[0]), np.diff(path[1]))))
    start = cdist[ind]

    # FIX 1: Generate a dynamically feasible distance profile.
    v_curr = state[3]
    pts = [start]
    s_ref = start
    v_ref = v_curr
    v_profile = [v_curr]

    for _ in range(K):
        if v_ref < target_v:
            v_ref = min(target_v, v_ref + 3.0 * DT)  # smooth acceleration limit
        elif v_ref > target_v:
            v_ref = max(target_v, v_ref - 5.0 * DT)  # smooth braking limit
        
        s_ref += max(0.0, v_ref) * DT
        pts.append(s_ref)
        v_profile.append(max(0.0, v_ref))

    xref[0] = np.interp(pts, cdist, path[0])
    xref[1] = np.interp(pts, cdist, path[1])
    xref[2] = np.array(v_profile)
    xref[3] = np.interp(pts, cdist, path[2])

    if ego_frame:
        dx = xref[0] - pose[0]
        dy = xref[1] - pose[1]
        xref[0] = dx * np.cos(-pose[2]) - dy * np.sin(-pose[2])
        xref[1] = dy * np.cos(-pose[2]) + dx * np.sin(-pose[2])
        xref[3] = xref[3] - pose[2]

    xref[3] = (xref[3] + np.pi) % (2 * np.pi) - np.pi
    xref[3] = xref[3, 0] + np.unwrap(xref[3] - xref[3, 0])
    return xref


def ego_to_global(state, traj_ego):
    """Map a (>=2, N) ego-frame trajectory back to global XY. Returns (2, N)."""
    ct, st = np.cos(state[2]), np.sin(state[2])
    R = np.array([[ct, -st], [st, ct]])
    g = R @ traj_ego[:2]
    g[0] += state[0]
    g[1] += state[1]
    return g


class AdaptiveSpeed:
    """
    Gives the MPC human-like speed behaviour by modulating the target speed
    reference based on what a driver would naturally do:

      1. OBSTACLE APPROACH  - smooth deceleration as an obstacle enters sensor
         range, proportional to proximity. Braking starts at obs_brake_start
         metres and reaches v_min at obs_brake_end metres.

      2. CORNER SLOWDOWN - looks ahead lookahead_s metres along the path,
         finds the tightest curvature, and sets a speed limit so lateral
         acceleration stays within a_lat_comfort. Speed rises automatically on
         straights and drops into corners, exactly like a driver reading ahead.

      3. SMOOTH TRANSITIONS - a first-order low-pass filter (time-constant tau)
         prevents sudden speed jumps; the car accelerates and brakes gradually.

    Usage (replaces constant target_v in the main loop):
        speed_ctrl = AdaptiveSpeed(base_speed=target_v)
        adaptive_v = speed_ctrl.update(state, path, ego_obs, dt=C.DT)
        tgt = get_ref_trajectory(state, path, adaptive_v, ...)
    """

    def __init__(
        self,
        base_speed:      float = 11.0,
        a_lat_comfort:   float =  4.5,   # m/s^2 comfortable lateral accel (CARLA passes 1.5)
        lookahead_s:     float = 25.0,   # metres to scan ahead for corners
        obs_brake_start: float = 35.0,   # metres: start braking for obstacle
        obs_brake_end:   float =  5.0,   # metres: v_min reached here
        v_min:           float =  4.0,   # m/s: ~14 km/h - keeps 12 m of MPC horizon
        tau:             float =  1.2,   # s: smoothing time constant
    ):
        self.base            = base_speed
        self.a_lat           = a_lat_comfort
        self.lookahead       = lookahead_s
        self.obs_brake_start = obs_brake_start
        self.obs_brake_end   = obs_brake_end
        self.v_min           = v_min
        self.tau             = tau
        self._v_smooth       = base_speed
        self.reason          = "FREE"   # for HUD display

    def update(self, state, path, obstacle_ego, dt: float) -> float:
        """Call once per control tick. Returns adaptive target speed [m/s]."""
        v_target = self.base
        reasons  = []

        # 1. obstacle proximity
        if obstacle_ego is not None:
            d = math.hypot(float(obstacle_ego[0]), float(obstacle_ego[1])) \
                - float(obstacle_ego[2])
            d = max(d, 0.)
            if d < self.obs_brake_start:
                t = max(0., (d - self.obs_brake_end)
                           / (self.obs_brake_start - self.obs_brake_end))
                v_obs = self.v_min + (self.base - self.v_min) * t
                if v_obs < v_target:
                    v_target = v_obs
                    reasons.append("OBS")

        # 2. path curvature ahead
        i0 = _nn_idx(state[:3], path)
        s, i, n, kmax = 0., i0, path.shape[1], 0.
        while s < self.lookahead and i < n - 2:
            dx = path[0,i+1]-path[0,i]; dy = path[1,i+1]-path[1,i]
            ds = math.hypot(dx, dy)
            dth = abs((path[2,i+1]-path[2,i]+math.pi) % (2*math.pi) - math.pi)
            if ds > 1e-6:
                kmax = max(kmax, dth/ds)
            s += ds; i += 1

        if kmax > 1e-4:
            v_corner = max(math.sqrt(self.a_lat / kmax), self.v_min)
            if v_corner < v_target:
                v_target = v_corner
                reasons.append("CORNER")

        # 3. smooth low-pass
        alpha = dt / (self.tau + dt)
        self._v_smooth += alpha * (v_target - self._v_smooth)
        self.reason = "+".join(reasons) if reasons else "FREE"
        return max(self.v_min, self._v_smooth)

    def reset(self, speed: float = None):
        self._v_smooth = speed if speed is not None else self.base
        self.reason = "FREE"

    def set_base(self, speed: float):
        self.base = speed

# ============================================================================ #
#  Config-driven factories: one place that maps config.py to the constructors,
#  so every entry point (sim, viewers, CARLA) builds the controller and speed
#  governor identically from the central config. All values use getattr with the
#  controller's own defaults as fallback, so an older/trimmed config still works.
# ============================================================================ #
def build_mpc(C, dt=None, horizon_time=None):
    """Construct a DynamicBicycleMPC from a config module C."""
    p = CarParams()
    for k, v in C.CAR.items():
        setattr(p, k, v)
    return DynamicBicycleMPC(
        params=p,
        dt=dt if dt is not None else C.DT,
        horizon_time=horizon_time if horizon_time is not None else C.HORIZON_TIME,
        max_speed=getattr(C, "MAX_SPEED", 30.0),
        max_acc=getattr(C, "MAX_ACC", 4.0),
        max_d_acc=getattr(C, "MAX_D_ACC", 6.0),
        max_steer=getattr(C, "MAX_STEER", 0.5),
        max_d_steer=getattr(C, "MAX_D_STEER", 0.6),
        state_cost=getattr(C, "STATE_COST", (5, 150, 8, 40, 2, 2)),
        final_state_cost=getattr(C, "FINAL_STATE_COST", (5, 80, 8, 40, 2, 2)),
        input_cost=getattr(C, "INPUT_COST", (1, 5)),
        input_rate_cost=getattr(C, "INPUT_RATE_COST", (1, 250)),
        safety_margin=getattr(C, "OBSTACLE_SAFETY_MARGIN", 1.0),
        slack_penalty=getattr(C, "SLACK_PENALTY", 1e5),
        road_halfwidth=getattr(C, "ROAD_HALFWIDTH", 5.0),
        cross_band=getattr(C, "CROSS_BAND", 0.0),
        pass_zone=getattr(C, "PASS_ZONE", 6.0),
    )


def build_adaptive_speed(C, base_speed=None, carla=False):
    """Construct an AdaptiveSpeed governor from a config module C.

    carla=True selects the CARLA-specific tuning (separate from the pygame
    values) so the two front-ends can behave differently.
    """
    if carla:
        return AdaptiveSpeed(
            base_speed=base_speed if base_speed is not None
            else getattr(C, "CARLA_TARGET_SPEED", 7.0),
            a_lat_comfort=getattr(C, "CARLA_A_LAT_COMFORT", 1.5),
            lookahead_s=getattr(C, "ADAPT_LOOKAHEAD", 25.0),
            obs_brake_start=getattr(C, "CARLA_OBS_BRAKE_START", 25.0),
            obs_brake_end=getattr(C, "CARLA_OBS_BRAKE_END", 5.0),
            v_min=getattr(C, "CARLA_V_MIN", 3.0),
            tau=getattr(C, "ADAPT_TAU", 1.2),
        )
    return AdaptiveSpeed(
        base_speed=base_speed if base_speed is not None
        else getattr(C, "TARGET_SPEED", 11.0),
        a_lat_comfort=getattr(C, "ADAPT_A_LAT_COMFORT", 4.5),
        lookahead_s=getattr(C, "ADAPT_LOOKAHEAD", 25.0),
        obs_brake_start=getattr(C, "ADAPT_OBS_BRAKE_START", 35.0),
        obs_brake_end=getattr(C, "ADAPT_OBS_BRAKE_END", 5.0),
        v_min=getattr(C, "ADAPT_V_MIN", 4.0),
        tau=getattr(C, "ADAPT_TAU", 1.2),
    )