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

import numpy as np
import numpy.typing as npt
import cvxpy as opt
from scipy.linalg import expm


def vehicle_dynamics(X, u, p):
    """Continuous dynamic-bicycle dynamics f(X, u) for car parameters `p`.

    X = [x, y, psi, vx, vy, r], u = [a, delta]. Reused by the controller's
    linearization and by the open-loop simulator (manual_drive.py)."""
    _, _, psi, vx, vy, r = X
    a, delta = u
    vxs = vx if abs(vx) > p.v_min else (p.v_min if vx >= 0 else -p.v_min)
    alpha_f = delta - (vy + p.lf * r) / vxs
    alpha_r = -(vy - p.lr * r) / vxs
    Fyf = p.Cf * alpha_f
    Fyr = p.Cr * alpha_r
    return np.array([
        vx * np.cos(psi) - vy * np.sin(psi),
        vx * np.sin(psi) + vy * np.cos(psi),
        r,
        a + vy * r,
        (Fyf + Fyr) / p.m - vx * r,
        (p.lf * Fyf - p.lr * Fyr) / p.Iz,
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

    @property
    def wheelbase(self):
        return self.lf + self.lr


class DynamicBicycleMPC:
    STATE_DIM = 6     # [x, y, psi, vx, vy, r]
    CONTROL_DIM = 2   # [a, delta]

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
        state_cost: tuple[float, ...] = (5, 80, 8, 40, 2, 2),
        final_state_cost: tuple[float, ...] = (5, 80, 8, 40, 2, 2),
        input_cost: tuple[float, ...] = (1, 5),
        input_rate_cost: tuple[float, ...] = (1, 60),
        safety_margin: float = 0.5,
        slack_penalty: float = 1e5,
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

        self.q = np.diag(state_cost)
        self.qf = np.diag(final_state_cost)
        self.r = np.diag(input_cost)
        self._rr_sqrt = np.sqrt(np.asarray(input_rate_cost, dtype=float))

        self._slack_penalty = slack_penalty
        self._vehicle_buffer = self.width / 2.0 + safety_margin

        N = self.control_horizon
        n, m = self.STATE_DIM, self.CONTROL_DIM

        # --- decision variables ---
        self._states = opt.Variable((n, N + 1), name="states")
        self._controls = opt.Variable((m, N), name="controls")
        self._slack = opt.Variable(N, nonneg=True, name="obstacle_slack")
        self._err = opt.Variable((n, N + 1), name="track_error")

        # --- parameters (all in Parameter-matrix @ Variable form for DPP) ---
        self._x0 = opt.Parameter(n, name="x0")
        self._last_u = opt.Parameter(m, name="last_u")
        self._A = [opt.Parameter((n, n)) for _ in range(N)]
        self._B = [opt.Parameter((n, m)) for _ in range(N)]
        self._C = [opt.Parameter(n) for _ in range(N)]
        self._M = [opt.Parameter((n, n)) for _ in range(N + 1)]
        self._ref = [opt.Parameter(n) for _ in range(N + 1)]
        self._obs_n = [opt.Parameter(n) for _ in range(N)]
        self._obs_safe = opt.Parameter(N)

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

        # err_k == M_k @ state_k - ref_k  (DPP-blessed parameter @ variable)
        for k in range(N + 1):
            constraints += [self._err[:, k] == self._M[k] @ self._states[:, k]
                            - self._ref[k]]

        for k in range(N):
            constraints += [
                self._states[:, k + 1]
                == self._A[k] @ self._states[:, k]
                + self._B[k] @ self._controls[:, k]
                + self._C[k]
            ]
            cost += opt.quad_form(self._err[:, k], self.q)

            constraints += [
                self._obs_n[k] @ self._states[:, k + 1]
                >= self._obs_safe[k] - self._slack[k]
            ]
            cost += self._slack_penalty * self._slack[k]

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
        """Solve robustly: CLARABEL warm -> CLARABEL cold -> OSQP. The QP can
        occasionally fail numerically on a warm start near tight turns; a cold
        re-solve or a second solver almost always recovers, which is gentler
        than an emergency brake. Returns True if a usable solution was found."""
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
    def solve(self, initial_state, target, obstacle=None, max_iter=3, tolerance=1e-2):
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

        x_ref, y_ref = target[0], target[1]
        vx_ref, psi_ref = target[2], target[3]
        cos_v, sin_v = np.cos(psi_ref), np.sin(psi_ref)
        along_ref = cos_v * x_ref + sin_v * y_ref
        cross_ref = -sin_v * x_ref + cos_v * y_ref

        # reference params (depend only on the path) -> set once
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

        if obstacle is not None:
            ox, oy, orad, ovx, ovy = obstacle
        else:
            ox = oy = orad = ovx = ovy = None

        obs_n = np.zeros((N, n))
        obs_safe = np.zeros(N)
        for k in range(N):
            if obstacle is None:
                obs_n[k, 0] = 1.0
                obs_safe[k] = -1e6
            else:
                # Half-plane tangent to the keep-out disc, oriented from the
                # obstacle toward the reference path. This points longitudinally
                # when the car is still approaching and rotates to lateral as it
                # comes alongside, giving a smooth avoidance bulge. (Assumes the
                # obstacle is offset from the centreline; an obstacle dead-centre
                # on the path needs the lane-change variant -- see README.)
                dx = x_ref[k + 1] - ox
                dy = y_ref[k + 1] - oy
                d = max(np.hypot(dx, dy), 1e-3)
                nx, ny = dx / d, dy / d
                obs_n[k, 0], obs_n[k, 1] = nx, ny
                obs_safe[k] = nx * ox + ny * oy + orad + self._vehicle_buffer
        for k in range(N):
            self._obs_n[k].value = obs_n[k]
        self._obs_safe.value = obs_safe

        # warm-start guess for the iMPC loop
        if self._prev_traj is not None and self._prev_u is not None:
            x_guess = np.roll(self._prev_traj, -1, axis=1)
            x_guess[:, -1] = self._prev_traj[:, -1]
            u_guess = np.roll(self._prev_u, -1, axis=1)
            u_guess[:, -1] = self._prev_u[:, -1]
        else:
            x_guess = np.tile(np.asarray(initial_state, dtype=float).reshape(n, 1),
                              (1, N + 1))
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
    """Sample a (4, K+1) reference [x, y, vx, psi] over the horizon.

    `state` here is the GLOBAL pose [x, y, psi, ...]; only x,y,psi are used.
    """
    K = int(T / DT)
    xref = np.zeros((4, K + 1))
    pose = np.array([state[0], state[1], state[2]])  # x, y, psi
    ind = _nn_idx(pose, path)
    cdist = np.append([0.0], np.cumsum(np.hypot(np.diff(path[0]), np.diff(path[1]))))
    start = cdist[ind]
    pts = [d * DT * target_v + start for d in range(K + 1)]
    xref[0] = np.interp(pts, cdist, path[0])
    xref[1] = np.interp(pts, cdist, path[1])
    xref[2] = target_v
    xref[3] = np.interp(pts, cdist, path[2])
    reached = np.where(np.interp(pts, cdist, cdist) >= cdist[-1] - 1e-9)
    xref[2, reached] = 0.0

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
