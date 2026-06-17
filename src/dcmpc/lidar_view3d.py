"""
3D semantic point-cloud viewer for the CARLA virtual LiDAR.

Renders the live point cloud in a real 3D perspective window where each detected
cluster gets its own distinct colour (cars, poles, scenery), the ground is dimmed,
and the ego vehicle sits at the origin as a forward-pointing arrowhead.

Two backends, picked automatically:
  Open3D  (preferred) - true GPU 3D window with mouse orbit/zoom.
    Install with: pip install open3d
  matplotlib 3D (fallback) - works with no extra install, lower fidelity.

This module is PERCEPTION-VISUALISATION ONLY. It does its own clustering purely
for colouring and never feeds the controller, so it cannot affect driving.
"""

from __future__ import annotations
import math
import numpy as np


# ── distinct colour palette for clusters (RGB 0-1) ──────────────────────────
_PALETTE = np.array([
    [0.95, 0.20, 0.20],   # red
    [0.20, 0.85, 0.30],   # green
    [0.25, 0.55, 1.00],   # blue
    [1.00, 0.65, 0.10],   # orange
    [0.85, 0.30, 0.95],   # purple
    [0.10, 0.90, 0.90],   # cyan
    [1.00, 0.85, 0.20],   # yellow
    [1.00, 0.40, 0.70],   # pink
    [0.55, 0.85, 0.30],   # lime
    [0.50, 0.60, 1.00],   # periwinkle
], dtype=np.float32)

_GROUND_COL = np.array([0.16, 0.18, 0.22], dtype=np.float32)   # dim grey-blue
_DROP_COL   = np.array([0.10, 0.11, 0.13], dtype=np.float32)   # near-black


def _grid_clusters(xy, cell, min_pts):
    """Lightweight grid + 8-neighbour flood-fill clustering for colouring.
    Returns an int label per row of xy (-1 = unclustered)."""
    n = xy.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return labels
    keys = np.floor(xy / cell).astype(np.int64)
    buckets = {}
    for i, k in enumerate(map(tuple, keys)):
        buckets.setdefault(k, []).append(i)
    seen, cid = set(), 0
    for key in buckets:
        if key in seen:
            continue
        stack, comp = [key], []
        while stack:
            c = stack.pop()
            if c in seen or c not in buckets:
                continue
            seen.add(c); comp.extend(buckets[c])
            cx, cy = c
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (cx + dx, cy + dy)
                    if nb in buckets and nb not in seen:
                        stack.append(nb)
        if len(comp) >= min_pts:
            for i in comp:
                labels[i] = cid
            cid += 1
    return labels


def _colourise(pts4, z_min, z_max, fov_deg, ego_excl, max_range, cell, min_pts,
               max_radius):
    """Return (xyz, rgb) for the whole cloud: ground dim, clusters coloured."""
    xyz = pts4[:, :3].copy()
    xyz[:, 1] *= -1.0   # flip CARLA y to our convention
    rgb = np.tile(_DROP_COL, (xyz.shape[0], 1))

    z = pts4[:, 2]
    on_band = (z > z_min) & (z < z_max)

    bx, by = xyz[:, 0], xyz[:, 1]
    dist = np.hypot(bx, by)
    bearing = np.arctan2(by, bx)
    fwd = (bx > 0) & (dist <= max_range) & (dist >= ego_excl) & \
          (np.abs(bearing) <= math.radians(fov_deg / 2))
    sel = on_band & fwd

    rgb[on_band & ~fwd] = _GROUND_COL

    idx = np.where(sel)[0]
    if idx.size:
        labels = _grid_clusters(xyz[idx, :2], cell, min_pts)
        for li in range(labels.max() + 1 if labels.size else 0):
            members = idx[labels == li]
            if members.size == 0:
                continue
            cx = xyz[members, 0].mean(); cy = xyz[members, 1].mean()
            spread = np.max(np.hypot(xyz[members, 0] - cx, xyz[members, 1] - cy))
            col = _PALETTE[li % len(_PALETTE)]
            rgb[members] = col if spread <= max_radius else _GROUND_COL
        rgb[idx[labels == -1]] = _GROUND_COL * 1.4
    return xyz.astype(np.float32), rgb.astype(np.float32)


# ============================================================================ #
#  Open3D backend (preferred)
# ============================================================================ #
class _Open3DView:
    def __init__(self, max_range, fov_deg, cfg):
        import open3d as o3d
        self.o3d = o3d
        self.cfg = cfg
        self.max_range = max_range
        self.fov_deg = fov_deg
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("dcmpc - LiDAR 3D - semantic clusters",
                               width=1280, height=720)
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0, 0, 0], dtype=np.float64)
        opt.point_size = 2.0
        self.pcd = o3d.geometry.PointCloud()
        self._init = False

        # Ego marker: arrow pointing forward (+x = car heading).
        # create_arrow points along +Z by default, so rotate 90 deg around Y to +X.
        self.ego = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.3,
            cone_radius=0.7,
            cylinder_height=3.0,
            cone_height=1.5,
        )
        R = self.ego.get_rotation_matrix_from_xyz((0, math.pi / 2, 0))
        self.ego.rotate(R, center=(0, 0, 0))
        self.ego.paint_uniform_color([0.0, 0.83, 1.0])   # cyan
        self.vis.add_geometry(self.ego)

    def update(self, xyz, rgb, obstacles):
        self.pcd.points = self.o3d.utility.Vector3dVector(xyz)
        self.pcd.colors = self.o3d.utility.Vector3dVector(rgb)
        if not self._init:
            self.vis.add_geometry(self.pcd)
            self._init = True
        else:
            self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        try:
            self.vis.destroy_window()
        except Exception:
            pass


# ============================================================================ #
#  matplotlib 3D backend (fallback)
# ============================================================================ #
class _MplView:
    def __init__(self, max_range, fov_deg, cfg):
        import matplotlib
        import importlib
        cur = matplotlib.get_backend().lower()
        if "agg" in cur and "tkagg" not in cur and "qt" not in cur:
            for be, mod in [("TkAgg", "tkinter"), ("QtAgg", "PyQt5")]:
                try:
                    importlib.import_module(mod)
                    matplotlib.use(be, force=True)
                    break
                except Exception:
                    continue
        import matplotlib.pyplot as plt
        self.plt = plt
        self.max_range = max_range
        self.fig = plt.figure(figsize=(10, 7), facecolor="black")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor("black")
        plt.ion()
        self.fig.show()

    def update(self, xyz, rgb, obstacles):
        ax = self.ax
        ax.clear()
        ax.set_facecolor("black")

        # subsample for speed in the fallback
        n = xyz.shape[0]
        step = max(1, n // 8000)
        p = xyz[::step]
        c = rgb[::step]
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=c, s=2, depthshade=False)

        # Ego marker: cyan arrow pointing forward along +x (car heading)
        ax.quiver(
            0, 0, 0,    # tail at origin
            4, 0, 0,    # points forward along +x
            color="#00d4ff",
            linewidth=2.5,
            arrow_length_ratio=0.4,
        )

        ax.set_xlim(0, self.max_range)
        ax.set_ylim(-self.max_range / 2, self.max_range / 2)
        ax.set_zlim(-3, 3)
        ax.view_init(elev=22, azim=-70)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.set_pane_color((0, 0, 0, 1.0))
        ax.tick_params(colors="#445")
        self.fig.canvas.draw_idle()
        try:
            self.plt.pause(0.001)
        except Exception:
            pass

    def close(self):
        self.plt.close(self.fig)


# ============================================================================ #
#  Public viewer - picks a backend, exposes the same update() signature as the
#  2-D LidarViewer so it drops into carla_bridge with no logic change.
# ============================================================================ #
class LidarView3D:
    def __init__(self, max_range=45.0, fov_deg=90.0, cfg=None):
        self.cfg = cfg or {}
        self.backend_name = None
        self._impl = None
        try:
            import open3d  # noqa: F401
            self._impl = _Open3DView(max_range, fov_deg, self.cfg)
            self.backend_name = "open3d"
        except Exception:
            try:
                self._impl = _MplView(max_range, fov_deg, self.cfg)
                self.backend_name = "matplotlib3d"
            except Exception as e:
                self.backend_name = f"none ({e})"
                self._impl = None

    def update(self, raw_pts, obstacles, z_min, z_max, ego_exclusion,
               max_range, fov_deg):
        if self._impl is None or raw_pts is None or raw_pts.shape[0] == 0:
            return
        pts4 = raw_pts if raw_pts.shape[1] >= 4 else \
            np.column_stack([raw_pts, np.full(raw_pts.shape[0], 0.5, np.float32)])
        xyz, rgb = _colourise(
            pts4, z_min, z_max, fov_deg, ego_exclusion, max_range,
            cell=self.cfg.get("cell", 1.0),
            min_pts=self.cfg.get("min_pts", 8),
            max_radius=self.cfg.get("max_radius", 3.5))
        self._impl.update(xyz, rgb, obstacles)

    def close(self):
        if self._impl is not None:
            self._impl.close()