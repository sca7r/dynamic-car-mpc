"""
3D chase-camera visualizer (pure pygame, software-rendered).

Real 3D: world points in 3D, a perspective camera that follows behind the car,
near-plane clipping, painter's-algorithm depth sorting, and directional shading.
No OpenGL / no extra deps -- it draws shaded polygons with pygame.

Modes (TAB):
  AUTO    MPC drives, draws its live plan ribbon on the road
  MANUAL  Arrow keys drive (feel the dynamic tire model)

Keys:  TAB mode | R reset | + / - target speed | C camera distance | ESC quit
Run:   python drive_pygame_3d.py
"""

from __future__ import annotations
import os, sys, math, time
import numpy as np

import config as C
from dynamic_bicycle_mpc import (
    DynamicBicycleMPC, CarParams, rk4_step,
    compute_path_from_wp, get_ref_trajectory, ego_to_global,
)

SELFTEST = "--selftest" in sys.argv
if SELFTEST:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame

W, H = 1280, 800
FPS = 60
NEAR = 0.5

# palette
SKY_TOP   = (18, 22, 34)
SKY_BOT   = (42, 52, 74)
GRASS_A   = (34, 52, 38)
GRASS_B   = (40, 60, 44)
TARMAC    = (46, 48, 54)
TARMAC_LO = (32, 34, 39)
KERB_R    = (190, 45, 45)
KERB_W    = (220, 220, 220)
CENTRE    = (210, 200, 90)
PLAN      = (57, 255, 180)
CAR_BODY  = (20, 110, 210)
CAR_CABIN = (60, 170, 255)
CAR_NOSE  = (240, 120, 30)
WHEEL     = (24, 24, 28)
OBS       = (210, 55, 55)
OBS_TOP   = (250, 110, 90)
WHITE     = (236, 240, 245)
DIM       = (120, 126, 138)
ACCENT    = (57, 255, 180)
AMBER     = (255, 185, 40)
RED       = (255, 70, 70)
PANEL     = (16, 18, 24, 205)

LIGHT = np.array([-0.4, -0.55, 0.73]); LIGHT /= np.linalg.norm(LIGHT)


def build_world():
    path = compute_path_from_wp(C.TRACK_X, C.TRACK_Y, step=0.25)
    cdist = np.append([0.], np.cumsum(np.hypot(np.diff(path[0]), np.diff(path[1]))))
    obs = []
    for o in C.OBSTACLES:
        if "x" in o and "y" in o:
            obs.append((float(o["x"]), float(o["y"]), float(o["radius"])))
        else:
            s = float(o["along"]) * cdist[-1]
            i = min(max(int(np.searchsorted(cdist, s)), 0), path.shape[1]-1)
            th = path[2, i]; perp = np.array([-math.sin(th), math.cos(th)])
            off = float(o.get("offset", 0.))
            obs.append((path[0,i]+perp[0]*off, path[1,i]+perp[1]*off, float(o["radius"])))
    return path, obs


def sense(state, obstacles):
    best, best_d = None, 1e18
    for ox, oy, orad in obstacles:
        dx, dy = ox-state[0], oy-state[1]; d = math.hypot(dx, dy)
        if d - orad > C.SENSOR_RANGE: continue
        rel = (math.atan2(dy, dx) - state[2] + math.pi) % (2*math.pi) - math.pi
        if abs(rel) - math.asin(min(1., orad/max(d,1e-6))) > math.radians(C.SENSOR_FOV_DEG)/2:
            continue
        if d < best_d:
            ct, st = math.cos(-state[2]), math.sin(-state[2])
            best = (dx*ct-dy*st, dx*st+dy*ct, orad, 0., 0.); best_d = d
    return best


# ── camera ────────────────────────────────────────────────────────────
class Camera:
    def __init__(self, fov_deg=62):
        self.pos = np.array([0., -12., 5.])
        self.look = np.array([0., 0., 0.8])
        self.focal = (H / 2) / math.tan(math.radians(fov_deg) / 2)
        self.cx, self.cy = W / 2, H / 2
        self._build()

    def _build(self):
        f = self.look - self.pos; f = f / (np.linalg.norm(f) + 1e-9)
        r = np.cross(f, np.array([0., 0., 1.])); r = r / (np.linalg.norm(r) + 1e-9)
        u = np.cross(r, f)
        self.f, self.r, self.u = f, r, u

    def follow(self, state, back, height, ahead):
        psi = state[2]
        fwd = np.array([math.cos(psi), math.sin(psi), 0.])
        car = np.array([state[0], state[1], 0.])
        tgt_pos  = car - fwd * back + np.array([0., 0., height])
        tgt_look = car + fwd * ahead + np.array([0., 0., 0.8])
        # smooth follow
        self.pos  += (tgt_pos  - self.pos)  * 0.15
        self.look += (tgt_look - self.look) * 0.18
        self._build()

    def to_cam(self, P):
        d = P - self.pos
        return np.array([d @ self.r, d @ self.u, d @ self.f])

    def project(self, cv):
        z = cv[2]
        return (self.cx + self.focal * cv[0] / z, self.cy - self.focal * cv[1] / z)


def clip_near(cam_verts):
    """Sutherland-Hodgman clip a polygon (camera space) against zc >= NEAR."""
    out = []
    n = len(cam_verts)
    for i in range(n):
        a = cam_verts[i]; b = cam_verts[(i + 1) % n]
        ina = a[2] >= NEAR; inb = b[2] >= NEAR
        if ina:
            out.append(a)
        if ina != inb:
            t = (NEAR - a[2]) / (b[2] - a[2])
            out.append(a + t * (b - a))
    return out


def shade(base, normal):
    i = 0.42 + 0.58 * max(0., float(normal @ LIGHT))
    return (min(255, int(base[0]*i)), min(255, int(base[1]*i)), min(255, int(base[2]*i)))


class Scene:
    """Collects shaded faces, then paints them far-to-near."""
    def __init__(self, cam):
        self.cam = cam
        self.faces = []

    def add(self, world_verts, normal, base_col, outline=None):
        cam = self.cam
        cv = [cam.to_cam(P) for P in world_verts]
        if all(c[2] < NEAR for c in cv):
            return
        cv = clip_near(cv)
        if len(cv) < 3:
            return
        pts = [cam.project(c) for c in cv]
        depth = sum(c[2] for c in cv) / len(cv)
        self.faces.append((depth, pts, shade(base_col, normal), outline))

    def paint(self, surf):
        self.faces.sort(key=lambda x: -x[0])
        for _, pts, col, outline in self.faces:
            if len(pts) >= 3:
                pygame.draw.polygon(surf, col, pts)
                if outline:
                    pygame.draw.polygon(surf, outline, pts, 1)


def oriented_box(cx, cy, z0, length, width, height, yaw):
    """Return list of (verts, normal) faces for a yawed box on the ground."""
    c, s = math.cos(yaw), math.sin(yaw)
    def W_(lx, ly, lz):
        return np.array([cx + lx*c - ly*s, cy + lx*s + ly*c, lz])
    def N_(nx, ny, nz):
        return np.array([nx*c - ny*s, nx*s + ny*c, nz])
    L, Wd = length/2, width/2
    z1 = z0 + height
    faces = [
        ([W_(L,Wd,z1), W_(L,-Wd,z1), W_(-L,-Wd,z1), W_(-L,Wd,z1)], N_(0,0,1)),    # top
        ([W_(L,Wd,z0), W_(L,Wd,z1), W_(L,-Wd,z1), W_(L,-Wd,z0)],  N_(1,0,0)),     # front
        ([W_(-L,Wd,z0), W_(-L,-Wd,z0), W_(-L,-Wd,z1), W_(-L,Wd,z1)], N_(-1,0,0)), # back
        ([W_(L,Wd,z0), W_(-L,Wd,z0), W_(-L,Wd,z1), W_(L,Wd,z1)],  N_(0,1,0)),     # left
        ([W_(L,-Wd,z0), W_(L,-Wd,z1), W_(-L,-Wd,z1), W_(-L,-Wd,z0)], N_(0,-1,0)), # right
    ]
    return faces


def add_car(scene, state, steer):
    x, y, psi = state[0], state[1], state[2]
    # body
    for verts, n in oriented_box(x, y, 0.25, 4.4, 1.95, 0.7, psi):
        scene.add(verts, n, CAR_BODY, outline=(8, 40, 80))
    # cabin (pushed back a bit)
    c, s = math.cos(psi), math.sin(psi)
    cab_x = x + (-0.2)*c; cab_y = y + (-0.2)*s
    for verts, n in oriented_box(cab_x, cab_y, 0.95, 2.2, 1.6, 0.6, psi):
        scene.add(verts, n, CAR_CABIN, outline=(20, 80, 130))
    # nose accent
    nose_x = x + 1.9*c; nose_y = y + 1.9*s
    for verts, n in oriented_box(nose_x, nose_y, 0.3, 0.5, 1.8, 0.45, psi):
        scene.add(verts, n, CAR_NOSE)
    # wheels
    for lx, ly, fr in [(1.4, 1.05, True), (1.4, -1.05, True),
                       (-1.4, 1.05, False), (-1.4, -1.05, False)]:
        wx = x + lx*c - ly*s; wy = y + lx*s + ly*c
        wyaw = psi + (steer if fr else 0.)
        for verts, n in oriented_box(wx, wy, 0.0, 0.9, 0.4, 0.55, wyaw):
            scene.add(verts, n, WHEEL)


def add_obstacle(scene, ox, oy, orad, sides=10, height=1.6):
    ring = [(ox + orad*math.cos(2*math.pi*k/sides),
             oy + orad*math.sin(2*math.pi*k/sides)) for k in range(sides)]
    top = [np.array([px, py, height]) for px, py in ring]
    scene.add(top, np.array([0., 0., 1.]), OBS_TOP)
    for k in range(sides):
        x0, y0 = ring[k]; x1, y1 = ring[(k+1) % sides]
        verts = [np.array([x0, y0, 0.]), np.array([x1, y1, 0.]),
                 np.array([x1, y1, height]), np.array([x0, y0, height])]
        mx, my = (x0+x1)/2 - ox, (y0+y1)/2 - oy
        nrm = np.array([mx, my, 0.]); nrm /= (np.linalg.norm(nrm)+1e-9)
        scene.add(verts, nrm, OBS)


def add_ground(scene, cam, car_xy, tile=18.0, reach=150.0):
    cx0 = int(car_xy[0] // tile); cy0 = int(car_xy[1] // tile)
    nt = int(reach // tile) + 1
    for i in range(cx0-nt, cx0+nt+1):
        for j in range(cy0-nt, cy0+nt+1):
            x0, y0 = i*tile, j*tile
            cxw, cyw = x0 + tile/2, y0 + tile/2
            if (cxw-car_xy[0])**2 + (cyw-car_xy[1])**2 > reach*reach:
                continue
            ctr = cam.to_cam(np.array([cxw, cyw, 0.]))
            if ctr[2] < NEAR:
                continue
            col = GRASS_A if (i + j) % 2 == 0 else GRASS_B
            verts = [np.array([x0, y0, 0.]), np.array([x0+tile, y0, 0.]),
                     np.array([x0+tile, y0+tile, 0.]), np.array([x0, y0+tile, 0.])]
            scene.add(verts, np.array([0., 0., 1.]), col)


def add_road(scene, cam, path, car_xy, hw=4.0, reach=150.0, step=10):
    n = path.shape[1]
    idxs = list(range(0, n-step, step))
    for i in idxs:
        xi, yi, thi = path[0, i], path[1, i], path[2, i]
        if (xi-car_xy[0])**2 + (yi-car_xy[1])**2 > reach*reach:
            continue
        j = min(i+step, n-1)
        xj, yj, thj = path[0, j], path[1, j], path[2, j]
        pi = np.array([-math.sin(thi), math.cos(thi)])
        pj = np.array([-math.sin(thj), math.cos(thj)])
        Li = np.array([xi+pi[0]*hw, yi+pi[1]*hw, 0.02]); Ri = np.array([xi-pi[0]*hw, yi-pi[1]*hw, 0.02])
        Lj = np.array([xj+pj[0]*hw, yj+pj[1]*hw, 0.02]); Rj = np.array([xj-pj[0]*hw, yj-pj[1]*hw, 0.02])
        col = TARMAC if (i // step) % 2 == 0 else TARMAC_LO
        scene.add([Li, Ri, Rj, Lj], np.array([0., 0., 1.]), col)
        # kerbs
        kw = 0.5
        kcol = KERB_R if (i // step) % 2 == 0 else KERB_W
        Lo_i = np.array([xi+pi[0]*(hw+kw), yi+pi[1]*(hw+kw), 0.04])
        Lo_j = np.array([xj+pj[0]*(hw+kw), yj+pj[1]*(hw+kw), 0.04])
        scene.add([Li, Lo_i, Lo_j, Lj], np.array([0., 0., 1.]), kcol)
        Ro_i = np.array([xi-pi[0]*(hw+kw), yi-pi[1]*(hw+kw), 0.04])
        Ro_j = np.array([xj-pj[0]*(hw+kw), yj-pj[1]*(hw+kw), 0.04])
        scene.add([Ri, Ro_i, Ro_j, Rj], np.array([0., 0., 1.]), kcol)
        # centre dashes
        if (i // step) % 2 == 0:
            cw = 0.18
            Ci = np.array([xi+pi[0]*cw, yi+pi[1]*cw, 0.05]); Cii = np.array([xi-pi[0]*cw, yi-pi[1]*cw, 0.05])
            Cj = np.array([xj+pj[0]*cw, yj+pj[1]*cw, 0.05]); Cjj = np.array([xj-pj[0]*cw, yj-pj[1]*cw, 0.05])
            scene.add([Ci, Cii, Cjj, Cj], np.array([0., 0., 1.]), CENTRE)


def draw_sky(surf):
    for yy in range(0, H//2, 2):
        t = yy / (H/2)
        col = (int(SKY_TOP[0]+(SKY_BOT[0]-SKY_TOP[0])*t),
               int(SKY_TOP[1]+(SKY_BOT[1]-SKY_TOP[1])*t),
               int(SKY_TOP[2]+(SKY_BOT[2]-SKY_TOP[2])*t))
        pygame.draw.rect(surf, col, (0, yy, W, 2))
    pygame.draw.rect(surf, GRASS_A, (0, H//2, W, H//2))


def draw_plan(surf, cam, plan_xy):
    if plan_xy is None or plan_xy.shape[1] < 2:
        return
    pts = []
    for k in range(plan_xy.shape[1]):
        cv = cam.to_cam(np.array([plan_xy[0, k], plan_xy[1, k], 0.12]))
        if cv[2] < NEAR:
            pts = []  # break the ribbon at the camera plane
            continue
        pts.append(cam.project(cv))
        if len(pts) >= 2:
            pygame.draw.line(surf, PLAN, pts[-2], pts[-1], 3)
    for p in pts[::3]:
        pygame.draw.circle(surf, PLAN, (int(p[0]), int(p[1])), 3)


def draw_hud(surf, state, ucmd, mode, target_v, lap_t, best_lap, laps, fonts):
    fB, fM, fS = fonts
    vx, vy, r = state[3], state[4], state[5]
    steer_deg = math.degrees(ucmd[1]) if ucmd is not None else 0.
    lat_g = vx*r/9.81
    sideslip = math.degrees(math.atan2(vy, max(abs(vx), .5))) if abs(vx) > .3 else 0.

    panel = pygame.Surface((250, 250), pygame.SRCALPHA); panel.fill(PANEL)
    surf.blit(panel, (10, 10))
    pygame.draw.rect(surf, ACCENT, (10, 10, 250, 250), 1, border_radius=6)

    badge = ACCENT if mode == "AUTO" else AMBER
    bs = fB.render("● AUTO – MPC" if mode == "AUTO" else "◆ MANUAL – YOU", True, badge)
    surf.blit(bs, (22, 20))
    sp = fB.render(f"{vx*3.6:5.1f}", True, WHITE)
    surf.blit(sp, (22, 52)); surf.blit(fS.render("km/h", True, DIM), (22+sp.get_width()+5, 72))

    y = 96
    def row(lbl, val, unit, col=WHITE):
        nonlocal y
        surf.blit(fS.render(lbl, True, DIM), (22, y))
        v = fM.render(f"{val:+6.2f}", True, col)
        surf.blit(v, (250-v.get_width()-44, y-2))
        surf.blit(fS.render(unit, True, DIM), (250-40, y))
        y += 26
    row("STEER", steer_deg, "deg", AMBER if abs(steer_deg) > 20 else WHITE)
    row("YAW", math.degrees(r), "°/s")
    row("SLIP", sideslip, "deg", RED if abs(sideslip) > 3 else WHITE)
    row("LAT", lat_g, "g", RED if abs(lat_g) > .8 else AMBER if abs(lat_g) > .5 else WHITE)

    surf.blit(fS.render(f"LAP {laps+1}   {lap_t:5.1f}s", True, WHITE), (22, y+2))
    best = f"BEST {best_lap:5.1f}s" if best_lap < 9999 else "BEST  --.-s"
    surf.blit(fS.render(best, True, ACCENT), (22, y+22))
    surf.blit(fS.render("TAB mode  R reset  +/- speed  ESC", True, DIM), (14, H-24))


def main():
    pygame.display.init(); pygame.font.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Dynamic-bicycle car MPC — 3D chase view")
    clock = pygame.time.Clock()
    fB = pygame.font.SysFont("consolas,menlo,monospace", 24, bold=True)
    fM = pygame.font.SysFont("consolas,menlo,monospace", 19)
    fS = pygame.font.SysFont("consolas,menlo,monospace", 14)

    path, obstacles = build_world()
    p = CarParams()
    for k, v in C.CAR.items(): setattr(p, k, v)
    mpc = DynamicBicycleMPC(params=p, dt=C.DT, horizon_time=C.HORIZON_TIME)
    cam = Camera()

    target_v = C.TARGET_SPEED
    cam_back = 11.0
    def fresh():
        return np.array([path[0,0], path[1,0], path[2,0], target_v, 0., 0.])
    state = fresh(); mode = "AUTO"; steer = 0.; ucmd = np.zeros(2); plan_xy = None
    lap_start = time.time(); best_lap = 9999.; laps = 0
    running = True; frames = 0

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE: running = False
                elif e.key == pygame.K_TAB:
                    mode = "MANUAL" if mode == "AUTO" else "AUTO"
                    mpc._prev_traj = None; mpc._prev_u = None; plan_xy = None
                elif e.key == pygame.K_r:
                    state = fresh(); mpc._prev_traj=None; mpc._prev_u=None; steer=0.; lap_start=time.time()
                elif e.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS): target_v = min(target_v+1, 25)
                elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS): target_v = max(target_v-1, 3)
                elif e.key == pygame.K_c: cam_back = 11.0 if cam_back > 14 else 18.0

        keys = pygame.key.get_pressed()
        if mode == "AUTO":
            tgt = get_ref_trajectory(state, path, target_v, mpc.control_horizon*C.DT, C.DT)
            ego_obs = sense(state, obstacles)
            es = np.array([0.,0.,0., state[3], state[4], state[5]])
            traj, u = mpc.solve(es, tgt, obstacle=ego_obs, max_iter=3)
            plan_xy = ego_to_global(state, traj) if traj is not None else None
            ucmd = u[:,0]; steer = ucmd[1]
            state = rk4_step(state, ucmd, C.DT, p, substeps=5)
            if np.hypot(state[0]-path[0,-1], state[1]-path[1,-1]) < 4.:
                t_now = time.time()-lap_start
                if laps > 0: best_lap = min(best_lap, t_now)
                laps += 1; lap_start = time.time(); state = fresh()
                mpc._prev_traj=None; mpc._prev_u=None
        else:
            dt = 0.02
            a = 3.0 if keys[pygame.K_UP] else (-4.5 if keys[pygame.K_DOWN] else 0.)
            if keys[pygame.K_LEFT]: steer = min(mpc.max_steer, steer+0.9*dt)
            elif keys[pygame.K_RIGHT]: steer = max(-mpc.max_steer, steer-0.9*dt)
            else: steer *= 0.92
            ucmd = np.array([a, steer]); state = rk4_step(state, ucmd, dt, p, substeps=2)
            plan_xy = None
        if not np.all(np.isfinite(state)):
            state = fresh(); mpc._prev_traj=None; mpc._prev_u=None; steer=0.

        cam.follow(state, back=cam_back, height=4.6, ahead=12.0)

        draw_sky(screen)
        scene = Scene(cam)
        add_ground(scene, cam, (state[0], state[1]))
        add_road(scene, cam, path, (state[0], state[1]))
        for ox, oy, orad in obstacles:
            add_obstacle(scene, ox, oy, orad)
        add_car(scene, state, steer)
        scene.paint(screen)
        draw_plan(screen, cam, plan_xy)
        draw_hud(screen, state, ucmd, mode, target_v, time.time()-lap_start, best_lap, laps, (fB, fM, fS))

        pygame.display.flip(); clock.tick(FPS)
        frames += 1
        if SELFTEST and frames == 20:
            mode = "MANUAL"; mpc._prev_traj=None; mpc._prev_u=None; plan_xy=None
        if SELFTEST and frames > 40: running = False

    pygame.quit()
    if SELFTEST: print("selftest OK:", frames, "frames")


if __name__ == "__main__":
    main()
