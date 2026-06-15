"""
3D chase-camera visualizer — standalone, same renderer as drive.py.

Modes (TAB): AUTO (MPC drives) / MANUAL (arrow keys)
Keys: TAB mode | R reset | + / - target speed | ESC quit
Run:  python drive_pygame_3d.py
"""

from __future__ import annotations
import os, sys, math, time
import numpy as np

import config as C
from dynamic_bicycle_mpc import (
    DynamicBicycleMPC, CarParams, rk4_step,
    compute_path_from_wp, get_ref_trajectory, ego_to_global,
    AdaptiveSpeed,
)

SELFTEST = "--selftest" in sys.argv
if SELFTEST:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame

W, H       = 1280, 800
FPS        = 60
NEAR       = 0.8
FOG_START  = 65.0
FOG_END    = 160.0

# palette
SKY_TOP  = (12,  16,  28)
SKY_BOT  = (52,  72,  110)
FOG_COL  = (52,  72,  110)
GRASS_A  = (32,  50,  36)
GRASS_B  = (38,  58,  42)
ROAD_A   = (50,  52,  58)
ROAD_B   = (40,  42,  48)
KERB_R   = (185, 40,  40)
KERB_W   = (215, 215, 215)
CENTRE   = (205, 195, 80)
PLAN     = (57,  255, 180)
BODY     = (18,  108, 210)
CABIN    = (55,  160, 255)
NOSE     = (235, 115, 25)
WHEEL_C  = (22,  22,  26)
OBS_SIDE = (205, 50,  50)
OBS_TOP_C= (245, 105, 85)
ACCENT   = (57,  255, 180)
AMBER    = (255, 182, 35)
RED      = (255, 65,  65)
C_WHITE  = (235, 238, 242)
C_DIM    = (110, 115, 125)
C_GBAR_BG= (35,  38,  44)
C_HUD_LINE=(40,  44,  52)

SUN = np.array([-0.45, -0.5, 0.74])
SUN /= np.linalg.norm(SUN)


# ── world ─────────────────────────────────────────────────────────────
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
            th = path[2, i]; pr = np.array([-math.sin(th), math.cos(th)])
            off = float(o.get("offset", 0.))
            obs.append((path[0,i]+pr[0]*off, path[1,i]+pr[1]*off, float(o["radius"])))
    return path, obs


def sense(state, obstacles):
    best, bd = None, 1e18
    for ox, oy, orad in obstacles:
        dx, dy = ox-state[0], oy-state[1]; d = math.hypot(dx, dy)
        if d - orad > C.SENSOR_RANGE: continue
        rel = (math.atan2(dy, dx) - state[2] + math.pi) % (2*math.pi) - math.pi
        if abs(rel) - math.asin(min(1., orad/max(d,1e-6))) > math.radians(C.SENSOR_FOV_DEG)/2:
            continue
        if d < bd:
            ct, st = math.cos(-state[2]), math.sin(-state[2])
            best = (dx*ct-dy*st, dx*st+dy*ct, orad, 0., 0.); bd = d
    return best


# ── camera ────────────────────────────────────────────────────────────
class Camera:
    def __init__(self, fov=63):
        self.pos  = np.array([0., -12., 5.])
        self.look = np.array([0.,   0., 0.8])
        self._fov = fov
        self.focal = (H/2) / math.tan(math.radians(fov)/2)
        self._rebuild()

    def _rebuild(self):
        f = self.look - self.pos; f /= np.linalg.norm(f) + 1e-9
        r = np.cross(f, np.array([0.,0.,1.])); r /= np.linalg.norm(r) + 1e-9
        self.f, self.r, self.u = f, r, np.cross(r, f)

    def follow(self, state, back=12., height=5., ahead=14.):
        psi = state[2]; fwd = np.array([math.cos(psi), math.sin(psi), 0.])
        car = np.array([state[0], state[1], 0.])
        self.pos  += (car - fwd*back + np.array([0.,0.,height]) - self.pos)  * 0.12
        self.look += (car + fwd*ahead + np.array([0.,0.,0.6])   - self.look) * 0.15
        self._rebuild()

    def to_cam(self, P):
        d = P - self.pos; return np.array([d@self.r, d@self.u, d@self.f])

    def project(self, cv):
        z = cv[2]; return (W/2 + self.focal*cv[0]/z, H/2 - self.focal*cv[1]/z)

    def dist(self, P): return float(np.linalg.norm(P - self.pos))


# ── clip + project ────────────────────────────────────────────────────
def _clip_near(cvs):
    out = []; n = len(cvs)
    for i in range(n):
        a, b = cvs[i], cvs[(i+1)%n]; ia, ib = a[2] >= NEAR, b[2] >= NEAR
        if ia: out.append(a)
        if ia != ib:
            t = (NEAR - a[2]) / (b[2] - a[2]); out.append(a + t*(b-a))
    return out

def clip_project(cam, verts):
    cvs = _clip_near([cam.to_cam(P) for P in verts])
    if len(cvs) < 3: return []
    return [cam.project(c) for c in cvs]


# ── shading + fog ─────────────────────────────────────────────────────
def fog(col, dist):
    t = max(0., min(1., (dist - FOG_START) / (FOG_END - FOG_START)))
    return (int(col[0]+(FOG_COL[0]-col[0])*t),
            int(col[1]+(FOG_COL[1]-col[1])*t),
            int(col[2]+(FOG_COL[2]-col[2])*t))

def shade_fog(base, normal, dist):
    i = 0.38 + 0.62 * max(0., float(normal @ SUN))
    lit = (min(255,int(base[0]*i)), min(255,int(base[1]*i)), min(255,int(base[2]*i)))
    return fog(lit, dist)


# ── layered scene ─────────────────────────────────────────────────────
class Scene:
    """
    Layered painter: ground → road → markings → props.
    Within each layer, sorts far-to-near.
    This stops grass from painting over road and road from painting over the car.
    """
    LAYERS = ("ground", "road", "mark", "prop")

    def __init__(self):
        self.layers = {k: [] for k in self.LAYERS}

    def add(self, cam, verts, normal, col, dist=None, layer="prop"):
        pts = clip_project(cam, verts)
        if not pts: return
        if dist is None: dist = cam.dist(np.mean(verts, axis=0))
        self.layers[layer].append((dist, pts, shade_fog(col, normal, dist)))

    def paint(self, surf):
        for name in self.LAYERS:
            for _, pts, col in sorted(self.layers[name], key=lambda x: -x[0]):
                if len(pts) >= 3:
                    pygame.draw.polygon(surf, col, pts)


# ── geometry ──────────────────────────────────────────────────────────
def box_faces(cx, cy, z0, length, width, height, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    def W_(lx, ly, lz): return np.array([cx+lx*c-ly*s, cy+lx*s+ly*c, lz])
    def N_(nx, ny, nz): return np.array([nx*c-ny*s, nx*s+ny*c, nz])
    L, Wd, z1 = length/2, width/2, z0+height
    return [
        ([W_(L,Wd,z1), W_(L,-Wd,z1), W_(-L,-Wd,z1), W_(-L,Wd,z1)],  N_(0,0,1)),
        ([W_(L,Wd,z0), W_(L,Wd,z1),  W_(L,-Wd,z1),  W_(L,-Wd,z0)],  N_(1,0,0)),
        ([W_(-L,Wd,z0),W_(-L,-Wd,z0),W_(-L,-Wd,z1), W_(-L,Wd,z1)],  N_(-1,0,0)),
        ([W_(L,Wd,z0), W_(-L,Wd,z0), W_(-L,Wd,z1),  W_(L,Wd,z1)],   N_(0,1,0)),
        ([W_(L,-Wd,z0),W_(L,-Wd,z1), W_(-L,-Wd,z1), W_(-L,-Wd,z0)], N_(0,-1,0)),
    ]


def add_car(scene, cam, state, steer):
    x, y, psi = state[0], state[1], state[2]
    c, s = math.cos(psi), math.sin(psi)
    # each face uses dist=None → own centroid distance → correct painter sort
    for v, n in box_faces(x, y, 0.25, 4.4, 1.95, 0.7, psi):
        scene.add(cam, v, n, BODY)
    for v, n in box_faces(x-0.2*c, y-0.2*s, 0.95, 2.2, 1.6, 0.6, psi):
        scene.add(cam, v, n, CABIN)
    for v, n in box_faces(x+1.9*c, y+1.9*s, 0.28, 0.5, 1.85, 0.46, psi):
        scene.add(cam, v, n, NOSE)
    for lx, ly, fr in [(1.35,1.05,True),(1.35,-1.05,True),(-1.35,1.05,False),(-1.35,-1.05,False)]:
        wx = x + lx*c - ly*s; wy = y + lx*s + ly*c
        wa = psi + (steer if fr else 0.)
        for v, n in box_faces(wx, wy, 0., 0.9, 0.38, 0.54, wa):
            scene.add(cam, v, n, WHEEL_C)


def add_obstacle(scene, cam, ox, oy, orad, sides=16, ht=1.65):
    ring = [(ox + orad*math.cos(2*math.pi*k/sides),
             oy + orad*math.sin(2*math.pi*k/sides)) for k in range(sides)]
    # top cap — own centroid distance
    scene.add(cam, [np.array([px, py, ht]) for px, py in ring],
              np.array([0.,0.,1.]), OBS_TOP_C)
    for k in range(sides):
        x0, y0 = ring[k]; x1, y1 = ring[(k+1)%sides]
        mx, my = (x0+x1)/2 - ox, (y0+y1)/2 - oy
        nrm = np.array([mx, my, 0.]); nrm /= np.linalg.norm(nrm) + 1e-9
        # backface cull: skip panels facing away from camera
        cam_dir = np.array([ox - cam.pos[0], oy - cam.pos[1], ht/2 - cam.pos[2]])
        if float(nrm @ cam_dir[:3]) > 0:
            continue
        scene.add(cam, [np.array([x0,y0,0.]), np.array([x1,y1,0.]),
                        np.array([x1,y1,ht]), np.array([x0,y0,ht])], nrm, OBS_SIDE)


def add_ground(scene, cam, car_xy, tile=20., reach=130.):
    cx0 = int(car_xy[0]//tile); cy0 = int(car_xy[1]//tile); nt = int(reach//tile)+1
    for i in range(cx0-nt, cx0+nt+1):
        for j in range(cy0-nt, cy0+nt+1):
            x0, y0 = i*tile, j*tile; cxw, cyw = x0+tile/2, y0+tile/2
            if (cxw-car_xy[0])**2 + (cyw-car_xy[1])**2 > reach*reach: continue
            if cam.to_cam(np.array([cxw, cyw, -0.05]))[2] < NEAR: continue
            col = GRASS_A if (i+j)%2==0 else GRASS_B
            scene.add(cam,
                      [np.array([x0,y0,-0.05]), np.array([x0+tile,y0,-0.05]),
                       np.array([x0+tile,y0+tile,-0.05]), np.array([x0,y0+tile,-0.05])],
                      np.array([0.,0.,1.]), col,
                      cam.dist(np.array([cxw, cyw, -0.05])), layer="ground")


def add_road(scene, cam, path, car_xy, hw=4., reach=120., step=2):
    n = path.shape[1]
    for i in range(0, n-step, step):
        xi, yi, thi = path[0,i], path[1,i], path[2,i]
        if (xi-car_xy[0])**2 + (yi-car_xy[1])**2 > reach*reach: continue
        j = min(i+step, n-1)
        if cam.to_cam(np.array([(xi+path[0,j])/2, (yi+path[1,j])/2, 0.]))[2] < NEAR: continue
        xj, yj, thj = path[0,j], path[1,j], path[2,j]
        pi_ = np.array([-math.sin(thi), math.cos(thi)])
        pj_ = np.array([-math.sin(thj), math.cos(thj)])
        Li = np.array([xi+pi_[0]*hw, yi+pi_[1]*hw, 0.])
        Ri = np.array([xi-pi_[0]*hw, yi-pi_[1]*hw, 0.])
        Lj = np.array([xj+pj_[0]*hw, yj+pj_[1]*hw, 0.])
        Rj = np.array([xj-pj_[0]*hw, yj-pj_[1]*hw, 0.])
        dist = cam.dist(np.array([(xi+xj)/2, (yi+yj)/2, 0.]))
        idx  = i // step
        scene.add(cam, [Li,Ri,Rj,Lj], np.array([0.,0.,1.]),
                  ROAD_A if idx%2==0 else ROAD_B, dist, layer="road")
        kw = 0.6; kc = KERB_R if idx%2==0 else KERB_W
        Lo  = np.array([xi+pi_[0]*(hw+kw), yi+pi_[1]*(hw+kw), 0.02])
        Lo2 = np.array([xj+pj_[0]*(hw+kw), yj+pj_[1]*(hw+kw), 0.02])
        Ro  = np.array([xi-pi_[0]*(hw+kw), yi-pi_[1]*(hw+kw), 0.02])
        Ro2 = np.array([xj-pj_[0]*(hw+kw), yj-pj_[1]*(hw+kw), 0.02])
        scene.add(cam, [Li,Lo,Lo2,Lj], np.array([0.,0.,1.]), kc, dist, layer="mark")
        scene.add(cam, [Ri,Ro,Ro2,Rj], np.array([0.,0.,1.]), kc, dist, layer="mark")
        if idx%2==0:
            cw = 0.16
            Ci  = np.array([xi+pi_[0]*cw, yi+pi_[1]*cw, 0.01])
            Ci2 = np.array([xi-pi_[0]*cw, yi-pi_[1]*cw, 0.01])
            Cj  = np.array([xj+pj_[0]*cw, yj+pj_[1]*cw, 0.01])
            Cj2 = np.array([xj-pj_[0]*cw, yj-pj_[1]*cw, 0.01])
            scene.add(cam, [Ci,Ci2,Cj2,Cj], np.array([0.,0.,1.]), CENTRE, dist, layer="mark")


# ── sky + plan ────────────────────────────────────────────────────────
def draw_sky(surf):
    half = H//2 + 20
    for yy in range(0, half, 2):
        t = yy / half
        col = (int(SKY_TOP[0]+(SKY_BOT[0]-SKY_TOP[0])*t),
               int(SKY_TOP[1]+(SKY_BOT[1]-SKY_TOP[1])*t),
               int(SKY_TOP[2]+(SKY_BOT[2]-SKY_TOP[2])*t))
        pygame.draw.rect(surf, col, (0, yy, W, 2))
    if half < H: pygame.draw.rect(surf, GRASS_A, (0, half, W, H-half))


def draw_plan(surf, cam, plan_xy):
    if plan_xy is None or plan_xy.shape[1] < 2: return
    prev = None
    for k in range(plan_xy.shape[1]):
        cv = cam.to_cam(np.array([plan_xy[0,k], plan_xy[1,k], 0.12]))
        if cv[2] < NEAR: prev = None; continue
        pt = cam.project(cv)
        if prev: pygame.draw.line(surf, PLAN, prev, pt, 3)
        pygame.draw.circle(surf, PLAN, (int(pt[0]), int(pt[1])), 3)
        prev = pt


# ── HUD ───────────────────────────────────────────────────────────────
def _g_col(g):
    a = min(abs(g)/1.2, 1.)
    return (int(80+175*a), int(220-155*a), int(80-80*a))

def _bar_col(val, lo, hi):
    t = max(0., min(1., (val-lo)/(hi-lo)))
    if t < 0.5: return (int(80+175*t*2), 220, 80)
    return (255, int(220-155*(t-0.5)*2), 80)

def draw_hud(surf, state, ucmd, mode, target_v, lap_t, best_lap, laps, fonts,
             adaptive_v=None, reason="FREE"):
    fB, fM, fS = fonts
    vx, vy, r  = state[3], state[4], state[5]
    steer_d = math.degrees(ucmd[1]) if ucmd is not None else 0.
    lon_g   = (ucmd[0]/9.81) if ucmd is not None else 0.
    lat_g   = vx * r / 9.81
    slip    = math.degrees(math.atan2(vy, max(abs(vx), .5))) if abs(vx) > .3 else 0.

    B, M, S = fB.get_height(), fM.get_height(), fS.get_height()
    PAD, PW = 14, 250
    BAR_H, GAP, ROW_H, GR = 8, 10, M+4, 32
    PH = 2*PAD + 2*B + 54 + 3*GAP + 5*ROW_H + 2*GR + S + 2*M

    panel = pygame.Surface((PW, PH), pygame.SRCALPHA)
    panel.fill((15, 17, 23, 225))
    surf.blit(panel, (12, 12))
    pygame.draw.rect(surf, ACCENT, (12, 12, PW, PH), 1, border_radius=8)

    x, y, bw = 12+PAD, 12+PAD, PW-2*PAD

    # mode badge
    bc = ACCENT if mode == "AUTO" else AMBER
    bh = B + 12
    chip = pygame.Surface((bw, bh), pygame.SRCALPHA); chip.fill((*bc, 35))
    surf.blit(chip, (x, y))
    pygame.draw.rect(surf, bc, (x, y, bw, bh), 1, border_radius=6)
    label = "AUTO – MPC" if mode == "AUTO" else "MANUAL – YOU"
    surf.blit(fB.render(f"● {label}", True, bc), (x+8, y+(bh-B)//2))
    y += bh + GAP

    # speed + bar
    sp = fB.render(f"{vx*3.6:5.1f}", True, C_WHITE)
    surf.blit(sp, (x, y))
    surf.blit(fS.render("km/h", True, C_DIM), (x+sp.get_width()+8, y+B-S))
    y += B + 6
    span = target_v * 3.6 * 1.4
    pygame.draw.rect(surf, C_GBAR_BG, (x, y, bw, BAR_H), border_radius=4)
    fill = int(bw * min(max(vx*3.6, 0)/span, 1.))
    if fill > 0:
        pygame.draw.rect(surf, _bar_col(vx*3.6, 0, span), (x, y, fill, BAR_H), border_radius=4)
    tx = x + int(bw * min(target_v*3.6/span, 1.))
    pygame.draw.line(surf, C_WHITE, (tx, y-2), (tx, y+BAR_H+2), 2)
    y += BAR_H + GAP

    if adaptive_v is not None:
        reason_col = {"FREE": ACCENT, "CORNER": AMBER,
                      "OBS": RED, "OBS+CORNER": RED}.get(reason, ACCENT)
        surf.blit(fS.render(f"ADAPT  {adaptive_v*3.6:4.1f} km/h", True, reason_col), (x, y))
        rs = fS.render(reason, True, reason_col)
        surf.blit(rs, (x+bw-rs.get_width(), y))
        y += S + 4

    pygame.draw.line(surf, C_HUD_LINE, (x,y), (x+bw,y), 1); y += GAP

    # telemetry rows
    def row(lbl, val, unit, col=C_WHITE):
        nonlocal y
        surf.blit(fS.render(lbl, True, C_DIM), (x, y+(ROW_H-S)//2))
        vs = fM.render(f"{val:+6.2f}", True, col)
        surf.blit(vs, (x+bw-vs.get_width()-34, y+(ROW_H-M)//2))
        surf.blit(fS.render(unit, True, C_DIM), (x+bw-30, y+(ROW_H-S)//2))
        y += ROW_H
    row("STEER",    steer_d,          "deg", AMBER if abs(steer_d) > 20 else C_WHITE)
    row("YAW RATE", math.degrees(r),  "°/s")
    row("SIDESLIP", slip,             "deg", RED   if abs(slip)    >  3 else C_WHITE)
    row("LAT G",    lat_g,            "g",   RED   if abs(lat_g)   > .8 else AMBER if abs(lat_g) > .5 else C_WHITE)
    row("LON G",    lon_g,            "g")

    pygame.draw.line(surf, C_HUD_LINE, (x,y+GAP//2), (x+bw,y+GAP//2), 1); y += GAP

    # G-force circle
    gcx, gcy = x + bw//2, y + GR
    pygame.draw.circle(surf, C_GBAR_BG, (gcx, gcy), GR)
    pygame.draw.circle(surf, C_HUD_LINE, (gcx, gcy), GR, 1)
    pygame.draw.circle(surf, C_HUD_LINE, (gcx, gcy), GR//2, 1)
    pygame.draw.line(surf, C_HUD_LINE, (gcx-GR, gcy), (gcx+GR, gcy), 1)
    pygame.draw.line(surf, C_HUD_LINE, (gcx, gcy-GR), (gcx, gcy+GR), 1)
    dxg = int(max(-1,min(1, lat_g)) * GR * .8)
    dyg = int(max(-1,min(1,-lon_g)) * GR * .8)
    gcol = _g_col(math.hypot(lat_g, lon_g))
    pygame.draw.circle(surf, gcol, (gcx+dxg, gcy+dyg), 6)
    pygame.draw.circle(surf, (255,255,255), (gcx+dxg, gcy+dyg), 6, 2)
    y += 2*GR + 4
    gl = fS.render("G-FORCE", True, C_DIM)
    surf.blit(gl, (gcx - gl.get_width()//2, y)); y += S + GAP

    pygame.draw.line(surf, C_HUD_LINE, (x, y-GAP//2), (x+bw, y-GAP//2), 1)

    # lap info
    surf.blit(fM.render(f"LAP {laps+1}", True, C_WHITE), (x, y))
    ts = fM.render(f"{lap_t:5.1f}s", True, C_WHITE)
    surf.blit(ts, (x+bw-ts.get_width(), y)); y += M+4
    best = f"BEST {best_lap:5.1f}s" if best_lap < 9999 else "BEST  --.-s"
    surf.blit(fM.render(best, True, ACCENT), (x, y))


# ── main ──────────────────────────────────────────────────────────────
def main():
    pygame.display.init(); pygame.font.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Dynamic-bicycle car MPC — 3D")
    clock = pygame.time.Clock()
    fB = pygame.font.SysFont("consolas,menlo,monospace", 24, bold=True)
    fM = pygame.font.SysFont("consolas,menlo,monospace", 19)
    fS = pygame.font.SysFont("consolas,menlo,monospace", 14)

    path, obstacles = build_world()
    p = CarParams()
    for k, v in C.CAR.items(): setattr(p, k, v)
    mpc = DynamicBicycleMPC(params=p, dt=C.DT, horizon_time=C.HORIZON_TIME,
                            road_halfwidth=getattr(C, "ROAD_HALFWIDTH", 5.),
                            pass_zone=getattr(C, "PASS_ZONE", 6.),
                            safety_margin=getattr(C, "OBSTACLE_SAFETY_MARGIN", 1.5))
    speed_ctrl = AdaptiveSpeed(base_speed=C.TARGET_SPEED)
    cam = Camera()

    target_v = C.TARGET_SPEED
    def fresh():
        return np.array([path[0,0], path[1,0], path[2,0], target_v, 0., 0.])

    state = fresh(); mode = "AUTO"; steer = 0.
    ucmd = np.zeros(2); plan_xy = None
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
                    state = fresh(); mpc._prev_traj = None; mpc._prev_u = None
                    steer = 0.; lap_start = time.time(); speed_ctrl.reset(target_v)
                elif e.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    target_v = min(target_v+1, 25); speed_ctrl.set_base(target_v)
                elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    target_v = max(target_v-1, 3);  speed_ctrl.set_base(target_v)

        keys = pygame.key.get_pressed()
        if mode == "AUTO":
            ego_obs = sense(state, obstacles)
            adaptive_v = speed_ctrl.update(state, path, ego_obs, C.DT)
            tgt = get_ref_trajectory(state, path, adaptive_v, mpc.control_horizon*C.DT, C.DT)
            traj, u = mpc.solve(np.array([0.,0.,0.,state[3],state[4],state[5]]),
                                tgt, obstacle=ego_obs, max_iter=3)
            plan_xy = ego_to_global(state, traj) if traj is not None else None
            ucmd = u[:,0]; steer = ucmd[1]
            state = rk4_step(state, ucmd, C.DT, p, substeps=5)
            if np.hypot(state[0]-path[0,-1], state[1]-path[1,-1]) < 4.:
                t_now = time.time() - lap_start
                if laps > 0: best_lap = min(best_lap, t_now)
                laps += 1; lap_start = time.time()
                state = fresh(); mpc._prev_traj = None; mpc._prev_u = None
                speed_ctrl.reset(target_v)
        else:
            dt = 0.02
            a = 3.0 if keys[pygame.K_UP] else (-4.5 if keys[pygame.K_DOWN] else 0.)
            if keys[pygame.K_LEFT]:    steer = min(mpc.max_steer,  steer + 0.9*dt)
            elif keys[pygame.K_RIGHT]: steer = max(-mpc.max_steer, steer - 0.9*dt)
            else: steer *= 0.92
            ucmd = np.array([a, steer])
            state = rk4_step(state, ucmd, dt, p, substeps=2)
            plan_xy = None
        if not np.all(np.isfinite(state)):
            state = fresh(); mpc._prev_traj = None; mpc._prev_u = None; steer = 0.

        cam.follow(state)

        draw_sky(screen)
        sc = Scene()
        add_ground(sc, cam, (state[0], state[1]))
        add_road(sc, cam, path, (state[0], state[1]))
        for ox, oy, orad in obstacles:
            add_obstacle(sc, cam, ox, oy, orad)
        add_car(sc, cam, state, steer)
        sc.paint(screen)
        draw_plan(screen, cam, plan_xy)
        draw_hud(screen, state, ucmd, mode, target_v,
                 time.time()-lap_start, best_lap, laps, (fB, fM, fS),
                 adaptive_v=speed_ctrl._v_smooth if mode=="AUTO" else None,
                 reason=speed_ctrl.reason)
        screen.blit(fS.render("TAB  mode   R  reset   + / -  speed   ESC  quit",
                               True, C_DIM), (12, H-22))

        pygame.display.flip()
        clock.tick(FPS)
        frames += 1
        if SELFTEST and frames == 20:
            mode = "MANUAL"; mpc._prev_traj = None; mpc._prev_u = None; plan_xy = None
        if SELFTEST and frames > 40: running = False

    pygame.quit()
    if SELFTEST: print("selftest OK:", frames, "frames")


if __name__ == "__main__":
    main()