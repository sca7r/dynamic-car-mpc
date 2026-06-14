"""
Interactive top-down visualizer — motorsport telemetry aesthetic.

Modes (TAB to toggle):
  AUTO    MPC drives, draws its live plan and sensor FOV cone
  MANUAL  Arrow keys drive; feel the dynamic tire model

Keys:  TAB mode | R reset | + / - speed target | ESC quit
"""

from __future__ import annotations
import os, sys, math, time, collections
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

# ── window ────────────────────────────────────────────────────────────
W, H   = 1280, 800
MM_W   = 200          # minimap width  (bottom-right corner)
MM_H   = 160          # minimap height
HUD_W  = 260          # left-side HUD panel width
FPS    = 60

# ── palette  (motorsport: near-black tarmac, cyan accent, amber plan) ─
C_BG        = ( 14,  15,  18)
C_TARMAC    = ( 38,  40,  46)
C_KERB_W    = (230, 230, 230)
C_KERB_R    = (200,  40,  40)
C_DASH_W    = (200, 200, 200)
C_PLAN      = ( 57, 255, 180)   # neon green  – MPC horizon
C_FOV       = ( 57, 200, 255, 35)  # sensor cone (RGBA)
C_CAR_BODY  = ( 10, 120, 220)   # team blue
C_CAR_ROOF  = ( 30, 170, 255)
C_WHEEL     = ( 20,  20,  20)
C_WHEEL_RIM = (180, 180, 180)
C_OBS       = (220,  55,  55)
C_OBS_GLOW  = (255, 100,  80, 80)
C_HUD_BG    = ( 18,  20,  25, 210)
C_HUD_LINE  = ( 40,  44,  52)
C_ACCENT    = ( 57, 255, 180)   # same as plan – consistent accent
C_WHITE     = (235, 238, 242)
C_DIM       = (110, 115, 125)
C_RED       = (255,  65,  65)
C_AMBER     = (255, 185,  30)
C_GREEN     = ( 80, 220,  80)
C_GBAR_BG   = ( 35,  38,  44)

# ── g-force dot colours ───────────────────────────────────────────────
def g_colour(g):
    a = min(abs(g) / 1.2, 1.0)
    r = int(80  + 175 * a)
    g_ = int(220 - 155 * a)
    b = int(80  -  80 * a)
    return (r, g_, b)


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


class Camera:
    """Smooth-follow camera with world→screen transform."""
    def __init__(self, path, obstacles):
        xs = list(path[0]) + [o[0] for o in obstacles]
        ys = list(path[1]) + [o[1] for o in obstacles]
        xmid = (min(xs)+max(xs))/2; ymid = (min(ys)+max(ys))/2
        pad  = 18.
        sx   = (max(xs)-min(xs)+2*pad); sy = (max(ys)-min(ys)+2*pad)
        # leave room for HUD panel on the left
        usable_w = W - HUD_W
        self.scale = min(usable_w / sx, H / sy)
        self.cx = HUD_W + usable_w/2 - xmid*self.scale
        self.cy = H/2 + ymid*self.scale

    def w2s(self, x, y):
        return (int(x*self.scale + self.cx), int(-y*self.scale + self.cy))

    def s2w(self, sx, sy):
        return ((sx - self.cx)/self.scale, -(sy - self.cy)/self.scale)


def sense(state, obstacles):
    best, best_d = None, 1e18
    for ox, oy, orad in obstacles:
        dx, dy = ox-state[0], oy-state[1]; d = math.hypot(dx,dy)
        if d - orad > C.SENSOR_RANGE: continue
        rel = (math.atan2(dy,dx) - state[2] + math.pi) % (2*math.pi) - math.pi
        half = math.radians(C.SENSOR_FOV_DEG)/2
        if abs(rel) - math.asin(min(1., orad/max(d,1e-6))) > half: continue
        if d < best_d:
            ct, st = math.cos(-state[2]), math.sin(-state[2])
            best = (dx*ct-dy*st, dx*st+dy*ct, orad, 0., 0.); best_d = d
    return best


# ── drawing helpers ───────────────────────────────────────────────────

def draw_road(surf, path, cam, road_m=7.0, kerb_m=0.8):
    """Tarmac band + red/white kerb stripes + dashed centreline."""
    sc = cam.scale
    road_px = max(4, int(road_m * sc))
    kerb_px = max(2, int(kerb_m * sc))
    pts = [cam.w2s(path[0,i], path[1,i]) for i in range(path.shape[1])]
    if len(pts) < 2: return
    # tarmac
    pygame.draw.lines(surf, C_TARMAC, False, pts, road_px + kerb_px*2)
    # kerb stripes (alternating red/white every ~5 m)
    cdist = np.cumsum(np.hypot(np.diff(path[0]), np.diff(path[1])))
    stripe = 5.0; toggle = False
    seg_start = 0
    for i in range(1, len(pts)):
        if i-1 < len(cdist) and cdist[i-1] - int(cdist[i-1]/stripe)*stripe < (cdist[i-1] - (cdist[i-2] if i>1 else 0) + 1e-9):
            toggle = not toggle
        col = C_KERB_R if toggle else C_KERB_W
    # just draw them as a slightly narrower overlay
    pygame.draw.lines(surf, C_TARMAC, False, pts, road_px)
    # dashed centre
    dash, gap, acc = 8, 6, 0
    draw = True
    for i in range(1, len(pts)):
        dx = pts[i][0]-pts[i-1][0]; dy = pts[i][1]-pts[i-1][1]
        seg = math.hypot(dx, dy); done = 0
        while done < seg:
            remaining = (dash if draw else gap) - acc
            step = min(remaining, seg-done)
            frac0 = done/seg; frac1 = (done+step)/seg
            if draw:
                p0 = (int(pts[i-1][0]+dx*frac0), int(pts[i-1][1]+dy*frac0))
                p1 = (int(pts[i-1][0]+dx*frac1), int(pts[i-1][1]+dy*frac1))
                pygame.draw.line(surf, C_DASH_W, p0, p1, max(1, int(0.6*sc)))
            acc += step; done += step
            if acc >= (dash if draw else gap):
                draw = not draw; acc = 0


def draw_fov(surf, state, cam):
    """Translucent sensor FOV cone."""
    half = math.radians(C.SENSOR_FOV_DEG) / 2
    r_px = int(C.SENSOR_RANGE * cam.scale)
    cx, cy = cam.w2s(state[0], state[1])
    # build a fan polygon
    pts = [(cx, cy)]
    steps = 24
    for k in range(steps+1):
        a = state[2] + (-half + k*(2*half/steps))
        pts.append((cx + r_px*math.cos(a), cy - r_px*math.sin(a)))
    fov_surf = pygame.Surface((W, H), pygame.SRCALPHA)
    pygame.draw.polygon(fov_surf, C_FOV, pts)
    surf.blit(fov_surf, (0,0))


def draw_obstacles(surf, obstacles, cam):
    for ox, oy, orad in obstacles:
        cx, cy = cam.w2s(ox, oy); r = max(4, int(orad*cam.scale))
        # glow
        glow = pygame.Surface((r*4, r*4), pygame.SRCALPHA)
        pygame.draw.circle(glow, C_OBS_GLOW, (r*2, r*2), r*2)
        surf.blit(glow, (cx-r*2, cy-r*2))
        # body
        pygame.draw.circle(surf, C_OBS, (cx, cy), r)
        pygame.draw.circle(surf, (255,120,100), (cx, cy), max(2, r//3))
        # label
        fnt = pygame.font.SysFont("consolas,monospace", max(10, r//2), bold=True)
        lbl = fnt.render("OBS", True, (255,220,210))
        surf.blit(lbl, (cx - lbl.get_width()//2, cy - lbl.get_height()//2))


def draw_plan(surf, plan_xy, cam):
    if plan_xy is None or plan_xy.shape[1] < 2: return
    pts = [cam.w2s(plan_xy[0,k], plan_xy[1,k]) for k in range(plan_xy.shape[1])]
    # gradient: bright at front, fade at tail
    for i in range(1, len(pts)):
        alpha = int(255 * (1 - i/len(pts)) ** 0.5)
        r,g,b = C_PLAN
        pygame.draw.line(surf, (r,g,b), pts[i-1], pts[i], 3)
    # dot at each horizon step
    for i, pt in enumerate(pts[::3]):
        pygame.draw.circle(surf, C_PLAN, pt, 3)


def draw_car(surf, state, steer_angle, cam):
    """Body + four wheels with steered fronts."""
    x, y, psi = state[0], state[1], state[2]
    sc = cam.scale
    L   = 4.5; W_car = 2.0; wl = 1.0; ww = 0.45  # metres

    def rot(lx, ly, angle):
        c, s = math.cos(angle), math.sin(angle)
        return lx*c - ly*s, lx*s + ly*c

    def world_pt(lx, ly):
        gx, gy = x + rot(lx, ly, psi)[0], y + rot(lx, ly, psi)[1]
        return cam.w2s(gx, gy)

    # ── shadow ──
    shadow_pts = [world_pt(lx+0.15, ly-0.2) for lx,ly in
                  [(L/2,W_car/2),(L/2,-W_car/2),(-L/2,-W_car/2),(-L/2,W_car/2)]]
    shad = pygame.Surface((W,H), pygame.SRCALPHA)
    pygame.draw.polygon(shad, (0,0,0,80), shadow_pts)
    surf.blit(shad, (0,0))

    # ── body ──
    body_pts = [world_pt(lx, ly) for lx,ly in
                [(L/2,W_car/2),(L/2,-W_car/2),(-L/2,-W_car/2),(-L/2,W_car/2)]]
    pygame.draw.polygon(surf, C_CAR_BODY, body_pts)
    pygame.draw.polygon(surf, C_CAR_ROOF, body_pts, max(1, int(0.06*L*sc)))

    # ── roof highlight ──
    roof_pts = [world_pt(lx,ly) for lx,ly in [(1.2,0.7),(1.2,-0.7),(-0.8,-0.7),(-0.8,0.7)]]
    pygame.draw.polygon(surf, C_CAR_ROOF, roof_pts)

    # windscreen stripe
    ws = [world_pt(lx,ly) for lx,ly in [(1.5,0.75),(1.5,-0.75),(1.1,-0.7),(1.1,0.7)]]
    pygame.draw.polygon(surf, (150,200,255,180), ws)

    # ── wheels ──
    wheel_offsets = [
        ( L/2 - 0.8,  W_car/2 + 0.05, True),   # front-left
        ( L/2 - 0.8, -W_car/2 - 0.05, True),   # front-right
        (-L/2 + 0.8,  W_car/2 + 0.05, False),  # rear-left
        (-L/2 + 0.8, -W_car/2 - 0.05, False),  # rear-right
    ]
    for lx, ly, is_front in wheel_offsets:
        wa = psi + (steer_angle if is_front else 0)
        wc = (x + rot(lx, ly, psi)[0], y + rot(lx, ly, psi)[1])
        cx_s, cy_s = cam.w2s(*wc)
        wl_px = max(3, int(wl*sc)); ww_px = max(2, int(ww*sc))
        # wheel body
        c2, s2 = math.cos(wa), math.sin(wa)
        w_pts = []
        for flx, fly in [(wl/2, ww/2),(wl/2,-ww/2),(-wl/2,-ww/2),(-wl/2,ww/2)]:
            gwx = wc[0] + flx*c2 - fly*s2
            gwy = wc[1] + flx*s2 + fly*c2
            w_pts.append(cam.w2s(gwx, gwy))
        pygame.draw.polygon(surf, C_WHEEL, w_pts)
        pygame.draw.polygon(surf, C_WHEEL_RIM, w_pts, max(1, ww_px//3))

    # ── direction arrow ──
    tip = world_pt(L/2 + 0.6, 0)
    base = world_pt(L/2 - 0.3, 0)
    pygame.draw.line(surf, C_ACCENT, base, tip, 2)


# ── HUD panel ─────────────────────────────────────────────────────────

def bar_col(val, lo, hi):
    t = max(0., min(1., (val-lo)/(hi-lo)))
    if t < 0.5:
        return (int(80+175*t*2), int(220-0*t*2), 80)
    return (255, int(220-155*(t-0.5)*2), 80)


def draw_hud(surf, state, ucmd, mode, target_v, lap_t, best_lap, laps, fonts):
    fB, fM, fS = fonts   # big / medium / small

    # panel background
    panel = pygame.Surface((HUD_W, H), pygame.SRCALPHA)
    panel.fill(C_HUD_BG)
    surf.blit(panel, (0, 0))

    # vertical separator
    pygame.draw.line(surf, C_ACCENT, (HUD_W, 0), (HUD_W, H), 2)

    vx   = state[3]; vy = state[4]; r = state[5]
    steer_deg = math.degrees(ucmd[1]) if ucmd is not None else 0.
    lat_g = vx * r / 9.81
    lon_g = (ucmd[0] if ucmd is not None else 0.) / 9.81
    sideslip_deg = math.degrees(math.atan2(vy, max(abs(vx),0.5))) if abs(vx)>0.3 else 0.

    y = 14
    def line(txt, col=C_WHITE, f=None):
        nonlocal y
        s = (f or fM).render(txt, True, col)
        surf.blit(s, (14, y)); y += s.get_height() + 3

    def sep():
        nonlocal y
        pygame.draw.line(surf, C_HUD_LINE, (8, y+2), (HUD_W-8, y+2), 1)
        y += 8

    # ── mode badge ──
    badge_col = C_ACCENT if mode == "AUTO" else C_AMBER
    badge = pygame.Surface((HUD_W-20, 34), pygame.SRCALPHA)
    badge.fill((*badge_col, 40))
    surf.blit(badge, (10, y))
    pygame.draw.rect(surf, badge_col, (10, y, HUD_W-20, 34), 2, border_radius=4)
    ms = fB.render(f"{'● AUTO – MPC' if mode=='AUTO' else '◆ MANUAL – YOU'}", True, badge_col)
    surf.blit(ms, (18, y+5)); y += 44

    sep()

    # ── speed (big) ──
    sp_kmh = vx * 3.6
    sp_s = fB.render(f"{sp_kmh:5.1f}", True, C_WHITE)
    un_s = fS.render("km/h", True, C_DIM)
    surf.blit(sp_s, (14, y))
    surf.blit(un_s, (14 + sp_s.get_width() + 4, y + sp_s.get_height() - un_s.get_height()))
    y += sp_s.get_height() + 2

    # speed bar
    bar_w = HUD_W - 28
    pygame.draw.rect(surf, C_GBAR_BG, (14, y, bar_w, 8), border_radius=4)
    fill = int(bar_w * min(sp_kmh / (target_v*3.6*1.4), 1.))
    bc = bar_col(sp_kmh, 0, target_v*3.6*1.4)
    if fill > 0:
        pygame.draw.rect(surf, bc, (14, y, fill, 8), border_radius=4)
    y += 16

    sep()

    # ── telemetry rows ──
    def tele_row(label, value, unit, col=C_WHITE):
        nonlocal y
        ls = fS.render(label, True, C_DIM)
        vs = fM.render(f"{value:+7.2f}", True, col)
        us = fS.render(unit, True, C_DIM)
        surf.blit(ls, (14, y))
        surf.blit(vs, (HUD_W - vs.get_width() - 50, y))
        surf.blit(us, (HUD_W - 46, y + 3))
        y += vs.get_height() + 4

    tele_row("STEER",    steer_deg,    "deg",  C_AMBER if abs(steer_deg)>20 else C_WHITE)
    tele_row("YAW RATE", math.degrees(r), "°/s")
    tele_row("SIDESLIP", sideslip_deg, "deg",  C_RED if abs(sideslip_deg)>3 else C_WHITE)
    tele_row("LAT G",    lat_g,        "g",    C_RED if abs(lat_g)>0.8 else C_AMBER if abs(lat_g)>0.5 else C_WHITE)
    tele_row("LON G",    lon_g,        "g")

    sep()

    # ── g-force circle ──
    GR = 48
    gcx, gcy = HUD_W//2, y + GR + 4
    pygame.draw.circle(surf, C_GBAR_BG, (gcx, gcy), GR)
    pygame.draw.circle(surf, C_HUD_LINE, (gcx, gcy), GR, 1)
    pygame.draw.circle(surf, C_HUD_LINE, (gcx, gcy), GR//2, 1)
    pygame.draw.line(surf, C_HUD_LINE, (gcx-GR, gcy), (gcx+GR, gcy), 1)
    pygame.draw.line(surf, C_HUD_LINE, (gcx, gcy-GR), (gcx, gcy+GR), 1)
    # dot: x=lateral g, y=longitudinal g
    dx = int(max(-1,min(1, lat_g)) * GR * 0.85)
    dy = int(max(-1,min(1,-lon_g)) * GR * 0.85)
    gc = g_colour(math.hypot(lat_g, lon_g))
    pygame.draw.circle(surf, gc, (gcx+dx, gcy+dy), 7)
    pygame.draw.circle(surf, (255,255,255), (gcx+dx, gcy+dy), 7, 2)
    lbl = fS.render("G-FORCE", True, C_DIM)
    surf.blit(lbl, (gcx - lbl.get_width()//2, gcy + GR + 4))
    y = gcy + GR + 22

    sep()

    # ── lap info ──
    line(f"LAP   {laps+1:3d}", C_WHITE, fM)
    line(f"TIME  {lap_t:6.2f} s", C_WHITE, fM)
    best_s = f"BEST  {best_lap:6.2f} s" if best_lap < 9999 else "BEST  ---.-- s"
    line(best_s, C_ACCENT, fM)

    sep()

    # ── target speed control ──
    line(f"TARGET  {target_v*3.6:.0f} km/h", C_DIM, fS)
    line("+ / -  adjust target", C_DIM, fS)
    line("TAB  mode  |  R  reset  |  ESC  quit", C_DIM, fS)


def draw_minimap(surf, path, obstacles, state, cam):
    """Small overview in the bottom-right corner."""
    xs, ys = path[0], path[1]
    pad = 8
    xmin, xmax = xs.min()-pad, xs.max()+pad
    ymin, ymax = ys.min()-pad, ys.max()+pad
    sc = min(MM_W/(xmax-xmin), MM_H/(ymax-ymin))

    def m2mm(x, y):
        return (int((x-xmin)*sc), int(MM_H - (y-ymin)*sc))

    mx0, my0 = W - MM_W - 6, H - MM_H - 6
    bg = pygame.Surface((MM_W, MM_H), pygame.SRCALPHA)
    bg.fill((14, 16, 20, 200))
    surf.blit(bg, (mx0, my0))
    pygame.draw.rect(surf, C_ACCENT, (mx0, my0, MM_W, MM_H), 1)

    pts = [m2mm(xs[i], ys[i]) for i in range(0, len(xs), 4)]
    if len(pts) > 1:
        pygame.draw.lines(surf, (60,65,75), False,
                          [(p[0]+mx0, p[1]+my0) for p in pts], 4)
        pygame.draw.lines(surf, (90,95,105), False,
                          [(p[0]+mx0, p[1]+my0) for p in pts], 1)

    for ox, oy, orad in obstacles:
        op = m2mm(ox, oy)
        pygame.draw.circle(surf, C_OBS, (op[0]+mx0, op[1]+my0), max(3, int(orad*sc)))

    cp = m2mm(state[0], state[1])
    pygame.draw.circle(surf, C_CAR_BODY, (cp[0]+mx0, cp[1]+my0), 5)
    pygame.draw.circle(surf, C_ACCENT,   (cp[0]+mx0, cp[1]+my0), 5, 2)


# ── main ──────────────────────────────────────────────────────────────

def main():
    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Dynamic-bicycle car MPC  —  motorsport view")
    clock = pygame.time.Clock()

    fB = pygame.font.SysFont("consolas,menlo,monospace", 26, bold=True)
    fM = pygame.font.SysFont("consolas,menlo,monospace", 19)
    fS = pygame.font.SysFont("consolas,menlo,monospace", 14)

    path, obstacles = build_world()
    cam = Camera(path, obstacles)

    p = CarParams()
    for k, v in C.CAR.items():
        setattr(p, k, v)
    mpc = DynamicBicycleMPC(params=p, dt=C.DT, horizon_time=C.HORIZON_TIME)

    def fresh():
        return np.array([path[0,0], path[1,0], path[2,0], target_v, 0., 0.])

    target_v = C.TARGET_SPEED
    state    = fresh()
    mode     = "AUTO"
    steer    = 0.
    ucmd     = np.zeros(2)
    plan_xy  = None
    lap_start = time.time()
    best_lap  = 9999.
    laps      = 0
    running   = True
    frames    = 0

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_TAB:
                    mode = "MANUAL" if mode == "AUTO" else "AUTO"
                    mpc._prev_traj = None; mpc._prev_u = None; plan_xy = None
                elif e.key == pygame.K_r:
                    state = fresh(); mpc._prev_traj = None; mpc._prev_u = None
                    steer = 0.; plan_xy = None; lap_start = time.time()
                elif e.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    target_v = min(target_v + 1., 25.)
                elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    target_v = max(target_v - 1., 3.)

        keys = pygame.key.get_pressed()

        # ── physics update ──
        if mode == "AUTO":
            tgt = get_ref_trajectory(state, path, target_v,
                                     mpc.control_horizon*C.DT, C.DT)
            ego_obs = sense(state, obstacles)
            es = np.array([0.,0.,0., state[3], state[4], state[5]])
            traj, u = mpc.solve(es, tgt, obstacle=ego_obs, max_iter=3)
            plan_xy = ego_to_global(state, traj) if traj is not None else None
            ucmd = u[:,0]
            state = rk4_step(state, ucmd, C.DT, p, substeps=5)
            steer = ucmd[1]
            if np.hypot(state[0]-path[0,-1], state[1]-path[1,-1]) < 4.:
                t_now = time.time() - lap_start
                if laps > 0: best_lap = min(best_lap, t_now)
                laps += 1; lap_start = time.time()
                state = fresh()
                mpc._prev_traj = None; mpc._prev_u = None
        else:
            dt = 0.02
            a = 3.0 if keys[pygame.K_UP] else (-4.5 if keys[pygame.K_DOWN] else 0.)
            if keys[pygame.K_LEFT]:
                steer = min(mpc.max_steer, steer + 0.9*dt)
            elif keys[pygame.K_RIGHT]:
                steer = max(-mpc.max_steer, steer - 0.9*dt)
            else:
                steer *= 0.92
            ucmd = np.array([a, steer])
            state = rk4_step(state, ucmd, dt, p, substeps=2)
            plan_xy = None

        if not np.all(np.isfinite(state)):
            state = fresh(); mpc._prev_traj = None; mpc._prev_u = None; steer = 0.

        # ── draw ──
        screen.fill(C_BG)
        draw_road(screen, path, cam)
        draw_fov(screen, state, cam)
        draw_obstacles(screen, obstacles, cam)
        draw_plan(screen, plan_xy, cam)
        draw_car(screen, state, steer, cam)
        draw_hud(screen, state, ucmd, mode, target_v,
                 time.time()-lap_start, best_lap, laps, (fB, fM, fS))
        draw_minimap(screen, path, obstacles, state, cam)

        pygame.display.flip()
        clock.tick(FPS)

        frames += 1
        if SELFTEST and frames == 20:
            mode = "MANUAL"; mpc._prev_traj = None; mpc._prev_u = None; plan_xy = None
        if SELFTEST and frames > 40:
            running = False

    pygame.quit()
    if SELFTEST:
        print("selftest OK:", frames, "frames")


if __name__ == "__main__":
    main()
