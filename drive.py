"""
Unified live visualizer — press V to switch layout:
    SPLIT   fixed 3D chase (left half)  +  top-down motorsport (right half)
    3D      full-window 3D chase camera
    TOP     full-window top-down motorsport view

Modes (TAB): AUTO (MPC drives) / MANUAL (arrow keys)
Keys:  V layout | TAB mode | R reset | + / - speed | ESC quit
Run:   python drive.py
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

W, H  = 1800, 1000
FPS   = 60
NEAR  = 0.8
FOG_START = 65.0
FOG_END   = 160.0

# ── palette shared ─────────────────────────────────────────────────────
SKY_TOP=(12,16,28); SKY_BOT=(52,72,110); FOG_COL=(52,72,110)
GRASS_A=(32,50,36); GRASS_B=(38,58,42)
ROAD_A=(50,52,58);  ROAD_B=(40,42,48)
KERB_R=(185,40,40); KERB_W=(215,215,215); CENTRE=(205,195,80)
PLAN=(57,255,180)
BODY=(18,108,210); CABIN=(55,160,255); NOSE=(235,115,25); WHEEL_C=(22,22,26)
OBS_SIDE=(205,50,50); OBS_TOP_C=(245,105,85)
ACCENT=(57,255,180); AMBER=(255,182,35); RED=(255,65,65)

# top-down palette
C_BG=(14,15,18); C_TARMAC=(38,40,46); C_KERB_W2=(230,230,230); C_KERB_R2=(200,40,40)
C_DASH_W=(200,200,200); C_FOV=(57,200,255,35)
C_CAR_BODY=(10,120,220); C_CAR_ROOF=(30,170,255)
C_WHEEL=(20,20,20); C_WHEEL_RIM=(180,180,180)
C_OBS=(220,55,55); C_OBS_GLOW=(255,100,80,80)
C_HUD_BG=(18,20,25,210); C_HUD_LINE=(40,44,52)
C_WHITE=(235,238,242); C_DIM=(110,115,125); C_RED=(255,65,65); C_AMBER=(255,185,30)
C_GBAR_BG=(35,38,44); HUD_W=260; MM_W=180; MM_H=140

SUN=np.array([-0.45,-0.5,0.74]); SUN/=np.linalg.norm(SUN)


# ── shared world ───────────────────────────────────────────────────────
def build_world():
    path = compute_path_from_wp(C.TRACK_X, C.TRACK_Y, step=0.25)
    cdist = np.append([0.], np.cumsum(np.hypot(np.diff(path[0]), np.diff(path[1]))))
    obs = []
    for o in C.OBSTACLES:
        if "x" in o and "y" in o:
            obs.append((float(o["x"]), float(o["y"]), float(o["radius"])))
        else:
            s = float(o["along"])*cdist[-1]
            i = min(max(int(np.searchsorted(cdist,s)),0),path.shape[1]-1)
            th=path[2,i]; pr=np.array([-math.sin(th),math.cos(th)]); off=float(o.get("offset",0.))
            obs.append((path[0,i]+pr[0]*off, path[1,i]+pr[1]*off, float(o["radius"])))
    return path, obs


def sense(state, obstacles):
    best,bd=None,1e18
    for ox,oy,orad in obstacles:
        dx,dy=ox-state[0],oy-state[1]; d=math.hypot(dx,dy)
        if d-orad>C.SENSOR_RANGE: continue
        rel=(math.atan2(dy,dx)-state[2]+math.pi)%(2*math.pi)-math.pi
        if abs(rel)-math.asin(min(1.,orad/max(d,1e-6)))>math.radians(C.SENSOR_FOV_DEG)/2: continue
        if d<bd:
            ct,st=math.cos(-state[2]),math.sin(-state[2])
            best=(dx*ct-dy*st,dx*st+dy*ct,orad,0.,0.); bd=d
    return best


# ══════════════════════════════════════════════════════════════════════
#  3D RENDERER
# ══════════════════════════════════════════════════════════════════════
class Camera3D:
    def __init__(self, fov=63):
        self.pos=np.array([0.,-12.,5.]); self.look=np.array([0.,0.,0.8])
        self.focal=None; self._fov=fov; self._rebuild()
    def _rebuild(self):
        f=self.look-self.pos; f/=np.linalg.norm(f)+1e-9
        r=np.cross(f,np.array([0.,0.,1.])); r/=np.linalg.norm(r)+1e-9
        self.f,self.r,self.u=f,r,np.cross(r,f)
    def set_viewport(self,x,y,w,h):
        self.vx,self.vy,self.vw,self.vh=x,y,w,h
        self.cx=x+w/2; self.cy=y+h/2
        self.focal=(h/2)/math.tan(math.radians(self._fov)/2)
    def follow(self,state,back=12.,height=5.,ahead=14.):
        psi=state[2]; fwd=np.array([math.cos(psi),math.sin(psi),0.])
        car=np.array([state[0],state[1],0.])
        self.pos+=(car-fwd*back+np.array([0.,0.,height])-self.pos)*0.12
        self.look+=(car+fwd*ahead+np.array([0.,0.,0.6])-self.look)*0.15
        self._rebuild()
    def to_cam(self,P):
        d=P-self.pos; return np.array([d@self.r,d@self.u,d@self.f])
    def project(self,cv):
        z=cv[2]; return (self.cx+self.focal*cv[0]/z, self.cy-self.focal*cv[1]/z)
    def dist(self,P): return np.linalg.norm(P-self.pos)


def _clip_near(cvs):
    out=[]; n=len(cvs)
    for i in range(n):
        a,b=cvs[i],cvs[(i+1)%n]; ia,ib=a[2]>=NEAR,b[2]>=NEAR
        if ia: out.append(a)
        if ia!=ib:
            t=(NEAR-a[2])/(b[2]-a[2]); out.append(a+t*(b-a))
    return out

def _clip_project(cam, verts):
    cvs=[cam.to_cam(P) for P in verts]
    cvs=_clip_near(cvs)
    if len(cvs)<3: return []
    return [cam.project(c) for c in cvs]

def _fog(col, dist):
    t=max(0.,min(1.,(dist-FOG_START)/(FOG_END-FOG_START)))
    return (int(col[0]+(FOG_COL[0]-col[0])*t),
            int(col[1]+(FOG_COL[1]-col[1])*t),
            int(col[2]+(FOG_COL[2]-col[2])*t))

def _shade_fog(base, normal, dist):
    i=0.38+0.62*max(0.,float(normal@SUN))
    lit=(min(255,int(base[0]*i)),min(255,int(base[1]*i)),min(255,int(base[2]*i)))
    return _fog(lit, dist)


class Scene3D:
    # draw order: ground (bottom) -> road -> markings -> props (car/obstacles, top)
    LAYERS = ("ground", "road", "mark", "prop")
    def __init__(self):
        self.layers = {k: [] for k in self.LAYERS}
    def add(self,cam,verts,normal,col,dist=None,layer="prop"):
        pts=_clip_project(cam,verts)
        if not pts: return
        if dist is None: dist=cam.dist(np.mean(verts,axis=0))
        self.layers[layer].append((dist,pts,_shade_fog(col,normal,dist)))
    def paint(self,surf,clip_rect):
        surf.set_clip(clip_rect)
        # each layer painted far->near; layers stacked bottom->top so road always
        # covers ground, markings cover road, props cover everything.
        for name in self.LAYERS:
            faces = self.layers[name]
            faces.sort(key=lambda x:-x[0])
            for _,pts,col in faces:
                if len(pts)>=3: pygame.draw.polygon(surf,col,pts)
        surf.set_clip(None)


def _box_faces(cx,cy,z0,L,W_,H_,yaw):
    c,s=math.cos(yaw),math.sin(yaw)
    def Wp(lx,ly,lz): return np.array([cx+lx*c-ly*s,cy+lx*s+ly*c,lz])
    def Np(nx,ny,nz): return np.array([nx*c-ny*s,nx*s+ny*c,nz])
    l,w,z1=L/2,W_/2,z0+H_
    return [
        ([Wp(l,w,z1),Wp(l,-w,z1),Wp(-l,-w,z1),Wp(-l,w,z1)],Np(0,0,1)),
        ([Wp(l,w,z0),Wp(l,w,z1),Wp(l,-w,z1),Wp(l,-w,z0)],Np(1,0,0)),
        ([Wp(-l,w,z0),Wp(-l,-w,z0),Wp(-l,-w,z1),Wp(-l,w,z1)],Np(-1,0,0)),
        ([Wp(l,w,z0),Wp(-l,w,z0),Wp(-l,w,z1),Wp(l,w,z1)],Np(0,1,0)),
        ([Wp(l,-w,z0),Wp(l,-w,z1),Wp(-l,-w,z1),Wp(-l,-w,z0)],Np(0,-1,0)),
    ]

def _add_car3d(sc,cam,state,steer):
    x,y,psi=state[0],state[1],state[2]; c,s=math.cos(psi),math.sin(psi)
    # dist=None → each face computes its own centroid distance, so the
    # painter's algorithm can correctly order front/back/side faces
    for v,n in _box_faces(x,y,0.25,4.4,1.95,0.7,psi): sc.add(cam,v,n,BODY)
    for v,n in _box_faces(x-0.2*c,y-0.2*s,0.95,2.2,1.6,0.6,psi): sc.add(cam,v,n,CABIN)
    for v,n in _box_faces(x+1.9*c,y+1.9*s,0.28,0.5,1.85,0.46,psi): sc.add(cam,v,n,NOSE)
    for lx,ly,fr in [(1.35,1.05,True),(1.35,-1.05,True),(-1.35,1.05,False),(-1.35,-1.05,False)]:
        wx=x+lx*c-ly*s; wy=y+lx*s+ly*c; wa=psi+(steer if fr else 0.)
        for v,n in _box_faces(wx,wy,0.,0.9,0.38,0.54,wa): sc.add(cam,v,n,WHEEL_C)

def _add_obs3d(sc,cam,ox,oy,orad,sides=16,ht=1.65):
    # each face passes dist=None so it uses its own centroid — this is what
    # makes the cylinder look round instead of flat (faces sort correctly)
    ring=[(ox+orad*math.cos(2*math.pi*k/sides),oy+orad*math.sin(2*math.pi*k/sides)) for k in range(sides)]
    sc.add(cam,[np.array([px,py,ht]) for px,py in ring],np.array([0.,0.,1.]),OBS_TOP_C)
    for k in range(sides):
        x0,y0=ring[k]; x1,y1=ring[(k+1)%sides]
        mx,my=(x0+x1)/2-ox,(y0+y1)/2-oy
        nrm=np.array([mx,my,0.]); nrm/=np.linalg.norm(nrm)+1e-9
        # skip faces pointing away from camera (backface cull → ~50% fewer faces)
        cam_dir=np.array([ox-cam.pos[0],oy-cam.pos[1],ht/2-cam.pos[2]])
        if float(nrm@cam_dir[:3]) > 0:  # dot>0 means face points away
            continue
        sc.add(cam,[np.array([x0,y0,0.]),np.array([x1,y1,0.]),
                    np.array([x1,y1,ht]),np.array([x0,y0,ht])],nrm,OBS_SIDE)

def _add_ground3d(sc,cam,car_xy,tile=20.,reach=130.):
    cx0=int(car_xy[0]//tile); cy0=int(car_xy[1]//tile); nt=int(reach//tile)+1
    for i in range(cx0-nt,cx0+nt+1):
        for j in range(cy0-nt,cy0+nt+1):
            x0,y0=i*tile,j*tile; cxw,cyw=x0+tile/2,y0+tile/2
            if (cxw-car_xy[0])**2+(cyw-car_xy[1])**2>reach*reach: continue
            if cam.to_cam(np.array([cxw,cyw,-0.05]))[2]<NEAR: continue
            col=GRASS_A if (i+j)%2==0 else GRASS_B
            sc.add(cam,[np.array([x0,y0,-0.05]),np.array([x0+tile,y0,-0.05]),
                        np.array([x0+tile,y0+tile,-0.05]),np.array([x0,y0+tile,-0.05])],
                   np.array([0.,0.,1.]),col,cam.dist(np.array([cxw,cyw,-0.05])),layer="ground")

def _add_road3d(sc,cam,path,car_xy,hw=4.,reach=120.,step=2):
    n=path.shape[1]
    for i in range(0, n-step, step):
        xi,yi,thi=path[0,i],path[1,i],path[2,i]
        if (xi-car_xy[0])**2+(yi-car_xy[1])**2>reach*reach: continue
        j=min(i+step,n-1)
        if cam.to_cam(np.array([(xi+path[0,j])/2,(yi+path[1,j])/2,0.]))[2]<NEAR: continue
        xj,yj,thj=path[0,j],path[1,j],path[2,j]
        pi_=np.array([-math.sin(thi),math.cos(thi)]); pj_=np.array([-math.sin(thj),math.cos(thj)])
        Li=np.array([xi+pi_[0]*hw,yi+pi_[1]*hw,0.]); Ri=np.array([xi-pi_[0]*hw,yi-pi_[1]*hw,0.])
        Lj=np.array([xj+pj_[0]*hw,yj+pj_[1]*hw,0.]); Rj=np.array([xj-pj_[0]*hw,yj-pj_[1]*hw,0.])
        dist=cam.dist(np.array([(xi+xj)/2,(yi+yj)/2,0.]))
        idx=i//step
        sc.add(cam,[Li,Ri,Rj,Lj],np.array([0.,0.,1.]),ROAD_A if idx%2==0 else ROAD_B,dist,layer="road")
        # kerbs share the road edge vertices (Li/Lj, Ri/Rj) so no gap
        kw=0.6; kc=KERB_R if idx%2==0 else KERB_W
        Lo =np.array([xi+pi_[0]*(hw+kw),yi+pi_[1]*(hw+kw),0.02]); Lo2=np.array([xj+pj_[0]*(hw+kw),yj+pj_[1]*(hw+kw),0.02])
        Ro =np.array([xi-pi_[0]*(hw+kw),yi-pi_[1]*(hw+kw),0.02]); Ro2=np.array([xj-pj_[0]*(hw+kw),yj-pj_[1]*(hw+kw),0.02])
        sc.add(cam,[Li,Lo,Lo2,Lj],np.array([0.,0.,1.]),kc,dist,layer="mark")
        sc.add(cam,[Ri,Ro,Ro2,Rj],np.array([0.,0.,1.]),kc,dist,layer="mark")
        # dashed centre line
        if idx%2==0:
            cw=0.16
            Ci =np.array([xi+pi_[0]*cw,yi+pi_[1]*cw,0.01]); Ci2=np.array([xi-pi_[0]*cw,yi-pi_[1]*cw,0.01])
            Cj =np.array([xj+pj_[0]*cw,yj+pj_[1]*cw,0.01]); Cj2=np.array([xj-pj_[0]*cw,yj-pj_[1]*cw,0.01])
            sc.add(cam,[Ci,Ci2,Cj2,Cj],np.array([0.,0.,1.]),CENTRE,dist,layer="mark")

def draw_overlay_hud(surf, state, ucmd, mode, target_v, lap_t, best_lap, laps, fonts,
                     adaptive_v=None, reason="FREE", ox=12, oy=12):
    """Readable telemetry + G-force panel for the 3D / split views.

    Layout (top -> bottom), all spacing derived from the font metrics so the
    panel always sizes itself to fit its content with no overlap:
      mode badge -> speed + target bar -> telemetry rows -> G-force meter
      -> lap / best time
    """
    fB, fM, fS = fonts
    vx, vy, r = state[3], state[4], state[5]
    steer_d = math.degrees(ucmd[1]) if ucmd is not None else 0.
    lon_g   = (ucmd[0]/9.81) if ucmd is not None else 0.
    lat_g   = vx*r/9.81
    slip    = math.degrees(math.atan2(vy, max(abs(vx),.5))) if abs(vx)>.3 else 0.

    B, M, S = fB.get_height(), fM.get_height(), fS.get_height()
    PAD, PW   = 14, 250
    BAR_H, GAP, ROW_H, GR = 8, 10, M+4, 32
    PH = 2*PAD + 2*B + 54 + 3*GAP + 5*ROW_H + 2*GR + S + 2*M

    panel = pygame.Surface((PW, PH), pygame.SRCALPHA)
    panel.fill((15,17,23,225))
    surf.blit(panel, (ox, oy))
    pygame.draw.rect(surf, ACCENT, (ox,oy,PW,PH), 1, border_radius=8)

    x, y, bw = ox+PAD, oy+PAD, PW-2*PAD

    # ── mode badge ──────────────────────────────────────────────────
    badge_col = ACCENT if mode=="AUTO" else AMBER
    badge_h   = B+12
    chip = pygame.Surface((bw, badge_h), pygame.SRCALPHA)
    chip.fill((*badge_col, 35))
    surf.blit(chip, (x, y))
    pygame.draw.rect(surf, badge_col, (x,y,bw,badge_h), 1, border_radius=6)
    label = "AUTO – MPC" if mode=="AUTO" else "MANUAL – YOU"
    surf.blit(fB.render(f"● {label}", True, badge_col), (x+8, y+(badge_h-B)//2))
    y += badge_h + GAP

    # ── speed + target bar ─────────────────────────────────────────
    sp = fB.render(f"{vx*3.6:5.1f}", True, C_WHITE)
    surf.blit(sp, (x, y))
    surf.blit(fS.render("km/h", True, C_DIM), (x+sp.get_width()+8, y+B-S))
    y += B+6
    span = target_v*3.6*1.4
    pygame.draw.rect(surf, C_GBAR_BG, (x,y,bw,BAR_H), border_radius=4)
    fill = int(bw*min(max(vx*3.6,0)/span, 1.))
    if fill>0:
        pygame.draw.rect(surf, _bar_col(vx*3.6,0,span), (x,y,fill,BAR_H), border_radius=4)
    tgt_x = x+int(bw*min(target_v*3.6/span, 1.))  # marks the target speed on the bar
    pygame.draw.line(surf, C_WHITE, (tgt_x,y-2), (tgt_x,y+BAR_H+2), 2)
    y += BAR_H+GAP

    # adaptive speed + reason (when in AUTO mode)
    if adaptive_v is not None:
        reason_col = {
            "FREE":         ACCENT,
            "CORNER":       AMBER,
            "OBS":          RED,
            "OBS+CORNER":   RED,
        }.get(reason, ACCENT)
        av_str = f"ADAPT  {adaptive_v*3.6:4.1f} km/h"
        av_surf = fS.render(av_str, True, reason_col)
        surf.blit(av_surf, (x, y))
        rs = fS.render(reason, True, reason_col)
        surf.blit(rs, (x+bw-rs.get_width(), y))
        y += S+4

    pygame.draw.line(surf, C_HUD_LINE, (x,y), (x+bw,y), 1); y += GAP

    # ── telemetry rows ──────────────────────────────────────────────
    def trow(lbl, val, unit, col=C_WHITE):
        nonlocal y
        surf.blit(fS.render(lbl, True, C_DIM), (x, y+(ROW_H-S)//2))
        vs = fM.render(f"{val:+6.2f}", True, col)
        surf.blit(vs, (x+bw-vs.get_width()-34, y+(ROW_H-M)//2))
        surf.blit(fS.render(unit, True, C_DIM), (x+bw-30, y+(ROW_H-S)//2))
        y += ROW_H
    trow("STEER",    steer_d,         "deg", AMBER if abs(steer_d)>20 else C_WHITE)
    trow("YAW RATE", math.degrees(r), "°/s")
    trow("SIDESLIP", slip,            "deg", RED if abs(slip)>3 else C_WHITE)
    trow("LAT G",    lat_g,           "g",   RED if abs(lat_g)>.8 else AMBER if abs(lat_g)>.5 else C_WHITE)
    trow("LON G",    lon_g,           "g")

    pygame.draw.line(surf, C_HUD_LINE, (x,y+GAP//2), (x+bw,y+GAP//2), 1); y += GAP

    # ── G-force meter ──────────────────────────────────────────────
    gcx, gcy = x+bw//2, y+GR
    pygame.draw.circle(surf, C_GBAR_BG, (gcx,gcy), GR)
    pygame.draw.circle(surf, C_HUD_LINE, (gcx,gcy), GR, 1)
    pygame.draw.circle(surf, C_HUD_LINE, (gcx,gcy), GR//2, 1)
    pygame.draw.line(surf, C_HUD_LINE, (gcx-GR,gcy), (gcx+GR,gcy), 1)
    pygame.draw.line(surf, C_HUD_LINE, (gcx,gcy-GR), (gcx,gcy+GR), 1)
    dxg = int(max(-1,min(1,lat_g))*GR*.8)
    dyg = int(max(-1,min(1,-lon_g))*GR*.8)
    gcol = _g_col(math.hypot(lat_g,lon_g))
    pygame.draw.circle(surf, gcol, (gcx+dxg,gcy+dyg), 6)
    pygame.draw.circle(surf, (255,255,255), (gcx+dxg,gcy+dyg), 6, 2)
    y += 2*GR+4
    glabel = fS.render("G-FORCE", True, C_DIM)
    surf.blit(glabel, (gcx-glabel.get_width()//2, y))
    y += S+GAP

    pygame.draw.line(surf, C_HUD_LINE, (x,y-GAP//2), (x+bw,y-GAP//2), 1)

    # ── lap info ──────────────────────────────────────────────────────
    surf.blit(fM.render(f"LAP {laps+1}", True, C_WHITE), (x, y))
    tstr = fM.render(f"{lap_t:5.1f}s", True, C_WHITE)
    surf.blit(tstr, (x+bw-tstr.get_width(), y))
    y += M+4
    best = f"BEST {best_lap:5.1f}s" if best_lap<9999 else "BEST  --.-s"
    surf.blit(fM.render(best, True, ACCENT), (x, y))


def draw_sky3d(surf,vx,vy,vw,vh):
    half=vy+vh//2+20
    for yy in range(vy,min(half,vy+vh),2):
        t=(yy-vy)/(vh/2)
        col=(int(SKY_TOP[0]+(SKY_BOT[0]-SKY_TOP[0])*t),
             int(SKY_TOP[1]+(SKY_BOT[1]-SKY_TOP[1])*t),
             int(SKY_TOP[2]+(SKY_BOT[2]-SKY_TOP[2])*t))
        pygame.draw.rect(surf,col,(vx,yy,vw,2))
    if half<vy+vh: pygame.draw.rect(surf,GRASS_A,(vx,half,vw,vy+vh-half))

def draw_plan3d(surf,cam,plan_xy):
    if plan_xy is None or plan_xy.shape[1]<2: return
    prev=None
    for k in range(plan_xy.shape[1]):
        cv=cam.to_cam(np.array([plan_xy[0,k],plan_xy[1,k],0.12]))
        if cv[2]<NEAR: prev=None; continue
        pt=cam.project(cv)
        if prev: pygame.draw.line(surf,PLAN,prev,pt,3)
        pygame.draw.circle(surf,PLAN,(int(pt[0]),int(pt[1])),3)
        prev=pt

def render_3d(surf, cam, path, obstacles, state, steer, plan_xy, vx, vy, vw, vh):
    cam.set_viewport(vx, vy, vw, vh)
    clip=pygame.Rect(vx,vy,vw,vh)
    surf.set_clip(clip)
    draw_sky3d(surf,vx,vy,vw,vh)
    surf.set_clip(None)
    sc=Scene3D()
    _add_ground3d(sc,cam,(state[0],state[1]))
    _add_road3d(sc,cam,path,(state[0],state[1]))
    for ox,oy,orad in obstacles: _add_obs3d(sc,cam,ox,oy,orad)
    _add_car3d(sc,cam,state,steer)
    sc.paint(surf,clip)
    surf.set_clip(clip)
    draw_plan3d(surf,cam,plan_xy)
    surf.set_clip(None)


# ══════════════════════════════════════════════════════════════════════
#  TOP-DOWN RENDERER  (from drive_pygame.py — full quality)
# ══════════════════════════════════════════════════════════════════════
class CameraTop:
    def __init__(self,path,obstacles,x0,y0,w,h):
        self.x0,self.y0,self.w,self.h=x0,y0,w,h
        xs=list(path[0])+[o[0] for o in obstacles]
        ys=list(path[1])+[o[1] for o in obstacles]
        xmid=(min(xs)+max(xs))/2; ymid=(min(ys)+max(ys))/2; pad=16.
        self.scale=min(w/((max(xs)-min(xs))+2*pad), h/((max(ys)-min(ys))+2*pad))
        self.cx=x0+w/2-xmid*self.scale; self.cy=y0+h/2+ymid*self.scale
    def w2s(self,x,y): return (int(x*self.scale+self.cx),int(-y*self.scale+self.cy))

def _g_col(g):
    a=min(abs(g)/1.2,1.); return (int(80+175*a),int(220-155*a),int(80-80*a))

def _bar_col(val,lo,hi):
    t=max(0.,min(1.,(val-lo)/(hi-lo)))
    if t<0.5: return (int(80+175*t*2),int(220),80)
    return (255,int(220-155*(t-0.5)*2),80)

def render_top(surf, cam, path, obstacles, state, steer, plan_xy, show_hud=True, show_minimap=True, ucmd=None, mode="AUTO", target_v=11., lap_t=0., best_lap=9999., laps=0, fonts=None, layout="TOP"):
    x0,y0,w,h=cam.x0,cam.y0,cam.w,cam.h
    clip=pygame.Rect(x0,y0,w,h); surf.set_clip(clip)
    surf.fill(C_BG,clip)

    sc=cam.scale
    # road
    pts=[cam.w2s(path[0,i],path[1,i]) for i in range(path.shape[1])]
    if len(pts)>1:
        road_px=max(4,int(7.*sc)); kerb_px=max(2,int(0.8*sc))
        pygame.draw.lines(surf,C_TARMAC,False,pts,road_px+kerb_px*2)
        pygame.draw.lines(surf,C_TARMAC,False,pts,road_px)
        # dashes
        dash,gap,acc=8,6,0; draw=True
        for i in range(1,len(pts)):
            dx=pts[i][0]-pts[i-1][0]; dy=pts[i][1]-pts[i-1][1]; seg=math.hypot(dx,dy); done=0
            while done<seg:
                remaining=(dash if draw else gap)-acc; step=min(remaining,seg-done)
                frac0=done/seg; frac1=(done+step)/seg
                if draw:
                    p0=(int(pts[i-1][0]+dx*frac0),int(pts[i-1][1]+dy*frac0))
                    p1=(int(pts[i-1][0]+dx*frac1),int(pts[i-1][1]+dy*frac1))
                    pygame.draw.line(surf,C_DASH_W,p0,p1,max(1,int(0.6*sc)))
                acc+=step; done+=step
                if acc>=(dash if draw else gap): draw=not draw; acc=0

    # FOV
    half=math.radians(C.SENSOR_FOV_DEG)/2; r_px=int(C.SENSOR_RANGE*sc)
    cx2,cy2=cam.w2s(state[0],state[1]); fov_pts=[(cx2,cy2)]
    for k in range(25): a=state[2]+(-half+k*2*half/24); fov_pts.append((cx2+r_px*math.cos(a),cy2-r_px*math.sin(a)))
    fov_s=pygame.Surface((w,h),pygame.SRCALPHA); pygame.draw.polygon(fov_s,C_FOV,[(p[0]-x0,p[1]-y0) for p in fov_pts]); surf.blit(fov_s,(x0,y0))

    # obstacles
    for ox,oy,orad in obstacles:
        cx3,cy3=cam.w2s(ox,oy); r=max(4,int(orad*sc))
        gls=pygame.Surface((r*4,r*4),pygame.SRCALPHA); pygame.draw.circle(gls,C_OBS_GLOW,(r*2,r*2),r*2); surf.blit(gls,(cx3-r*2,cy3-r*2))
        pygame.draw.circle(surf,C_OBS,(cx3,cy3),r); pygame.draw.circle(surf,(255,120,100),(cx3,cy3),max(2,r//3))

    # plan
    if plan_xy is not None and plan_xy.shape[1]>1:
        ppts=[cam.w2s(plan_xy[0,k],plan_xy[1,k]) for k in range(plan_xy.shape[1])]
        for i in range(1,len(ppts)): pygame.draw.line(surf,PLAN,ppts[i-1],ppts[i],3)
        for pt in ppts[::3]: pygame.draw.circle(surf,PLAN,pt,3)

    # car
    x,y,psi=state[0],state[1],state[2]; L=4.5; Wc=2.0; wl=1.0; ww=0.45
    def rot(lx,ly,a):
        c2,s2=math.cos(a),math.sin(a); return lx*c2-ly*s2,lx*s2+ly*c2
    def wp(lx,ly): gx,gy=x+rot(lx,ly,psi)[0],y+rot(lx,ly,psi)[1]; return cam.w2s(gx,gy)
    shad_s=pygame.Surface((w,h),pygame.SRCALPHA)
    pygame.draw.polygon(shad_s,(0,0,0,80),[wp(lx+.15,ly-.2) for lx,ly in [(L/2,Wc/2),(L/2,-Wc/2),(-L/2,-Wc/2),(-L/2,Wc/2)]]); surf.blit(shad_s,(x0,y0))
    body=[wp(lx,ly) for lx,ly in [(L/2,Wc/2),(L/2,-Wc/2),(-L/2,-Wc/2),(-L/2,Wc/2)]]
    pygame.draw.polygon(surf,C_CAR_BODY,body); pygame.draw.polygon(surf,C_CAR_ROOF,body,max(1,int(0.06*L*sc)))
    roof=[wp(lx,ly) for lx,ly in [(1.2,.7),(1.2,-.7),(-.8,-.7),(-.8,.7)]]; pygame.draw.polygon(surf,C_CAR_ROOF,roof)
    for lx,ly,fr in [(L/2-.8,Wc/2+.05,True),(L/2-.8,-Wc/2-.05,True),(-L/2+.8,Wc/2+.05,False),(-L/2+.8,-Wc/2-.05,False)]:
        wa=psi+(steer if fr else 0.); wc2=(x+rot(lx,ly,psi)[0],y+rot(lx,ly,psi)[1])
        c3,s3=math.cos(wa),math.sin(wa)
        wpts=[cam.w2s(wc2[0]+flx*c3-fly*s3,wc2[1]+flx*s3+fly*c3) for flx,fly in [(wl/2,ww/2),(wl/2,-ww/2),(-wl/2,-ww/2),(-wl/2,ww/2)]]
        pygame.draw.polygon(surf,C_WHEEL,wpts); pygame.draw.polygon(surf,C_WHEEL_RIM,wpts,max(1,int(ww*sc//3)))
    tip=wp(L/2+.6,0); base=wp(L/2-.3,0); pygame.draw.line(surf,ACCENT,base,tip,2)

    surf.set_clip(None)

    # HUD (only on top-down full or split-right — when viewport starts at x0=0 or right half)
    if show_hud and fonts and ucmd is not None:
        fB,fM,fS=fonts; vx2=state[3]; vy2=state[4]; r2=state[5]
        steer_d=math.degrees(ucmd[1]); lat_g=vx2*r2/9.81; lon_g=ucmd[0]/9.81
        slip=math.degrees(math.atan2(vy2,max(abs(vx2),.5))) if abs(vx2)>.3 else 0.
        hx=x0  # HUD always at left edge of this viewport
        panel=pygame.Surface((HUD_W,h),pygame.SRCALPHA); panel.fill(C_HUD_BG); surf.blit(panel,(hx,y0))
        pygame.draw.line(surf,ACCENT,(hx+HUD_W,y0),(hx+HUD_W,y0+h),2)
        yy=y0+14
        def line_(txt,col=C_WHITE,f=None):
            nonlocal yy; s=(f or fM).render(txt,True,col); surf.blit(s,(hx+14,yy)); yy+=s.get_height()+3
        def sep_():
            nonlocal yy; pygame.draw.line(surf,C_HUD_LINE,(hx+8,yy+2),(hx+HUD_W-8,yy+2),1); yy+=8
        bc=ACCENT if mode=="AUTO" else AMBER
        badge=pygame.Surface((HUD_W-20,34),pygame.SRCALPHA); badge.fill((*bc,40)); surf.blit(badge,(hx+10,yy))
        pygame.draw.rect(surf,bc,(hx+10,yy,HUD_W-20,34),2,border_radius=4)
        surf.blit(fB.render("● AUTO – MPC" if mode=="AUTO" else "◆ MANUAL – YOU",True,bc),(hx+18,yy+5)); yy+=44
        sep_()
        sp_s=fB.render(f"{vx2*3.6:5.1f}",True,C_WHITE); surf.blit(sp_s,(hx+14,yy))
        surf.blit(fS.render("km/h",True,C_DIM),(hx+14+sp_s.get_width()+4,yy+sp_s.get_height()-fS.get_height())); yy+=sp_s.get_height()+2
        bw=HUD_W-28; pygame.draw.rect(surf,C_GBAR_BG,(hx+14,yy,bw,8),border_radius=4)
        fill=int(bw*min(vx2*3.6/(target_v*3.6*1.4),1.))
        if fill>0: pygame.draw.rect(surf,_bar_col(vx2*3.6,0,target_v*3.6*1.4),(hx+14,yy,fill,8),border_radius=4)
        yy+=16; sep_()
        def trow(lbl,val,unit,col=C_WHITE):
            nonlocal yy
            surf.blit(fS.render(lbl,True,C_DIM),(hx+14,yy))
            vs=fM.render(f"{val:+7.2f}",True,col); surf.blit(vs,(hx+HUD_W-vs.get_width()-50,yy))
            surf.blit(fS.render(unit,True,C_DIM),(hx+HUD_W-46,yy+3)); yy+=vs.get_height()+4
        trow("STEER",steer_d,"deg",C_AMBER if abs(steer_d)>20 else C_WHITE)
        trow("YAW RATE",math.degrees(r2),"°/s")
        trow("SIDESLIP",slip,"deg",C_RED if abs(slip)>3 else C_WHITE)
        trow("LAT G",lat_g,"g",C_RED if abs(lat_g)>.8 else C_AMBER if abs(lat_g)>.5 else C_WHITE)
        trow("LON G",lon_g,"g"); sep_()
        GR=44; gcx=hx+HUD_W//2; gcy=yy+GR+4
        pygame.draw.circle(surf,C_GBAR_BG,(gcx,gcy),GR); pygame.draw.circle(surf,C_HUD_LINE,(gcx,gcy),GR,1)
        pygame.draw.circle(surf,C_HUD_LINE,(gcx,gcy),GR//2,1)
        pygame.draw.line(surf,C_HUD_LINE,(gcx-GR,gcy),(gcx+GR,gcy),1); pygame.draw.line(surf,C_HUD_LINE,(gcx,gcy-GR),(gcx,gcy+GR),1)
        dx2=int(max(-1,min(1,lat_g))*GR*.85); dy2=int(max(-1,min(1,-lon_g))*GR*.85)
        gc=_g_col(math.hypot(lat_g,lon_g)); pygame.draw.circle(surf,gc,(gcx+dx2,gcy+dy2),7); pygame.draw.circle(surf,(255,255,255),(gcx+dx2,gcy+dy2),7,2)
        surf.blit(fS.render("G-FORCE",True,C_DIM),(gcx-fS.size("G-FORCE")[0]//2,gcy+GR+4))
        yy=gcy+GR+22; sep_()
        line_(f"LAP  {laps+1:3d}",C_WHITE,fM); line_(f"TIME  {lap_t:6.2f} s",C_WHITE,fM)
        line_(f"BEST  {best_lap:6.2f} s" if best_lap<9999 else "BEST  ---.-- s",ACCENT,fM); sep_()
        line_(f"TARGET  {target_v*3.6:.0f} km/h",C_DIM,fS)
        line_(f"V layout ({layout})   TAB mode   R reset",C_DIM,fS)
        line_("+ / -  speed   ESC quit",C_DIM,fS)

    # minimap
    if show_minimap:
        xs2,ys2=path[0],path[1]; pad=8
        xmn,xmx=xs2.min()-pad,xs2.max()+pad; ymn,ymx=ys2.min()-pad,ys2.max()+pad
        msc=min(MM_W/(xmx-xmn),MM_H/(ymx-ymn))
        def m2mm(x2,y2): return (int((x2-xmn)*msc),int(MM_H-(y2-ymn)*msc))
        mx0=x0+w-MM_W-6; my0=y0+h-MM_H-6
        bg=pygame.Surface((MM_W,MM_H),pygame.SRCALPHA); bg.fill((14,16,20,200)); surf.blit(bg,(mx0,my0))
        pygame.draw.rect(surf,ACCENT,(mx0,my0,MM_W,MM_H),1)
        mpts=[(m2mm(xs2[i],ys2[i])) for i in range(0,len(xs2),4)]
        if len(mpts)>1:
            pygame.draw.lines(surf,(60,65,75),False,[(p[0]+mx0,p[1]+my0) for p in mpts],4)
            pygame.draw.lines(surf,(90,95,105),False,[(p[0]+mx0,p[1]+my0) for p in mpts],1)
        for ox,oy,orad in obstacles:
            op=m2mm(ox,oy); pygame.draw.circle(surf,C_OBS,(op[0]+mx0,op[1]+my0),max(3,int(orad*msc)))
        cp=m2mm(state[0],state[1]); pygame.draw.circle(surf,C_CAR_BODY,(cp[0]+mx0,cp[1]+my0),5); pygame.draw.circle(surf,ACCENT,(cp[0]+mx0,cp[1]+my0),5,2)


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    pygame.display.init(); pygame.font.init()
    screen=pygame.display.set_mode((W,H))
    pygame.display.set_caption("Dynamic-bicycle car MPC")
    clock=pygame.time.Clock()
    fB=pygame.font.SysFont("consolas,menlo,monospace",24,bold=True)
    fM=pygame.font.SysFont("consolas,menlo,monospace",19)
    fS=pygame.font.SysFont("consolas,menlo,monospace",14)

    path,obstacles=build_world()
    p=CarParams()
    for k,v in C.CAR.items(): setattr(p,k,v)
    mpc=DynamicBicycleMPC(params=p,dt=C.DT,horizon_time=C.HORIZON_TIME,
                          road_halfwidth=getattr(C,"ROAD_HALFWIDTH",5.),
                          pass_zone=getattr(C,"PASS_ZONE",6.),
                          safety_margin=getattr(C,"OBSTACLE_SAFETY_MARGIN",1.5))
    speed_ctrl=AdaptiveSpeed(base_speed=C.TARGET_SPEED)
    cam3d=Camera3D()

    LAYOUTS=["SPLIT","3D","TOP"]; li=0

    def make_top_cam(layout):
        if layout=="SPLIT": return CameraTop(path,obstacles,W//2,0,W-W//2,H)
        if layout=="TOP":   return CameraTop(path,obstacles,HUD_W,0,W-HUD_W,H)
        return None

    top_cam=make_top_cam(LAYOUTS[li])

    target_v=C.TARGET_SPEED
    def fresh(): return np.array([path[0,0],path[1,0],path[2,0],target_v,0.,0.])
    state=fresh(); mode="AUTO"; steer=0.; ucmd=np.zeros(2); plan_xy=None
    lap_start=time.time(); best_lap=9999.; laps=0; running=True; frames=0

    while running:
        for e in pygame.event.get():
            if e.type==pygame.QUIT: running=False
            elif e.type==pygame.KEYDOWN:
                if e.key==pygame.K_ESCAPE: running=False
                elif e.key==pygame.K_v:
                    li=(li+1)%len(LAYOUTS); top_cam=make_top_cam(LAYOUTS[li])
                elif e.key==pygame.K_TAB:
                    mode="MANUAL" if mode=="AUTO" else "AUTO"
                    mpc._prev_traj=None; mpc._prev_u=None; plan_xy=None
                elif e.key==pygame.K_r:
                    state=fresh(); mpc._prev_traj=None; mpc._prev_u=None
                    steer=0.; lap_start=time.time(); speed_ctrl.reset(target_v)
                elif e.key in (pygame.K_EQUALS,pygame.K_PLUS,pygame.K_KP_PLUS):
                    target_v=min(target_v+1,25); speed_ctrl.set_base(target_v)
                elif e.key in (pygame.K_MINUS,pygame.K_KP_MINUS):
                    target_v=max(target_v-1,3);  speed_ctrl.set_base(target_v)

        keys=pygame.key.get_pressed()
        if mode=="AUTO":
            ego_obs=sense(state,obstacles)
            adaptive_v=speed_ctrl.update(state,path,ego_obs,C.DT)
            tgt=get_ref_trajectory(state,path,adaptive_v,mpc.control_horizon*C.DT,C.DT)
            traj,u=mpc.solve(np.array([0.,0.,0.,state[3],state[4],state[5]]),tgt,obstacle=ego_obs,max_iter=3)
            plan_xy=ego_to_global(state,traj) if traj is not None else None
            ucmd=u[:,0]; steer=ucmd[1]; state=rk4_step(state,ucmd,C.DT,p,substeps=5)
            if np.hypot(state[0]-path[0,-1],state[1]-path[1,-1])<4.:
                t=time.time()-lap_start
                if laps>0: best_lap=min(best_lap,t)
                laps+=1; lap_start=time.time(); state=fresh()
                mpc._prev_traj=None; mpc._prev_u=None; speed_ctrl.reset(target_v)
        else:
            dt=0.02; a=3.0 if keys[pygame.K_UP] else (-4.5 if keys[pygame.K_DOWN] else 0.)
            if keys[pygame.K_LEFT]: steer=min(mpc.max_steer,steer+0.9*dt)
            elif keys[pygame.K_RIGHT]: steer=max(-mpc.max_steer,steer-0.9*dt)
            else: steer*=0.92
            ucmd=np.array([a,steer]); state=rk4_step(state,ucmd,dt,p,substeps=2); plan_xy=None
        if not np.all(np.isfinite(state)):
            state=fresh(); mpc._prev_traj=None; mpc._prev_u=None; steer=0.

        cam3d.follow(state)
        layout=LAYOUTS[li]
        screen.fill(C_BG)

        if layout=="SPLIT":
            render_3d(screen,cam3d,path,obstacles,state,steer,plan_xy,0,0,W//2,H)
            render_top(screen,top_cam,path,obstacles,state,steer,plan_xy,
                       show_hud=False,show_minimap=True,ucmd=ucmd,mode=mode,
                       target_v=target_v,lap_t=time.time()-lap_start,
                       best_lap=best_lap,laps=laps,fonts=(fB,fM,fS),layout=layout)
            pygame.draw.line(screen,ACCENT,(W//2,0),(W//2,H),2)
            draw_overlay_hud(screen,state,ucmd,mode,target_v,time.time()-lap_start,best_lap,laps,(fB,fM,fS),
                             adaptive_v=speed_ctrl._v_smooth if mode=="AUTO" else None,
                             reason=speed_ctrl.reason)
            screen.blit(fS.render("V  switch view   TAB  mode   R  reset   + / -  speed   ESC  quit", True, C_DIM), (12, H-22))
        elif layout=="3D":
            render_3d(screen,cam3d,path,obstacles,state,steer,plan_xy,0,0,W,H)
            draw_overlay_hud(screen,state,ucmd,mode,target_v,time.time()-lap_start,best_lap,laps,(fB,fM,fS),
                             adaptive_v=speed_ctrl._v_smooth if mode=="AUTO" else None,
                             reason=speed_ctrl.reason)
            screen.blit(fS.render("V  switch view   TAB  mode   R  reset   + / -  speed   ESC  quit", True, C_DIM), (12, H-22))
        else:  # TOP
            render_top(screen,top_cam,path,obstacles,state,steer,plan_xy,
                       show_hud=True,show_minimap=True,ucmd=ucmd,mode=mode,
                       target_v=target_v,lap_t=time.time()-lap_start,
                       best_lap=best_lap,laps=laps,fonts=(fB,fM,fS),layout=layout)

        pygame.display.flip(); clock.tick(FPS); frames+=1
        if SELFTEST and frames in (14,28): li=(li+1)%len(LAYOUTS); top_cam=make_top_cam(LAYOUTS[li])
        if SELFTEST and frames==35: mode="MANUAL"
        if SELFTEST and frames>42: running=False

    pygame.quit()
    if SELFTEST: print("selftest OK:",frames,"frames")

if __name__=="__main__":
    main()