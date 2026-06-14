"""
Unified live visualizer — 3D chase view and top-down view, together.

Press V to cycle layout:
    SPLIT  3D chase (left)  +  top-down (right)
    3D     full-window 3D chase camera
    TOP    full-window top-down

Modes (TAB):  AUTO (MPC drives)  /  MANUAL (arrow keys)
Keys:  V layout | TAB mode | R reset | + / - target speed | ESC quit
Run:   python drive.py

Everything (road, obstacles, car) comes from config.py; the controller and
dynamics are the same dynamic_bicycle_mpc used everywhere else. Only drawing
differs between the two views.
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

W, H = 1360, 800
FPS = 60
NEAR = 0.5

# ── palette ───────────────────────────────────────────────────────────
SKY_TOP=(18,22,34); SKY_BOT=(42,52,74)
GRASS_A=(34,52,38); GRASS_B=(40,60,44)
TARMAC=(46,48,54); TARMAC_LO=(32,34,39)
KERB_R=(190,45,45); KERB_W=(220,220,220); CENTRE=(210,200,90)
PLAN=(57,255,180); CAR_BODY=(20,110,210); CAR_CABIN=(60,170,255)
CAR_NOSE=(240,120,30); WHEEL=(24,24,28)
OBS=(210,55,55); OBS_TOP=(250,110,90); OBS_GLOW=(255,100,80,80)
WHITE=(236,240,245); DIM=(120,126,138); ACCENT=(57,255,180)
AMBER=(255,185,40); RED=(255,70,70); PANEL=(16,18,24,205)
TD_BG=(20,22,27); TD_ROAD=(52,55,62); FOV=(57,200,255,32)
LIGHT=np.array([-0.4,-0.55,0.73]); LIGHT/=np.linalg.norm(LIGHT)


def build_world():
    path = compute_path_from_wp(C.TRACK_X, C.TRACK_Y, step=0.25)
    cd = np.append([0.], np.cumsum(np.hypot(np.diff(path[0]), np.diff(path[1]))))
    obs=[]
    for o in C.OBSTACLES:
        if "x" in o and "y" in o:
            obs.append((float(o["x"]),float(o["y"]),float(o["radius"])))
        else:
            s=float(o["along"])*cd[-1]; i=min(max(int(np.searchsorted(cd,s)),0),path.shape[1]-1)
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


# ── viewport (a sub-rect of the window with its own projection centre) ─
class Viewport:
    def __init__(self, x, y, w, h, fov_deg=62):
        self.x,self.y,self.w,self.h=x,y,w,h
        self.cx=x+w/2; self.cy=y+h/2
        self.focal=(h/2)/math.tan(math.radians(fov_deg)/2)
    def rect(self): return pygame.Rect(self.x,self.y,self.w,self.h)


# ── 3D chase camera ───────────────────────────────────────────────────
class Camera:
    def __init__(self):
        self.pos=np.array([0.,-12.,5.]); self.look=np.array([0.,0.,0.8])
        self._build()
    def _build(self):
        f=self.look-self.pos; f/=np.linalg.norm(f)+1e-9
        r=np.cross(f,np.array([0.,0.,1.])); r/=np.linalg.norm(r)+1e-9
        u=np.cross(r,f); self.f,self.r,self.u=f,r,u
    def follow(self, state, back=11., height=4.6, ahead=12.):
        psi=state[2]; fwd=np.array([math.cos(psi),math.sin(psi),0.]); car=np.array([state[0],state[1],0.])
        self.pos+=(car-fwd*back+np.array([0.,0.,height])-self.pos)*0.15
        self.look+=(car+fwd*ahead+np.array([0.,0.,0.8])-self.look)*0.18
        self._build()
    def to_cam(self,P):
        d=P-self.pos; return np.array([d@self.r,d@self.u,d@self.f])
    def project(self,cv,vp):
        z=cv[2]; return (vp.cx+vp.focal*cv[0]/z, vp.cy-vp.focal*cv[1]/z)


def clip_near(cv):
    out=[]; n=len(cv)
    for i in range(n):
        a=cv[i]; b=cv[(i+1)%n]; ina=a[2]>=NEAR; inb=b[2]>=NEAR
        if ina: out.append(a)
        if ina!=inb:
            t=(NEAR-a[2])/(b[2]-a[2]); out.append(a+t*(b-a))
    return out

def shade(base,n):
    i=0.42+0.58*max(0.,float(n@LIGHT))
    return (min(255,int(base[0]*i)),min(255,int(base[1]*i)),min(255,int(base[2]*i)))


class Scene:
    def __init__(self,cam,vp): self.cam=cam; self.vp=vp; self.faces=[]
    def add(self,verts,n,base,outline=None):
        cv=[self.cam.to_cam(P) for P in verts]
        if all(c[2]<NEAR for c in cv): return
        cv=clip_near(cv)
        if len(cv)<3: return
        pts=[self.cam.project(c,self.vp) for c in cv]
        depth=sum(c[2] for c in cv)/len(cv)
        self.faces.append((depth,pts,shade(base,n),outline))
    def paint(self,surf):
        self.faces.sort(key=lambda x:-x[0])
        for _,pts,col,ol in self.faces:
            if len(pts)>=3:
                pygame.draw.polygon(surf,col,pts)
                if ol: pygame.draw.polygon(surf,ol,pts,1)


def oriented_box(cx,cy,z0,length,width,height,yaw):
    c,s=math.cos(yaw),math.sin(yaw)
    def Wp(lx,ly,lz): return np.array([cx+lx*c-ly*s, cy+lx*s+ly*c, lz])
    def Np(nx,ny,nz): return np.array([nx*c-ny*s, nx*s+ny*c, nz])
    L,Wd=length/2,width/2; z1=z0+height
    return [
        ([Wp(L,Wd,z1),Wp(L,-Wd,z1),Wp(-L,-Wd,z1),Wp(-L,Wd,z1)],Np(0,0,1)),
        ([Wp(L,Wd,z0),Wp(L,Wd,z1),Wp(L,-Wd,z1),Wp(L,-Wd,z0)],Np(1,0,0)),
        ([Wp(-L,Wd,z0),Wp(-L,-Wd,z0),Wp(-L,-Wd,z1),Wp(-L,Wd,z1)],Np(-1,0,0)),
        ([Wp(L,Wd,z0),Wp(-L,Wd,z0),Wp(-L,Wd,z1),Wp(L,Wd,z1)],Np(0,1,0)),
        ([Wp(L,-Wd,z0),Wp(L,-Wd,z1),Wp(-L,-Wd,z1),Wp(-L,-Wd,z0)],Np(0,-1,0)),
    ]

def add_car3d(sc,state,steer):
    x,y,psi=state[0],state[1],state[2]; c,s=math.cos(psi),math.sin(psi)
    for v,n in oriented_box(x,y,0.25,4.4,1.95,0.7,psi): sc.add(v,n,CAR_BODY,(8,40,80))
    for v,n in oriented_box(x-0.2*c,y-0.2*s,0.95,2.2,1.6,0.6,psi): sc.add(v,n,CAR_CABIN,(20,80,130))
    for v,n in oriented_box(x+1.9*c,y+1.9*s,0.3,0.5,1.8,0.45,psi): sc.add(v,n,CAR_NOSE)
    for lx,ly,fr in [(1.4,1.05,True),(1.4,-1.05,True),(-1.4,1.05,False),(-1.4,-1.05,False)]:
        wx=x+lx*c-ly*s; wy=y+lx*s+ly*c; wy_a=psi+(steer if fr else 0.)
        for v,n in oriented_box(wx,wy,0.,0.9,0.4,0.55,wy_a): sc.add(v,n,WHEEL)

def add_obstacle3d(sc,ox,oy,orad,sides=10,height=1.6):
    ring=[(ox+orad*math.cos(2*math.pi*k/sides),oy+orad*math.sin(2*math.pi*k/sides)) for k in range(sides)]
    sc.add([np.array([px,py,height]) for px,py in ring],np.array([0.,0.,1.]),OBS_TOP)
    for k in range(sides):
        x0,y0=ring[k]; x1,y1=ring[(k+1)%sides]
        v=[np.array([x0,y0,0.]),np.array([x1,y1,0.]),np.array([x1,y1,height]),np.array([x0,y0,height])]
        nrm=np.array([(x0+x1)/2-ox,(y0+y1)/2-oy,0.]); nrm/=np.linalg.norm(nrm)+1e-9
        sc.add(v,nrm,OBS)

def add_ground3d(sc,cam,car_xy,tile=18.,reach=140.):
    cx0=int(car_xy[0]//tile); cy0=int(car_xy[1]//tile); nt=int(reach//tile)+1
    for i in range(cx0-nt,cx0+nt+1):
        for j in range(cy0-nt,cy0+nt+1):
            x0,y0=i*tile,j*tile; cxw,cyw=x0+tile/2,y0+tile/2
            if (cxw-car_xy[0])**2+(cyw-car_xy[1])**2>reach*reach: continue
            if cam.to_cam(np.array([cxw,cyw,0.]))[2]<NEAR: continue
            col=GRASS_A if (i+j)%2==0 else GRASS_B
            sc.add([np.array([x0,y0,0.]),np.array([x0+tile,y0,0.]),
                    np.array([x0+tile,y0+tile,0.]),np.array([x0,y0+tile,0.])],np.array([0.,0.,1.]),col)

def add_road3d(sc,cam,path,car_xy,hw=4.,reach=140.,step=10):
    n=path.shape[1]
    for i in range(0,n-step,step):
        xi,yi,thi=path[0,i],path[1,i],path[2,i]
        if (xi-car_xy[0])**2+(yi-car_xy[1])**2>reach*reach: continue
        j=min(i+step,n-1); xj,yj,thj=path[0,j],path[1,j],path[2,j]
        pi=np.array([-math.sin(thi),math.cos(thi)]); pj=np.array([-math.sin(thj),math.cos(thj)])
        Li=np.array([xi+pi[0]*hw,yi+pi[1]*hw,0.02]); Ri=np.array([xi-pi[0]*hw,yi-pi[1]*hw,0.02])
        Lj=np.array([xj+pj[0]*hw,yj+pj[1]*hw,0.02]); Rj=np.array([xj-pj[0]*hw,yj-pj[1]*hw,0.02])
        sc.add([Li,Ri,Rj,Lj],np.array([0.,0.,1.]),TARMAC if (i//step)%2==0 else TARMAC_LO)
        kw=0.5; kc=KERB_R if (i//step)%2==0 else KERB_W
        sc.add([Li,np.array([xi+pi[0]*(hw+kw),yi+pi[1]*(hw+kw),0.04]),
                np.array([xj+pj[0]*(hw+kw),yj+pj[1]*(hw+kw),0.04]),Lj],np.array([0.,0.,1.]),kc)
        sc.add([Ri,np.array([xi-pi[0]*(hw+kw),yi-pi[1]*(hw+kw),0.04]),
                np.array([xj-pj[0]*(hw+kw),yj-pj[1]*(hw+kw),0.04]),Rj],np.array([0.,0.,1.]),kc)
        if (i//step)%2==0:
            cw=0.18
            sc.add([np.array([xi+pi[0]*cw,yi+pi[1]*cw,0.05]),np.array([xi-pi[0]*cw,yi-pi[1]*cw,0.05]),
                    np.array([xj-pj[0]*cw,yj-pj[1]*cw,0.05]),np.array([xj+pj[0]*cw,yj+pj[1]*cw,0.05])],
                   np.array([0.,0.,1.]),CENTRE)

def draw_sky(surf,vp):
    for yy in range(0,vp.h//2,2):
        t=yy/(vp.h/2)
        col=(int(SKY_TOP[0]+(SKY_BOT[0]-SKY_TOP[0])*t),int(SKY_TOP[1]+(SKY_BOT[1]-SKY_TOP[1])*t),
             int(SKY_TOP[2]+(SKY_BOT[2]-SKY_TOP[2])*t))
        pygame.draw.rect(surf,col,(vp.x,vp.y+yy,vp.w,2))
    pygame.draw.rect(surf,GRASS_A,(vp.x,vp.y+vp.h//2,vp.w,vp.h-vp.h//2))

def draw_plan3d(surf,cam,vp,plan_xy):
    if plan_xy is None or plan_xy.shape[1]<2: return
    pts=[]
    for k in range(plan_xy.shape[1]):
        cv=cam.to_cam(np.array([plan_xy[0,k],plan_xy[1,k],0.12]))
        if cv[2]<NEAR: pts=[]; continue
        pts.append(cam.project(cv,vp))
        if len(pts)>=2: pygame.draw.line(surf,PLAN,pts[-2],pts[-1],3)
    for p in pts[::3]: pygame.draw.circle(surf,PLAN,(int(p[0]),int(p[1])),3)

def render_3d(surf,vp,cam,path,obstacles,state,steer,plan_xy):
    surf.set_clip(vp.rect())
    draw_sky(surf,vp)
    sc=Scene(cam,vp)
    add_ground3d(sc,cam,(state[0],state[1]))
    add_road3d(sc,cam,path,(state[0],state[1]))
    for ox,oy,orad in obstacles: add_obstacle3d(sc,ox,oy,orad)
    add_car3d(sc,state,steer)
    sc.paint(surf)
    draw_plan3d(surf,cam,vp,plan_xy)
    surf.set_clip(None)


# ── top-down renderer ─────────────────────────────────────────────────
class TopDown:
    def __init__(self,vp,path,obstacles):
        self.vp=vp
        xs=list(path[0])+[o[0] for o in obstacles]; ys=list(path[1])+[o[1] for o in obstacles]
        xmid=(min(xs)+max(xs))/2; ymid=(min(ys)+max(ys))/2; pad=16.
        self.scale=min(vp.w/((max(xs)-min(xs))+2*pad), vp.h/((max(ys)-min(ys))+2*pad))
        self.ox=vp.x+vp.w/2-xmid*self.scale; self.oy=vp.y+vp.h/2+ymid*self.scale
        self.centre=[self.w2s(path[0,i],path[1,i]) for i in range(0,path.shape[1],3)]
    def w2s(self,x,y): return (int(x*self.scale+self.ox),int(-y*self.scale+self.oy))
    def car_poly(self,state,L,Wd):
        x,y,psi=state[0],state[1],state[2]; c,s=math.cos(psi),math.sin(psi); out=[]
        for lx,ly in [(L/2,Wd/2),(L/2,-Wd/2),(-L/2,-Wd/2),(-L/2,Wd/2)]:
            out.append(self.w2s(x+lx*c-ly*s, y+lx*s+ly*c))
        return out
    def render(self,surf,path,obstacles,state,steer,plan_xy):
        vp=self.vp; surf.set_clip(vp.rect()); surf.fill(TD_BG,vp.rect())
        sc=self.scale
        if len(self.centre)>1:
            pygame.draw.lines(surf,TD_ROAD,False,self.centre,max(4,int(8*sc)))
            # dashed centre
            for k in range(0,len(self.centre)-1,2):
                pygame.draw.line(surf,(200,195,120),self.centre[k],self.centre[k+1],max(1,int(0.5*sc)))
        # FOV cone
        half=math.radians(C.SENSOR_FOV_DEG)/2; rng=C.SENSOR_RANGE*sc
        cx,cy=self.w2s(state[0],state[1]); fan=[(cx,cy)]
        for kk in range(13):
            a=state[2]+(-half+kk*(2*half/12)); fan.append((cx+rng*math.cos(a),cy-rng*math.sin(a)))
        cone=pygame.Surface((vp.w,vp.h),pygame.SRCALPHA)
        pygame.draw.polygon(cone,FOV,[(p[0]-vp.x,p[1]-vp.y) for p in fan]); surf.blit(cone,(vp.x,vp.y))
        # obstacles
        for ox,oy,orad in obstacles:
            p=self.w2s(ox,oy); r=max(4,int(orad*sc))
            glow=pygame.Surface((r*4,r*4),pygame.SRCALPHA); pygame.draw.circle(glow,OBS_GLOW,(r*2,r*2),r*2)
            surf.blit(glow,(p[0]-r*2,p[1]-r*2)); pygame.draw.circle(surf,OBS,p,r)
        # plan
        if plan_xy is not None and plan_xy.shape[1]>1:
            pts=[self.w2s(plan_xy[0,k],plan_xy[1,k]) for k in range(plan_xy.shape[1])]
            pygame.draw.lines(surf,PLAN,False,pts,2)
        # car body + wheels + heading
        pygame.draw.polygon(surf,CAR_BODY,self.car_poly(state,4.4,1.95))
        pygame.draw.polygon(surf,CAR_CABIN,self.car_poly(state,4.4,1.95),max(1,int(0.15*sc)))
        x,y,psi=state[0],state[1],state[2]; c,s=math.cos(psi),math.sin(psi)
        for lx,ly,fr in [(1.3,1.05,True),(1.3,-1.05,True),(-1.3,1.05,False),(-1.3,-1.05,False)]:
            wx=x+lx*c-ly*s; wy=y+lx*s+ly*c; wa=psi+(steer if fr else 0.); wc,ws=math.cos(wa),math.sin(wa)
            wp=[self.w2s(wx+dx*wc-dy*ws,wy+dx*ws+dy*wc) for dx,dy in [(0.5,0.2),(0.5,-0.2),(-0.5,-0.2),(-0.5,0.2)]]
            pygame.draw.polygon(surf,WHEEL,wp)
        tip=self.w2s(x+3.0*c,y+3.0*s); base=self.w2s(x+1.0*c,y+1.0*s)
        pygame.draw.line(surf,ACCENT,base,tip,2)
        surf.set_clip(None)
        pygame.draw.rect(surf,(60,64,72),vp.rect(),1)


# ── HUD ───────────────────────────────────────────────────────────────
def draw_hud(surf,state,ucmd,mode,target_v,lap_t,best_lap,laps,layout,fonts):
    fB,fM,fS=fonts; vx,vy,r=state[3],state[4],state[5]
    steer_deg=math.degrees(ucmd[1]) if ucmd is not None else 0.; lat_g=vx*r/9.81
    slip=math.degrees(math.atan2(vy,max(abs(vx),.5))) if abs(vx)>.3 else 0.
    panel=pygame.Surface((250,232),pygame.SRCALPHA); panel.fill(PANEL); surf.blit(panel,(10,10))
    pygame.draw.rect(surf,ACCENT,(10,10,250,232),1,border_radius=6)
    badge=ACCENT if mode=="AUTO" else AMBER
    surf.blit(fB.render("● AUTO – MPC" if mode=="AUTO" else "◆ MANUAL – YOU",True,badge),(22,18))
    sp=fB.render(f"{vx*3.6:5.1f}",True,WHITE); surf.blit(sp,(22,48))
    surf.blit(fS.render("km/h",True,DIM),(22+sp.get_width()+5,66))
    y=90
    def row(l,v,u,col=WHITE):
        nonlocal y
        surf.blit(fS.render(l,True,DIM),(22,y))
        vs=fM.render(f"{v:+6.2f}",True,col); surf.blit(vs,(250-vs.get_width()-44,y-2))
        surf.blit(fS.render(u,True,DIM),(250-40,y)); y+=25
    row("STEER",steer_deg,"deg",AMBER if abs(steer_deg)>20 else WHITE)
    row("YAW",math.degrees(r),"°/s"); row("SLIP",slip,"deg",RED if abs(slip)>3 else WHITE)
    row("LAT",lat_g,"g",RED if abs(lat_g)>.8 else AMBER if abs(lat_g)>.5 else WHITE)
    surf.blit(fS.render(f"LAP {laps+1}   {lap_t:5.1f}s",True,WHITE),(22,y+2))
    best=f"BEST {best_lap:5.1f}s" if best_lap<9999 else "BEST  --.-s"
    surf.blit(fS.render(best,True,ACCENT),(22,y+22))
    surf.blit(fS.render(f"VIEW: {layout}   (V to switch)",True,ACCENT),(14,H-44))
    surf.blit(fS.render("TAB mode  R reset  +/- speed  ESC quit",True,DIM),(14,H-24))


def main():
    pygame.display.init(); pygame.font.init()
    screen=pygame.display.set_mode((W,H))
    pygame.display.set_caption("Dynamic-bicycle car MPC — 3D + top-down")
    clock=pygame.time.Clock()
    fB=pygame.font.SysFont("consolas,menlo,monospace",24,bold=True)
    fM=pygame.font.SysFont("consolas,menlo,monospace",19)
    fS=pygame.font.SysFont("consolas,menlo,monospace",14)

    path,obstacles=build_world()
    p=CarParams()
    for k,v in C.CAR.items(): setattr(p,k,v)
    mpc=DynamicBicycleMPC(params=p,dt=C.DT,horizon_time=C.HORIZON_TIME)
    cam=Camera()

    LAYOUTS=["SPLIT","3D","TOP"]; li=0
    def make_views(layout):
        if layout=="SPLIT":
            return (Viewport(0,0,W//2,H), TopDown(Viewport(W//2,0,W-W//2,H),path,obstacles))
        if layout=="3D":
            return (Viewport(0,0,W,H), None)
        return (None, TopDown(Viewport(0,0,W,H),path,obstacles))
    vp3d, td = make_views(LAYOUTS[li])

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
                    li=(li+1)%len(LAYOUTS); vp3d,td=make_views(LAYOUTS[li])
                elif e.key==pygame.K_TAB:
                    mode="MANUAL" if mode=="AUTO" else "AUTO"; mpc._prev_traj=None; mpc._prev_u=None; plan_xy=None
                elif e.key==pygame.K_r:
                    state=fresh(); mpc._prev_traj=None; mpc._prev_u=None; steer=0.; lap_start=time.time()
                elif e.key in (pygame.K_EQUALS,pygame.K_PLUS,pygame.K_KP_PLUS): target_v=min(target_v+1,25)
                elif e.key in (pygame.K_MINUS,pygame.K_KP_MINUS): target_v=max(target_v-1,3)

        keys=pygame.key.get_pressed()
        if mode=="AUTO":
            tgt=get_ref_trajectory(state,path,target_v,mpc.control_horizon*C.DT,C.DT)
            es=np.array([0.,0.,0.,state[3],state[4],state[5]])
            traj,u=mpc.solve(es,tgt,obstacle=sense(state,obstacles),max_iter=3)
            plan_xy=ego_to_global(state,traj) if traj is not None else None
            ucmd=u[:,0]; steer=ucmd[1]; state=rk4_step(state,ucmd,C.DT,p,substeps=5)
            if np.hypot(state[0]-path[0,-1],state[1]-path[1,-1])<4.:
                t=time.time()-lap_start
                if laps>0: best_lap=min(best_lap,t)
                laps+=1; lap_start=time.time(); state=fresh(); mpc._prev_traj=None; mpc._prev_u=None
        else:
            dt=0.02; a=3.0 if keys[pygame.K_UP] else (-4.5 if keys[pygame.K_DOWN] else 0.)
            if keys[pygame.K_LEFT]: steer=min(mpc.max_steer,steer+0.9*dt)
            elif keys[pygame.K_RIGHT]: steer=max(-mpc.max_steer,steer-0.9*dt)
            else: steer*=0.92
            ucmd=np.array([a,steer]); state=rk4_step(state,ucmd,dt,p,substeps=2); plan_xy=None
        if not np.all(np.isfinite(state)):
            state=fresh(); mpc._prev_traj=None; mpc._prev_u=None; steer=0.

        cam.follow(state)
        screen.fill((10,11,14))
        if vp3d is not None: render_3d(screen,vp3d,cam,path,obstacles,state,steer,plan_xy)
        if td  is not None: td.render(screen,path,obstacles,state,steer,plan_xy)
        if vp3d is not None and td is not None:
            pygame.draw.line(screen,ACCENT,(W//2,0),(W//2,H),2)
        draw_hud(screen,state,ucmd,mode,target_v,time.time()-lap_start,best_lap,laps,LAYOUTS[li],(fB,fM,fS))

        pygame.display.flip(); clock.tick(FPS); frames+=1
        if SELFTEST and frames in (14,28): li=(li+1)%len(LAYOUTS); vp3d,td=make_views(LAYOUTS[li])
        if SELFTEST and frames==35: mode="MANUAL"
        if SELFTEST and frames>42: running=False

    pygame.quit()
    if SELFTEST: print("selftest OK:",frames,"frames, all layouts+modes")


if __name__=="__main__":
    main()
