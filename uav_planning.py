import heapq,itertools,math,time,json
from dataclasses import dataclass,asdict
import numpy as np
VERSION='0.2.0'
@dataclass
class Settings:
    g:float=9.81
    max_speed:float=5.
    max_acceleration:float=2.
    max_vertical_speed:float=2.
    vehicle_radius:float=.25
    clearance:float=.5
    grid_step:float=2.
    population:int=24
    generations:int=25
    max_waypoints:int=16
    rrt_iterations:int=500
    energy_reference_s:float=100.
    length_reference_m:float=100.
    def __post_init__(self):
        for name in ('g','max_speed','max_acceleration','max_vertical_speed','grid_step','energy_reference_s','length_reference_m'):
            x=getattr(self,name)
            if not np.isfinite(x) or x<=0: raise ValueError(name+' must be finite and positive')
        for name in ('vehicle_radius','clearance'):
            x=getattr(self,name)
            if not np.isfinite(x) or x<0: raise ValueError(name+' must be finite and nonnegative')
        for name,minimum in [('population',4),('generations',0),('max_waypoints',2),('rrt_iterations',0)]:
            x=getattr(self,name)
            if isinstance(x,bool) or not isinstance(x,(int,np.integer)) or x<minimum: raise ValueError('Invalid '+name)
@dataclass
class World:
    name:str
    lower:np.ndarray
    upper:np.ndarray
    boxes:np.ndarray
    start:np.ndarray
    goal:np.ndarray
    def expanded_boxes(self,cfg):
        b=np.asarray(self.boxes,float).reshape(-1,2,3).copy()
        b[:,0]-=cfg.vehicle_radius+cfg.clearance;b[:,1]+=cfg.vehicle_radius+cfg.clearance
        return b
    def valid_point(self,p,cfg):
        p=np.asarray(p,float)
        if p.shape!=(3,) or not np.all(np.isfinite(p)):return False
        if np.any(p<self.lower) or np.any(p>self.upper):return False
        b=self.expanded_boxes(cfg)
        return not np.any(np.all(p>=b[:,0],axis=1)&np.all(p<=b[:,1],axis=1))
    def clear_segment(self,p,q,cfg):
        if not self.valid_point(p,cfg) or not self.valid_point(q,cfg):return False
        p,q=np.asarray(p,float),np.asarray(q,float);d=q-p
        for low,high in self.expanded_boxes(cfg):
            enter,leave=0.,1.
            for axis in range(3):
                if abs(d[axis])<1e-12:
                    if p[axis]<low[axis] or p[axis]>high[axis]:enter,leave=1.,0.;break
                else:
                    a,b=sorted(((low[axis]-p[axis])/d[axis],(high[axis]-p[axis])/d[axis]))
                    enter,leave=max(enter,a),min(leave,b)
                    if enter>leave:break
            if enter<=leave:return False
        return True

def scenario(name='wall'):
    scenes={'sparse':[[[18,18,2],[22,22,8]]],'wall':[[[18,0,2],[22,28,12]]],
    'culdesac':[[[18,10,2],[22,30,14]],[[2,10,2],[18,12,14]],[[2,28,2],[18,30,14]]]}
    if name not in scenes:raise ValueError('Unknown scene')
    return World(name,np.array([0.,0.,2.]),np.array([40.,40.,18.]),np.array(scenes[name],float),
                 np.array([10.,20.,4.] if name=='culdesac' else [4.,20.,4.]),np.array([36.,20.,4.]))
def valid_endpoints(w,c):
    if not w.valid_point(w.start,c) or not w.valid_point(w.goal,c):raise ValueError('Infeasible endpoints')
def shortcut(path,w,c):
    p=np.asarray(path,float)
    if len(p)<2:return p
    out,i=[p[0]],0
    while i<len(p)-1:
        j=len(p)-1
        while j>i+1 and not w.clear_segment(p[i],p[j],c):j-=1
        out.append(p[j]);i=j
    return np.array(out)
def quintic(tau):
    u=np.asarray(tau)
    return 10*u**3-15*u**4+6*u**5,30*u**2-60*u**3+30*u**4,60*u-180*u**2+120*u**3

def durations(path,c):
    d=np.diff(path,axis=0);L=np.linalg.norm(d,axis=1)
    return 1.01*np.maximum.reduce([1.875*L/c.max_speed,np.sqrt((10*np.sqrt(3)/3)*L/c.max_acceleration),1.875*np.abs(d[:,2])/c.max_vertical_speed,np.full(len(d),.2)])
def energy_quotient(path,times,c,quadrature_order=24):
    nodes,weights=np.polynomial.legendre.leggauss(quadrature_order)
    _,_,d2=quintic((nodes+1)/2);delta=np.diff(path,axis=0)
    a=delta[:,None,:]*d2[None,:,None]/times[:,None,None]**2;a[:,:,2]+=c.g
    return float(np.sum(times[:,None]*weights[None,:]*np.linalg.norm(a,axis=2)**1.5/2))
def assess(raw,w,c):
    if raw is None:return dict(success=False,reason='no_geometric_path',violation=1e9)
    p=np.asarray(raw,float)
    if p.ndim!=2 or p.shape[1]!=3 or len(p)<2 or not np.all(np.isfinite(p)):return dict(success=False,reason='invalid_path',violation=1e9)
    if not np.allclose(p[0],w.start,rtol=0,atol=1e-8) or not np.allclose(p[-1],w.goal,rtol=0,atol=1e-8):return dict(success=False,reason='endpoint_mismatch',violation=1e9)
    p=p[np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-10]]
    if len(p)<2:return dict(success=False,reason='zero_distance_mission',violation=1e9)
    p=shortcut(p,w,c)
    bad=sum(not w.valid_point(x,c) for x in p)+sum(not w.clear_segment(a,b,c) for a,b in zip(p[:-1],p[1:]))
    if bad:return dict(success=False,reason='collision_or_bounds',violation=float(bad))
    ts=durations(p,c);eq=energy_quotient(p,ts,c)
    return dict(success=True,reason='passed_translational_checks',violation=0.,path=p.tolist(),segment_durations_s=ts.tolist(),
      length_m=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()),flight_duration_s=float(ts.sum()),
      energy_quotient_m1p5_sminus2=eq,equivalent_hover_time_s=eq/c.g**1.5,battery_energy_j=None,decoded_waypoints=len(p))
def ranking(r,objective,c):
    if not r['success']:return 1,r['violation']
    if objective=='distance':return 0,r['length_m']/c.length_reference_m
    if objective=='proxy':return 0,r['equivalent_hover_time_s']/c.energy_reference_s
    raise ValueError('Unknown objective')
def astar(w,c):
    valid_endpoints(w,c);step=c.grid_step
    def index(p):
        r=(p-w.lower)/step
        if not np.allclose(r,np.round(r),rtol=0,atol=1e-8):raise ValueError('Grid-aligned endpoints required')
        return tuple(np.round(r).astype(int))
    def point(k):return w.lower+np.array(k)*step
    start,goal=index(w.start),index(w.goal);dims=np.floor((w.upper-w.lower)/step).astype(int)
    moves=[m for m in itertools.product((-1,0,1),repeat=3) if m!=(0,0,0)]
    costs,parents={start:0.},{};heap=[(float(np.linalg.norm(w.goal-w.start)),0.,start)]
    while heap:
        _,cost,cur=heapq.heappop(heap)
        if cost>costs[cur]+1e-10:continue
        if cur==goal:
            chain=[cur]
            while chain[-1]!=start:chain.append(parents[chain[-1]])
            return np.array([point(k) for k in chain[::-1]])
        p=point(cur)
        for move in moves:
            nxt=tuple(cur[i]+move[i] for i in range(3))
            if any(nxt[i]<0 or nxt[i]>dims[i] for i in range(3)):continue
            q=point(nxt);nc=cost+float(np.linalg.norm(q-p))
            if nc>=costs.get(nxt,float('inf'))-1e-10:continue
            if w.clear_segment(p,q,c):
                costs[nxt],parents[nxt]=nc,cur;heapq.heappush(heap,(nc+float(np.linalg.norm(q-w.goal)),nc,nxt))
    return None
def rrt_star(w,c,seed=0):
    valid_endpoints(w,c);rng=np.random.default_rng(seed)
    points,parents,costs,children=[w.start.copy()],[-1],[0.],[set()]
    for _ in range(c.rrt_iterations):
        target=w.goal if rng.random()<.15 else rng.uniform(w.lower,w.upper)
        distances=np.linalg.norm(np.array(points)-target,axis=1);nearest=int(np.argmin(distances))
        if distances[nearest]<1e-9:continue
        q=points[nearest]+(target-points[nearest])*min(1.,4./distances[nearest])
        if not w.clear_segment(points[nearest],q,c):continue
        ds=np.linalg.norm(np.array(points)-q,axis=1);radius=min(12.,35.*(math.log(len(points)+1)/(len(points)+1))**(1/3))
        near=set(np.flatnonzero(ds<=radius).tolist())|{nearest}
        candidates=sorted(near,key=lambda j:costs[j]+ds[j]);parent=next(j for j in candidates if w.clear_segment(points[j],q,c))
        idx=len(points);points.append(q);parents.append(parent);costs.append(costs[parent]+float(ds[parent]));children.append(set());children[parent].add(idx)
        ancestors,cur=set(),parent
        while cur!=-1:ancestors.add(cur);cur=parents[cur]
        for j in near:
            if j in ancestors or j==0:continue
            nc=costs[idx]+float(ds[j])
            if nc+1e-10<costs[j] and w.clear_segment(q,points[j],c):
                children[parents[j]].remove(j);parents[j]=idx;children[idx].add(j);change=nc-costs[j];stack=[j]
                while stack:
                    k=stack.pop();costs[k]+=change;stack.extend(children[k])
    candidates=[i for i,p in enumerate(points) if w.clear_segment(p,w.goal,c)]
    if not candidates:return None
    j=min(candidates,key=lambda i:costs[i]+np.linalg.norm(points[i]-w.goal));chain=[w.goal]
    while j!=-1:chain.append(points[j]);j=parents[j]
    return np.array(chain[::-1])
def mutate(path,w,c,rng):
    p=path.copy();op=rng.choice(['shift','insert','delete'])
    if op=='insert' and len(p)<c.max_waypoints:
        i=int(rng.integers(len(p)-1));q=(p[i]+p[i+1])/2+rng.normal(0,3.,3);p=np.insert(p,i+1,np.clip(q,w.lower,w.upper),axis=0)
    elif op=='delete' and len(p)>2:p=np.delete(p,int(rng.integers(1,len(p)-1)),axis=0)
    elif len(p)>2:
        i=int(rng.integers(1,len(p)-1));p[i]=np.clip(p[i]+rng.normal(0,3.,3),w.lower,w.upper)
    return p

def genetic(w,c,seed=0,objective='proxy'):
    valid_endpoints(w,c);rng=np.random.default_rng(seed)
    warm=astar(w,c);warm=shortcut(warm,w,c) if warm is not None else np.array([w.start,w.goal])
    if len(warm)>c.max_waypoints:
        warm=warm[np.linspace(0,len(warm)-1,c.max_waypoints,dtype=int)]
    population=[warm]
    while len(population)<c.population:
        if rng.random()<.5:population.append(mutate(warm,w,c,rng))
        else:
            maximum=min(4,c.max_waypoints-2);k=int(rng.integers(1,maximum+1)) if maximum else 0
            population.append(np.vstack((w.start,rng.uniform(w.lower,w.upper,(k,3)),w.goal)))
    history,cache=[],{}
    def score(p):
        key=p.tobytes()
        if key not in cache:cache[key]=assess(p,w,c)
        return cache[key]
    for generation in range(c.generations+1):
        order=sorted(range(len(population)),key=lambda i:ranking(score(population[i]),objective,c));best=score(population[order[0]])
        history.append(dict(generation=generation,feasible_count=sum(score(p)['success'] for p in population),best_score=ranking(best,objective,c)[1],best_is_feasible=best['success']))
        if generation==c.generations:break
        def select():
            ids=rng.integers(0,len(population),3)
            return population[min(ids,key=lambda i:ranking(score(population[i]),objective,c))]
        nxt=[population[order[0]].copy(),population[order[1]].copy()]
        while len(nxt)<c.population:
            a,b=select(),select();child=a.copy()
            if rng.random()<.85:
                i,j=int(rng.integers(1,len(a))),int(rng.integers(1,len(b)));candidate=np.vstack((a[:i],b[j:]))
                if 2<=len(candidate)<=c.max_waypoints:child=candidate
            if rng.random()<.4:child=mutate(child,w,c,rng)
            nxt.append(child)
        population=nxt
    return population[order[0]],history

def sample_trajectory(path,ts,samples_per_segment=101):
    rows,elapsed=[],0.
    for j,(p,q,T) in enumerate(zip(path[:-1],path[1:],ts)):
        u=np.linspace(0,1,samples_per_segment);s,ds,d2=quintic(u);d=q-p
        r=np.column_stack((elapsed+u*T,p+s[:,None]*d,ds[:,None]*d/T,d2[:,None]*d/T**2))
        rows.append(r if j==0 else r[1:]);elapsed+=T
    return np.vstack(rows)
def final_check(r,w,c):
    if not r.get('success'):return False
    try:
        p=np.asarray(r['path'],float);ts=np.asarray(r['segment_durations_s'],float)
        if p.ndim!=2 or p.shape[1]!=3 or len(p)<2 or ts.shape!=(len(p)-1,):return False
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(ts)) or np.any(ts<=0):return False
        if not np.allclose(p[0],w.start,rtol=0,atol=1e-8) or not np.allclose(p[-1],w.goal,rtol=0,atol=1e-8):return False
        rows=sample_trajectory(p,ts,401)
        return bool(np.max(np.linalg.norm(rows[:,4:7],axis=1))<=c.max_speed+1e-8 and np.max(np.linalg.norm(rows[:,7:10],axis=1))<=c.max_acceleration+1e-8 and np.max(np.abs(rows[:,6]))<=c.max_vertical_speed+1e-8 and all(w.valid_point(x,c) for x in rows[:,1:4]) and all(w.clear_segment(a,b,c) for a,b in zip(p[:-1],p[1:])))
    except (ValueError,TypeError,KeyError):return False
independent_check=final_check # Backward-compatible name only; shared geometry is not independent.
