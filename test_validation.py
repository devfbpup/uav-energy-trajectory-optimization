import unittest, copy
import numpy as np
import uav_planning as u
from uav_planning import *

def sat_hit(p,q,low,high):
    # Independent segment/AABB separating-axis formulation, not the slab code.
    c=(p+q)/2-(low+high)/2;d=(q-p)/2;e=(high-low)/2;ad=np.abs(d)
    if np.any(np.abs(c)>e+ad):return False
    for i,j,k in [(0,1,2),(1,2,0),(2,0,1)]:
        if abs(c[j]*d[k]-c[k]*d[j])>e[j]*ad[k]+e[k]*ad[j]+1e-12:return False
    return True
class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.c=Settings(population=10,generations=3,rrt_iterations=100);self.w=scenario('wall')
    def empty(self,size=8):
        return World('empty',np.zeros(3),np.full(3,float(size)),np.empty((0,2,3)),np.zeros(3),np.full(3,float(size)))
    def test01_quintic_boundaries(self):
        s,v,a=quintic(np.array([0.,1.]));np.testing.assert_allclose(s,[0,1]);np.testing.assert_allclose(v,0,atol=1e-12);np.testing.assert_allclose(a,0,atol=1e-12)
    def test02_segment_collision(self):
        self.assertTrue(self.w.valid_point(self.w.start,self.c));self.assertTrue(self.w.valid_point(self.w.goal,self.c));self.assertFalse(self.w.clear_segment(self.w.start,self.w.goal,self.c))
    def test03_grazing_parallel(self):
        z=12.75;self.assertFalse(self.w.clear_segment([4,20,z],[36,20,z],self.c));self.assertTrue(self.w.clear_segment([4,20,z+.001],[36,20,z+.001],self.c));self.assertTrue(self.w.clear_segment([4,20,4],[4,30,4],self.c))
    def test04_symmetry(self):
        rng=np.random.default_rng(19)
        for _ in range(100):
            a,b=rng.uniform(self.w.lower,self.w.upper,(2,3));self.assertEqual(self.w.clear_segment(a,b,self.c),self.w.clear_segment(b,a,self.c))
    def test05_nonfinite_and_bounds(self):
        self.assertFalse(self.w.valid_point([np.nan,2,3],self.c));self.assertFalse(self.w.valid_point([41,2,3],self.c))
    def test06_failed_energy(self):
        r=assess(np.array([self.w.start,self.w.goal]),self.w,self.c);self.assertFalse(r['success']);self.assertNotIn('equivalent_hover_time_s',r)
    def test07_hover_identity(self):
        p=np.array([[1.,1,4],[1.,1,4]]);self.assertAlmostEqual(energy_quotient(p,np.array([7.]),self.c),7*self.c.g**1.5,places=10)
    def test08_duration_bounds(self):
        rng=np.random.default_rng(23)
        for _ in range(30):
            p=rng.uniform([0,0,2],[40,40,18],(2,3));r=sample_trajectory(p,durations(p,self.c),2001)
            self.assertLessEqual(np.linalg.norm(r[:,4:7],axis=1).max(),self.c.max_speed);self.assertLessEqual(np.linalg.norm(r[:,7:10],axis=1).max(),self.c.max_acceleration);self.assertLessEqual(np.abs(r[:,6]).max(),self.c.max_vertical_speed)
    def test09_quadrature(self):
        p=np.array([[0.,0.,3.],[10.,5.,8.]]);ts=durations(p,self.c);a=energy_quotient(p,ts,self.c,24);b=energy_quotient(p,ts,self.c,64);r=sample_trajectory(p,ts,10001);acc=r[:,7:10].copy();acc[:,2]+=self.c.g
        other=np.trapezoid(np.linalg.norm(acc,axis=1)**1.5,r[:,0]);self.assertLess(abs(a-b)/b,1e-10);self.assertLess(abs(a-other)/other,1e-7)
    def test10_feasibility_ranking(self):
        self.assertLess(ranking(dict(success=True,length_m=1e9,equivalent_hover_time_s=1e9),'proxy',self.c),ranking(dict(success=False,violation=.0001),'proxy',self.c));self.assertLess(ranking(dict(success=False,violation=1),'proxy',self.c),ranking(dict(success=False,violation=2),'proxy',self.c))
    def test11_astar_empty(self):
        w=self.empty();p=astar(w,self.c);self.assertAlmostEqual(np.linalg.norm(np.diff(p,axis=0),axis=1).sum(),np.linalg.norm(w.goal-w.start))
    def test12_astar_obstacle(self):
        r=assess(astar(self.w,self.c),self.w,self.c);self.assertTrue(r['success']);self.assertTrue(final_check(r,self.w,self.c));self.assertIsNone(r['battery_energy_j'])
    def test13_rrt_reproducible(self):
        w=self.empty();a=rrt_star(w,self.c,3);b=rrt_star(w,self.c,3);np.testing.assert_allclose(a,b);self.assertTrue(assess(a,w,self.c)['success'])
    def test14_ga_reproducible_elitism(self):
        w=scenario('sparse');a,h=genetic(w,self.c,3,'proxy');b,h2=genetic(w,self.c,3,'proxy');np.testing.assert_allclose(a,b);self.assertEqual(h,h2);self.assertTrue(assess(a,w,self.c)['success']);scores=[x['best_score'] for x in h];self.assertTrue(all(y<=x+1e-10 for x,y in zip(scores[:-1],scores[1:])))
    def test15_mutation_endpoints(self):
        rng=np.random.default_rng(8);p=np.array([self.w.start,[4,32,4],[36,32,4],self.w.goal])
        for _ in range(50):
            p=mutate(p,self.w,self.c,rng);np.testing.assert_allclose(p[0],self.w.start);np.testing.assert_allclose(p[-1],self.w.goal);self.assertTrue(2<=len(p)<=self.c.max_waypoints)
    def test16_positive_parameters(self):
        for key in ('g','max_speed','max_acceleration','max_vertical_speed','grid_step','energy_reference_s','length_reference_m'):
            for value in (0,-1,float('nan'),float('inf')):
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):Settings(**{key:value})
    def test17_clearance_parameters(self):
        for key in ('vehicle_radius','clearance'):
            for value in (-1,float('nan')):
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):Settings(**{key:value})
    def test18_search_parameters(self):
        for kw in ({'population':3},{'population':4.5},{'generations':-1},{'rrt_iterations':-1},{'max_waypoints':1}):
            with self.subTest(kw=kw),self.assertRaises(ValueError):Settings(**kw)
    def test19_absolute_endpoints(self):
        w=self.empty(100);w.start=np.full(3,20.);w.goal=np.full(3,80.);p=np.array([w.start,w.goal]);p[0,0]+=1e-4;self.assertFalse(assess(p,w,self.c)['success'])
    def test20_changed_endpoints(self):
        w=self.empty();r=assess(np.array([w.start,w.goal]),w,self.c);r['path'][0][0]+=1;self.assertFalse(final_check(r,w,self.c))
    def test21_negative_times(self):
        w=self.empty();r=assess(np.array([w.start,w.goal]),w,self.c);r['segment_durations_s']=[-100.];self.assertFalse(final_check(r,w,self.c))
    def test22_nonfinite_times(self):
        w=self.empty();r=assess(np.array([w.start,w.goal]),w,self.c);r['segment_durations_s']=[float('inf')];self.assertFalse(final_check(r,w,self.c))
    def test23_initial_waypoint_cap(self):
        w=self.empty();c=Settings(max_waypoints=2,population=6,generations=0);original=u.assess;seen=[]
        def spy(p,*args):seen.append(len(p));return original(p,*args)
        u.assess=spy
        try:u.genetic(w,c,2,'distance')
        finally:u.assess=original
        self.assertTrue(seen);self.assertLessEqual(max(seen),2)
    def test24_independent_collision_formulation(self):
        rng=np.random.default_rng(6301)
        for scene in ('sparse','wall','culdesac'):
            w=scenario(scene)
            for _ in range(200):
                a,b=rng.uniform(w.lower,w.upper,(2,3));hit=any(sat_hit(a,b,lo,hi) for lo,hi in w.expanded_boxes(self.c))
                self.assertEqual(w.clear_segment(a,b,self.c),not hit)
if __name__=='__main__':unittest.main(verbosity=2)
