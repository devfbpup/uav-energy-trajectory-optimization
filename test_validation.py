"""Independent validation suite for uav_planning and urban_scenes (v0.3.0).

The collision test in this file (``sat_hit``) is written independently of the
slab test used by the planner, using the separating-axis theorem for a segment
against an axis-aligned box. It is used both by the unit tests and by
reproduce.py to re-check every accepted trajectory.

Run with:  python3 -m unittest test_validation -v
"""

import math
import unittest

import numpy as np

import uav_planning as u
import urban_scenes as us


def sat_hit(a, b, lo, hi):
    """True if segment a-b intersects the axis-aligned box [lo, hi].

    Separating-axis theorem: three box face normals plus three cross products
    of the segment direction with the box axes.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    centre = (lo + hi) / 2.0
    extent = (hi - lo) / 2.0
    m = (a + b) / 2.0 - centre
    d = (b - a) / 2.0
    ad = np.abs(d)
    for i in range(3):
        if abs(m[i]) > extent[i] + ad[i]:
            return False
    for i, j in ((0, 1), (0, 2), (1, 2)):
        lhs = abs(m[i] * d[j] - m[j] * d[i])
        rhs = extent[i] * ad[j] + extent[j] * ad[i]
        if lhs > rhs + 1e-12:
            return False
    return True


CFG = u.Settings()
URBAN_CFG = u.Settings(grid_step=4.0)


class TestGeometry(unittest.TestCase):
    def test_sat_agrees_with_slab_on_random_segments(self):
        rng = np.random.default_rng(7)
        w = u.scenario("wall")
        boxes = w.expanded_boxes(CFG)
        mismatches = 0
        for _ in range(600):
            p = rng.uniform(w.lower, w.upper)
            q = rng.uniform(w.lower, w.upper)
            slab = w.clear_segment(p, q, CFG)
            sat = not any(sat_hit(p, q, lo, hi) for lo, hi in boxes)
            inside = w.valid_point(p, CFG) and w.valid_point(q, CFG)
            if inside and slab != sat:
                mismatches += 1
        self.assertEqual(mismatches, 0)

    def test_known_intersection_and_miss(self):
        self.assertTrue(sat_hit([0, 0, 0], [10, 10, 10], [4, 4, 4], [6, 6, 6]))
        self.assertFalse(sat_hit([0, 0, 0], [1, 0, 0], [4, 4, 4], [6, 6, 6]))

    def test_endpoint_validity(self):
        w = u.scenario("wall")
        self.assertTrue(w.valid_point(w.start, CFG))
        self.assertFalse(w.valid_point(np.array([20.0, 10.0, 6.0]), CFG))

    def test_unknown_scene_rejected(self):
        with self.assertRaises(ValueError):
            u.scenario("atlantis")


class TestKinematics(unittest.TestCase):
    def test_quintic_boundary_conditions(self):
        s, ds, d2 = u.quintic(np.array([0.0, 1.0]))
        self.assertTrue(np.allclose(s, [0, 1]))
        self.assertTrue(np.allclose(ds, [0, 0]))
        self.assertTrue(np.allclose(d2, [0, 0]))

    def test_durations_respect_limits(self):
        w = u.scenario("sparse")
        p = np.array([w.start, [20.0, 8.0, 10.0], w.goal])
        ts = u.durations(p, CFG)
        rows = u.sample_trajectory(p, ts, 401)
        self.assertLessEqual(np.max(np.linalg.norm(rows[:, 4:7], axis=1)), CFG.max_speed + 1e-8)
        self.assertLessEqual(np.max(np.linalg.norm(rows[:, 7:10], axis=1)), CFG.max_acceleration + 1e-8)
        self.assertLessEqual(np.max(np.abs(rows[:, 6])), CFG.max_vertical_speed + 1e-8)

    def test_minimum_segment_duration(self):
        p = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]])
        self.assertGreaterEqual(u.durations(p, CFG)[0], 0.2)

    def test_quadrature_convergence(self):
        w = u.scenario("wall")
        p = np.array([w.start, [20.0, 30.0, 10.0], w.goal])
        ts = u.durations(p, CFG)
        a = u.energy_quotient(p, ts, CFG, 24)
        b = u.energy_quotient(p, ts, CFG, 64)
        self.assertLess(abs(a - b) / b, 1e-12)

    def test_energy_quotient_hover_limit(self):
        p = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 5.0 + 1e-9]])
        ts = np.array([10.0])
        eq = u.energy_quotient(p, ts, CFG)
        self.assertAlmostEqual(eq / CFG.g ** 1.5, 10.0, places=6)


class TestPropulsion(unittest.TestCase):
    def test_hover_power_matches_closed_form(self):
        prop = u.Propulsion()
        thrust = prop.hover_thrust_n
        vi = math.sqrt((thrust / prop.rotors) / (2 * prop.air_density_kgpm3 * prop.disk_area_m2))
        expected = (thrust * vi / prop.figure_of_merit + prop.hover_profile_power_w)
        expected = expected / prop.electrical_efficiency + prop.ancillary_power_w
        self.assertAlmostEqual(prop.hover_power_w(), expected, places=9)

    def test_power_increases_with_speed_and_acceleration(self):
        prop = u.Propulsion()
        base = prop.electrical_power_w(np.zeros(3), np.zeros(3))
        faster = prop.electrical_power_w(np.zeros(3), np.array([10.0, 0.0, 0.0]))
        accelerating = prop.electrical_power_w(np.array([0.0, 0.0, 2.0]), np.zeros(3))
        self.assertGreater(faster, base)
        self.assertGreater(accelerating, base)

    def test_energy_of_pure_hover(self):
        prop = u.Propulsion()
        p = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 5.0 + 1e-9]])
        ts = np.array([12.0])
        energy = u.electrical_energy_j(p, ts, CFG)
        self.assertAlmostEqual(energy, prop.hover_power_w() * 12.0, places=4)

    def test_reserve_reduces_usable_energy(self):
        prop = u.Propulsion(mission_energy_allowance_j=1000.0, reserve_fraction=0.25)
        self.assertAlmostEqual(prop.usable_energy_j, 750.0)

    def test_energy_quadrature_convergence(self):
        w = u.scenario("culdesac")
        p = np.array([w.start, [10.0, 34.0, 10.0], w.goal])
        ts = u.durations(p, CFG)
        a = u.electrical_energy_j(p, ts, CFG, 24)
        b = u.electrical_energy_j(p, ts, CFG, 64)
        self.assertLess(abs(a - b) / b, 1e-9)

    def test_invalid_propulsion_parameters_rejected(self):
        for kwargs in (
            dict(mass_kg=0.0),
            dict(rotors=0),
            dict(reserve_fraction=1.0),
            dict(figure_of_merit=1.5),
            dict(electrical_efficiency=2.0),
            dict(ancillary_power_w=-1.0),
        ):
            with self.assertRaises(ValueError):
                u.Propulsion(**kwargs)


class TestEnergyConstraint(unittest.TestCase):
    def test_exceeding_allowance_is_rejected(self):
        w = u.scenario("wall")
        cfg = u.Settings(propulsion=u.Propulsion(mission_energy_allowance_j=100.0))
        r = u.assess(u.astar(w, cfg), w, cfg)
        self.assertFalse(r["success"])
        self.assertEqual(r["reason"], "energy_budget_exceeded")

    def test_generous_allowance_is_accepted(self):
        w = u.scenario("wall")
        r = u.assess(u.astar(w, CFG), w, CFG)
        self.assertTrue(r["success"])
        self.assertTrue(r["energy_feasible"])
        self.assertLessEqual(r["model_electrical_energy_j"], r["mission_usable_energy_j"])

    def test_constraint_can_be_disabled(self):
        w = u.scenario("wall")
        cfg = u.Settings(propulsion=u.Propulsion(mission_energy_allowance_j=100.0), enforce_energy_budget=False)
        r = u.assess(u.astar(w, cfg), w, cfg)
        self.assertTrue(r["success"])
        self.assertFalse(r["energy_feasible"])

    def test_final_check_rejects_energy_violation(self):
        w = u.scenario("wall")
        r = u.assess(u.astar(w, CFG), w, CFG)
        tight = u.Settings(propulsion=u.Propulsion(mission_energy_allowance_j=100.0))
        self.assertTrue(u.final_check(r, w, CFG))
        self.assertFalse(u.final_check(r, w, tight))

    def test_ranking_prefers_feasible(self):
        feasible = dict(success=True, length_m=100.0, equivalent_hover_time_s=10.0, flight_duration_s=10.0, model_electrical_energy_j=1000.0)
        infeasible = dict(success=False, violation=0.1)
        self.assertLess(u.ranking(feasible, "energy", CFG), u.ranking(infeasible, "energy", CFG))


class TestAssess(unittest.TestCase):
    def test_rejects_endpoint_mismatch(self):
        w = u.scenario("wall")
        r = u.assess(np.array([[0.0, 0.0, 4.0], w.goal]), w, CFG)
        self.assertEqual(r["reason"], "endpoint_mismatch")

    def test_rejects_none_and_malformed(self):
        w = u.scenario("wall")
        self.assertEqual(u.assess(None, w, CFG)["reason"], "no_geometric_path")
        self.assertEqual(u.assess(np.array([1.0, 2.0, 3.0]), w, CFG)["reason"], "invalid_path")
        self.assertEqual(u.assess(np.array([[np.nan, 0, 0], [1, 1, 1]]), w, CFG)["reason"], "invalid_path")

    def test_rejects_collision(self):
        w = u.scenario("wall")
        cfg = u.Settings(shortcutting=False)
        r = u.assess(np.array([w.start, [20.0, 14.0, 6.0], w.goal]), w, cfg)
        self.assertEqual(r["reason"], "collision_or_bounds")

    def test_reports_stops_and_duration_together(self):
        w = u.scenario("wall")
        r = u.assess(u.astar(w, CFG), w, CFG)
        self.assertEqual(r["retained_stops"], r["decoded_waypoints"] - 2)
        self.assertAlmostEqual(r["flight_duration_s"], sum(r["segment_durations_s"]), places=9)
        self.assertAlmostEqual(
            r["mean_electrical_power_w"], r["model_electrical_energy_j"] / r["flight_duration_s"], places=6
        )


class TestPlanners(unittest.TestCase):
    def test_astar_deterministic_and_feasible(self):
        w = u.scenario("culdesac")
        a = u.astar(w, CFG)
        b = u.astar(w, CFG)
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(u.assess(a, w, CFG)["success"])

    def test_astar_requires_grid_alignment(self):
        w = u.scenario("sparse")
        w.start = w.start + 0.37
        with self.assertRaises(ValueError):
            u.astar(w, CFG)

    def test_rrt_star_reproducible(self):
        w = u.scenario("wall")
        a = u.rrt_star(w, CFG, 3)
        b = u.rrt_star(w, CFG, 3)
        self.assertTrue(np.array_equal(a, b))

    def test_genetic_reproducible_per_seed(self):
        w = u.scenario("wall")
        a, _ = u.genetic(w, CFG, 1, "proxy")
        b, _ = u.genetic(w, CFG, 1, "proxy")
        self.assertTrue(np.array_equal(a, b))

    def test_all_objectives_available_and_monotone_history(self):
        w = u.scenario("sparse")
        for objective in u.OBJECTIVES:
            best, history = u.genetic(w, CFG, 0, objective)
            r = u.assess(best, w, CFG)
            self.assertTrue(r["success"], objective)
            scores = [h["best_score"] for h in history if h["best_is_feasible"]]
            self.assertTrue(all(b <= a + 1e-12 for a, b in zip(scores[:-1], scores[1:])), objective)

    def test_time_objective_beats_proxy_on_duration(self):
        w = u.scenario("wall")
        durations = {}
        for objective in ("time", "distance"):
            best, _ = u.genetic(w, CFG, 0, objective)
            durations[objective] = u.assess(best, w, CFG)["flight_duration_s"]
        self.assertLessEqual(durations["time"], durations["distance"] + 1e-9)

    def test_unknown_objective_rejected(self):
        w = u.scenario("sparse")
        with self.assertRaises(ValueError):
            u.genetic(w, CFG, 0, "fuel")

    def test_evaluation_budget_is_respected(self):
        w = u.scenario("wall")
        cfg = u.Settings(evaluation_budget=120, generations=200)
        _, history = u.genetic(w, cfg, 0, "proxy")
        self.assertLessEqual(history[-1]["evaluations"], 120 + cfg.population)
        self.assertGreaterEqual(history[-1]["evaluations"], 120 - cfg.population)

    def test_ablation_switches_change_nothing_structurally_invalid(self):
        w = u.scenario("wall")
        for cfg in (
            u.Settings(warm_start=False),
            u.Settings(shortcutting=False),
            u.Settings(warm_start=False, shortcutting=False),
        ):
            best, _ = u.genetic(w, cfg, 0, "proxy")
            r = u.assess(best, w, cfg)
            self.assertTrue(r["success"])
            self.assertTrue(u.final_check(r, w, cfg))

    def test_shortcutting_disabled_keeps_all_waypoints(self):
        w = u.scenario("sparse")
        path = np.array([w.start, [10.0, 20.0, 4.0], [20.0, 30.0, 12.0], w.goal])
        cfg = u.Settings(shortcutting=False)
        self.assertEqual(len(u.shortcut(path, w, cfg)), len(path))
        self.assertLessEqual(len(u.shortcut(path, w, CFG)), len(path))

    def test_invalid_settings_rejected(self):
        for kwargs in (
            dict(max_speed=0.0),
            dict(population=2),
            dict(max_waypoints=1),
            dict(evaluation_budget=0),
            dict(grid_step=-1.0),
        ):
            with self.assertRaises(ValueError):
                u.Settings(**kwargs)


class TestSceneIngestion(unittest.TestCase):
    def test_obj_roundtrip_recovers_box(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "box.obj"
            vertices, faces = us._box_mesh((2.0, 2.0, 0.0), (6.0, 6.0, 8.0))
            lines = [f"v {x} {y} {z}" for x, y, z in vertices] + [f"f {a} {b} {c}" for a, b, c in faces]
            path.write_text("\n".join(lines) + "\n")
            triangles = us.read_obj_triangles(path)
            self.assertEqual(triangles.shape, (12, 3, 3))
            boxes, provenance = us.ingest_mesh(path, (0, 0, 0), (10, 10, 10), voxel_size=1.0)
            self.assertGreater(len(boxes), 0)
            self.assertEqual(len(provenance["asset_sha256"]), 64)
            union_lower = boxes[:, 0].min(axis=0)
            union_upper = boxes[:, 1].max(axis=0)
            self.assertTrue(np.all(union_lower <= np.array([2.0, 2.0, 0.0]) + 1e-9))
            self.assertTrue(np.all(union_upper >= np.array([6.0, 6.0, 8.0]) - 1e-9))

    def test_ingestion_is_conservative_for_interior_points(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "box.obj"
            vertices, faces = us._box_mesh((2.0, 2.0, 0.0), (6.0, 6.0, 8.0))
            lines = [f"v {x} {y} {z}" for x, y, z in vertices] + [f"f {a} {b} {c}" for a, b, c in faces]
            path.write_text("\n".join(lines) + "\n")
            boxes, _ = us.ingest_mesh(path, (0, 0, 0), (10, 10, 10), voxel_size=1.0)
            interior = np.array([4.0, 4.0, 4.0])
            covered = np.any(
                np.all(interior >= boxes[:, 0], axis=1) & np.all(interior <= boxes[:, 1], axis=1)
            )
            self.assertTrue(bool(covered))

    def test_city_assets_deterministic_checksums(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            first = us.build_city_mesh(f"{tmp}/a.obj", 101)
            second = us.build_city_mesh(f"{tmp}/b.obj", 101)
            third = us.build_city_mesh(f"{tmp}/c.obj", 202)
            self.assertEqual(first[0], second[0])
            self.assertNotEqual(first[0], third[0])
            self.assertGreater(first[1], 20)

    def test_urban_worlds_are_navigable(self):
        for name in us.URBAN_SEEDS:
            for route_set, count in (("development", 3), ("heldout", 2)):
                for i in range(count):
                    w = us.urban_world(name, i, route_set=route_set)
                    u.valid_endpoints(w, URBAN_CFG)
                    self.assertGreater(len(w.boxes), 20)
                    self.assertEqual(w.provenance["kind"], "ingested_mesh")

    def test_urban_planning_succeeds_and_validates(self):
        w = us.urban_world("urban_a", 0)
        path = u.astar(w, URBAN_CFG)
        self.assertIsNotNone(path)
        r = u.assess(path, w, URBAN_CFG)
        self.assertTrue(r["success"])
        self.assertTrue(u.final_check(r, w, URBAN_CFG))
        boxes = w.expanded_boxes(URBAN_CFG)
        p = np.array(r["path"])
        self.assertFalse(
            any(sat_hit(a, b, lo, hi) for a, b in zip(p[:-1], p[1:]) for lo, hi in boxes)
        )

    def test_urban_scene_denser_than_synthetic(self):
        self.assertGreater(len(us.urban_world("urban_b").boxes), len(u.scenario("culdesac").boxes))

    def test_unknown_urban_scene_rejected(self):
        with self.assertRaises(ValueError):
            us.urban_world("metropolis")


class TestValidationHelpers(unittest.TestCase):
    def test_final_check_rejects_broken_record(self):
        w = u.scenario("wall")
        self.assertFalse(u.final_check(dict(success=False), w, CFG))
        self.assertFalse(u.final_check(dict(success=True, path=[[0, 0, 0]], segment_durations_s=[]), w, CFG))

    def test_sample_trajectory_shape_and_continuity(self):
        w = u.scenario("sparse")
        p = np.array([w.start, [20.0, 8.0, 10.0], w.goal])
        ts = u.durations(p, CFG)
        rows = u.sample_trajectory(p, ts, 51)
        self.assertEqual(rows.shape, (101, 10))
        self.assertAlmostEqual(rows[-1, 0], ts.sum(), places=9)
        self.assertTrue(np.allclose(rows[0, 1:4], w.start))
        self.assertTrue(np.allclose(rows[-1, 1:4], w.goal))

    def test_backward_compatible_alias(self):
        self.assertIs(u.independent_check, u.final_check)

    def test_with_overrides_does_not_mutate(self):
        cfg = u.Settings()
        other = u.with_overrides(cfg, warm_start=False)
        self.assertTrue(cfg.warm_start)
        self.assertFalse(other.warm_start)


if __name__ == "__main__":
    unittest.main(verbosity=2)
