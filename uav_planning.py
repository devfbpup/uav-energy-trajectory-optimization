"""Planning, timing, energy and feasibility model for energy-constrained UAV
trajectory optimisation.

Version 0.3.0 extends 0.2.0 with

* a parameterised electrical propulsion model (momentum-theory induced power,
  profile power, parasite power, drive efficiency, ancillary load) that turns a
  decoded trajectory into joules,
* a hard mission-energy (battery) constraint applied inside feasibility ranking,
  so the planners are energy-constrained and not only energy-scored,
* a minimum-time objective and a minimum-electrical-energy objective in
  addition to the distance and acceleration-proxy objectives,
* an explicit evaluation budget so genetic variants can be compared under
  matched computational accounting, and
* switches for the A* warm start and the visibility shortcutting stage so both
  can be ablated.

The propulsion constants are literature-typical values for a small quadrotor.
They are a *parameterised reference model*, not a vehicle-specific calibration
fitted to measured flight data; energies are therefore model energies in joules
rather than measured battery consumption.
"""

import heapq
import itertools
import math
from dataclasses import dataclass, field, replace

import numpy as np

VERSION = "0.3.0"


# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------
@dataclass
class Propulsion:
    """Parameterised multirotor electrical power model."""

    g: float = 9.81
    mass_kg: float = 1.5
    rotors: int = 4
    rotor_radius_m: float = 0.12
    air_density_kgpm3: float = 1.225
    figure_of_merit: float = 0.70
    hover_profile_power_w: float = 22.0
    drag_area_m2: float = 0.06
    electrical_efficiency: float = 0.85
    ancillary_power_w: float = 10.0
    # Energy allocated to a single mission leg, before reserve.
    mission_energy_allowance_j: float = 30000.0
    reserve_fraction: float = 0.20

    def __post_init__(self):
        positive = (
            "g",
            "mass_kg",
            "rotor_radius_m",
            "air_density_kgpm3",
            "figure_of_merit",
            "electrical_efficiency",
            "mission_energy_allowance_j",
        )
        for name in positive:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        for name in ("hover_profile_power_w", "drag_area_m2", "ancillary_power_w"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(name + " must be finite and nonnegative")
        if isinstance(self.rotors, bool) or not isinstance(self.rotors, (int, np.integer)) or self.rotors < 1:
            raise ValueError("rotors must be a positive integer")
        if not 0.0 <= self.reserve_fraction < 1.0:
            raise ValueError("reserve_fraction must lie in [0, 1)")
        if self.figure_of_merit > 1.0 or self.electrical_efficiency > 1.0:
            raise ValueError("efficiencies cannot exceed one")

    @property
    def disk_area_m2(self):
        return math.pi * self.rotor_radius_m ** 2

    @property
    def hover_thrust_n(self):
        return self.mass_kg * self.g

    @property
    def usable_energy_j(self):
        """Allowance left after the reserve is withheld."""
        return self.mission_energy_allowance_j * (1.0 - self.reserve_fraction)

    def electrical_power_w(self, acceleration, velocity):
        """Instantaneous electrical power for arrays of shape (..., 3).

        Thrust follows from the rigid-body requirement T = m * ||a + g||.
        Induced power uses static momentum theory with a figure of merit,
        profile power is scaled from its hover value by (T / T_hover)^{3/2},
        parasite power uses a fixed equivalent flat-plate area, and the sum is
        divided by the drive efficiency before the ancillary load is added.
        """
        acceleration = np.asarray(acceleration, float)
        velocity = np.asarray(velocity, float)
        proper = acceleration.copy()
        proper[..., 2] += self.g
        thrust = self.mass_kg * np.linalg.norm(proper, axis=-1)
        per_rotor = thrust / self.rotors
        induced_velocity = np.sqrt(
            np.maximum(per_rotor, 0.0) / (2.0 * self.air_density_kgpm3 * self.disk_area_m2)
        )
        p_induced = thrust * induced_velocity / self.figure_of_merit
        p_profile = self.hover_profile_power_w * (thrust / self.hover_thrust_n) ** 1.5
        speed = np.linalg.norm(velocity, axis=-1)
        p_parasite = 0.5 * self.air_density_kgpm3 * self.drag_area_m2 * speed ** 3
        shaft = p_induced + p_profile + p_parasite
        return shaft / self.electrical_efficiency + self.ancillary_power_w

    def hover_power_w(self):
        return float(self.electrical_power_w(np.zeros(3), np.zeros(3)))


@dataclass
class Settings:
    g: float = 9.81
    max_speed: float = 5.0
    max_acceleration: float = 2.0
    max_vertical_speed: float = 2.0
    vehicle_radius: float = 0.25
    clearance: float = 0.5
    grid_step: float = 2.0
    population: int = 24
    generations: int = 25
    max_waypoints: int = 16
    rrt_iterations: int = 500
    energy_reference_s: float = 100.0
    length_reference_m: float = 100.0
    time_reference_s: float = 100.0
    energy_reference_j: float = 10000.0
    # Matched computational accounting: maximum number of distinct trajectory
    # evaluations a genetic run may consume. None means "generation limited".
    evaluation_budget: int = None
    warm_start: bool = True
    shortcutting: bool = True
    enforce_energy_budget: bool = True
    propulsion: Propulsion = field(default_factory=Propulsion)

    def __post_init__(self):
        for name in (
            "g",
            "max_speed",
            "max_acceleration",
            "max_vertical_speed",
            "grid_step",
            "energy_reference_s",
            "length_reference_m",
            "time_reference_s",
            "energy_reference_j",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        for name in ("vehicle_radius", "clearance"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(name + " must be finite and nonnegative")
        for name, minimum in [
            ("population", 4),
            ("generations", 0),
            ("max_waypoints", 2),
            ("rrt_iterations", 0),
        ]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError("Invalid " + name)
        if self.evaluation_budget is not None:
            if (
                isinstance(self.evaluation_budget, bool)
                or not isinstance(self.evaluation_budget, (int, np.integer))
                or self.evaluation_budget < 1
            ):
                raise ValueError("Invalid evaluation_budget")
        if isinstance(self.propulsion, dict):
            self.propulsion = Propulsion(**self.propulsion)


OBJECTIVES = ("distance", "proxy", "time", "energy")


# ----------------------------------------------------------------------------
# worlds
# ----------------------------------------------------------------------------
@dataclass
class World:
    name: str
    lower: np.ndarray
    upper: np.ndarray
    boxes: np.ndarray
    start: np.ndarray
    goal: np.ndarray
    provenance: dict = field(default_factory=dict)

    def expanded_boxes(self, cfg):
        key = (float(cfg.vehicle_radius), float(cfg.clearance))
        cached = getattr(self, "_expanded_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        b = np.asarray(self.boxes, float).reshape(-1, 2, 3).copy()
        b[:, 0] -= cfg.vehicle_radius + cfg.clearance
        b[:, 1] += cfg.vehicle_radius + cfg.clearance
        object.__setattr__(self, "_expanded_cache", (key, b))
        return b

    def valid_point(self, p, cfg):
        p = np.asarray(p, float)
        if p.shape != (3,) or not np.all(np.isfinite(p)):
            return False
        if np.any(p < self.lower) or np.any(p > self.upper):
            return False
        b = self.expanded_boxes(cfg)
        if len(b) == 0:
            return True
        return not bool(np.any(np.all(p >= b[:, 0], axis=1) & np.all(p <= b[:, 1], axis=1)))

    def clear_segment(self, p, q, cfg):
        """Vectorised full-segment slab test against every inflated box."""
        if not self.valid_point(p, cfg) or not self.valid_point(q, cfg):
            return False
        p = np.asarray(p, float)
        q = np.asarray(q, float)
        boxes = self.expanded_boxes(cfg)
        if len(boxes) == 0:
            return True
        d = q - p
        low = boxes[:, 0]
        high = boxes[:, 1]
        enter = np.zeros(len(boxes))
        leave = np.ones(len(boxes))
        alive = np.ones(len(boxes), bool)
        for axis in range(3):
            if abs(d[axis]) < 1e-12:
                outside = (p[axis] < low[:, axis]) | (p[axis] > high[:, axis])
                alive &= ~outside
            else:
                t0 = (low[:, axis] - p[axis]) / d[axis]
                t1 = (high[:, axis] - p[axis]) / d[axis]
                lo = np.minimum(t0, t1)
                hi = np.maximum(t0, t1)
                enter = np.maximum(enter, lo)
                leave = np.minimum(leave, hi)
            alive &= enter <= leave
            if not alive.any():
                return True
        return not bool(alive.any())


def scenario(name="wall"):
    scenes = {
        "sparse": [[[18, 18, 2], [22, 22, 8]]],
        "wall": [[[18, 0, 2], [22, 28, 12]]],
        "culdesac": [
            [[18, 10, 2], [22, 30, 14]],
            [[2, 10, 2], [18, 12, 14]],
            [[2, 28, 2], [18, 30, 14]],
        ],
    }
    if name not in scenes:
        raise ValueError("Unknown scene")
    return World(
        name,
        np.array([0.0, 0.0, 2.0]),
        np.array([40.0, 40.0, 18.0]),
        np.array(scenes[name], float),
        np.array([10.0, 20.0, 4.0] if name == "culdesac" else [4.0, 20.0, 4.0]),
        np.array([36.0, 20.0, 4.0]),
        {"kind": "synthetic_boxes", "source": "hand specified"},
    )


def valid_endpoints(w, c):
    if not w.valid_point(w.start, c) or not w.valid_point(w.goal, c):
        raise ValueError("Infeasible endpoints")


# ----------------------------------------------------------------------------
# trajectory construction
# ----------------------------------------------------------------------------
def shortcut(path, w, c):
    p = np.asarray(path, float)
    if len(p) < 2 or not getattr(c, "shortcutting", True):
        return p
    out, i = [p[0]], 0
    while i < len(p) - 1:
        j = len(p) - 1
        while j > i + 1 and not w.clear_segment(p[i], p[j], c):
            j -= 1
        out.append(p[j])
        i = j
    return np.array(out)


def quintic(tau):
    u = np.asarray(tau, float)
    return (
        10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5,
        30 * u ** 2 - 60 * u ** 3 + 30 * u ** 4,
        60 * u - 180 * u ** 2 + 120 * u ** 3,
    )


def durations(path, c):
    d = np.diff(path, axis=0)
    L = np.linalg.norm(d, axis=1)
    return 1.01 * np.maximum.reduce(
        [
            1.875 * L / c.max_speed,
            np.sqrt((10 * np.sqrt(3) / 3) * L / c.max_acceleration),
            1.875 * np.abs(d[:, 2]) / c.max_vertical_speed,
            np.full(len(d), 0.2),
        ]
    )


def _quadrature(order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return (nodes + 1) / 2, weights / 2


def energy_quotient(path, times, c, quadrature_order=24):
    tau, weights = _quadrature(quadrature_order)
    _, _, d2 = quintic(tau)
    delta = np.diff(path, axis=0)
    a = delta[:, None, :] * d2[None, :, None] / times[:, None, None] ** 2
    a[:, :, 2] += c.g
    return float(np.sum(times[:, None] * weights[None, :] * np.linalg.norm(a, axis=2) ** 1.5))


def electrical_energy_j(path, times, c, quadrature_order=24):
    """Model electrical energy of the decoded trajectory, in joules."""
    tau, weights = _quadrature(quadrature_order)
    _, d1, d2 = quintic(tau)
    delta = np.diff(path, axis=0)
    velocity = delta[:, None, :] * d1[None, :, None] / times[:, None, None]
    acceleration = delta[:, None, :] * d2[None, :, None] / times[:, None, None] ** 2
    power = c.propulsion.electrical_power_w(acceleration, velocity)
    return float(np.sum(times[:, None] * weights[None, :] * power))


def assess(raw, w, c):
    """Decode, check and score a candidate polyline."""
    if raw is None:
        return dict(success=False, reason="no_geometric_path", violation=1e9)
    p = np.asarray(raw, float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 2 or not np.all(np.isfinite(p)):
        return dict(success=False, reason="invalid_path", violation=1e9)
    if not np.allclose(p[0], w.start, rtol=0, atol=1e-8) or not np.allclose(p[-1], w.goal, rtol=0, atol=1e-8):
        return dict(success=False, reason="endpoint_mismatch", violation=1e9)
    p = p[np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-10]]
    if len(p) < 2:
        return dict(success=False, reason="zero_distance_mission", violation=1e9)
    p = shortcut(p, w, c)
    bad = sum(not w.valid_point(x, c) for x in p) + sum(
        not w.clear_segment(a, b, c) for a, b in zip(p[:-1], p[1:])
    )
    if bad:
        return dict(success=False, reason="collision_or_bounds", violation=float(bad))
    ts = durations(p, c)
    eq = energy_quotient(p, ts, c)
    energy = electrical_energy_j(p, ts, c)
    usable = c.propulsion.usable_energy_j
    record = dict(
        success=True,
        reason="passed_translational_and_energy_checks",
        violation=0.0,
        path=p.tolist(),
        segment_durations_s=ts.tolist(),
        length_m=float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()),
        flight_duration_s=float(ts.sum()),
        energy_quotient_m1p5_sminus2=eq,
        equivalent_hover_time_s=eq / c.g ** 1.5,
        model_electrical_energy_j=energy,
        mission_usable_energy_j=usable,
        energy_margin_j=usable - energy,
        energy_feasible=bool(energy <= usable),
        mean_electrical_power_w=energy / float(ts.sum()),
        decoded_waypoints=len(p),
        retained_stops=len(p) - 2,
        battery_energy_j=energy,
    )
    if c.enforce_energy_budget and not record["energy_feasible"]:
        return dict(
            record,
            success=False,
            reason="energy_budget_exceeded",
            violation=float(energy / usable),
        )
    return record


def objective_value(r, objective, c):
    if objective == "distance":
        return r["length_m"] / c.length_reference_m
    if objective == "proxy":
        return r["equivalent_hover_time_s"] / c.energy_reference_s
    if objective == "time":
        return r["flight_duration_s"] / c.time_reference_s
    if objective == "energy":
        return r["model_electrical_energy_j"] / c.energy_reference_j
    raise ValueError("Unknown objective")


def ranking(r, objective, c):
    """Feasibility-first ranking (Deb's rule): feasible before infeasible."""
    if not r["success"]:
        return 1, r["violation"]
    return 0, objective_value(r, objective, c)


# ----------------------------------------------------------------------------
# baseline planners
# ----------------------------------------------------------------------------
def astar(w, c):
    valid_endpoints(w, c)
    step = c.grid_step

    def index(p):
        r = (p - w.lower) / step
        if not np.allclose(r, np.round(r), rtol=0, atol=1e-8):
            raise ValueError("Grid-aligned endpoints required")
        return tuple(np.round(r).astype(int))

    def point(k):
        return w.lower + np.array(k) * step

    start, goal = index(w.start), index(w.goal)
    dims = np.floor((w.upper - w.lower) / step).astype(int)
    moves = [m for m in itertools.product((-1, 0, 1), repeat=3) if m != (0, 0, 0)]
    costs, parents = {start: 0.0}, {}
    heap = [(float(np.linalg.norm(w.goal - w.start)), 0.0, start)]
    while heap:
        _, cost, cur = heapq.heappop(heap)
        if cost > costs[cur] + 1e-10:
            continue
        if cur == goal:
            chain = [cur]
            while chain[-1] != start:
                chain.append(parents[chain[-1]])
            return np.array([point(k) for k in chain[::-1]])
        p = point(cur)
        for move in moves:
            nxt = tuple(cur[i] + move[i] for i in range(3))
            if any(nxt[i] < 0 or nxt[i] > dims[i] for i in range(3)):
                continue
            q = point(nxt)
            nc = cost + float(np.linalg.norm(q - p))
            if nc >= costs.get(nxt, float("inf")) - 1e-10:
                continue
            if w.clear_segment(p, q, c):
                costs[nxt], parents[nxt] = nc, cur
                heapq.heappush(heap, (nc + float(np.linalg.norm(q - w.goal)), nc, nxt))
    return None


def rrt_star(w, c, seed=0, iterations=None):
    valid_endpoints(w, c)
    iterations = c.rrt_iterations if iterations is None else int(iterations)
    rng = np.random.default_rng(seed)
    span = float(np.linalg.norm(w.upper - w.lower))
    step_length = max(4.0, span / 12.0)
    points, parents, costs, children = [w.start.copy()], [-1], [0.0], [set()]
    for _ in range(iterations):
        target = w.goal if rng.random() < 0.15 else rng.uniform(w.lower, w.upper)
        distances = np.linalg.norm(np.array(points) - target, axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] < 1e-9:
            continue
        q = points[nearest] + (target - points[nearest]) * min(1.0, step_length / distances[nearest])
        if not w.clear_segment(points[nearest], q, c):
            continue
        ds = np.linalg.norm(np.array(points) - q, axis=1)
        radius = min(3 * step_length, 8 * step_length * (math.log(len(points) + 1) / (len(points) + 1)) ** (1 / 3))
        near = set(np.flatnonzero(ds <= radius).tolist()) | {nearest}
        candidates = sorted(near, key=lambda j: costs[j] + ds[j])
        parent = next(j for j in candidates if w.clear_segment(points[j], q, c))
        idx = len(points)
        points.append(q)
        parents.append(parent)
        costs.append(costs[parent] + float(ds[parent]))
        children.append(set())
        children[parent].add(idx)
        ancestors, cur = set(), parent
        while cur != -1:
            ancestors.add(cur)
            cur = parents[cur]
        for j in near:
            if j in ancestors or j == 0:
                continue
            nc = costs[idx] + float(ds[j])
            if nc + 1e-10 < costs[j] and w.clear_segment(q, points[j], c):
                children[parents[j]].remove(j)
                parents[j] = idx
                children[idx].add(j)
                change = nc - costs[j]
                stack = [j]
                while stack:
                    k = stack.pop()
                    costs[k] += change
                    stack.extend(children[k])
    candidates = [i for i, p in enumerate(points) if w.clear_segment(p, w.goal, c)]
    if not candidates:
        return None
    j = min(candidates, key=lambda i: costs[i] + np.linalg.norm(points[i] - w.goal))
    chain = [w.goal]
    while j != -1:
        chain.append(points[j])
        j = parents[j]
    return np.array(chain[::-1])


# ----------------------------------------------------------------------------
# genetic search
# ----------------------------------------------------------------------------
def mutate(path, w, c, rng):
    p = path.copy()
    op = rng.choice(["shift", "insert", "delete"])
    if op == "insert" and len(p) < c.max_waypoints:
        i = int(rng.integers(len(p) - 1))
        q = (p[i] + p[i + 1]) / 2 + rng.normal(0, 3.0, 3)
        p = np.insert(p, i + 1, np.clip(q, w.lower, w.upper), axis=0)
    elif op == "delete" and len(p) > 2:
        p = np.delete(p, int(rng.integers(1, len(p) - 1)), axis=0)
    elif len(p) > 2:
        i = int(rng.integers(1, len(p) - 1))
        p[i] = np.clip(p[i] + rng.normal(0, 3.0, 3), w.lower, w.upper)
    return p


def genetic(w, c, seed=0, objective="proxy", warm_path=None):
    """Variable-length GA. Returns (best_chromosome, history).

    The run stops at ``c.generations`` generation updates or when the distinct
    trajectory evaluations reach ``c.evaluation_budget``, whichever comes first.
    ``warm_path`` allows a pre-computed A* solution to be reused so that a
    cached deterministic warm start is not recomputed for every seed; the time
    spent on it is accounted for separately by the caller.
    """
    if objective not in OBJECTIVES:
        raise ValueError("Unknown objective")
    valid_endpoints(w, c)
    rng = np.random.default_rng(seed)
    if c.warm_start:
        warm = warm_path if warm_path is not None else astar(w, c)
        warm = shortcut(np.asarray(warm, float), w, c) if warm is not None else np.array([w.start, w.goal])
    else:
        warm = np.array([w.start, w.goal])
    if len(warm) > c.max_waypoints:
        warm = warm[np.linspace(0, len(warm) - 1, c.max_waypoints, dtype=int)]
    population = [warm]
    while len(population) < c.population:
        if c.warm_start and rng.random() < 0.5:
            population.append(mutate(warm, w, c, rng))
        else:
            maximum = min(4, c.max_waypoints - 2)
            k = int(rng.integers(1, maximum + 1)) if maximum else 0
            population.append(np.vstack((w.start, rng.uniform(w.lower, w.upper, (k, 3)), w.goal)))

    history, cache = [], {}
    counter = {"evaluations": 0}

    def score(p):
        key = p.tobytes()
        if key not in cache:
            cache[key] = assess(p, w, c)
            counter["evaluations"] += 1
        return cache[key]

    budget = c.evaluation_budget
    order = None
    for generation in range(c.generations + 1):
        order = sorted(range(len(population)), key=lambda i: ranking(score(population[i]), objective, c))
        best = score(population[order[0]])
        history.append(
            dict(
                generation=generation,
                evaluations=counter["evaluations"],
                feasible_count=sum(score(p)["success"] for p in population),
                best_score=ranking(best, objective, c)[1],
                best_is_feasible=best["success"],
                best_length_m=best.get("length_m"),
                best_duration_s=best.get("flight_duration_s"),
                best_proxy_s=best.get("equivalent_hover_time_s"),
                best_energy_j=best.get("model_electrical_energy_j"),
            )
        )
        if generation == c.generations:
            break
        if budget is not None and counter["evaluations"] >= budget:
            break

        def select():
            ids = rng.integers(0, len(population), 3)
            return population[min(ids, key=lambda i: ranking(score(population[i]), objective, c))]

        nxt = [population[order[0]].copy(), population[order[1]].copy()]
        while len(nxt) < c.population:
            a, b = select(), select()
            child = a.copy()
            if rng.random() < 0.85:
                i, j = int(rng.integers(1, len(a))), int(rng.integers(1, len(b)))
                candidate = np.vstack((a[:i], b[j:]))
                if 2 <= len(candidate) <= c.max_waypoints:
                    child = candidate
            if rng.random() < 0.4:
                child = mutate(child, w, c, rng)
            nxt.append(child)
        population = nxt
    return population[order[0]], history


# ----------------------------------------------------------------------------
# validation helpers
# ----------------------------------------------------------------------------
def sample_trajectory(path, ts, samples_per_segment=101):
    rows, elapsed = [], 0.0
    for j, (p, q, T) in enumerate(zip(path[:-1], path[1:], ts)):
        u = np.linspace(0, 1, samples_per_segment)
        s, ds, d2 = quintic(u)
        d = q - p
        r = np.column_stack((elapsed + u * T, p + s[:, None] * d, ds[:, None] * d / T, d2[:, None] * d / T ** 2))
        rows.append(r if j == 0 else r[1:])
        elapsed += T
    return np.vstack(rows)


def final_check(r, w, c):
    if not r.get("success"):
        return False
    try:
        p = np.asarray(r["path"], float)
        ts = np.asarray(r["segment_durations_s"], float)
        if p.ndim != 2 or p.shape[1] != 3 or len(p) < 2 or ts.shape != (len(p) - 1,):
            return False
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(ts)) or np.any(ts <= 0):
            return False
        if not np.allclose(p[0], w.start, rtol=0, atol=1e-8) or not np.allclose(p[-1], w.goal, rtol=0, atol=1e-8):
            return False
        if c.enforce_energy_budget:
            energy = electrical_energy_j(p, ts, c)
            if energy > c.propulsion.usable_energy_j:
                return False
        rows = sample_trajectory(p, ts, 401)
        return bool(
            np.max(np.linalg.norm(rows[:, 4:7], axis=1)) <= c.max_speed + 1e-8
            and np.max(np.linalg.norm(rows[:, 7:10], axis=1)) <= c.max_acceleration + 1e-8
            and np.max(np.abs(rows[:, 6])) <= c.max_vertical_speed + 1e-8
            and all(w.valid_point(x, c) for x in rows[:, 1:4])
            and all(w.clear_segment(a, b, c) for a, b in zip(p[:-1], p[1:]))
        )
    except (ValueError, TypeError, KeyError):
        return False


independent_check = final_check  # Backward-compatible name only.


def with_overrides(cfg, **kwargs):
    """Return a copy of a Settings object with fields replaced."""
    return replace(cfg, **kwargs)
