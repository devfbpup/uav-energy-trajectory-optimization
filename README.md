# Evolutionary Trajectory Optimization for Energy-Constrained UAVs in Dense Urban Environments

Reference implementation, test suite, and reproduction script for the numerical
results reported in the article of the same title.

Authors: Devjyoti Saha (Polytechnic University of the Philippines) and
Jo-Ann V. Magsumbol (De La Salle University).

## What this code is

A deterministic, dependency-light study of whether ranking quintic
rest-to-rest trajectories by an acceleration-based energy proxy produces
different routes than ranking them by path length.

- `uav_planning.py` — planners (grid A*, RRT*, genetic algorithm), the quintic
  time-scaling model, the segment/axis-aligned-box collision test, and the
  energy proxy. Version `0.2.0`.
  SHA-256: `98fde7c4e91652f377a80fedc51b007db0b3534261dc46d3b009c4aa0bbf0a37`
- `test_validation.py` — 24 unit tests, including a 600-sample cross-check of
  the collision routine against an independently written separating-axis
  formulation, an analytic hover identity, and a quadrature convergence test.
- `reproduce.py` — the full experimental protocol: 3 synthetic scenes x 3
  routes x 5 seeds x 4 planners = 180 attempts. Writes `runs/`, `protocol.json`
  and `summary.json`, and with `--check` compares the aggregates against the
  values reported in the article.

## What this code is not

These limits are stated in the article and repeated here so the code is not
misread:

- **The geometry is synthetic.** Three hand-specified axis-aligned box scenes
  in a 40 x 40 x 18 m volume. No UrbanScene3D mesh, no OpenStreetMap extract,
  and no registered real city model is used anywhere in this repository.
- **The energy quantity is an uncalibrated proxy, not joules.** It is the
  integral of the 3/2 power of proper acceleration, divided by g^1.5, so its
  unit is seconds — an equivalent hover time. No motor, propeller, battery, or
  aerodynamic parameters were fitted or measured, and `battery_energy_j` is
  deliberately `null` in every saved record.
- **There is no controller and no attitude dynamics.** Trajectories are C^2
  rest-to-rest straight segments; yaw, drag, wind, and closed-loop tracking are
  out of scope.
- **The planner comparisons are descriptive.** Five seeds per route with no
  matched computational budget and no held-out routes; no significance testing
  is claimed.

## Requirements

Python 3.13 and NumPy. The results reported in the article were produced with
Python 3.13.12 and NumPy 2.4.6 and independently reproduced bit-for-bit on a
second machine.

```bash
pip install numpy
```

## Running

```bash
python3 -m unittest test_validation -v     # 24 tests, ~1 s
python3 reproduce.py --check               # 180 attempts, ~1-3 min
```

A successful reproduction prints:

```
All reference values reproduced exactly.
```

Wall-clock runtimes in `summary.json` are machine dependent and are the only
quantities that legitimately differ between runs. Every geometric and
proxy-energy value is bit-reproducible under the same NumPy version.

## Reported aggregates

All 180 attempts were accepted (45 per planner). Planner order below is
A*, RRT*, distance-objective GA, proxy-objective GA.

| Quantity | A* | RRT* | Distance GA | Proxy GA |
| --- | --- | --- | --- | --- |
| Mean path length (m) | 35.398 | 36.293 | 35.364 | 36.562 |
| Mean proxy (s) | 19.630 | 20.111 | 19.558 | 18.647 |

Pooled kinematic maxima across all accepted attempts: 4.9505 m/s speed
(limit 5), 1.9606 m/s^2 acceleration (limit 2), 1.9802 m/s vertical speed
(limit 2). Maximum relative difference between 24- and 64-node Gauss-Legendre
quadrature: 4.18e-16.

Mean route-level proxy reduction of the proxy-objective GA: 4.29% against A*
and 4.02% against the distance-objective GA, averaged over the nine
scene-route pairs. The reduction is concentrated in the wall scene; in the
sparse scene every planner returns the same route and the difference is
exactly zero.

## License

MIT. See `LICENSE`. If you use this code, please cite the article; see
`CITATION.cff`.
