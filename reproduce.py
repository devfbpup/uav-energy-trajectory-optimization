"""Full experimental protocol for the article.

Stages
------
1. Asset generation and mesh ingestion (dense urban scenes) with checksums.
2. Main protocol: 6 scenes x (3 development + 2 held-out routes) x 5 seeds x
   6 planners, each output revalidated with the shared checker, an independently
   written separating-axis collision test, analytic kinematic bounds, a
   24-versus-64 node quadrature check, and the mission-energy constraint.
3. Matched-budget sweep: every genetic objective at 150/300/600/1200 trajectory
   evaluations and RRT* at 250-4000 iterations, on a fixed route subset.
4. Ablations: A* warm start and visibility shortcutting switched off.
5. Aggregation: per-planner means, route-level paired differences, cluster
   bootstrap confidence intervals and Holm-adjusted permutation tests.

Usage
-----
    python3 reproduce.py                 # run everything, write runs/ + summary.json
    python3 reproduce.py --check         # additionally compare with reference.json
    python3 reproduce.py --write-reference
    python3 reproduce.py --stage main    # run a single stage

The comparison in --check is a numerical comparison within an explicit
tolerance (1e-9 relative and absolute); it is not a bitwise file comparison.
"""

import argparse
import json
import math
import platform
import statistics
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import uav_planning as u
import urban_scenes as us
from test_validation import sat_hit

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs"
TOLERANCE = 1e-9

PLANNERS = ["astar", "rrt_star", "distance_ga", "proxy_ga", "time_ga", "energy_ga"]
GA_OBJECTIVE = {"distance_ga": "distance", "proxy_ga": "proxy", "time_ga": "time", "energy_ga": "energy"}
SYNTHETIC_SCENES = ["sparse", "wall", "culdesac"]
URBAN_SCENES = ["urban_a", "urban_b", "urban_c"]
SYNTHETIC_DEV_Y = [16.0, 20.0, 24.0]
SYNTHETIC_HELDOUT_Y = [14.0, 26.0]
SEEDS = list(range(5))
BUDGETS = [150, 300, 600, 1200]
RRT_BUDGETS = [250, 500, 1000, 2000, 4000]
SUBSET_SEEDS = [0, 1, 2]

SYNTHETIC_CFG = dict(grid_step=2.0)
URBAN_CFG = dict(grid_step=4.0)


def settings_for(scene, **overrides):
    base = URBAN_CFG if scene in URBAN_SCENES else SYNTHETIC_CFG
    return u.Settings(**dict(base, **overrides))


def routes(route_set="development"):
    """Yield (scene, route_id, world_factory) for the requested route set."""
    ys = SYNTHETIC_DEV_Y if route_set == "development" else SYNTHETIC_HELDOUT_Y
    for scene in SYNTHETIC_SCENES:
        for y in ys:
            yield scene, f"{scene}_y{int(y)}", (lambda s=scene, y=y: us.synthetic_world(s, y))
    count = 3 if route_set == "development" else 2
    for scene in URBAN_SCENES:
        for i in range(count):
            yield scene, f"{scene}_r{i}", (
                lambda s=scene, i=i, rs=route_set: us.urban_world(s, i, route_set=rs)
            )


def validate(record, world, cfg):
    """Post-hoc validation of an accepted output."""
    p = np.array(record["path"])
    ts = np.array(record["segment_durations_s"])
    record["final_check"] = u.final_check(record, world, cfg)
    record["sat_collision_check"] = not any(
        sat_hit(a, b, lo, hi)
        for a, b in zip(p[:-1], p[1:])
        for lo, hi in world.expanded_boxes(cfg)
    )
    delta = np.diff(p, axis=0)
    L = np.linalg.norm(delta, axis=1)
    record["exact_max_speed_mps"] = float(np.max(1.875 * L / ts))
    record["exact_max_acceleration_mps2"] = float(np.max((10 * np.sqrt(3) / 3) * L / ts ** 2))
    record["exact_max_vertical_mps"] = float(np.max(1.875 * np.abs(delta[:, 2]) / ts))
    record["analytic_bounds_check"] = bool(
        record["exact_max_speed_mps"] <= cfg.max_speed + 1e-8
        and record["exact_max_acceleration_mps2"] <= cfg.max_acceleration + 1e-8
        and record["exact_max_vertical_mps"] <= cfg.max_vertical_speed + 1e-8
    )
    eq64 = u.energy_quotient(p, ts, cfg, 64)
    record["quadrature_relative_difference"] = abs(eq64 - record["energy_quotient_m1p5_sminus2"]) / eq64
    e64 = u.electrical_energy_j(p, ts, cfg, 64)
    record["energy_quadrature_relative_difference"] = abs(e64 - record["model_electrical_energy_j"]) / e64
    record["quadrature_check"] = bool(
        record["quadrature_relative_difference"] < 1e-9
        and record["energy_quadrature_relative_difference"] < 1e-9
    )
    record["energy_constraint_check"] = bool(
        record["model_electrical_energy_j"] <= cfg.propulsion.usable_energy_j
    )
    keys = [
        "final_check",
        "sat_collision_check",
        "analytic_bounds_check",
        "quadrature_check",
        "energy_constraint_check",
    ]
    if not all(record[k] for k in keys):
        record.update(success=False, reason="post_validation_failure")
    return record


def plan(planner, world, cfg, seed, warm_cache):
    """Run one planner. Returns (raw_path, history, search_time_s)."""
    key = (world.name, tuple(world.start), tuple(world.goal), cfg.grid_step)
    if planner == "astar":
        if key not in warm_cache:
            t0 = time.perf_counter()
            warm_cache[key] = (u.astar(world, cfg), time.perf_counter() - t0)
        raw, elapsed = warm_cache[key]
        return (None if raw is None else raw.copy()), [], elapsed
    if planner == "rrt_star":
        t0 = time.perf_counter()
        raw = u.rrt_star(world, cfg, seed)
        return raw, [], time.perf_counter() - t0
    if key not in warm_cache:
        t0 = time.perf_counter()
        warm_cache[key] = (u.astar(world, cfg), time.perf_counter() - t0)
    warm, warm_time = warm_cache[key]
    t0 = time.perf_counter()
    raw, history = u.genetic(world, cfg, seed, GA_OBJECTIVE[planner], warm_path=warm)
    # The deterministic A* warm start is computed once per route and reused for
    # every seed; its measured cost is added back so GA timing stays honest.
    return raw, history, (time.perf_counter() - t0) + (warm_time if cfg.warm_start else 0.0)


def run_main(route_set="development"):
    OUT.mkdir(exist_ok=True)
    records = []
    warm_cache = {}
    for scene, route_id, factory in routes(route_set):
        world = factory()
        cfg = settings_for(scene)
        u.valid_endpoints(world, cfg)
        for seed in SEEDS:
            for planner in PLANNERS:
                raw, history, elapsed = plan(planner, world, cfg, seed, warm_cache)
                r = u.assess(raw, world, cfg)
                r.update(
                    planner=planner,
                    scene=scene,
                    route=route_id,
                    route_set=route_set,
                    seed=seed,
                    runtime_s=elapsed,
                    scene_kind="urban" if scene in URBAN_SCENES else "synthetic",
                    provenance=world.provenance,
                )
                if r["success"]:
                    r = validate(r, world, cfg)
                r["history"] = history
                (OUT / f"{route_set}_{route_id}_seed{seed}_{planner}.json").write_text(json.dumps(r, indent=2))
                records.append(r)
        print(f"completed {route_set} {route_id}: {len(records)} attempts", flush=True)
    return records


def subset_routes():
    """One development route per scene, used for sweeps and ablations."""
    chosen = []
    for scene, route_id, factory in routes("development"):
        if not any(r[0] == scene for r in chosen):
            chosen.append((scene, route_id, factory))
    return chosen


def _merge(path, rows, keys):
    """Merge freshly computed rows into a JSON artefact, replacing overlaps."""
    existing = json.loads(path.read_text()) if path.exists() else []
    fresh = {tuple(r[k] for k in keys) for r in rows}
    kept = [r for r in existing if tuple(r[k] for k in keys) not in fresh]
    merged = kept + rows
    path.write_text(json.dumps(merged, indent=2))
    return merged


def run_budget_sweep(scene_filter=None):
    rows = []
    warm_cache = {}
    for scene, route_id, factory in subset_routes():
        if scene_filter and scene not in scene_filter:
            continue
        world = factory()
        for budget in BUDGETS:
            cfg = settings_for(scene, evaluation_budget=budget, generations=200)
            for planner in ["distance_ga", "proxy_ga", "time_ga", "energy_ga"]:
                for seed in SUBSET_SEEDS:
                    raw, history, elapsed = plan(planner, world, cfg, seed, warm_cache)
                    r = u.assess(raw, world, cfg)
                    rows.append(
                        dict(
                            kind="genetic",
                            scene=scene,
                            route=route_id,
                            planner=planner,
                            seed=seed,
                            budget=budget,
                            evaluations=history[-1]["evaluations"] if history else None,
                            runtime_s=elapsed,
                            success=r["success"],
                            length_m=r.get("length_m"),
                            flight_duration_s=r.get("flight_duration_s"),
                            equivalent_hover_time_s=r.get("equivalent_hover_time_s"),
                            model_electrical_energy_j=r.get("model_electrical_energy_j"),
                        )
                    )
        for iterations in RRT_BUDGETS:
            cfg = settings_for(scene, rrt_iterations=iterations)
            for seed in SUBSET_SEEDS:
                t0 = time.perf_counter()
                raw = u.rrt_star(world, cfg, seed, iterations)
                elapsed = time.perf_counter() - t0
                r = u.assess(raw, world, cfg)
                rows.append(
                    dict(
                        kind="rrt",
                        scene=scene,
                        route=route_id,
                        planner="rrt_star",
                        seed=seed,
                        budget=iterations,
                        evaluations=None,
                        runtime_s=elapsed,
                        success=r["success"],
                        length_m=r.get("length_m"),
                        flight_duration_s=r.get("flight_duration_s"),
                        equivalent_hover_time_s=r.get("equivalent_hover_time_s"),
                        model_electrical_energy_j=r.get("model_electrical_energy_j"),
                    )
                )
        print(f"budget sweep done for {route_id}", flush=True)
    return _merge(ROOT / "budget_sweep.json", rows, ("route", "planner", "seed", "budget", "kind"))


ABLATIONS = {
    "full": dict(),
    "no_warm_start": dict(warm_start=False),
    "no_shortcutting": dict(shortcutting=False),
    "no_warm_start_no_shortcutting": dict(warm_start=False, shortcutting=False),
}


def run_ablations(scene_filter=None):
    rows = []
    warm_cache = {}
    for scene, route_id, factory in subset_routes():
        if scene_filter and scene not in scene_filter:
            continue
        world = factory()
        for label, overrides in ABLATIONS.items():
            cfg = settings_for(scene, **overrides)
            for planner in ["proxy_ga", "energy_ga", "time_ga", "distance_ga"]:
                for seed in SUBSET_SEEDS:
                    raw, history, elapsed = plan(planner, world, cfg, seed, warm_cache)
                    r = u.assess(raw, world, cfg)
                    rows.append(
                        dict(
                            ablation=label,
                            scene=scene,
                            route=route_id,
                            planner=planner,
                            seed=seed,
                            runtime_s=elapsed,
                            success=r["success"],
                            length_m=r.get("length_m"),
                            flight_duration_s=r.get("flight_duration_s"),
                            equivalent_hover_time_s=r.get("equivalent_hover_time_s"),
                            model_electrical_energy_j=r.get("model_electrical_energy_j"),
                            decoded_waypoints=r.get("decoded_waypoints"),
                        )
                    )
        print(f"ablations done for {route_id}", flush=True)
    return _merge(ROOT / "ablations.json", rows, ("route", "planner", "seed", "ablation"))


# ----------------------------------------------------------------------------
# aggregation and inference
# ----------------------------------------------------------------------------
METRICS = [
    ("length_m", "mean_length_m"),
    ("flight_duration_s", "mean_duration_s"),
    ("equivalent_hover_time_s", "mean_proxy_s"),
    ("model_electrical_energy_j", "mean_energy_j"),
    ("decoded_waypoints", "mean_decoded_waypoints"),
    ("retained_stops", "mean_retained_stops"),
]


def cluster_bootstrap(values, draws=10000, seed=12345):
    """Percentile bootstrap over route-level values (routes are the units)."""
    values = np.asarray(values, float)
    if len(values) == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), (draws, len(values)))
    means = values[idx].mean(axis=1)
    return dict(
        mean=float(values.mean()),
        ci_low=float(np.percentile(means, 2.5)),
        ci_high=float(np.percentile(means, 97.5)),
        routes=int(len(values)),
    )


def sign_flip_p(values, draws=100000, seed=999):
    """Two-sided paired permutation (sign-flip) test on route-level differences."""
    values = np.asarray(values, float)
    if len(values) == 0 or np.allclose(values, 0):
        return 1.0
    rng = np.random.default_rng(seed)
    observed = abs(values.mean())
    signs = rng.choice([-1.0, 1.0], (draws, len(values)))
    null = np.abs((signs * values).mean(axis=1))
    return float((np.sum(null >= observed - 1e-15) + 1) / (draws + 1))


def holm(pairs):
    """Holm-Bonferroni adjustment. pairs: list of (label, p)."""
    ordered = sorted(pairs, key=lambda kv: kv[1])
    m = len(ordered)
    adjusted, running = {}, 0.0
    for i, (label, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[label] = running
    return adjusted


def route_means(records, planner, metric, route_set):
    out = {}
    for r in records:
        if r["planner"] == planner and r["route_set"] == route_set and r["success"]:
            out.setdefault(r["route"], []).append(r[metric])
    return {k: statistics.mean(v) for k, v in out.items()}


def summarize(records):
    summary = {
        "version": u.VERSION,
        "attempts": len(records),
        "accepted": sum(r["success"] for r in records),
        "energy_model": "parameterised electrical propulsion model (not fitted to measured flight data)",
        "mission_usable_energy_j": u.Settings().propulsion.usable_energy_j,
        "hover_power_w": u.Settings().propulsion.hover_power_w(),
        "by_planner": {},
        "by_planner_scene_kind": {},
        "route_level": {},
        "max_quadrature_relative_difference": max(
            r.get("quadrature_relative_difference", 0) for r in records
        ),
        "max_energy_quadrature_relative_difference": max(
            r.get("energy_quadrature_relative_difference", 0) for r in records
        ),
    }
    for route_set in ("development", "heldout"):
        subset = [r for r in records if r["route_set"] == route_set]
        block = {}
        for planner in PLANNERS:
            rows = [r for r in subset if r["planner"] == planner]
            good = [r for r in rows if r["success"]]
            entry = {"attempts": len(rows), "accepted": len(good), "mean_runtime_s": statistics.mean(r["runtime_s"] for r in rows)}
            for key, label in METRICS:
                entry[label] = statistics.mean(r[key] for r in good) if good else None
            block[planner] = entry
        summary["by_planner"][route_set] = block

        kinds = {}
        for kind in ("synthetic", "urban"):
            kinds[kind] = {}
            for planner in PLANNERS:
                good = [r for r in subset if r["planner"] == planner and r["scene_kind"] == kind and r["success"]]
                rows = [r for r in subset if r["planner"] == planner and r["scene_kind"] == kind]
                kinds[kind][planner] = {
                    "attempts": len(rows),
                    "accepted": len(good),
                    "mean_proxy_s": statistics.mean(r["equivalent_hover_time_s"] for r in good) if good else None,
                    "mean_energy_j": statistics.mean(r["model_electrical_energy_j"] for r in good) if good else None,
                    "mean_duration_s": statistics.mean(r["flight_duration_s"] for r in good) if good else None,
                    "mean_length_m": statistics.mean(r["length_m"] for r in good) if good else None,
                }
        summary["by_planner_scene_kind"][route_set] = kinds

        # route-level paired comparisons with cluster bootstrap and permutation
        comparisons = {}
        raw_p = []
        for metric, contrasts in (
            (
                "model_electrical_energy_j",
                [("energy_ga", "astar"), ("energy_ga", "distance_ga"), ("energy_ga", "time_ga"), ("proxy_ga", "astar")],
            ),
            (
                "equivalent_hover_time_s",
                [("proxy_ga", "astar"), ("proxy_ga", "distance_ga"), ("proxy_ga", "time_ga")],
            ),
            ("flight_duration_s", [("time_ga", "astar"), ("time_ga", "proxy_ga")]),
        ):
            for treatment, baseline in contrasts:
                a = route_means(records, treatment, metric, route_set)
                b = route_means(records, baseline, metric, route_set)
                shared = sorted(set(a) & set(b))
                pct = [100.0 * (b[k] - a[k]) / b[k] for k in shared]
                label = f"{treatment}_vs_{baseline}_{metric}"
                p = sign_flip_p(pct)
                comparisons[label] = dict(
                    metric=metric,
                    percent_reduction=cluster_bootstrap(pct),
                    per_route={k: v for k, v in zip(shared, pct)},
                    permutation_p=p,
                )
                raw_p.append((label, p))
        for label, adj in holm(raw_p).items():
            comparisons[label]["holm_adjusted_p"] = adj
        summary["route_level"][route_set] = comparisons

    summary["pooled_max_speed_mps"] = max(r.get("exact_max_speed_mps", 0) for r in records)
    summary["pooled_max_acceleration_mps2"] = max(r.get("exact_max_acceleration_mps2", 0) for r in records)
    summary["pooled_max_vertical_mps"] = max(r.get("exact_max_vertical_mps", 0) for r in records)
    summary["pooled_max_energy_fraction"] = max(
        (r["model_electrical_energy_j"] / r["mission_usable_energy_j"]) for r in records if r.get("success")
    )
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    return summary


def summarize_sweep(rows):
    out = {}
    for row in rows:
        key = f"{row['planner']}@{row['budget']}"
        out.setdefault(key, {"runtime_s": [], "energy_j": [], "proxy_s": [], "duration_s": [], "accepted": 0, "attempts": 0})
        entry = out[key]
        entry["attempts"] += 1
        entry["runtime_s"].append(row["runtime_s"])
        if row["success"]:
            entry["accepted"] += 1
            entry["energy_j"].append(row["model_electrical_energy_j"])
            entry["proxy_s"].append(row["equivalent_hover_time_s"])
            entry["duration_s"].append(row["flight_duration_s"])
    return {
        k: {
            "attempts": v["attempts"],
            "accepted": v["accepted"],
            "mean_runtime_s": statistics.mean(v["runtime_s"]),
            "mean_energy_j": statistics.mean(v["energy_j"]) if v["energy_j"] else None,
            "mean_proxy_s": statistics.mean(v["proxy_s"]) if v["proxy_s"] else None,
            "mean_duration_s": statistics.mean(v["duration_s"]) if v["duration_s"] else None,
        }
        for k, v in sorted(out.items())
    }


def summarize_ablations(rows):
    out = {}
    for row in rows:
        key = f"{row['planner']}@{row['ablation']}"
        out.setdefault(key, {"energy_j": [], "proxy_s": [], "duration_s": [], "length_m": [], "runtime_s": [], "accepted": 0, "attempts": 0})
        entry = out[key]
        entry["attempts"] += 1
        entry["runtime_s"].append(row["runtime_s"])
        if row["success"]:
            entry["accepted"] += 1
            entry["energy_j"].append(row["model_electrical_energy_j"])
            entry["proxy_s"].append(row["equivalent_hover_time_s"])
            entry["duration_s"].append(row["flight_duration_s"])
            entry["length_m"].append(row["length_m"])
    return {
        k: {
            "attempts": v["attempts"],
            "accepted": v["accepted"],
            "mean_runtime_s": statistics.mean(v["runtime_s"]),
            "mean_energy_j": statistics.mean(v["energy_j"]) if v["energy_j"] else None,
            "mean_proxy_s": statistics.mean(v["proxy_s"]) if v["proxy_s"] else None,
            "mean_duration_s": statistics.mean(v["duration_s"]) if v["duration_s"] else None,
            "mean_length_m": statistics.mean(v["length_m"]) if v["length_m"] else None,
        }
        for k, v in sorted(out.items())
    }


def reference_view(summary, sweep_summary, ablation_summary):
    """The subset of aggregates that --check compares."""
    view = {"attempts": summary["attempts"], "accepted": summary["accepted"]}
    for route_set, block in summary["by_planner"].items():
        for planner, entry in block.items():
            for label in ("mean_length_m", "mean_duration_s", "mean_proxy_s", "mean_energy_j", "mean_retained_stops"):
                view[f"{route_set}/{planner}/{label}"] = entry[label]
            view[f"{route_set}/{planner}/accepted"] = entry["accepted"]
    for route_set, block in summary["route_level"].items():
        for label, entry in block.items():
            view[f"{route_set}/{label}/percent"] = entry["percent_reduction"]["mean"]
            view[f"{route_set}/{label}/ci_low"] = entry["percent_reduction"]["ci_low"]
            view[f"{route_set}/{label}/ci_high"] = entry["percent_reduction"]["ci_high"]
    for key in ("pooled_max_speed_mps", "pooled_max_acceleration_mps2", "pooled_max_vertical_mps", "pooled_max_energy_fraction"):
        view[key] = summary[key]
    for key, entry in sweep_summary.items():
        for label in ("mean_energy_j", "mean_proxy_s", "mean_duration_s"):
            view[f"sweep/{key}/{label}"] = entry[label]
    for key, entry in ablation_summary.items():
        for label in ("mean_energy_j", "mean_proxy_s", "mean_length_m"):
            view[f"ablation/{key}/{label}"] = entry[label]
    return view


def compare_reference(view, reference):
    failures = []
    for key, expected in reference.items():
        got = view.get(key)
        if expected is None or got is None:
            if expected != got:
                failures.append(f"{key}: got {got!r}, expected {expected!r}")
            continue
        if isinstance(expected, bool) or isinstance(got, bool):
            if got != expected:
                failures.append(f"{key}: got {got!r}, expected {expected!r}")
        elif not math.isclose(got, expected, rel_tol=TOLERANCE, abs_tol=TOLERANCE):
            failures.append(f"{key}: got {got!r}, expected {expected!r}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare aggregates with reference.json within the stated tolerance")
    parser.add_argument("--write-reference", action="store_true", help="write reference.json from this run")
    parser.add_argument("--stage", choices=["all", "main", "sweep", "ablations", "summarize"], default="all")
    parser.add_argument("--routes", default="", help="comma separated scene names to restrict sweep/ablation stages")
    args = parser.parse_args()

    started = time.monotonic()
    assets = us.ensure_urban_assets()
    cfg = u.Settings()
    protocol = {
        "version": u.VERSION,
        "synthetic_scenes": SYNTHETIC_SCENES,
        "urban_scenes": URBAN_SCENES,
        "synthetic_development_route_y": SYNTHETIC_DEV_Y,
        "synthetic_heldout_route_y": SYNTHETIC_HELDOUT_Y,
        "urban_routes": us.URBAN_ROUTES,
        "urban_assets": assets,
        "urban_workspace": [list(us.URBAN_WORKSPACE_LOWER), list(us.URBAN_WORKSPACE_UPPER)],
        "seeds": SEEDS,
        "planners": PLANNERS,
        "budgets": BUDGETS,
        "rrt_budgets": RRT_BUDGETS,
        "ablations": sorted(ABLATIONS),
        "settings_synthetic": asdict(u.Settings(**SYNTHETIC_CFG)),
        "settings_urban": asdict(u.Settings(**URBAN_CFG)),
        "tolerance": TOLERANCE,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    (ROOT / "protocol.json").write_text(json.dumps(protocol, indent=2, default=str))

    records, sweep, ablation = [], [], []
    if args.stage in ("all", "main"):
        records = run_main("development") + run_main("heldout")
        (ROOT / "records.json").write_text(json.dumps(records, indent=2))
    else:
        path = ROOT / "records.json"
        if path.exists():
            records = json.loads(path.read_text())
    scene_filter = [s for s in args.routes.split(",") if s]
    if args.stage in ("all", "sweep"):
        sweep = run_budget_sweep(scene_filter)
    elif (ROOT / "budget_sweep.json").exists():
        sweep = json.loads((ROOT / "budget_sweep.json").read_text())
    if args.stage in ("all", "ablations"):
        ablation = run_ablations(scene_filter)
    elif (ROOT / "ablations.json").exists():
        ablation = json.loads((ROOT / "ablations.json").read_text())

    summary = summarize(records)
    summary["budget_sweep"] = summarize_sweep(sweep) if sweep else {}
    summary["ablations"] = summarize_ablations(ablation) if ablation else {}
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2))

    view = reference_view(summary, summary["budget_sweep"], summary["ablations"])
    if args.write_reference:
        (ROOT / "reference.json").write_text(json.dumps(view, indent=2))
        print(f"wrote reference.json with {len(view)} tracked aggregates")

    print(json.dumps({k: v for k, v in summary.items() if k not in ("route_level", "budget_sweep", "ablations", "by_planner_scene_kind")}, indent=2)[:4000])
    print(f"elapsed {time.monotonic() - started:.1f} s")

    if args.check:
        reference = json.loads((ROOT / "reference.json").read_text())
        failures = compare_reference(view, reference)
        if failures:
            print("\nREFERENCE MISMATCH")
            for line in failures:
                print(" ", line)
            raise SystemExit(1)
        print(
            f"\nAll {len(reference)} tracked aggregates matched the reference values within "
            f"the specified tolerance (relative and absolute {TOLERANCE:g}). "
            "This is a numerical comparison, not a bitwise file comparison."
        )


if __name__ == "__main__":
    main()
