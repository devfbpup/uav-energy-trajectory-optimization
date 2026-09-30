"""v0.4.0 stage: the v0.3.0 protocol on registered UrbanScene3D tiles.

Scenes are the three tiles in ``urbanscene3d_scenes.REAL_SCENES``, each with the
frozen routes in ``urbanscene3d/routes.json`` (3 development + 2 held-out).
Everything else is the v0.3.0 protocol, imported from ``reproduce.py`` rather
than copied: the six planners, the urban settings (4 m lattice), seeds 0-4,
post-hoc validation, the matched-budget sweep and the ablations on one
development route per scene, and the route-level cluster bootstrap,
sign-flip permutation tests and Holm adjustment.

Outputs go to ``v040/`` so the v0.3.0 artefacts and ``reproduce.py --check``
are untouched.

Usage
-----
    export UAV_DATASET_ROOT=/path/to/UrbanScene3D_data
    python3 reproduce_urbanscene3d.py                    # all stages
    python3 reproduce_urbanscene3d.py --stage main
    python3 reproduce_urbanscene3d.py --check            # compare with v040/reference.json
    python3 reproduce_urbanscene3d.py --write-reference
"""

import argparse
import json
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import reproduce as r3
import uav_planning as u
import urbanscene3d_scenes as reg

VERSION = "0.4.0"
ROOT = Path(__file__).resolve().parent
STAGE = ROOT / "v040"
OUT = STAGE / "runs"


def settings_for(**overrides):
    return u.Settings(**dict(r3.URBAN_CFG, **overrides))


def routes(route_set="development"):
    frozen = reg.load_routes()
    for scene in reg.REAL_SCENES:
        for i in range(len(frozen["scenes"][scene][route_set])):
            yield scene, f"{scene}_r{i}", (lambda s=scene, i=i, rs=route_set: reg.real_world(s, i, rs, routes=frozen))


def subset_routes():
    chosen = []
    for scene, route_id, factory in routes("development"):
        if not any(c[0] == scene for c in chosen):
            chosen.append((scene, route_id, factory))
    return chosen


def run_route(job):
    """All seeds and planners on one route (one worker process per route)."""
    route_set, scene, route_id, index = job
    OUT.mkdir(parents=True, exist_ok=True)
    world = reg.real_world(scene, index, route_set)
    cfg = settings_for()
    u.valid_endpoints(world, cfg)
    records, warm_cache = [], {}
    for seed in r3.SEEDS:
        for planner in r3.PLANNERS:
            raw, history, elapsed = r3.plan(planner, world, cfg, seed, warm_cache)
            rec = u.assess(raw, world, cfg)
            rec.update(planner=planner, scene=scene, route=route_id, route_set=route_set, seed=seed,
                       runtime_s=elapsed, scene_kind="urbanscene3d", provenance=world.provenance)
            if rec["success"]:
                rec = r3.validate(rec, world, cfg)
            rec["history"] = history
            (OUT / f"{route_set}_{route_id}_seed{seed}_{planner}.json").write_text(json.dumps(rec, indent=2))
            records.append(rec)
    print(f"completed {route_set} {route_id}: {len(records)} attempts", flush=True)
    return records


def run_main(workers=1):
    """Every route of both sets; records are returned in the canonical route order.

    Routes are independent and every planner is deterministic given its seed,
    so running routes in parallel changes only the measured runtimes.
    """
    jobs = []
    for route_set in ("development", "heldout"):
        for scene, route_id, _ in routes(route_set):
            jobs.append((route_set, scene, route_id, int(route_id.rsplit("_r", 1)[1])))
    if workers > 1:
        from multiprocessing import Pool

        with Pool(workers) as pool:
            parts = pool.map(run_route, jobs, chunksize=1)
    else:
        parts = [run_route(job) for job in jobs]
    return [rec for part in parts for rec in part]


def _row(base, rec):
    return dict(base, success=rec["success"], length_m=rec.get("length_m"),
                flight_duration_s=rec.get("flight_duration_s"),
                equivalent_hover_time_s=rec.get("equivalent_hover_time_s"),
                model_electrical_energy_j=rec.get("model_electrical_energy_j"))


def run_budget_sweep():
    rows, warm_cache = [], {}
    for scene, route_id, factory in subset_routes():
        world = factory()
        for budget in r3.BUDGETS:
            cfg = settings_for(evaluation_budget=budget, generations=200)
            for planner in ["distance_ga", "proxy_ga", "time_ga", "energy_ga"]:
                for seed in r3.SUBSET_SEEDS:
                    raw, history, elapsed = r3.plan(planner, world, cfg, seed, warm_cache)
                    rows.append(_row(dict(kind="genetic", scene=scene, route=route_id, planner=planner, seed=seed,
                                          budget=budget, evaluations=history[-1]["evaluations"] if history else None,
                                          runtime_s=elapsed), u.assess(raw, world, cfg)))
        for iterations in r3.RRT_BUDGETS:
            cfg = settings_for(rrt_iterations=iterations)
            for seed in r3.SUBSET_SEEDS:
                t0 = time.perf_counter()
                raw = u.rrt_star(world, cfg, seed, iterations)
                elapsed = time.perf_counter() - t0
                rows.append(_row(dict(kind="rrt", scene=scene, route=route_id, planner="rrt_star", seed=seed,
                                      budget=iterations, evaluations=None, runtime_s=elapsed),
                                 u.assess(raw, world, cfg)))
        print(f"budget sweep done for {route_id}", flush=True)
    return rows


def run_ablations():
    rows, warm_cache = [], {}
    for scene, route_id, factory in subset_routes():
        world = factory()
        for label, overrides in r3.ABLATIONS.items():
            cfg = settings_for(**overrides)
            for planner in ["proxy_ga", "energy_ga", "time_ga", "distance_ga"]:
                for seed in r3.SUBSET_SEEDS:
                    raw, history, elapsed = r3.plan(planner, world, cfg, seed, warm_cache)
                    rec = u.assess(raw, world, cfg)
                    row = _row(dict(ablation=label, scene=scene, route=route_id, planner=planner, seed=seed,
                                    runtime_s=elapsed), rec)
                    row["decoded_waypoints"] = rec.get("decoded_waypoints")
                    rows.append(row)
        print(f"ablations done for {route_id}", flush=True)
    return rows


def compare_with_procedural(summary):
    """Same statistics on the v0.3.0 procedural urban scenes (committed records, no rerun)."""
    v030 = [r for r in json.loads((ROOT / "records.json").read_text()) if r["scene_kind"] == "urban"]
    procedural = r3.summarize(v030)
    out = {"note": "procedural = v0.3.0 records restricted to urban_a/b/c (15 routes, 450 attempts); "
                   "urbanscene3d = this stage. Identical planners, settings, seeds and statistics.",
           "route_sets": {}}
    for route_set in ("development", "heldout"):
        block = {}
        for label in summary["route_level"][route_set]:
            block[label] = {}
            for name, s in (("procedural", procedural), ("urbanscene3d", summary)):
                e = s["route_level"][route_set][label]
                block[label][name] = dict(e["percent_reduction"], permutation_p=e["permutation_p"],
                                          holm_adjusted_p=e["holm_adjusted_p"])
        means = {name: {p: {k: s["by_planner"][route_set][p][k] for k in
                            ("attempts", "accepted", "mean_length_m", "mean_duration_s", "mean_energy_j", "mean_retained_stops")}
                        for p in r3.PLANNERS}
                 for name, s in (("procedural", procedural), ("urbanscene3d", summary))}
        out["route_sets"][route_set] = {"route_level": block, "by_planner": means}
    (STAGE / "comparison.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write-reference", action="store_true")
    parser.add_argument("--stage", choices=["all", "main", "sweep", "ablations", "summarize"], default="all")
    parser.add_argument("--workers", type=int, default=1, help="parallel route workers for the main stage")
    args = parser.parse_args()
    STAGE.mkdir(exist_ok=True)
    started = time.monotonic()

    frozen = reg.load_routes()
    tiles = {name: reg.tile_record(name) for name in reg.REAL_SCENES}
    protocol = {
        "version": VERSION,
        "stage": "UrbanScene3D tiles, v0.3.0 protocol",
        "scenes": reg.REAL_SCENES,
        "datasets": {n: t["dataset"] for n, t in tiles.items()},
        "tile_sha256": {n: t["output"]["sha256"] for n, t in tiles.items()},
        "routes_file": "urbanscene3d/routes.json",
        "routes": {n: {rs: [[row["start"], row["goal"]] for row in frozen["scenes"][n][rs]]
                       for rs in ("development", "heldout")} for n in reg.REAL_SCENES},
        "seeds": r3.SEEDS, "planners": r3.PLANNERS, "budgets": r3.BUDGETS, "rrt_budgets": r3.RRT_BUDGETS,
        "ablations": sorted(r3.ABLATIONS),
        "settings_urban": asdict(settings_for()),
        "tolerance": r3.TOLERANCE,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }
    (STAGE / "protocol.json").write_text(json.dumps(protocol, indent=2, default=str))

    def stage_file(name, compute, run):
        path = STAGE / name
        if run:
            rows = compute()
            path.write_text(json.dumps(rows, indent=2))
            return rows
        return json.loads(path.read_text()) if path.exists() else []

    protocol["main_stage_workers"] = args.workers
    (STAGE / "protocol.json").write_text(json.dumps(protocol, indent=2, default=str))
    records = stage_file("records.json", lambda: run_main(args.workers), args.stage in ("all", "main"))
    sweep = stage_file("budget_sweep.json", run_budget_sweep, args.stage in ("all", "sweep"))
    ablation = stage_file("ablations.json", run_ablations, args.stage in ("all", "ablations"))

    if not records:
        print("no main-stage records yet; summary skipped")
        return
    summary = r3.summarize(records)
    summary["version"] = VERSION
    summary["scene_kind"] = "urbanscene3d"
    summary["budget_sweep"] = r3.summarize_sweep(sweep) if sweep else {}
    summary["ablations"] = r3.summarize_ablations(ablation) if ablation else {}
    (STAGE / "summary.json").write_text(json.dumps(summary, indent=2))
    if records:
        compare_with_procedural(summary)

    view = r3.reference_view(summary, summary["budget_sweep"], summary["ablations"])
    if args.write_reference:
        (STAGE / "reference.json").write_text(json.dumps(view, indent=2))
        print(f"wrote v040/reference.json with {len(view)} tracked aggregates")
    print(f"attempts {summary['attempts']}, accepted {summary['accepted']}, elapsed {time.monotonic() - started:.1f} s")

    if args.check:
        failures = r3.compare_reference(view, json.loads((STAGE / "reference.json").read_text()))
        if failures:
            print("\nREFERENCE MISMATCH")
            for line in failures:
                print(" ", line)
            raise SystemExit(1)
        print(f"\nAll tracked v0.4.0 aggregates matched within {r3.TOLERANCE:g}.")


if __name__ == "__main__":
    main()
