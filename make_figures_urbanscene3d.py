"""Figures and CSV exports for the v0.4.0 UrbanScene3D stage.

Scene-independent figures reuse ``make_figures.py`` unchanged, redirected to
``v040/figures`` and ``v040/exports``. Figures that name specific scenes or
routes in v0.3.0 (scenes, paths, planner metrics, convergence, power profile)
are re-implemented here for the UrbanScene3D tiles, and fig11 compares the
route-level contrasts on the procedural urban scenes with those on the tiles.
Needs UAV_DATASET_ROOT (for the tile obstacles) and matplotlib.
"""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import make_figures as mf
import uav_planning as u
import urbanscene3d_scenes as reg

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / "v040"
mf.FIGURES = STAGE / "figures"
mf.EXPORTS = STAGE / "exports"
ORDER, LABEL = mf.ORDER, mf.PLANNER_LABEL


def load(name):
    return json.loads((STAGE / name).read_text())


def draw_boxes(ax, world):
    for lower, upper in world.boxes:
        ax.add_patch(plt.Rectangle((lower[0], lower[1]), upper[0] - lower[0], upper[1] - lower[1],
                                   facecolor="0.55", edgecolor="none"))
    ax.set_xlim(world.lower[0], world.upper[0])
    ax.set_ylim(world.lower[1], world.upper[1])
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def fig_scenes():
    routes = reg.load_routes()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    rows = []
    for ax, name in zip(axes, reg.REAL_SCENES):
        world = reg.real_world(name, 0, "development", routes=routes)
        draw_boxes(ax, world)
        for route_set, style in (("development", "-"), ("heldout", "--")):
            for i, row in enumerate(routes["scenes"][name][route_set]):
                s, g = np.array(row["start"]), np.array(row["goal"])
                ax.plot([s[0], g[0]], [s[1], g[1]], style, color="tab:blue" if route_set == "development" else "tab:orange",
                        linewidth=1.2, label=route_set if i == 0 else None)
                ax.plot(*s[:2], "o", color="tab:green", markersize=4)
                ax.plot(*g[:2], "*", color="tab:red", markersize=8)
        record = reg.tile_record(name)
        ax.set_title(f"{record['dataset']} ({len(world.boxes)} boxes, "
                     f"{record['nodata']['fraction']:.0%} no-data)", fontsize=9)
        rows.append([name, record["dataset"], len(world.boxes), record["nodata"]["fraction"],
                     record["tile_selection"]["mapped_fraction"], record["output"]["sha256"]])
    axes[0].legend(fontsize=8, loc="upper left")
    mf.export("fig01_scenes.csv", ["scene", "dataset", "boxes", "nodata_fraction", "mapped_fraction", "tile_sha256"], rows)
    mf.save(fig, "fig01_scenes")


def fig_paths(records):
    routes = reg.load_routes()
    targets = [(name, f"{name}_r0") for name in reg.REAL_SCENES]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    rows = []
    for ax, (scene, route) in zip(axes, targets):
        draw_boxes(ax, reg.real_world(scene, 0, "development", routes=routes))
        for planner in ORDER:
            match = [r for r in records if r["route"] == route and r["planner"] == planner and r["seed"] == 0 and r["success"]]
            if not match:
                continue
            p = np.array(match[0]["path"])
            ax.plot(p[:, 0], p[:, 1], marker="o", markersize=2.5, linewidth=1.2, label=LABEL[planner])
            rows += [[route, planner, i, *xyz] for i, xyz in enumerate(p.tolist())]
        ax.set_title(f"{route} (seed 0)", fontsize=10)
    axes[0].legend(fontsize=8)
    mf.export("fig02_paths.csv", ["route", "planner", "index", "x", "y", "z"], rows)
    mf.save(fig, "fig02_paths")


def fig_metric_bars(records, procedural):
    metrics = [("model_electrical_energy_j", "Model electrical energy [J]"), ("flight_duration_s", "Traversal time [s]"),
               ("length_m", "Path length [m]"), ("retained_stops", "Retained interior waypoints (stops)")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    rows = []
    for ax, (key, label) in zip(axes.ravel(), metrics):
        for offset, (name, source) in ((-0.2, ("procedural urban (v0.3.0)", procedural)), (0.2, ("UrbanScene3D (v0.4.0)", records))):
            values, errors = [], []
            for planner in ORDER:
                good = [r[key] for r in mf.accepted(source, "development") if r["planner"] == planner]
                values.append(statistics.mean(good) if good else np.nan)
                errors.append(statistics.pstdev(good) if len(good) > 1 else 0.0)
                rows.append([key, name, planner, values[-1], errors[-1], len(good)])
            ax.bar(np.arange(len(ORDER)) + offset, values, width=0.38, yerr=errors, capsize=2, label=name)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([LABEL[p] for p in ORDER], rotation=20, fontsize=8)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].legend(fontsize=8)
    mf.export("fig03_metric_bars.csv", ["metric", "scenes", "planner", "mean", "population_sd", "accepted"], rows)
    mf.save(fig, "fig03_planner_metrics")


def fig_convergence(records):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    rows = []
    for ax, name in zip(axes, reg.REAL_SCENES):
        route = f"{name}_r0"
        for planner in ("distance_ga", "proxy_ga", "time_ga", "energy_ga"):
            match = [r for r in records if r["route"] == route and r["planner"] == planner and r["seed"] == 0]
            if not match or not match[0]["history"]:
                continue
            history = match[0]["history"]
            ax.plot([h["evaluations"] for h in history], [h["best_score"] for h in history], marker="o", markersize=3, label=LABEL[planner])
            rows += [[route, planner, h["generation"], h["evaluations"], h["best_score"], h["feasible_count"]] for h in history]
        ax.set_xlabel("Trajectory evaluations")
        ax.set_ylabel("Normalised objective of the best feasible individual")
        ax.set_title(route, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    mf.export("fig05_convergence.csv", ["route", "planner", "generation", "evaluations", "best_score", "feasible_count"], rows)
    mf.save(fig, "fig05_convergence")


def fig_power_profile(records):
    route = f"{reg.REAL_SCENES[0]}_r0"
    match = [r for r in records if r["route"] == route and r["planner"] in ("energy_ga", "astar") and r["seed"] == 0 and r["success"]]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    cfg = u.Settings(grid_step=4.0)
    rows = []
    for r in match:
        p, ts = np.array(r["path"]), np.array(r["segment_durations_s"])
        samples = u.sample_trajectory(p, ts, 201)
        power = cfg.propulsion.electrical_power_w(samples[:, 7:10], samples[:, 4:7])
        cumulative = np.concatenate(([0.0], np.cumsum(np.diff(samples[:, 0]) * (power[1:] + power[:-1]) / 2)))
        axes[0].plot(samples[:, 0], power, label=LABEL[r["planner"]])
        axes[1].plot(samples[:, 0], cumulative, label=LABEL[r["planner"]])
        rows += [[r["planner"], t, pw, cu] for t, pw, cu in zip(samples[:, 0], power, cumulative)]
    axes[1].axhline(cfg.propulsion.usable_energy_j, color="crimson", linestyle="--", label="usable mission energy")
    axes[0].set_ylabel("Electrical power [W]")
    axes[1].set_ylabel("Cumulative energy [J]")
    axes[1].set_xlabel("Time [s]")
    axes[0].set_title(f"{route}, seed 0", fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    mf.export("fig09_power_profile.csv", ["planner", "time_s", "power_w", "cumulative_energy_j"], rows)
    mf.save(fig, "fig09_power_profile")


def fig_procedural_vs_real(comparison):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
    rows = []
    for ax, route_set in zip(axes, ("development", "heldout")):
        block = comparison["route_sets"][route_set]["route_level"]
        labels = list(block)
        for offset, name, colour in ((-0.15, "procedural", "tab:gray"), (0.15, "urbanscene3d", "tab:blue")):
            e = [block[k][name] for k in labels]
            m = np.array([x["mean"] for x in e])
            ax.errorbar(m, np.arange(len(labels)) + offset, xerr=[m - [x["ci_low"] for x in e], [x["ci_high"] for x in e] - m],
                        fmt="o", capsize=3, color=colour, label="procedural urban (v0.3.0)" if name == "procedural" else "UrbanScene3D (v0.4.0)")
            rows += [[route_set, k, name, x["mean"], x["ci_low"], x["ci_high"], x["routes"], x["permutation_p"], x["holm_adjusted_p"]]
                     for k, x in zip(labels, e)]
        ax.axvline(0.0, color="0.4", linewidth=1)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([l.replace("_", " ") for l in labels], fontsize=7)
        ax.set_title(route_set, fontsize=10)
        ax.set_xlabel("Route-level mean reduction [%], cluster bootstrap 95% CI")
        ax.grid(axis="x", alpha=0.3)
    axes[0].legend(fontsize=8)
    mf.export("fig11_procedural_vs_urbanscene3d.csv",
              ["route_set", "comparison", "scenes", "mean_percent", "ci_low", "ci_high", "routes", "permutation_p", "holm_p"], rows)
    mf.save(fig, "fig11_procedural_vs_urbanscene3d")


def main():
    records, summary = load("records.json"), load("summary.json")
    procedural = [r for r in json.loads((ROOT / "records.json").read_text()) if r["scene_kind"] == "urban"]
    fig_scenes()
    fig_paths(records)
    fig_metric_bars(records, procedural)
    mf.fig_energy_vs_duration(records)
    fig_convergence(records)
    if (STAGE / "budget_sweep.json").exists():
        mf.fig_budget(load("budget_sweep.json"))
    if (STAGE / "ablations.json").exists():
        mf.fig_ablations(load("ablations.json"))
    mf.fig_route_level(summary)
    fig_power_profile(records)
    mf.fig_heldout(records)
    fig_procedural_vs_real(load("comparison.json"))
    mf.tables(records, summary)
    print(f"figures in {mf.FIGURES}, exports in {mf.EXPORTS}")


if __name__ == "__main__":
    main()
