"""Generate and export every figure and table in the article.

This script is the published plotting/export stage requested by review: it
reads the artefacts written by ``reproduce.py`` (``records.json``,
``budget_sweep.json``, ``ablations.json``, ``summary.json``) and writes every
plotted figure to ``figures/`` plus machine-readable CSV exports of the plotted
series to ``exports/`` so that each figure can be checked numerically.

Usage:  python3 make_figures.py
"""

import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import uav_planning as u
import urban_scenes as us

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
EXPORTS = ROOT / "exports"
PLANNER_LABEL = {
    "astar": "A*",
    "rrt_star": "RRT*",
    "distance_ga": "Distance GA",
    "proxy_ga": "Proxy GA",
    "time_ga": "Time GA",
    "energy_ga": "Energy GA",
}
ORDER = ["astar", "rrt_star", "distance_ga", "proxy_ga", "time_ga", "energy_ga"]


def load(name):
    return json.loads((ROOT / name).read_text())


def export(name, header, rows):
    EXPORTS.mkdir(exist_ok=True)
    with open(EXPORTS / name, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=200)
    fig.savefig(FIGURES / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


def accepted(records, route_set=None, kind=None):
    return [
        r
        for r in records
        if r["success"]
        and (route_set is None or r["route_set"] == route_set)
        and (kind is None or r["scene_kind"] == kind)
    ]


# ---------------------------------------------------------------- figure 1
def fig_scenes():
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    scenes = [("sparse", None), ("wall", None), ("culdesac", None), ("urban_a", 0), ("urban_b", 0), ("urban_c", 0)]
    rows = []
    for ax, (name, route) in zip(axes.ravel(), scenes):
        world = u.scenario(name) if route is None else us.urban_world(name, route)
        for lower, upper in world.boxes:
            ax.add_patch(
                plt.Rectangle(
                    (lower[0], lower[1]),
                    upper[0] - lower[0],
                    upper[1] - lower[1],
                    facecolor="0.55",
                    edgecolor="0.25",
                    linewidth=0.3,
                )
            )
        ax.plot(*world.start[:2], "o", color="tab:green", label="start")
        ax.plot(*world.goal[:2], "*", color="tab:red", markersize=12, label="goal")
        ax.set_xlim(world.lower[0], world.upper[0])
        ax.set_ylim(world.lower[1], world.upper[1])
        ax.set_aspect("equal")
        ax.set_title(f"{name} ({len(world.boxes)} boxes)", fontsize=10)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        rows.append([name, len(world.boxes), world.provenance.get("kind"), world.provenance.get("asset_sha256", "")])
    axes[0][0].legend(fontsize=8, loc="upper left")
    export("fig01_scenes.csv", ["scene", "boxes", "provenance_kind", "asset_sha256"], rows)
    save(fig, "fig01_scenes")


# ---------------------------------------------------------------- figure 2
def fig_paths(records):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    targets = [("wall", "wall_y20", None), ("urban_a", "urban_a_r0", 0)]
    rows = []
    for ax, (scene, route, route_index) in zip(axes, targets):
        world = u.scenario(scene) if route_index is None else us.urban_world(scene, route_index)
        if route_index is None:
            world = us.synthetic_world(scene, 20.0)
        for lower, upper in world.boxes:
            ax.add_patch(
                plt.Rectangle(
                    (lower[0], lower[1]),
                    upper[0] - lower[0],
                    upper[1] - lower[1],
                    facecolor="0.6",
                    edgecolor="0.3",
                    linewidth=0.3,
                )
            )
        for planner in ORDER:
            match = [r for r in records if r["route"] == route and r["planner"] == planner and r["seed"] == 0 and r["success"]]
            if not match:
                continue
            p = np.array(match[0]["path"])
            ax.plot(p[:, 0], p[:, 1], marker="o", markersize=3, linewidth=1.4, label=PLANNER_LABEL[planner])
            rows += [[route, planner, i, *xyz] for i, xyz in enumerate(p.tolist())]
        ax.set_xlim(world.lower[0], world.upper[0])
        ax.set_ylim(world.lower[1], world.upper[1])
        ax.set_aspect("equal")
        ax.set_title(f"{route} (seed 0)", fontsize=10)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    axes[0].legend(fontsize=8)
    export("fig02_paths.csv", ["route", "planner", "index", "x", "y", "z"], rows)
    save(fig, "fig02_paths")


# ---------------------------------------------------------------- figure 3
def fig_metric_bars(records):
    metrics = [
        ("model_electrical_energy_j", "Model electrical energy [J]"),
        ("flight_duration_s", "Traversal time [s]"),
        ("equivalent_hover_time_s", "Proxy equivalent hover time [s]"),
        ("retained_stops", "Retained interior waypoints (stops)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    rows = []
    for ax, (key, label) in zip(axes.ravel(), metrics):
        for offset, kind in ((-0.2, "synthetic"), (0.2, "urban")):
            values, errors = [], []
            for planner in ORDER:
                good = [r[key] for r in accepted(records, "development", kind) if r["planner"] == planner]
                values.append(statistics.mean(good) if good else np.nan)
                errors.append(statistics.pstdev(good) if len(good) > 1 else 0.0)
                rows.append([key, kind, planner, values[-1], errors[-1], len(good)])
            ax.bar(np.arange(len(ORDER)) + offset, values, width=0.38, yerr=errors, capsize=2, label=kind)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([PLANNER_LABEL[p] for p in ORDER], rotation=20, fontsize=8)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].legend(fontsize=8)
    export("fig03_metric_bars.csv", ["metric", "scene_kind", "planner", "mean", "population_sd", "accepted"], rows)
    save(fig, "fig03_planner_metrics")


# ---------------------------------------------------------------- figure 4
def fig_energy_vs_duration(records):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    rows = []
    for planner in ORDER:
        good = accepted(records, "development")
        good = [r for r in good if r["planner"] == planner]
        x = [r["flight_duration_s"] for r in good]
        y = [r["model_electrical_energy_j"] for r in good]
        ax.scatter(x, y, s=18, alpha=0.75, label=PLANNER_LABEL[planner])
        rows += [[planner, r["route"], r["seed"], r["flight_duration_s"], r["model_electrical_energy_j"], r["retained_stops"]] for r in good]
    usable = u.Settings().propulsion.usable_energy_j
    ax.axhline(usable, color="crimson", linestyle="--", linewidth=1.2, label="usable mission energy")
    ax.set_xlabel("Traversal time [s]")
    ax.set_ylabel("Model electrical energy [J]")
    ax.set_title("Energy against traversal time, all development attempts", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    export("fig04_energy_vs_duration.csv", ["planner", "route", "seed", "duration_s", "energy_j", "retained_stops"], rows)
    save(fig, "fig04_energy_vs_duration")


# ---------------------------------------------------------------- figure 5
def fig_convergence(records):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rows = []
    for ax, route in zip(axes, ["wall_y20", "urban_a_r0"]):
        for planner in ("distance_ga", "proxy_ga", "time_ga", "energy_ga"):
            match = [r for r in records if r["route"] == route and r["planner"] == planner and r["seed"] == 0]
            if not match or not match[0]["history"]:
                continue
            history = match[0]["history"]
            x = [h["evaluations"] for h in history]
            y = [h["best_score"] for h in history]
            ax.plot(x, y, marker="o", markersize=3, label=PLANNER_LABEL[planner])
            rows += [[route, planner, h["generation"], h["evaluations"], h["best_score"], h["feasible_count"]] for h in history]
        ax.set_xlabel("Trajectory evaluations")
        ax.set_ylabel("Normalised objective of the best feasible individual")
        ax.set_title(route, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    export("fig05_convergence.csv", ["route", "planner", "generation", "evaluations", "best_score", "feasible_count"], rows)
    save(fig, "fig05_convergence")


# ---------------------------------------------------------------- figure 6
def fig_budget(sweep):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rows = []
    for planner in ("distance_ga", "proxy_ga", "time_ga", "energy_ga", "rrt_star"):
        subset = [r for r in sweep if r["planner"] == planner and r["success"]]
        if not subset:
            continue
        budgets = sorted({r["budget"] for r in subset})
        runtimes, energies, durations = [], [], []
        for budget in budgets:
            block = [r for r in subset if r["budget"] == budget]
            runtimes.append(statistics.mean(r["runtime_s"] for r in block))
            energies.append(statistics.mean(r["model_electrical_energy_j"] for r in block))
            durations.append(statistics.mean(r["flight_duration_s"] for r in block))
            rows.append([planner, budget, runtimes[-1], energies[-1], durations[-1], len(block)])
        axes[0].plot(runtimes, energies, marker="o", label=PLANNER_LABEL[planner])
        axes[1].plot(budgets, energies, marker="s", label=PLANNER_LABEL[planner])
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Mean planner runtime [s, log scale]")
    axes[0].set_ylabel("Mean model electrical energy [J]")
    axes[0].set_title("Quality against runtime", fontsize=10)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Budget (trajectory evaluations, or RRT* iterations)")
    axes[1].set_ylabel("Mean model electrical energy [J]")
    axes[1].set_title("Quality against matched budget", fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    export("fig06_budget.csv", ["planner", "budget", "mean_runtime_s", "mean_energy_j", "mean_duration_s", "attempts"], rows)
    save(fig, "fig06_budget_curves")


# ---------------------------------------------------------------- figure 7
def fig_ablations(ablations):
    labels = ["full", "no_warm_start", "no_shortcutting", "no_warm_start_no_shortcutting"]
    planners = ["distance_ga", "proxy_ga", "time_ga", "energy_ga"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    rows = []
    for idx, (key, ylabel) in enumerate(
        [("model_electrical_energy_j", "Mean model energy [J]"), ("runtime_s", "Mean runtime [s]")]
    ):
        ax = axes[idx]
        width = 0.2
        for k, planner in enumerate(planners):
            values = []
            for label in labels:
                block = [r for r in ablations if r["planner"] == planner and r["ablation"] == label and (r["success"] or key == "runtime_s")]
                block = [r[key] for r in block if r[key] is not None]
                values.append(statistics.mean(block) if block else np.nan)
                rows.append([key, planner, label, values[-1], len(block)])
            ax.bar(np.arange(len(labels)) + (k - 1.5) * width, values, width=width, label=PLANNER_LABEL[planner])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    export("fig07_ablations.csv", ["metric", "planner", "ablation", "mean", "observations"], rows)
    save(fig, "fig07_ablations")


# ---------------------------------------------------------------- figure 8
def fig_route_level(summary):
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    rows = []
    entries = summary["route_level"]["development"]
    labels = list(entries)
    means = [entries[k]["percent_reduction"]["mean"] for k in labels]
    low = [entries[k]["percent_reduction"]["mean"] - entries[k]["percent_reduction"]["ci_low"] for k in labels]
    high = [entries[k]["percent_reduction"]["ci_high"] - entries[k]["percent_reduction"]["mean"] for k in labels]
    ax.errorbar(means, np.arange(len(labels)), xerr=[low, high], fmt="o", capsize=3)
    ax.axvline(0.0, color="0.4", linewidth=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([l.replace("_", " ") for l in labels], fontsize=7)
    ax.set_xlabel("Route-level mean reduction [%] with cluster bootstrap 95% interval")
    ax.grid(axis="x", alpha=0.3)
    for k in labels:
        e = entries[k]
        rows.append(
            [
                k,
                e["metric"],
                e["percent_reduction"]["mean"],
                e["percent_reduction"]["ci_low"],
                e["percent_reduction"]["ci_high"],
                e["percent_reduction"]["routes"],
                e["permutation_p"],
                e["holm_adjusted_p"],
            ]
        )
    export(
        "fig08_route_level.csv",
        ["comparison", "metric", "mean_percent", "ci_low", "ci_high", "routes", "permutation_p", "holm_p"],
        rows,
    )
    save(fig, "fig08_route_level_intervals")


# ---------------------------------------------------------------- figure 9
def fig_power_profile(records):
    match = [r for r in records if r["route"] == "urban_a_r0" and r["planner"] in ("energy_ga", "astar") and r["seed"] == 0 and r["success"]]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    cfg = u.Settings(grid_step=4.0)
    rows = []
    for r in match:
        p = np.array(r["path"])
        ts = np.array(r["segment_durations_s"])
        samples = u.sample_trajectory(p, ts, 201)
        power = cfg.propulsion.electrical_power_w(samples[:, 7:10], samples[:, 4:7])
        cumulative = np.concatenate(([0.0], np.cumsum(np.diff(samples[:, 0]) * (power[1:] + power[:-1]) / 2)))
        axes[0].plot(samples[:, 0], power, label=PLANNER_LABEL[r["planner"]])
        axes[1].plot(samples[:, 0], cumulative, label=PLANNER_LABEL[r["planner"]])
        rows += [[r["planner"], t, pw, cu] for t, pw, cu in zip(samples[:, 0], power, cumulative)]
    axes[1].axhline(cfg.propulsion.usable_energy_j, color="crimson", linestyle="--", label="usable mission energy")
    axes[0].set_ylabel("Electrical power [W]")
    axes[1].set_ylabel("Cumulative energy [J]")
    axes[1].set_xlabel("Time [s]")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    export("fig09_power_profile.csv", ["planner", "time_s", "power_w", "cumulative_energy_j"], rows)
    save(fig, "fig09_power_profile")


# ---------------------------------------------------------------- figure 10
def fig_heldout(records):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    rows = []
    width = 0.38
    for offset, route_set in ((-0.19, "development"), (0.19, "heldout")):
        values, errors = [], []
        for planner in ORDER:
            good = [r["model_electrical_energy_j"] for r in accepted(records, route_set) if r["planner"] == planner]
            values.append(statistics.mean(good) if good else np.nan)
            errors.append(statistics.pstdev(good) if len(good) > 1 else 0.0)
            rows.append([route_set, planner, values[-1], errors[-1], len(good)])
        ax.bar(np.arange(len(ORDER)) + offset, values, width=width, yerr=errors, capsize=2, label=route_set)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([PLANNER_LABEL[p] for p in ORDER], rotation=20, fontsize=8)
    ax.set_ylabel("Mean model electrical energy [J]")
    ax.set_title("Development against held-out routes", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    export("fig10_heldout.csv", ["route_set", "planner", "mean_energy_j", "population_sd", "accepted"], rows)
    save(fig, "fig10_heldout")


def tables(records, summary):
    rows = []
    for route_set in ("development", "heldout"):
        for planner in ORDER:
            entry = summary["by_planner"][route_set][planner]
            rows.append(
                [
                    route_set,
                    planner,
                    entry["attempts"],
                    entry["accepted"],
                    entry["mean_length_m"],
                    entry["mean_duration_s"],
                    entry["mean_proxy_s"],
                    entry["mean_energy_j"],
                    entry["mean_retained_stops"],
                    entry["mean_runtime_s"],
                ]
            )
    export(
        "table01_planner_summary.csv",
        ["route_set", "planner", "attempts", "accepted", "length_m", "duration_s", "proxy_s", "energy_j", "retained_stops", "runtime_s"],
        rows,
    )
    per_route = []
    for r in records:
        per_route.append(
            [
                r["route_set"],
                r["scene"],
                r["route"],
                r["planner"],
                r["seed"],
                r["success"],
                r.get("reason"),
                r.get("length_m"),
                r.get("flight_duration_s"),
                r.get("equivalent_hover_time_s"),
                r.get("model_electrical_energy_j"),
                r.get("retained_stops"),
                r.get("runtime_s"),
            ]
        )
    export(
        "table02_all_attempts.csv",
        ["route_set", "scene", "route", "planner", "seed", "success", "reason", "length_m", "duration_s", "proxy_s", "energy_j", "retained_stops", "runtime_s"],
        per_route,
    )
    print("wrote tables")


def main():
    records = load("records.json")
    summary = load("summary.json")
    sweep = load("budget_sweep.json")
    ablations = load("ablations.json")
    fig_scenes()
    fig_paths(records)
    fig_metric_bars(records)
    fig_energy_vs_duration(records)
    fig_convergence(records)
    fig_budget(sweep)
    fig_ablations(ablations)
    fig_route_level(summary)
    fig_power_profile(records)
    fig_heldout(records)
    tables(records, summary)
    print(f"figures in {FIGURES}, exports in {EXPORTS}")


if __name__ == "__main__":
    main()
