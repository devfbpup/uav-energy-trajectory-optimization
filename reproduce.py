"""Regenerate every numerical figure reported in the article from a clean checkout.

Usage:
    python3 reproduce.py            # full 180-attempt protocol, writes runs/ and summary.json
    python3 reproduce.py --check    # additionally compare against the recorded reference values

The protocol is fully deterministic: three synthetic scenes, route y in {16, 20, 24} m,
seeds 0-4, four planners. Wall-clock runtimes will differ between machines; every geometric
and proxy-energy quantity is bit-reproducible under the same NumPy version.
"""
import argparse, json, math, platform, statistics, time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import uav_planning as u
from test_validation import sat_hit

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs"

# Reference aggregates from the run reported in the article, in planner order
# astar, rrt_star, distance_ga, proxy_ga.
REFERENCE = {
    "mean_length_m": [35.3975342066759, 36.29268593164662, 35.36419874882379, 36.56181663796508],
    "mean_proxy_s": [19.62978543800459, 20.11097140906021, 19.557757754062344, 18.64740518054749],
    "accepted_per_planner": [45, 45, 45, 45],
    "pooled_max_speed_mps": 4.950495049504951,
    "pooled_max_acceleration_mps2": 1.9605920988138423,
    "pooled_max_vertical_mps": 1.9801980198019804,
    "mean_route_improvement_vs_astar_pct": 4.2933857113201395,
    "mean_route_improvement_vs_distance_ga_pct": 4.022208441262114,
    "source_sha256": "98fde7c4e91652f377a80fedc51b007db0b3534261dc46d3b009c4aa0bbf0a37",
}

PLANNERS = ["astar", "rrt_star", "distance_ga", "proxy_ga"]
SCENES = ["sparse", "wall", "culdesac"]
ROUTES = [16.0, 20.0, 24.0]
SEEDS = list(range(5))


def run_protocol():
    OUT.mkdir(exist_ok=True)
    cfg = u.Settings()
    protocol = {
        "version": u.VERSION,
        "scenes": SCENES,
        "route_y": ROUTES,
        "seeds": SEEDS,
        "planners": PLANNERS,
        "settings": asdict(cfg),
        "expected_attempts": 180,
        "geometry": "synthetic axis-aligned boxes, not UrbanScene3D",
        "energy": "uncalibrated equivalent-hover-time proxy, in seconds",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    (ROOT / "protocol.json").write_text(json.dumps(protocol, indent=2))

    records = []
    for scene in SCENES:
        for y in ROUTES:
            w = u.scenario(scene)
            w.start[1] = y
            w.goal[1] = y
            u.valid_endpoints(w, cfg)
            for seed in SEEDS:
                for planner in PLANNERS:
                    history = []
                    t0 = time.perf_counter()
                    if planner == "astar":
                        raw = u.astar(w, cfg)
                    elif planner == "rrt_star":
                        raw = u.rrt_star(w, cfg, seed)
                    else:
                        objective = "distance" if planner == "distance_ga" else "proxy"
                        raw, history = u.genetic(w, cfg, seed, objective)
                    r = u.assess(raw, w, cfg)
                    elapsed = time.perf_counter() - t0
                    r.update(planner=planner, scene=scene, route_y=y, seed=seed, runtime_s=elapsed)
                    if r["success"]:
                        p = np.array(r["path"])
                        ts = np.array(r["segment_durations_s"])
                        r["final_check"] = u.final_check(r, w, cfg)
                        r["sat_collision_check"] = not any(
                            sat_hit(a, b, lo, hi)
                            for a, b in zip(p[:-1], p[1:])
                            for lo, hi in w.expanded_boxes(cfg)
                        )
                        delta = np.diff(p, axis=0)
                        L = np.linalg.norm(delta, axis=1)
                        r["exact_max_speed_mps"] = float(np.max(1.875 * L / ts))
                        r["exact_max_acceleration_mps2"] = float(np.max((10 * np.sqrt(3) / 3) * L / ts ** 2))
                        r["exact_max_vertical_mps"] = float(np.max(1.875 * np.abs(delta[:, 2]) / ts))
                        r["analytic_bounds_check"] = bool(
                            r["exact_max_speed_mps"] <= cfg.max_speed + 1e-8
                            and r["exact_max_acceleration_mps2"] <= cfg.max_acceleration + 1e-8
                            and r["exact_max_vertical_mps"] <= cfg.max_vertical_speed + 1e-8
                        )
                        eq64 = u.energy_quotient(p, ts, cfg, 64)
                        r["quadrature_relative_difference"] = abs(eq64 - r["energy_quotient_m1p5_sminus2"]) / eq64
                        r["quadrature_check"] = r["quadrature_relative_difference"] < 1e-9
                        if not all(r[k] for k in ["final_check", "sat_collision_check", "analytic_bounds_check", "quadrature_check"]):
                            r.update(success=False, reason="post_validation_failure")
                    r["history"] = history
                    name = f"{scene}_y{int(y)}_seed{seed}_{planner}.json"
                    (OUT / name).write_text(json.dumps(r, indent=2))
                    records.append(r)
            print(f"completed {scene} y={int(y)}: {len(records)} attempts", flush=True)
    return records


def summarize(records):
    summary = {
        "attempts": len(records),
        "accepted": sum(r["success"] for r in records),
        "battery_energy_computed": False,
        "by_planner": {},
        "max_quadrature_relative_difference": max(r.get("quadrature_relative_difference", 0) for r in records),
    }
    for planner in PLANNERS:
        rows = [r for r in records if r["planner"] == planner]
        good = [r for r in rows if r["success"]]
        summary["by_planner"][planner] = {
            "attempts": len(rows),
            "accepted": len(good),
            "mean_length_m": statistics.mean(r["length_m"] for r in good),
            "mean_proxy_s": statistics.mean(r["equivalent_hover_time_s"] for r in good),
            "mean_runtime_s": statistics.mean(r["runtime_s"] for r in rows),
        }
    paired = []
    for scene in SCENES:
        for y in ROUTES:
            means = {
                p: statistics.mean(
                    r["equivalent_hover_time_s"]
                    for r in records
                    if r["scene"] == scene and r["route_y"] == y and r["planner"] == p and r["success"]
                )
                for p in PLANNERS
            }
            paired.append({
                "scene": scene,
                "route_y": y,
                "proxy_ga_vs_astar_percent": (means["astar"] - means["proxy_ga"]) / means["astar"] * 100,
                "proxy_ga_vs_distance_ga_percent": (means["distance_ga"] - means["proxy_ga"]) / means["distance_ga"] * 100,
            })
    summary["paired_route_descriptive_comparisons"] = paired
    summary["pooled_max_speed_mps"] = max(r.get("exact_max_speed_mps", 0) for r in records)
    summary["pooled_max_acceleration_mps2"] = max(r.get("exact_max_acceleration_mps2", 0) for r in records)
    summary["pooled_max_vertical_mps"] = max(r.get("exact_max_vertical_mps", 0) for r in records)
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def check(summary):
    failures = []

    def compare(label, got, expected, tolerance=1e-9):
        if not math.isclose(got, expected, rel_tol=tolerance, abs_tol=tolerance):
            failures.append(f"{label}: got {got!r}, expected {expected!r}")

    for i, planner in enumerate(PLANNERS):
        row = summary["by_planner"][planner]
        compare(f"{planner} mean_length_m", row["mean_length_m"], REFERENCE["mean_length_m"][i])
        compare(f"{planner} mean_proxy_s", row["mean_proxy_s"], REFERENCE["mean_proxy_s"][i])
        if row["accepted"] != REFERENCE["accepted_per_planner"][i]:
            failures.append(f"{planner} accepted: got {row['accepted']}, expected 45")
    for key in ("pooled_max_speed_mps", "pooled_max_acceleration_mps2", "pooled_max_vertical_mps"):
        compare(key, summary[key], REFERENCE[key])
    compare(
        "mean_route_improvement_vs_astar_pct",
        statistics.mean(r["proxy_ga_vs_astar_percent"] for r in summary["paired_route_descriptive_comparisons"]),
        REFERENCE["mean_route_improvement_vs_astar_pct"],
    )
    compare(
        "mean_route_improvement_vs_distance_ga_pct",
        statistics.mean(r["proxy_ga_vs_distance_ga_percent"] for r in summary["paired_route_descriptive_comparisons"]),
        REFERENCE["mean_route_improvement_vs_distance_ga_pct"],
    )
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare results against the recorded reference values")
    args = parser.parse_args()

    started = time.monotonic()
    records = run_protocol()
    assert len(records) == 180, f"expected 180 attempts, produced {len(records)}"
    summary = summarize(records)
    print(json.dumps({k: v for k, v in summary.items() if k != "paired_route_descriptive_comparisons"}, indent=2))
    print(f"elapsed {time.monotonic() - started:.1f} s")

    if args.check:
        failures = check(summary)
        if failures:
            print("\nREFERENCE MISMATCH")
            for line in failures:
                print(" ", line)
                raise SystemExit(1)
        print("\nAll reference values reproduced exactly.")


if __name__ == "__main__":
    main()
