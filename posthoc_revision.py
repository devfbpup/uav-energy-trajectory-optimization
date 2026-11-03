# -*- coding: utf-8 -*-
"""Post-hoc analyses for the HNICEM 2026 camera-ready version (reviewer comments 3 and 4).

Both analyses re-score the paths already stored in records.json and v040/records.json;
no planner is re-run, so the paths are those chosen under the nominal model and the
rest-to-rest decoder.

A. Power-model sensitivity (comment 3): every accepted trajectory is re-integrated
   under perturbed propulsion parameters, one at a time (x0.7 and x1.3) and jointly
   (Monte Carlo, each parameter uniform in its range). For each setting the
   route-level reductions of the paper are recomputed.

B. Fly-through re-timing (comment 4): every path is re-timed with a velocity-continuous
   quintic Hermite profile that passes through the same retained waypoints without
   stopping. Waypoint speed starts at v_max (1 + cos theta) / 2 along the bisector of
   the two segment directions and is halved locally until every segment satisfies the
   speed, acceleration and vertical-speed limits with 1% margin AND stays within 0.5 m
   of its straight chord. Because the stored chords are clear of obstacles inflated by
   0.75 m (vehicle radius 0.25 m + 0.5 m clearance), a curve within 0.5 m of them is
   still clear of every obstacle inflated by the vehicle radius; only the extra
   clearance margin is used. Zero waypoint speed reproduces the paper's decoder.

    py -3.14 posthoc_revision.py      -> posthoc_revision.json
"""
import json
import numpy as np
from dataclasses import replace
import uav_planning as u
from reproduce import cluster_bootstrap, sign_flip_p

CFG = u.Settings()
NOM = CFG.propulsion
TAU, W = np.polynomial.legendre.leggauss(24)
TAU, W = (TAU + 1) / 2, W / 2
MAX_DEV = 0.5

GROUPS = [("main", "development", "records.json"), ("main", "heldout", "records.json"),
          ("tiles", "development", "v040/records.json"), ("tiles", "heldout", "v040/records.json")]
CONTRASTS = [("energy_ga", "astar"), ("energy_ga", "distance_ga"), ("energy_ga", "time_ga")]
PLANNERS = ("astar", "distance_ga", "proxy_ga", "time_ga", "energy_ga")


# ------------------------------------------------------------------ helpers
def load():
    recs = {}
    for bench, rset, fn in GROUPS:
        rows = [r for r in json.load(open(fn))
                if r["success"] and r["route_set"] == rset and r["planner"] in PLANNERS]
        for r in rows:
            r["_p"] = np.asarray(r["path"], float)
            r["_t"] = np.asarray(r["segment_durations_s"], float)
        recs[(bench, rset)] = rows
    return recs


def route_level(rows, key, stats=True):
    """Route means over accepted attempts, then % reduction per route (as in reproduce.py)."""
    out = {}
    for treat, base in CONTRASTS:
        a, b = {}, {}
        for r in rows:
            if r["planner"] == treat:
                a.setdefault(r["route"], []).append(r[key])
            elif r["planner"] == base:
                b.setdefault(r["route"], []).append(r[key])
        shared = sorted(set(a) & set(b))
        pct = [100.0 * (np.mean(b[k]) - np.mean(a[k])) / np.mean(b[k]) for k in shared]
        if not stats:
            out[treat + "_vs_" + base] = dict(mean=float(np.mean(pct)))
            continue
        bs = cluster_bootstrap(pct)
        out[treat + "_vs_" + base] = dict(mean=bs["mean"], ci=[bs["ci_low"], bs["ci_high"]],
                                         p=sign_flip_p(pct), routes=len(shared))
    return out


def rest_energy(path, times, prop):
    _, d1, d2 = u.quintic(TAU)
    delta = np.diff(path, axis=0)
    vel = delta[:, None, :] * d1[None, :, None] / times[:, None, None]
    acc = delta[:, None, :] * d2[None, :, None] / times[:, None, None] ** 2
    return float(np.sum(times[:, None] * W[None, :] * prop.electrical_power_w(acc, vel)))


# ------------------------------------------------------- A. sensitivity
RANGES = {"mass_kg": (0.7, 1.3), "rotor_radius_m": (0.7, 1.3), "figure_of_merit": (0.7, 1.3),
          "hover_profile_power_w": (0.7, 1.3), "drag_area_m2": (0.7, 1.3),
          "electrical_efficiency": (0.7, 1.15), "ancillary_power_w": (0.7, 1.3)}


def scaled(factors):
    return replace(NOM, **{k: getattr(NOM, k) * f for k, f in factors.items()})


def sensitivity(recs, draws=300, seed=20261103):
    settings = [("nominal", {})]
    for k, (lo, hi) in RANGES.items():
        settings += [("%s x%.2f" % (k, lo), {k: lo}), ("%s x%.2f" % (k, hi), {k: hi})]
    rng = np.random.default_rng(seed)
    for j in range(draws):
        settings.append(("mc%d" % j, {k: rng.uniform(lo, hi) for k, (lo, hi) in RANGES.items()}))
    out = {}
    for name, fac in settings:
        prop = scaled(fac)
        row = {"hover_w": prop.hover_power_w()}
        for g, rows in recs.items():
            for r in rows:
                r["_e"] = rest_energy(r["_p"], r["_t"], prop)
            rl = route_level(rows, "_e", stats=False)
            row["/".join(g)] = {c: v["mean"] for c, v in rl.items()}
        out[name] = row
    summary = {}
    for g in recs:
        key = "/".join(g)
        for c in (a + "_vs_" + b for a, b in CONTRASTS):
            vals = [out[s][key][c] for s in out]
            summary[key + " " + c] = dict(nominal=out["nominal"][key][c],
                                          min=float(np.min(vals)), max=float(np.max(vals)))
    hov = [out[s]["hover_w"] for s in out]
    return dict(settings=len(out), hover_range_w=[float(min(hov)), float(max(hov))],
                summary=summary, oat={s: out[s] for s in out if not s.startswith("mc")})


# ------------------------------------------------------- B. fly-through
S0 = lambda t: (10 * t**3 - 15 * t**4 + 6 * t**5, 30 * t**2 - 60 * t**3 + 30 * t**4,
                60 * t - 180 * t**2 + 120 * t**3)
H1 = lambda t: (t - 6 * t**3 + 8 * t**4 - 3 * t**5, 1 - 18 * t**2 + 32 * t**3 - 15 * t**4,
                -36 * t + 96 * t**2 - 60 * t**3)
H4 = lambda t: (-4 * t**3 + 7 * t**4 - 3 * t**5, -12 * t**2 + 28 * t**3 - 15 * t**4,
                -24 * t + 84 * t**2 - 60 * t**3)
TS = np.linspace(0, 1, 101)


def seg_eval(p0, p1, v0, v1, T, t):
    """Position, velocity, acceleration of the quintic Hermite segment (zero end accel)."""
    T = np.asarray(T, float)[..., None, None]
    s, h, k = S0(t), H1(t), H4(t)
    d = p1 - p0
    pos = p0 + d * s[0][:, None] + T * (v0 * h[0][:, None] + v1 * k[0][:, None])
    vel = (d * s[1][:, None]) / T + v0 * h[1][:, None] + v1 * k[1][:, None]
    acc = (d * s[2][:, None]) / T**2 + (v0 * h[2][:, None] + v1 * k[2][:, None]) / T
    return pos, vel, acc


def chord_dev(pos, p0, p1):
    d = p1 - p0
    L2 = float(d @ d)
    t = np.clip(((pos - p0) @ d) / L2, 0, 1)
    return np.linalg.norm(pos - (p0 + t[..., None] * d), axis=-1)


def min_time(p0, p1, v0, v1):
    """Smallest duration on a geometric grid meeting the limits and the chord bound."""
    L = float(np.linalg.norm(p1 - p0))
    lo = max(L / CFG.max_speed, 0.2)
    hi = 4.0 * float(u.durations(np.array([p0, p1]), CFG)[0])
    grid = np.geomspace(lo, hi, 160)
    pos, vel, acc = seg_eval(p0, p1, v0, v1, grid, TS)
    ok = ((np.linalg.norm(vel, axis=-1).max(-1) <= 0.99 * CFG.max_speed)
          & (np.linalg.norm(acc, axis=-1).max(-1) <= 0.99 * CFG.max_acceleration)
          & (np.abs(vel[..., 2]).max(-1) <= 0.99 * CFG.max_vertical_speed)
          & (chord_dev(pos, p0, p1).max(-1) <= MAX_DEV))
    i = np.flatnonzero(ok)
    return (float(grid[i[0]]), float(chord_dev(pos[i[0]], p0, p1).max())) if len(i) else (None, None)


def fly_through(path):
    n = len(path)
    if n == 2:
        return u.durations(path, CFG), np.zeros((2, 3)), 0.0
    dirs = np.diff(path, axis=0)
    unit = dirs / np.linalg.norm(dirs, axis=1)[:, None]
    speed = np.zeros(n)
    bis = np.zeros((n, 3))
    for k in range(1, n - 1):
        b = unit[k - 1] + unit[k]
        if np.linalg.norm(b) > 1e-9:
            bis[k] = b / np.linalg.norm(b)
            speed[k] = CFG.max_speed * (1 + float(unit[k - 1] @ unit[k])) / 2
            if abs(bis[k][2]) > 1e-12:
                speed[k] = min(speed[k], 0.99 * CFG.max_vertical_speed / abs(bis[k][2]))
    for _ in range(40):
        vel = speed[:, None] * bis
        times, devs, bad = np.zeros(n - 1), np.zeros(n - 1), []
        for j in range(n - 1):
            if speed[j] == 0 and speed[j + 1] == 0:
                times[j] = u.durations(path[j:j + 2], CFG)[0]
                continue
            T, dv = min_time(path[j], path[j + 1], vel[j], vel[j + 1])
            if T is None:
                bad.append(j)
            else:
                times[j], devs[j] = T, dv
        if not bad:
            return times, vel, float(devs.max())
        for j in bad:
            for k in (j, j + 1):
                speed[k] = 0.0 if speed[k] < 0.05 else 0.5 * speed[k]
    raise RuntimeError("fly-through did not converge")


def fly_energy(path, times, vel, prop=NOM):
    e = 0.0
    for j in range(len(path) - 1):
        _, v, a = seg_eval(path[j], path[j + 1], vel[j], vel[j + 1], times[j], TAU)
        e += float(np.sum(times[j] * W * prop.electrical_power_w(a[0], v[0])))
    return e


def flythrough(recs):
    out, maxdev, kept = {}, 0.0, []
    for g, rows in recs.items():
        for r in rows:
            t, v, dv = fly_through(r["_p"])
            r["_fe"] = fly_energy(r["_p"], t, v)
            r["_ft"] = float(np.sum(t))
            r["_rest_check"] = rest_energy(r["_p"], r["_t"], NOM) - r["model_electrical_energy_j"]
            interior = np.linalg.norm(v[1:-1], axis=1) if len(v) > 2 else np.array([])
            kept.append(np.mean(interior > 0) if len(interior) else 1.0)
            maxdev = max(maxdev, dv)
        key = "/".join(g)
        out[key] = dict(
            energy=route_level(rows, "_fe"),
            time=route_level(rows, "_ft"),
            rest=route_level(rows, "model_electrical_energy_j"),
            mean_energy={p: float(np.mean([r["_fe"] for r in rows if r["planner"] == p])) for p in PLANNERS},
            mean_rest_energy={p: float(np.mean([r["model_electrical_energy_j"] for r in rows if r["planner"] == p])) for p in PLANNERS},
            mean_time={p: float(np.mean([r["_ft"] for r in rows if r["planner"] == p])) for p in PLANNERS},
            max_rest_check_j=float(max(abs(r["_rest_check"]) for r in rows)))
    return dict(groups=out, max_chord_deviation_m=maxdev,
                share_of_waypoints_flown_through=float(np.mean(kept)))


if __name__ == "__main__":
    recs = load()
    print({"/".join(k): len(v) for k, v in recs.items()})
    res = dict(flythrough=flythrough(recs))
    print("fly-through done", flush=True)
    res["sensitivity"] = sensitivity(recs)
    json.dump(res, open("posthoc_revision.json", "w"), indent=1)
    F = res["flythrough"]
    print("max chord deviation %.3f m, waypoints flown through %.1f%%"
          % (F["max_chord_deviation_m"], 100 * F["share_of_waypoints_flown_through"]))
    for g, d in F["groups"].items():
        print("\n", g, " rest-check max |dE| %.2e J" % d["max_rest_check_j"])
        for p in PLANNERS:
            print("   %-12s rest E %8.1f   fly E %8.1f   fly T %6.1f s"
                  % (p, d["mean_rest_energy"][p], d["mean_energy"][p], d["mean_time"][p]))
        for c in d["energy"]:
            print("   %-24s rest %6.3f%%   fly %6.3f%% CI %s p=%.3f   (time %6.3f%%)"
                  % (c, d["rest"][c]["mean"], d["energy"][c]["mean"],
                     ["%.3f" % x for x in d["energy"][c]["ci"]], d["energy"][c]["p"],
                     d["time"][c]["mean"]))
    S = res["sensitivity"]
    print("\nsensitivity: %d settings, hover %.1f-%.1f W" % (S["settings"], *S["hover_range_w"]))
    for k, v in S["summary"].items():
        print("  %-48s nominal %6.3f%%  range %6.3f .. %6.3f" % (k, v["nominal"], v["min"], v["max"]))
