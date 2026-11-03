# -*- coding: utf-8 -*-
"""Extra checks on the fly-through re-timing of posthoc_revision.py, for the camera-ready text.

1. Workspace bounds: every re-timed trajectory, sampled densely, must stay inside the
   workspace box of its scene (all three coordinates).
2. Chord deviation, sampled at 1001 points per segment instead of 101.
3. The share of interior waypoints flown through, pooled over all interior waypoints.

    py -3.14 posthoc_bounds_check.py
"""
import json
import numpy as np
import uav_planning as u
import posthoc_revision as P

recs = P.load()


TS = np.linspace(0, 1, 1001)
lo_margin, hi_margin, maxdev = np.inf, np.inf, 0.0
interior_total = interior_flown = 0
for (bench, rset), rows in recs.items():
    for r in rows:
        p = r["_p"]
        box = P.workspace(r)
        t, v, _ = P.fly_through(p, box)
        for j in range(len(p) - 1):
            pos, _, _ = P.seg_eval(p[j], p[j + 1], v[j], v[j + 1], t[j], TS)   # (1001, 3)
            lo_margin = min(lo_margin, float((pos - box[0]).min()))
            hi_margin = min(hi_margin, float((box[1] - pos).min()))
            maxdev = max(maxdev, float(P.chord_dev(pos, p[j], p[j + 1]).max()))
        if len(p) > 2:
            sp = np.linalg.norm(v[1:-1], axis=1)
            interior_total += len(sp)
            interior_flown += int(np.sum(sp > 0))

print("min margin inside lower workspace bounds: %.4f m" % lo_margin)
print("min margin inside upper workspace bounds: %.4f m" % hi_margin)
print("max chord deviation at 1001 samples/segment: %.4f m" % maxdev)
print("interior waypoints flown through: %d of %d (%.1f%%)"
      % (interior_flown, interior_total, 100.0 * interior_flown / interior_total))
