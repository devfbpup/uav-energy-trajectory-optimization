"""Registry of UrbanScene3D tiles as benchmark scenes (v0.4.0 stage).

The tiles are produced by ``urbanscene3d_prep.py`` under ``$UAV_DATASET_ROOT``
and are never stored in the repository. What the repository stores is the
preparation record of each tile (``urbanscene3d/<tile>.json``: source
checksums, transform, crop, output checksum) and the frozen route file
(``urbanscene3d/routes.json``). A tile is only used when its OBJ matches the
recorded SHA-256.

Ingestion is the unchanged v0.3.0 path (``urban_scenes.ingest_mesh``) with the
v0.3.0 urban workspace, voxel size and lattice. Only the provenance ``dataset``
field differs: ``"UrbanScene3D:<scene>/<tile>"``.

Route rule (written down before any planner runs)
-------------------------------------------------
Candidate endpoints are the nodes of the 4 m A* lattice at z = 10 m that are
valid after obstacle inflation with the urban settings and lie in the largest
connected component of valid lattice nodes over the whole flight band, where
two 26-neighbouring nodes are connected when the segment between them is
clear. This guarantees that a collision-free path exists; it uses only the
collision checker, not a planner.

Five routes per tile are then chosen greedily: route k is the candidate pair
with the largest straight-line distance whose two endpoints are both at least
24 m from every endpoint of routes 0..k-1 (ties: lexicographically smallest
start, then goal, with start < goal). Routes 0, 2, 4 are development and
routes 1, 3 are held out, which interleaves route lengths across the split.
As in v0.3.0 there are 3 development and 2 held-out routes per scene, and no
setting is tuned on either set.

Revision note: a first version of this rule (snap the v0.3.0 corner pairs to
the nearest node connected within the z = 10 m plane) was dry-run on the
tiles before any planner ran. It moved endpoints by up to 115 m and produced
duplicated and near-identical development/held-out routes, so it was replaced
by the rule above. No planner output informed the change.
"""

import json
import os
from pathlib import Path

import numpy as np

import uav_planning as u
import urban_scenes as us

REGISTRY_VERSION = "0.4.0"
ROOT = Path(__file__).resolve().parent
RECORD_DIR = ROOT / "urbanscene3d"
ROUTES_FILE = RECORD_DIR / "routes.json"
REAL_SCENES = ["PolyTech_t0", "ArtSci_t0", "ArtSci_t1"]
ROUTE_Z = 10.0
VOXEL_SIZE = 1.0
URBAN_CFG = dict(grid_step=4.0)


def dataset_root():
    root = os.environ.get("UAV_DATASET_ROOT")
    if not root:
        raise RuntimeError("set UAV_DATASET_ROOT to the folder holding UrbanScene3D and derived/")
    return Path(root)


def tile_record(name, record_dir=RECORD_DIR):
    return json.loads((Path(record_dir) / f"{name}.json").read_text())


def tile_asset(name, record_dir=RECORD_DIR, data_root=None):
    """Path of the prepared tile OBJ, after checking it against its record."""
    record = tile_record(name, record_dir)
    path = Path(data_root or dataset_root()) / "derived" / record["output"]["file"]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run urbanscene3d_prep.py")
    digest = us.sha256_of(path)
    if digest != record["output"]["sha256"]:
        raise ValueError(f"{path.name}: SHA-256 {digest} does not match record {record['output']['sha256']}")
    return path, record


_INGEST_CACHE = {}


def ingest_tile(name, record_dir=RECORD_DIR, data_root=None):
    """Boxes and provenance for a tile, cached in memory and beside the OBJ."""
    key = (name, str(record_dir), str(data_root))
    if key in _INGEST_CACHE:
        return _INGEST_CACHE[key]
    path, record = tile_asset(name, record_dir, data_root)
    cache = path.with_name(f"{path.stem}.boxes-{record['output']['sha256'][:16]}-v{VOXEL_SIZE:g}.json")
    if cache.exists():
        stored = json.loads(cache.read_text())
        boxes, provenance = np.array(stored["boxes"], float).reshape(-1, 2, 3), stored["provenance"]
    else:
        boxes, provenance = us.ingest_mesh(
            path, us.URBAN_WORKSPACE_LOWER, us.URBAN_WORKSPACE_UPPER,
            voxel_size=VOXEL_SIZE, dataset=record["dataset"],
        )
        provenance["prep_record"] = f"urbanscene3d/{name}.json"
        provenance["nodata_fraction"] = record["nodata"]["fraction"]
        cache.write_text(json.dumps({"boxes": boxes.tolist(), "provenance": provenance}))
    _INGEST_CACHE[key] = (boxes, provenance)
    return boxes, provenance


def _world(name, boxes, provenance, start, goal):
    return u.World(
        name, np.array(us.URBAN_WORKSPACE_LOWER), np.array(us.URBAN_WORKSPACE_UPPER),
        boxes.copy(), np.array(start, float), np.array(goal, float), dict(provenance),
    )


# ----------------------------------------------------------------------------
# route rule
# ----------------------------------------------------------------------------
ENDPOINT_SEPARATION_M = 24.0
ROUTES_PER_TILE = 5
DEVELOPMENT_INDICES = (0, 2, 4)
HELDOUT_INDICES = (1, 3)
MOVES = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]


def lattice_component(world, cfg):
    """Valid A* lattice nodes and the largest 26-connected component."""
    step = cfg.grid_step
    lower = world.lower
    dims = np.floor((world.upper - world.lower) / step).astype(int) + 1
    pts = {k: lower + np.array(k) * step for k in np.ndindex(*dims)}
    valid = {k for k, p in pts.items() if world.valid_point(p, cfg)}
    label, components = {}, []
    for seed in sorted(valid):
        if seed in label:
            continue
        stack, members = [seed], []
        label[seed] = len(components)
        while stack:
            cur = stack.pop()
            members.append(cur)
            for move in MOVES:
                nxt = (cur[0] + move[0], cur[1] + move[1], cur[2] + move[2])
                if nxt not in valid or nxt in label:
                    continue
                if world.clear_segment(pts[cur], pts[nxt], cfg):
                    label[nxt] = len(components)
                    stack.append(nxt)
        components.append(members)
    largest = max(components, key=len) if components else []
    return pts, valid, set(largest)


def select_routes(candidates, count=ROUTES_PER_TILE, separation=ENDPOINT_SEPARATION_M):
    """Greedy longest pairs with endpoint separation (see the route rule)."""
    cand = np.array(sorted(map(tuple, candidates)), float)
    diff = cand[:, None, :] - cand[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    usable = np.ones(len(cand), bool)
    chosen = []
    for _ in range(count):
        d = np.where(usable[:, None] & usable[None, :], dist, -1.0)
        d = np.triu(d, 1) + np.tril(np.full_like(d, -1.0))
        best = d.max()
        if best <= 0:
            break
        i, j = np.argwhere(d == best)[0]  # row-major: smallest start, then goal
        chosen.append((cand[i], cand[j]))
        for p in (cand[i], cand[j]):
            usable &= np.linalg.norm(cand - p, axis=1) >= separation
    return chosen


def derive_routes(names=REAL_SCENES, record_dir=RECORD_DIR, data_root=None):
    cfg = u.Settings(**URBAN_CFG)
    out = {"rule": __doc__.split("Route rule (written down before any planner runs)")[1].strip(),
           "registry_version": REGISTRY_VERSION, "z_m": ROUTE_Z, "grid_step_m": cfg.grid_step, "scenes": {}}
    for name in names:
        boxes, provenance = ingest_tile(name, record_dir, data_root)
        world = _world(name, boxes, provenance, (0, 0, ROUTE_Z), (0, 0, ROUTE_Z))
        pts, valid, largest = lattice_component(world, cfg)
        candidates = [pts[k] for k in largest if abs(pts[k][2] - ROUTE_Z) < 1e-9]
        chosen = select_routes(candidates)
        if len(chosen) < ROUTES_PER_TILE:
            raise ValueError(f"{name}: only {len(chosen)} routes satisfy the rule")
        scene = {"valid_nodes": len(valid), "largest_component_nodes": len(largest),
                 "lattice_nodes": len(pts), "candidate_endpoints": len(candidates), "boxes": int(len(boxes))}
        for route_set, indices in (("development", DEVELOPMENT_INDICES), ("heldout", HELDOUT_INDICES)):
            scene[route_set] = [
                {"selection_rank": k, "start": chosen[k][0].tolist(), "goal": chosen[k][1].tolist(),
                 "straight_line_m": float(np.linalg.norm(chosen[k][1] - chosen[k][0]))}
                for k in indices
            ]
        out["scenes"][name] = scene
        print(f"{name}: {len(boxes)} boxes, {len(valid)}/{len(pts)} valid lattice nodes, "
              f"largest component {len(largest)}, {len(candidates)} candidate endpoints", flush=True)
    return out


def load_routes(path=ROUTES_FILE):
    return json.loads(Path(path).read_text())


def real_world(name, route_index=0, route_set="development", record_dir=RECORD_DIR, data_root=None, routes=None):
    if name not in REAL_SCENES:
        raise ValueError("Unknown UrbanScene3D tile")
    boxes, provenance = ingest_tile(name, record_dir, data_root)
    routes = routes or load_routes()
    row = routes["scenes"][name][route_set][route_index]
    world = _world(name, boxes, provenance, row["start"], row["goal"])
    world.provenance.update(route_set=route_set, route_index=route_index)
    return world


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ingest the tiles and freeze routes.json")
    parser.add_argument("--write-routes", action="store_true")
    args = parser.parse_args()
    routes = derive_routes()
    if args.write_routes:
        ROUTES_FILE.write_text(json.dumps(routes, indent=2) + "\n")
        print(f"wrote {ROUTES_FILE.relative_to(ROOT)}")
    else:
        print(json.dumps(routes["scenes"], indent=1)[:3000])
