"""Mesh-to-obstacle ingestion and dense urban benchmark scenes.

This module implements the scene-ingestion stage that an UrbanScene3D tile
requires: a triangle mesh (Wavefront OBJ) is voxelised into a conservative
occupancy grid inside a declared workspace, the occupancy grid is decomposed
into axis-aligned boxes, and the resulting obstacle set is recorded together
with the asset checksum, voxel size, inflation policy and workspace crop.

The pipeline is asset agnostic: it accepts any OBJ file, including an
UrbanScene3D export. The benchmark scenes shipped with the repository are
procedurally generated dense-city meshes written by ``build_city_mesh`` so that
the full ingestion path is exercised and checksummed offline. Provenance always
records which asset a scene came from, so a procedurally generated city is
never presented as photogrammetric data.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

import uav_planning as u

INGEST_VERSION = "0.3.0"


# ----------------------------------------------------------------------------
# mesh writing (deterministic procedural city)
# ----------------------------------------------------------------------------
def _box_mesh(lower, upper):
    x0, y0, z0 = lower
    x1, y1, z1 = upper
    vertices = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    faces = [
        (1, 2, 3), (1, 3, 4), (5, 6, 7), (5, 7, 8),
        (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6),
        (3, 4, 8), (3, 8, 7), (4, 1, 5), (4, 5, 8),
    ]
    return vertices, faces


def city_layout(seed, extent=160.0, block=16.0, street=8.0, min_height=18.0, max_height=60.0):
    """Deterministic dense-city building layout on a street grid."""
    rng = np.random.default_rng(seed)
    pitch = block + street
    buildings = []
    n = int(np.floor((extent - street) / pitch))
    for i in range(n):
        for j in range(n):
            x0 = street + i * pitch
            y0 = street + j * pitch
            if rng.random() < 0.12:
                continue  # plaza / open lot
            inset_x = rng.uniform(0.0, 2.0)
            inset_y = rng.uniform(0.0, 2.0)
            width = block - inset_x
            depth = block - inset_y
            if rng.random() < 0.3:  # split the block into two narrower towers
                depth = depth / 2 - 2.0
            height = float(rng.uniform(min_height, max_height))
            buildings.append(
                (
                    (round(x0 + inset_x / 2, 3), round(y0 + inset_y / 2, 3), 0.0),
                    (round(x0 + inset_x / 2 + width, 3), round(y0 + inset_y / 2 + depth, 3), round(height, 3)),
                )
            )
    return buildings


def build_city_mesh(path, seed, **kwargs):
    """Write a deterministic dense-city OBJ asset and return its checksum."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buildings = city_layout(seed, **kwargs)
    lines = [
        "# procedurally generated dense-city asset",
        f"# generator urban_scenes.build_city_mesh seed={seed} version={INGEST_VERSION}",
    ]
    offset = 0
    for k, (lower, upper) in enumerate(buildings):
        vertices, faces = _box_mesh(lower, upper)
        lines.append(f"o building_{k:03d}")
        lines += [f"v {x:.3f} {y:.3f} {z:.3f}" for x, y, z in vertices]
        lines += [f"f {a + offset} {b + offset} {c + offset}" for a, b, c in faces]
        offset += len(vertices)
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return sha256_of(path), len(buildings)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# ingestion: OBJ -> voxels -> axis-aligned boxes
# ----------------------------------------------------------------------------
def read_obj_triangles(path):
    vertices = []
    triangles = []
    with open(path, "r") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                idx = [i - 1 if i > 0 else len(vertices) + i for i in idx]
                for k in range(1, len(idx) - 1):
                    triangles.append((idx[0], idx[k], idx[k + 1]))
    v = np.asarray(vertices, float)
    f = np.asarray(triangles, int)
    if len(v) == 0 or len(f) == 0:
        raise ValueError("mesh contains no triangles")
    return v[f]


def voxelise(triangles, lower, upper, voxel_size, column_fill=True):
    """Conservative surface voxelisation with optional vertical column fill."""
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    dims = np.maximum(np.ceil((upper - lower) / voxel_size).astype(int), 1)
    grid = np.zeros(tuple(dims), bool)
    for tri in triangles:
        a, b, c = tri
        edge = max(
            np.linalg.norm(b - a), np.linalg.norm(c - a), np.linalg.norm(c - b)
        )
        n = int(max(2, np.ceil(2.0 * edge / voxel_size)))
        s = np.linspace(0.0, 1.0, n)
        uu, vv = np.meshgrid(s, s, indexing="ij")
        mask = (uu + vv) <= 1.0
        pts = a + uu[mask][:, None] * (b - a) + vv[mask][:, None] * (c - a)
        idx = np.floor((pts - lower) / voxel_size).astype(int)
        inside = np.all((idx >= 0) & (idx < dims), axis=1)
        idx = idx[inside]
        if len(idx):
            grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    if column_fill:
        # Buildings are vertical structures: fill each column between its
        # lowest and highest occupied voxel so hollow shells cannot be entered.
        occupied = grid.any(axis=2)
        zs = np.arange(dims[2])[None, None, :]
        first = np.argmax(grid, axis=2)
        last = dims[2] - 1 - np.argmax(grid[:, :, ::-1], axis=2)
        fill = (zs >= first[:, :, None]) & (zs <= last[:, :, None]) & occupied[:, :, None]
        grid |= fill
    return grid, lower, float(voxel_size)


def boxes_from_voxels(grid, lower, voxel_size):
    """Greedy decomposition of an occupancy grid into axis-aligned boxes."""
    grid = grid.copy()
    nx, ny, nz = grid.shape
    boxes = []
    for k in range(nz):
        layer = grid[:, :, k]
        while layer.any():
            i, j = np.argwhere(layer)[0]
            j1 = j
            while j1 + 1 < ny and layer[i, j1 + 1]:
                j1 += 1
            i1 = i
            while i1 + 1 < nx and layer[i1 + 1, j : j1 + 1].all():
                i1 += 1
            k1 = k
            while k1 + 1 < nz and grid[i : i1 + 1, j : j1 + 1, k1 + 1].all():
                k1 += 1
            grid[i : i1 + 1, j : j1 + 1, k : k1 + 1] = False
            layer = grid[:, :, k]
            boxes.append(
                [
                    [lower[0] + i * voxel_size, lower[1] + j * voxel_size, lower[2] + k * voxel_size],
                    [
                        lower[0] + (i1 + 1) * voxel_size,
                        lower[1] + (j1 + 1) * voxel_size,
                        lower[2] + (k1 + 1) * voxel_size,
                    ],
                ]
            )
    return np.array(boxes, float).reshape(-1, 2, 3)


def ingest_mesh(asset_path, lower, upper, voxel_size=2.0, name=None, dataset="procedural_city"):
    """Full ingestion: OBJ asset -> conservative box obstacle set + provenance."""
    asset_path = Path(asset_path)
    triangles = read_obj_triangles(asset_path)
    grid, origin, vs = voxelise(triangles, lower, upper, voxel_size)
    boxes = boxes_from_voxels(grid, origin, vs)
    provenance = {
        "kind": "ingested_mesh",
        "dataset": dataset,
        "asset": asset_path.name,
        "asset_sha256": sha256_of(asset_path),
        "triangles": int(len(triangles)),
        "voxel_size_m": vs,
        "workspace_lower": list(map(float, lower)),
        "workspace_upper": list(map(float, upper)),
        "occupied_voxels": int(grid.sum()),
        "boxes": int(len(boxes)),
        "ingest_version": INGEST_VERSION,
        "conservative": "voxels are rounded outward and vertically filled; free space is never enlarged",
    }
    return boxes, provenance


# ----------------------------------------------------------------------------
# benchmark scenes
# ----------------------------------------------------------------------------
URBAN_WORKSPACE_LOWER = (0.0, 0.0, 2.0)
URBAN_WORKSPACE_UPPER = (160.0, 160.0, 30.0)
URBAN_SEEDS = {"urban_a": 101, "urban_b": 202, "urban_c": 303}
URBAN_ROUTES = {
    # (start, goal) pairs, grid aligned to the 4 m urban A* lattice
    "development": [
        ((4.0, 4.0, 10.0), (156.0, 156.0, 10.0)),
        ((4.0, 76.0, 10.0), (156.0, 76.0, 10.0)),
        ((76.0, 4.0, 10.0), (76.0, 156.0, 10.0)),
    ],
    "heldout": [
        ((4.0, 156.0, 10.0), (156.0, 4.0, 10.0)),
        ((28.0, 4.0, 10.0), (132.0, 156.0, 10.0)),
    ],
}


def urban_asset_path(name, root="assets"):
    return Path(root) / f"{name}.obj"


def ensure_urban_assets(root="assets"):
    """Generate the dense-city OBJ assets if they are absent; return checksums."""
    info = {}
    for name, seed in URBAN_SEEDS.items():
        path = urban_asset_path(name, root)
        if not path.exists():
            digest, count = build_city_mesh(path, seed)
        else:
            digest, count = sha256_of(path), None
        info[name] = {"asset": str(path), "sha256": digest, "buildings": count}
    return info


_URBAN_CACHE = {}


def urban_world(name, route_index=0, route_set="development", root="assets", voxel_size=1.0):
    if name not in URBAN_SEEDS:
        raise ValueError("Unknown urban scene")
    key = (name, root, voxel_size)
    if key not in _URBAN_CACHE:
        ensure_urban_assets(root)
        boxes, provenance = ingest_mesh(
            urban_asset_path(name, root),
            URBAN_WORKSPACE_LOWER,
            URBAN_WORKSPACE_UPPER,
            voxel_size=voxel_size,
            dataset="procedural_dense_city",
        )
        _URBAN_CACHE[key] = (boxes, provenance)
    boxes, provenance = _URBAN_CACHE[key]
    start, goal = URBAN_ROUTES[route_set][route_index]
    return u.World(
        name,
        np.array(URBAN_WORKSPACE_LOWER),
        np.array(URBAN_WORKSPACE_UPPER),
        boxes.copy(),
        np.array(start, float),
        np.array(goal, float),
        dict(provenance, route_set=route_set, route_index=route_index),
    )


def synthetic_world(name, y):
    w = u.scenario(name)
    w.start[1] = y
    w.goal[1] = y
    w.provenance = {"kind": "synthetic_boxes", "source": "hand specified", "route_y": y}
    return w


if __name__ == "__main__":
    info = ensure_urban_assets()
    for name in URBAN_SEEDS:
        world = urban_world(name)
        print(name, info[name]["sha256"][:12], "boxes", len(world.boxes), json.dumps(world.provenance)[:120])
