"""UrbanScene3D tile preparation: photogrammetric OBJ tiles -> pipeline OBJ.

This script converts one UrbanScene3D real scene (the oblique-photography
reconstruction, delivered as ContextCapture ``Tile_*/*.obj`` files plus a
``metadata.xml`` naming the spatial reference system) into the input that
``urban_scenes.ingest_mesh`` already accepts: a triangle-only Wavefront OBJ in
metres, Z up, ground datum at z = 0, cropped to a 160 m x 160 m tile whose
lower corner is the origin.

Every rule below is fixed before any planner runs and is recorded in the JSON
written next to each tile:

* Transform. Translation only: output = source_local + translation, with the
  rotation the identity and the scale 1. Source coordinates are metres in the
  scene's local frame (EPSG code and origin from ``metadata.xml``), Z up.
* Tile placement. Mapped coverage is measured on a 4 m cell grid (a cell is
  mapped if it contains at least one mesh vertex). A 40 x 40 cell window is
  placed on that grid to maximise the number of mapped cells; ties go to the
  smallest x, then the smallest y. A scene with two tiles gets the
  non-overlapping pair of windows with the largest combined mapped count.
* Ground datum. The 10th percentile of the per-cell minimum vertex height over
  the mapped cells of the tile.
* Crop. A triangle is kept when its xy bounding box intersects the tile and its
  highest vertex is at or above the datum. Triangles are kept whole; the
  voxeliser clips them to the workspace.
* No-data policy. Unmapped cells, dilated by one cell (8-neighbourhood), are
  written as closed boxes spanning the full flight band (z = 2 m to 29.5 m), so
  that ingestion turns them into obstacles. Missing reconstruction is never
  treated as free space.
* Decimation. None. Vertex coordinates are written with 3 decimals (1 mm).

The output is deterministic: the same inputs give a byte-identical OBJ, whose
SHA-256 is recorded. The meshes themselves are governed by the UrbanScene3D
licence (non-commercial, no redistribution) and are written under the dataset
root, never into the repository.

Usage
-----
    export UAV_DATASET_ROOT=/path/to/UrbanScene3D_data   # contains PolyTech/, ArtSci/
    python3 urbanscene3d_prep.py                          # all scenes in SCENE_TILES
    python3 urbanscene3d_prep.py --scene PolyTech
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

PREP_VERSION = "0.4.0"
TILE_SIZE_M = 160.0
CELL_M = 4.0
DATUM_PERCENTILE = 10.0
NODATA_Z = (2.0, 29.5)
COORD_DECIMALS = 3
# Tiles per scene, fixed from the scene extents alone (ArtSci spans more than
# two tile widths in x, PolyTech does not).
SCENE_TILES = {"PolyTech": 1, "ArtSci": 2}
ROOT = Path(__file__).resolve().parent
RECORD_DIR = ROOT / "urbanscene3d"


def dataset_root(root=None):
    root = root or os.environ.get("UAV_DATASET_ROOT")
    if not root:
        raise RuntimeError("set UAV_DATASET_ROOT to the folder holding the UrbanScene3D scenes")
    return Path(root)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------------
def read_metadata(scene_dir):
    text = (Path(scene_dir) / "metadata.xml").read_text()
    srs = re.search(r"<SRS>(.*?)</SRS>", text).group(1).strip()
    origin = [float(x) for x in re.search(r"<SRSOrigin>(.*?)</SRSOrigin>", text).group(1).split(",")]
    return {"srs": srs, "srs_origin": origin}


def source_files(scene_dir):
    return sorted(Path(scene_dir).glob("Tile_*/*.obj"), key=lambda p: p.relative_to(scene_dir).as_posix())


def read_obj(path):
    """Vertices (N, 3) and fan-triangulated faces (M, 3), zero based."""
    vertices, faces = [], []
    with open(path, "r", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                idx = [i - 1 if i > 0 else len(vertices) + i for i in idx]
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
    return np.asarray(vertices, float).reshape(-1, 3), np.asarray(faces, np.int64).reshape(-1, 3)


def read_scene(scene_dir):
    """Concatenate every tile of a scene in sorted order."""
    scene_dir = Path(scene_dir)
    verts, faces, files, offset = [], [], [], 0
    for path in source_files(scene_dir):
        v, f = read_obj(path)
        verts.append(v)
        faces.append(f + offset)
        offset += len(v)
        files.append({"path": path.relative_to(scene_dir).as_posix(), "sha256": sha256_of(path),
                      "vertices": int(len(v)), "triangles": int(len(f))})
    if not verts:
        raise FileNotFoundError(f"no Tile_*/*.obj files under {scene_dir}")
    return np.concatenate(verts), np.concatenate(faces), files


# ----------------------------------------------------------------------------
# tile selection, datum, crop
# ----------------------------------------------------------------------------
def coverage_grid(vertices, cell=CELL_M):
    """Boolean mapped-cell grid, its native-frame origin, and per-cell min z."""
    origin = np.floor(vertices[:, :2].min(axis=0) / cell) * cell
    ij = np.floor((vertices[:, :2] - origin) / cell).astype(int)
    shape = tuple(ij.max(axis=0) + 1)
    mapped = np.zeros(shape, bool)
    mapped[ij[:, 0], ij[:, 1]] = True
    zmin = np.full(shape, np.inf)
    np.minimum.at(zmin, (ij[:, 0], ij[:, 1]), vertices[:, 2])
    return mapped, origin, zmin


def window_counts(mapped, n):
    """Mapped-cell count of every n x n window, indexed by its lower corner."""
    padded = np.zeros((max(mapped.shape[0], n), max(mapped.shape[1], n)), bool)
    padded[: mapped.shape[0], : mapped.shape[1]] = mapped
    s = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), np.int64)
    s[1:, 1:] = padded.cumsum(0).cumsum(1)
    return s[n:, n:] - s[:-n, n:] - s[n:, :-n] + s[:-n, :-n]


def choose_windows(mapped, count, n):
    """Non-overlapping windows maximising the total mapped-cell count.

    One window: the best window. Two windows: the best non-overlapping pair,
    listed in lexicographic order. Ties go to the lexicographically smallest
    corner(s), i.e. smallest x, then smallest y.
    """
    counts = window_counts(mapped, n)
    corners = [(i, j) for i in range(counts.shape[0]) for j in range(counts.shape[1])]
    if count == 1:
        best = max(corners, key=lambda c: (counts[c], -c[0], -c[1]))
        return [best], counts
    if count != 2:
        raise ValueError("at most two tiles per scene are supported")
    best, best_key = None, None
    for a_index, a in enumerate(corners):
        for b in corners[a_index + 1 :]:
            if abs(a[0] - b[0]) < n and abs(a[1] - b[1]) < n:
                continue
            key = (counts[a] + counts[b], -a[0], -a[1], -b[0], -b[1])
            if best_key is None or key > best_key:
                best, best_key = [a, b], key
    if best is None:
        raise ValueError("scene too small for the requested number of tiles")
    return best, counts


def pad_to(grid, n, fill):
    out = np.full((max(grid.shape[0], n), max(grid.shape[1], n)), fill, dtype=grid.dtype)
    out[: grid.shape[0], : grid.shape[1]] = grid
    return out


def nodata_mask(mapped_window):
    """Unmapped cells dilated by one cell in the 8-neighbourhood."""
    unmapped = ~mapped_window
    padded = np.pad(unmapped, 1, constant_values=False)
    out = np.zeros_like(unmapped)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            out |= padded[1 + di : 1 + di + unmapped.shape[0], 1 + dj : 1 + dj + unmapped.shape[1]]
    return out


def mask_to_rectangles(mask):
    """Row runs of a boolean cell mask, merged vertically when identical."""
    runs = []
    for i in range(mask.shape[0]):
        j = 0
        while j < mask.shape[1]:
            if mask[i, j]:
                j1 = j
                while j1 + 1 < mask.shape[1] and mask[i, j1 + 1]:
                    j1 += 1
                runs.append([i, i, j, j1])
                j = j1 + 1
            else:
                j += 1
    merged = []
    for run in runs:
        for m in merged:
            if m[1] + 1 == run[0] and m[2] == run[2] and m[3] == run[3]:
                m[1] = run[1]
                break
        else:
            merged.append(run)
    return [tuple(r) for r in merged]


def crop_faces(vertices, faces, xy_lower, xy_upper, z_floor):
    tri = vertices[faces]
    lo, hi = tri.min(axis=1), tri.max(axis=1)
    keep = (
        (hi[:, 0] >= xy_lower[0]) & (lo[:, 0] <= xy_upper[0])
        & (hi[:, 1] >= xy_lower[1]) & (lo[:, 1] <= xy_upper[1])
        & (hi[:, 2] >= z_floor)
    )
    return faces[keep]


# ----------------------------------------------------------------------------
# writing
# ----------------------------------------------------------------------------
def _box(lower, upper):
    x0, y0, z0 = lower
    x1, y1, z1 = upper
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    f = [(1, 2, 3), (1, 3, 4), (5, 6, 7), (5, 7, 8), (1, 2, 6), (1, 6, 5),
         (2, 3, 7), (2, 7, 6), (3, 4, 8), (3, 8, 7), (4, 1, 5), (4, 5, 8)]
    return v, f


def format_vertex_block(v):
    rounded = np.round(v, COORD_DECIMALS) + 0.0  # + 0.0 turns -0.0 into 0.0
    return "".join(f"v {x:.3f} {y:.3f} {z:.3f}\n" for x, y, z in rounded.tolist())


def write_tile_obj(path, header, vertices, faces, boxes):
    """Triangle OBJ: mesh object, then one object per no-data box (1-based)."""
    used, inverse = np.unique(faces, return_inverse=True)
    local = inverse.reshape(-1, 3) + 1
    parts = ["".join(f"# {line}\n" for line in header), "o urbanscene3d_mesh\n",
             format_vertex_block(vertices[used]),
             "".join(f"f {a} {b} {c}\n" for a, b, c in local.tolist())]
    offset = len(used)
    for k, (lower, upper) in enumerate(boxes):
        bv, bf = _box(lower, upper)
        parts.append(f"o nodata_{k:04d}\n")
        parts.append(format_vertex_block(np.array(bv)))
        parts.append("".join(f"f {a + offset} {b + offset} {c + offset}\n" for a, b, c in bf))
        offset += len(bv)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("".join(parts))
    return int(len(used)), int(len(local))


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
def prepare_scene(scene, data_root=None, out_dir=None, record_dir=RECORD_DIR, n_tiles=None):
    data_root = dataset_root(data_root)
    scene_dir = data_root / scene
    out_dir = Path(out_dir) if out_dir else data_root / "derived"
    n_tiles = n_tiles or SCENE_TILES[scene]
    meta = read_metadata(scene_dir)
    vertices, faces, files = read_scene(scene_dir)
    mapped, grid_origin, zmin = coverage_grid(vertices)
    n = int(round(TILE_SIZE_M / CELL_M))
    windows, _ = choose_windows(mapped, n_tiles, n)
    mapped_p, zmin_p = pad_to(mapped, n, False), pad_to(zmin, n, np.inf)
    results = []
    for t, (i, j) in enumerate(windows):
        tile = f"t{t}"
        m = mapped_p[i : i + n, j : j + n]
        zm = zmin_p[i : i + n, j : j + n][m]
        datum = float(np.round(np.percentile(zm, DATUM_PERCENTILE), COORD_DECIMALS))
        xy_lower = grid_origin + np.array([i, j]) * CELL_M
        xy_upper = xy_lower + TILE_SIZE_M
        translation = [float(-xy_lower[0]), float(-xy_lower[1]), -datum]
        kept = crop_faces(vertices, faces, xy_lower, xy_upper, datum)
        out_vertices = vertices + np.array(translation)
        mask = nodata_mask(m)
        boxes = [
            ((a * CELL_M, c * CELL_M, NODATA_Z[0]), ((b + 1) * CELL_M, (d + 1) * CELL_M, NODATA_Z[1]))
            for a, b, c, d in mask_to_rectangles(mask)
        ]
        name = f"{scene}_{tile}"
        obj_path = out_dir / f"{name}.obj"
        header = [
            f"UrbanScene3D {scene} tile {tile}, prepared by urbanscene3d_prep.py {PREP_VERSION}",
            "units metres, Z up, ground datum z = 0, tile lower corner at the origin",
            "derived from UrbanScene3D; non-commercial use, no redistribution (dataset licence)",
        ]
        n_vertices, n_triangles = write_tile_obj(obj_path, header, out_vertices, kept, boxes)
        record = {
            "prep_version": PREP_VERSION,
            "dataset": f"UrbanScene3D:{scene}/{tile}",
            "scene": scene,
            "tile": tile,
            "source": {
                "product": "oblique-photography reconstruction (Oblique.zip)",
                "srs": meta["srs"],
                "srs_origin": meta["srs_origin"],
                "files": files,
                "metadata_xml_sha256": sha256_of(scene_dir / "metadata.xml"),
                "total_vertices": int(len(vertices)),
                "total_triangles": int(len(faces)),
            },
            "transform": {
                "convention": "output = scale * rotation @ source_local + translation",
                "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "scale": 1.0,
                "translation": translation,
                "source_units": "metres",
                "up_axis": "Z",
            },
            "tile_selection": {
                "rule": "40x40 window of 4 m cells maximising mapped cells; ties smallest x then y; with two tiles, the non-overlapping pair with the largest combined count",
                "cell_m": CELL_M,
                "window_index": [int(i), int(j)],
                "mapped_cells": int(m.sum()),
                "mapped_fraction": float(m.mean()),
            },
            "crop_box_source": {"xy_lower": xy_lower.tolist(), "xy_upper": xy_upper.tolist(), "z_floor": datum},
            "crop_box_output": {"xy_lower": [0.0, 0.0], "xy_upper": [TILE_SIZE_M, TILE_SIZE_M], "z_floor": 0.0},
            "ground_datum": {"rule": f"{DATUM_PERCENTILE:g}th percentile of per-cell minimum vertex z over mapped cells",
                             "source_z": datum},
            "nodata": {"rule": "unmapped 4 m cells dilated by one cell become closed boxes over the flight band",
                       "z_range": list(NODATA_Z), "cells": int(mask.sum()),
                       "fraction": float(mask.mean()), "boxes": len(boxes)},
            "decimation": "none",
            "coordinate_decimals": COORD_DECIMALS,
            "output": {"file": obj_path.name, "sha256": sha256_of(obj_path),
                       "mesh_vertices": n_vertices, "mesh_triangles": n_triangles},
        }
        record_dir = Path(record_dir)
        record_dir.mkdir(parents=True, exist_ok=True)
        (record_dir / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
        results.append(record)
        print(f"{name}: mapped {m.mean():.1%}, datum {datum} m, {n_triangles} triangles, "
              f"{len(boxes)} no-data boxes, sha256 {record['output']['sha256'][:12]}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", choices=sorted(SCENE_TILES), action="append")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    for scene in args.scene or sorted(SCENE_TILES):
        prepare_scene(scene, args.data_root)


if __name__ == "__main__":
    main()
