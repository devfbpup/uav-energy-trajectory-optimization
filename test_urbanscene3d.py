"""Tests for the UrbanScene3D preparation script and tile registry (v0.4.0).

The tests build a small synthetic scene in the ContextCapture layout
(``Tile_*/*.obj`` + ``metadata.xml``), so they run without the dataset. When
``urbanscene3d/routes.json`` exists, its structure is checked as well.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import uav_planning as u
import urban_scenes as us
import urbanscene3d_prep as prep
import urbanscene3d_scenes as reg

GROUND_Z = -30.0
OFFSET = np.array([-50.0, -20.0])  # native frame is not anchored at the origin
HOLE = ((100.0, 124.0), (40.0, 64.0))  # unmapped patch, native frame minus OFFSET
BUILDING = ((60.0, 60.0), (80.0, 80.0), 25.0)  # footprint and height above ground


def _in_hole(x, y):
    return HOLE[0][0] <= x < HOLE[0][1] and HOLE[1][0] <= y < HOLE[1][1]


def write_synthetic_scene(scene_dir, extent=(180.0, 172.0), step=2.0):
    """Ground grid (with an unmapped hole) split over two tiles, plus a building."""
    scene_dir = Path(scene_dir)
    (scene_dir / "metadata.xml").parent.mkdir(parents=True, exist_ok=True)
    (scene_dir / "metadata.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<ModelMetadata version="1">\n'
        "\t<SRS>EPSG:4547</SRS>\n\t<SRSOrigin>1000,2000,30</SRSOrigin>\n</ModelMetadata>\n"
    )
    xs = np.arange(0.0, extent[0] + step / 2, step)
    ys = np.arange(0.0, extent[1] + step / 2, step)
    half = extent[0] / 2
    for t, (x_lo, x_hi) in enumerate([(0.0, half), (half, extent[0])]):
        lines, index = [], {}
        for i, x in enumerate(xs):
            if x < x_lo or x > x_hi:
                continue
            for j, y in enumerate(ys):
                if _in_hole(x, y):
                    continue
                index[(i, j)] = len(index) + 1
                lines.append(f"v {x + OFFSET[0]:.3f} {y + OFFSET[1]:.3f} {GROUND_Z:.3f}")
        for (i, j), a in sorted(index.items()):
            quad = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            if all(q in index for q in quad):
                lines.append("f " + " ".join(f"{index[q]}/{index[q]}" for q in quad))  # quad, textured syntax
        if t == 0:
            (x0, y0), (x1, y1), h = BUILDING
            base = len(index)
            vs, fs = us._box_mesh((x0 + OFFSET[0], y0 + OFFSET[1], GROUND_Z), (x1 + OFFSET[0], y1 + OFFSET[1], GROUND_Z + h))
            lines += [f"v {x:.3f} {y:.3f} {z:.3f}" for x, y, z in vs]
            lines += [f"f {a + base} {b + base} {c + base}" for a, b, c in fs]
        path = scene_dir / f"Tile_+00{t}_+000" / f"Tile_+00{t}_+000.obj"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")


class PrepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        write_synthetic_scene(root / "data" / "Synthetic")
        cls.root = root
        cls.records = prep.prepare_scene("Synthetic", data_root=root / "data", record_dir=root / "records", n_tiles=1)
        cls.record = cls.records[0]
        cls.obj = root / "data" / "derived" / "Synthetic_t0.obj"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_transform_is_translation_to_tile_corner_and_datum(self):
        t = self.record["transform"]
        self.assertEqual(t["rotation"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertEqual(t["scale"], 1.0)
        lower = self.record["crop_box_source"]["xy_lower"]
        self.assertEqual(t["translation"], [-lower[0], -lower[1], -GROUND_Z])
        self.assertAlmostEqual(self.record["ground_datum"]["source_z"], GROUND_Z)
        # every tile corner lies on the 4 m cell grid of the native frame
        grid_origin = np.floor(OFFSET / prep.CELL_M) * prep.CELL_M
        self.assertTrue(np.allclose(np.mod(np.array(lower) - grid_origin, prep.CELL_M), 0.0))

    def test_building_lands_at_transformed_position(self):
        triangles = us.read_obj_triangles(self.obj)
        pts = triangles.reshape(-1, 3)
        shift = np.array(self.record["transform"]["translation"])
        (x0, y0), (x1, y1), h = BUILDING
        corner = np.array([x1 + OFFSET[0], y1 + OFFSET[1], GROUND_Z + h]) + shift
        self.assertTrue(np.any(np.all(np.abs(pts - corner) < 1e-3, axis=1)))
        # ground at the datum maps to z = 0, the building roof to its height
        mesh = pts[: 3 * self.record["output"]["mesh_triangles"]]
        self.assertAlmostEqual(float(mesh[:, 2].min()), 0.0, places=3)
        self.assertAlmostEqual(float(mesh[:, 2].max()), h, places=3)

    def test_crop_keeps_only_triangles_touching_the_tile(self):
        triangles = us.read_obj_triangles(self.obj)[: self.record["output"]["mesh_triangles"]]
        lo, hi = triangles.min(axis=1), triangles.max(axis=1)
        self.assertTrue(np.all(hi[:, :2] >= -1e-6))
        self.assertTrue(np.all(lo[:, :2] <= prep.TILE_SIZE_M + 1e-6))
        self.assertLess(self.record["output"]["mesh_triangles"], self.record["source"]["total_triangles"])

    def test_crop_faces_unit(self):
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 10, 0], [11, 10, 0], [10, 11, 0], [0, 0, -5], [1, 0, -5], [0, 1, -5]], float)
        f = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        kept = prep.crop_faces(v, f, np.array([-1.0, -1.0]), np.array([5.0, 5.0]), z_floor=-1.0)
        self.assertEqual(kept.tolist(), [[0, 1, 2]])

    def test_unmapped_cells_become_nodata_boxes(self):
        nodata = self.record["nodata"]
        self.assertGreater(nodata["boxes"], 0)
        world = u.World("t", np.array(us.URBAN_WORKSPACE_LOWER), np.array(us.URBAN_WORKSPACE_UPPER),
                        us.ingest_mesh(self.obj, us.URBAN_WORKSPACE_LOWER, us.URBAN_WORKSPACE_UPPER, 1.0)[0],
                        np.zeros(3), np.zeros(3))
        cfg = u.Settings(grid_step=4.0)
        shift = np.array(self.record["transform"]["translation"][:2])
        centre = np.array([(HOLE[0][0] + HOLE[0][1]) / 2, (HOLE[1][0] + HOLE[1][1]) / 2]) + OFFSET + shift
        self.assertFalse(world.valid_point(np.array([centre[0], centre[1], 10.0]), cfg))
        mapped_point = np.array([20.0, 20.0]) + OFFSET + shift
        self.assertTrue(world.valid_point(np.array([mapped_point[0], mapped_point[1], 10.0]), cfg))

    def test_nodata_mask_dilates_by_one_cell(self):
        mapped = np.ones((5, 5), bool)
        mapped[2, 2] = False
        mask = prep.nodata_mask(mapped)
        self.assertEqual(int(mask.sum()), 9)
        self.assertTrue(mask[1:4, 1:4].all())

    def test_mask_to_rectangles_covers_mask_exactly(self):
        rng = np.random.default_rng(3)
        mask = rng.random((12, 9)) < 0.4
        rebuilt = np.zeros_like(mask)
        for a, b, c, d in prep.mask_to_rectangles(mask):
            self.assertFalse(rebuilt[a : b + 1, c : d + 1].any())
            rebuilt[a : b + 1, c : d + 1] = True
        self.assertTrue(np.array_equal(rebuilt, mask))

    def test_choose_windows_two_tiles_do_not_overlap(self):
        mapped = np.zeros((10, 4), bool)
        mapped[:3, :] = True
        mapped[7:, :] = True
        (a, b), counts = prep.choose_windows(mapped, 2, 4)
        self.assertTrue(abs(a[0] - b[0]) >= 4 or abs(a[1] - b[1]) >= 4)
        self.assertEqual(counts[a] + counts[b], 24)

    def test_output_is_deterministic(self):
        again = prep.prepare_scene("Synthetic", data_root=self.root / "data",
                                   out_dir=self.root / "again", record_dir=self.root / "records2", n_tiles=1)[0]
        self.assertEqual(again["output"]["sha256"], self.record["output"]["sha256"])
        self.assertEqual((self.root / "again" / "Synthetic_t0.obj").read_bytes(), self.obj.read_bytes())
        self.assertEqual(self.record["output"]["sha256"], us.sha256_of(self.obj))

    def test_select_routes_longest_first_with_separation(self):
        grid = [np.array([x, y, 10.0]) for x in range(0, 101, 4) for y in range(0, 41, 4)]
        chosen = reg.select_routes(grid, count=3, separation=24.0)
        self.assertEqual(len(chosen), 3)
        lengths = [float(np.linalg.norm(g - s)) for s, g in chosen]
        self.assertAlmostEqual(lengths[0], float(np.hypot(100, 40)))
        self.assertEqual(lengths, sorted(lengths, reverse=True))
        ends = [p for pair in chosen for p in pair]
        for a in range(len(ends)):
            for b in range(a + 1, len(ends)):
                if a // 2 != b // 2:
                    self.assertGreaterEqual(float(np.linalg.norm(ends[a] - ends[b])), 24.0)
        self.assertEqual([c[0].tolist() for c in chosen], [c[0].tolist() for c in reg.select_routes(grid[::-1], 3)])

    def test_record_holds_source_checksums(self):
        files = self.record["source"]["files"]
        self.assertEqual(len(files), 2)
        for entry in files:
            self.assertEqual(entry["sha256"], us.sha256_of(self.root / "data" / "Synthetic" / entry["path"]))
        self.assertEqual(self.record["source"]["srs"], "EPSG:4547")
        self.assertEqual(self.record["dataset"], "UrbanScene3D:Synthetic/t0")
        self.assertEqual(self.record["decimation"], "none")

    def test_registry_checks_checksum_and_labels_provenance(self):
        boxes, provenance = reg.ingest_tile("Synthetic_t0", record_dir=self.root / "records", data_root=self.root / "data")
        self.assertEqual(provenance["dataset"], "UrbanScene3D:Synthetic/t0")
        self.assertGreater(len(boxes), 0)
        routes = reg.derive_routes(["Synthetic_t0"], record_dir=self.root / "records", data_root=self.root / "data")
        scene = routes["scenes"]["Synthetic_t0"]
        cfg = u.Settings(**reg.URBAN_CFG)
        self.assertEqual(len(scene["development"]), 3)
        self.assertEqual(len(scene["heldout"]), 2)
        for route_set in ("development", "heldout"):
            for row in scene[route_set]:
                w = reg._world("Synthetic_t0", boxes, provenance, row["start"], row["goal"])
                u.valid_endpoints(w, cfg)
                for p in (row["start"], row["goal"]):
                    self.assertEqual(p[2], reg.ROUTE_Z)
                    self.assertTrue(np.allclose(np.mod(np.array(p[:2]), cfg.grid_step), 0.0))
        # the longest route is development rank 0 and spans most of the tile
        self.assertEqual(scene["development"][0]["selection_rank"], 0)
        self.assertGreater(scene["development"][0]["straight_line_m"], 150.0)
        # a corrupted tile is refused
        tampered = self.root / "records_tampered"
        tampered.mkdir(exist_ok=True)
        rec = dict(self.record, output=dict(self.record["output"], sha256="0" * 64))
        (tampered / "Synthetic_t0.json").write_text(json.dumps(rec))
        with self.assertRaises(ValueError):
            reg.tile_asset("Synthetic_t0", record_dir=tampered, data_root=self.root / "data")


@unittest.skipUnless(reg.ROUTES_FILE.exists(), "routes.json not frozen yet")
class FrozenRoutesTest(unittest.TestCase):
    def test_frozen_routes_follow_the_split_and_lattice(self):
        routes = reg.load_routes()
        self.assertEqual(sorted(routes["scenes"]), sorted(reg.REAL_SCENES))
        for name, scene in routes["scenes"].items():
            self.assertEqual(len(scene["development"]), 3)
            self.assertEqual(len(scene["heldout"]), 2)
            ranks = [row["selection_rank"] for rs in ("development", "heldout") for row in scene[rs]]
            self.assertEqual(ranks, list(reg.DEVELOPMENT_INDICES) + list(reg.HELDOUT_INDICES))
            for route_set in ("development", "heldout"):
                for row in scene[route_set]:
                    for p in (row["start"], row["goal"]):
                        self.assertEqual(p[2], reg.ROUTE_Z)
                        self.assertTrue(np.allclose(np.mod(np.array(p[:2]), 4.0), 0.0))
                    self.assertGreater(row["straight_line_m"], 0.0)

    def test_tile_records_exist_for_every_scene(self):
        for name in reg.REAL_SCENES:
            record = reg.tile_record(name)
            self.assertEqual(record["dataset"], f"UrbanScene3D:{record['scene']}/{record['tile']}")
            self.assertEqual(len(record["output"]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
