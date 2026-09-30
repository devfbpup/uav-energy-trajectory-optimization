# Evolutionary Trajectory Optimization for Energy-Constrained UAVs in Dense Urban Environments

Reproducible benchmark code for the paper *"Evolutionary Trajectory Optimization for
Energy-Constrained UAVs in Dense Urban Environments: A Data-Driven Approach using
UrbanScene3D"*.

**Version 0.4.0**: adds the v0.3.0 protocol on registered UrbanScene3D tiles (`v040/`). The v0.3.0
artefacts are unchanged and still verify with `reproduce.py --check` and `audit.py`.

Authors:

- Devjyoti Saha — Department of Electronics Engineering, Polytechnic University of the Philippines
- Jo-Ann V. Magsumbol — Department of Electronics Engineering, Polytechnic University of the Philippines

> Affiliations here are synchronised with the manuscript, `CITATION.cff` and the archive.
> Earlier releases listed a different affiliation for the second author in this README; the
> manuscript affiliation (PUP) is authoritative.

## What this release adds over v0.2.0

| Area | v0.2.0 | v0.3.0 |
| --- | --- | --- |
| Energy | acceleration proxy only | proxy **and** a parameterised electrical propulsion model in joules, with a hard mission-energy constraint (30 000 J allowance, 20 % reserve → 24 000 J usable) |
| Scenes | 3 synthetic box scenes | 3 synthetic + 3 mesh-ingested dense urban scenes (OBJ → voxels → conservative boxes, ~130 boxes each) |
| Objectives | distance, proxy | distance, proxy, **time**, **energy** |
| Routes | 9 development | 18 development + **12 held-out** |
| Statistics | pooled means | route-level cluster bootstrap (10 000 resamples) + sign-flip permutation tests with Holm correction |
| Budgets | fixed | matched evaluation-budget sweep and quality-versus-runtime curves |
| Ablations | none | warm start, shortcutting, both |
| Artefacts | reproduce script | plotting/export script, audit script, checksum manifest, per-attempt records |
| Claim wording | "reproduced exactly" | "matched the reference aggregates within the specified tolerance" (1e-9 rel/abs) |

## Files

| File | Purpose |
| --- | --- |
| `uav_planning.py` | world model, quintic decoder, proxy and propulsion energy, energy constraint, A*, RRT*, genetic planner (4 objectives) |
| `urban_scenes.py` | procedural city mesh generator, OBJ reader, voxeliser, greedy box decomposition, UrbanScene3D-compatible ingestion entry point, scene/route registry |
| `reproduce.py` | protocol runner: main run, budget sweep, ablations, aggregation, reference writing and tolerance checking |
| `make_figures.py` | all ten figures (PNG + PDF) and a CSV export of every plotted series, plus two summary tables |
| `audit.py` | writes and verifies `manifest.sha256` / `manifest.json` (bitwise SHA-256 over every tracked artefact) |
| `test_validation.py` | 46 unit tests (geometry, SAT cross-check, kinematics, quadrature, power model, constraint, planners, ingestion) |
| `paper/main.tex`, `paper/main.html`, `paper/main.pdf` | manuscript source and rendered PDF |

Generated artefacts: `protocol.json`, `records.json`, `runs/` (one JSON per attempt),
`budget_sweep.json`, `ablations.json`, `summary.json`, `reference.json`,
`figures/`, `exports/`, `manifest.sha256`, `manifest.json`, `assets/*.obj`.

## Reproducing

```bash
python3 -m unittest test_validation -v        # 46 tests
python3 reproduce.py                          # full protocol (main + sweep + ablations + summary)
python3 reproduce.py --check                  # compare aggregates against reference.json
python3 make_figures.py                       # figures + CSV exports
python3 audit.py --write                      # (re)build the checksum manifest
python3 audit.py                              # verify every manifest entry bitwise
```

The protocol is resumable, which matters on constrained machines:

```bash
python3 reproduce.py --stage main
python3 reproduce.py --stage sweep      --routes sparse,wall
python3 reproduce.py --stage ablations  --routes urban_a,urban_b
python3 reproduce.py --stage summarize  --write-reference
```

## What the two verification mechanisms mean

They are deliberately separate and are **not** the same claim:

- `audit.py` performs **bitwise** SHA-256 verification of file content against
  `manifest.sha256` (947 tracked files in this release).
- `reproduce.py --check` performs a **numerical** comparison of 243 tracked aggregates
  against `reference.json` with a tolerance of `1e-9` relative and absolute. On success it
  prints *"matched the reference values within the specified tolerance … This is a
  numerical comparison, not a bitwise file comparison."* No claim of bitwise-identical
  floating-point results across machines is made, and no second-machine records are
  included in this release.

## Scene provenance and the UrbanScene3D hook

`urban_scenes.ingest_mesh()` accepts any OBJ triangle mesh, including an UrbanScene3D
export, and records `dataset`, `asset_sha256`, triangle count, voxel size, workspace crop,
occupied voxel count and box count into the scene provenance. The urban scenes shipped
here were generated by the deterministic procedural city generator (`dataset =
procedural_dense_city`) because the execution environment used for the reported runs had
**no network access** to the dataset archive. Every result therefore states its provenance;
point `ensure_urban_assets()` at dataset meshes to run the identical protocol on
UrbanScene3D tiles.

Asset digests (voxel size 1.0 m):

| Scene | SHA-256 (prefix) | Buildings | Boxes |
| --- | --- | --- | --- |
| `urban_a` | `0a9045f9…` | 31 | 131 |
| `urban_b` | `316c7ebd…` | 30 | — |
| `urban_c` | `48db175d…` | 33 | — |

## Headline results (900 attempts, 899 accepted)

Route-level mean reduction in modelled electrical energy versus A*: **2.273 %**
(95 % CI [0.300, 5.374]) on development routes, **1.014 %** ([0.488, 1.638]) held-out.
Versus the minimum-time GA the energy GA gains only **0.049 %** (development) and is
**0.078 %** worse (held-out), neither significant after Holm correction — i.e. the gain over
A* is mostly shorter traversal time and fewer enforced stops, not a distinct energy effect.

## v0.4.0: the identical protocol on registered UrbanScene3D tiles

The three procedural urban scenes of v0.3.0 are joined by three tiles cut from the
UrbanScene3D **real-scene oblique-photography reconstructions** (Lin et al., ECCV 2022):
`UrbanScene3D:PolyTech/t0`, `UrbanScene3D:ArtSci/t0`, `UrbanScene3D:ArtSci/t1`.

| File | Role |
| --- | --- |
| `urbanscene3d_prep.py` | ContextCapture `Tile_*/*.obj` + `metadata.xml` → one deterministic 160 × 160 m OBJ per tile (Z up, metres, ground datum z = 0, unmapped cells written as no-fly boxes) |
| `urbanscene3d_scenes.py` | tile registry (SHA-256 checked), unchanged v0.3.0 ingestion, route rule |
| `reproduce_urbanscene3d.py` | the v0.3.0 protocol (same planners, settings, seeds, sweep, ablations, statistics) on the tiles; `--check` compares with `v040/reference.json` |
| `make_figures_urbanscene3d.py` | figures and CSV exports in `v040/`, including fig11 (procedural vs UrbanScene3D) |
| `test_urbanscene3d.py` | 14 tests: transform, crop, no-data policy, determinism, tile registry, route rule |
| `urbanscene3d/*.json` | per-tile preparation record: source file SHA-256s, EPSG:4547 origin, transform, crop box, datum, output SHA-256 |
| `urbanscene3d/routes.json` | the frozen routes and the written rule used to choose them |

**The dataset is not in this repository** (licence: non-commercial, no redistribution).
To reproduce, download the PolyTech and ArtSci "Oblique" reconstructions from
<https://vcc.tech/UrbanScene3D>, unzip each into `$UAV_DATASET_ROOT/<Scene>/`, then:

```bash
export UAV_DATASET_ROOT=/path/to/UrbanScene3D_data
python3 urbanscene3d_prep.py              # writes $UAV_DATASET_ROOT/derived/*.obj; SHA-256 must match urbanscene3d/*.json
python3 -m unittest test_urbanscene3d
python3 reproduce_urbanscene3d.py --workers 6 --check
```

Results (450 attempts, 384 accepted; all 66 rejections are RRT* finding no path within
500 iterations). Route-level mean reduction in modelled electrical energy, cluster
bootstrap 95 % CI, Holm-adjusted p:

| Comparison | Development (9 routes) | Held-out (6 routes) |
| --- | --- | --- |
| Energy GA vs A* | 2.207 % [1.396, 2.990], p = 0.034 | 3.213 % [1.669, 4.834], p = 0.272 |
| Energy GA vs Distance GA | 0.808 % [0.392, 1.264], p = 0.040 | 1.525 % [0.380, 2.795], p = 0.308 |
| Energy GA vs Time GA (negative control) | 0.030 % [−0.040, 0.130], p = 1.000 | 0.000 % [0.000, 0.000], p = 1.000 |

The negative-control finding survives on real geometry: the energy GA returns exactly the
time GA's path in 71 of 75 attempts. With 6 held-out clusters the sign-flip test cannot
reach p < 0.05 after Holm correction over nine comparisons. See `v040/comparison.json` for
the same statistics recomputed on the v0.3.0 procedural urban scenes alone.

## Energy semantics

Joule figures come from a parameterised propulsion model (m = 1.5 kg, 4 rotors, R = 0.12 m,
FM = 0.70, profile 22 W, CdA = 0.06 m², η = 0.85, ancillary 10 W; hover draw 178.36 W).
They are **model** energies, not measured battery consumption.

## Citation

Cite the specific archived **version DOI** for v0.4.0 alongside the concept DOI
`10.5281/zenodo.22679145`, which always resolves to the newest version. See `CITATION.cff`.

## License

MIT — see `LICENSE`.
