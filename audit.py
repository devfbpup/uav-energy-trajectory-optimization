"""Checksum manifest and artefact audit.

This is the published audit stage requested in review. It has two modes:

    python3 audit.py --write     # compute SHA-256 for every tracked artefact
                                 # and write manifest.sha256 + manifest.json
    python3 audit.py             # verify the working tree against the manifest

Tracked artefacts are the source files, the protocol description, the reference
aggregates, every per-attempt record in runs/, every figure in figures/, every
CSV export in exports/, and the ingested scene assets in assets/. The audit
reports the number of files verified, any checksum mismatch, any file that is
missing, and any file present in the tree but absent from the manifest.

The verification is a bitwise comparison of file content hashes. It is separate
from ``reproduce.py --check``, which compares numerical aggregates within a
stated tolerance. Neither is a claim that two independent runs produce
byte-identical floating-point output on different hardware.
"""

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.sha256"
MANIFEST_JSON = ROOT / "manifest.json"

TRACKED_FILES = [
    "uav_planning.py",
    "urban_scenes.py",
    "test_validation.py",
    "reproduce.py",
    "make_figures.py",
    "audit.py",
    "README.md",
    "CITATION.cff",
    "LICENSE",
    "protocol.json",
    "reference.json",
    "summary.json",
    "records.json",
    "budget_sweep.json",
    "ablations.json",
    # v0.4.0 UrbanScene3D stage (scripts and results only; no dataset meshes)
    "urbanscene3d_prep.py",
    "urbanscene3d_scenes.py",
    "reproduce_urbanscene3d.py",
    "make_figures_urbanscene3d.py",
    "test_urbanscene3d.py",
    ".gitignore",
]
TRACKED_DIRS = ["assets", "runs", "figures", "exports", "urbanscene3d", "v040"]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tracked_paths():
    paths = []
    for name in TRACKED_FILES:
        path = ROOT / name
        if path.exists():
            paths.append(path)
    for name in TRACKED_DIRS:
        directory = ROOT / name
        if directory.is_dir():
            paths += sorted(p for p in directory.rglob("*") if p.is_file())
    return sorted(paths, key=lambda p: p.relative_to(ROOT).as_posix())


def build():
    entries = {p.relative_to(ROOT).as_posix(): sha256_of(p) for p in tracked_paths()}
    MANIFEST.write_text("".join(f"{digest}  {name}\n" for name, digest in entries.items()))
    MANIFEST_JSON.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "file_count": len(entries),
                "algorithm": "sha256",
                "files": entries,
            },
            indent=2,
        )
    )
    print(f"wrote manifest.sha256 and manifest.json for {len(entries)} files")
    return entries


def read_manifest():
    entries = {}
    for line in MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        entries[name.strip()] = digest.strip()
    return entries


def verify():
    if not MANIFEST.exists():
        print("manifest.sha256 not found; run audit.py --write first")
        return 1
    expected = read_manifest()
    present = {p.relative_to(ROOT).as_posix(): p for p in tracked_paths()}
    missing = sorted(set(expected) - set(present))
    untracked = sorted(set(present) - set(expected))
    mismatched = []
    verified = 0
    for name, digest in expected.items():
        if name in present:
            if sha256_of(present[name]) == digest:
                verified += 1
            else:
                mismatched.append(name)
    print(f"audited files listed in manifest: {len(expected)}")
    print(f"verified identical: {verified}")
    print(f"checksum mismatches: {len(mismatched)}")
    print(f"missing from working tree: {len(missing)}")
    print(f"present but not in manifest: {len(untracked)}")
    for name in mismatched[:20]:
        print("  mismatch:", name)
    for name in missing[:20]:
        print("  missing:", name)
    for name in untracked[:20]:
        print("  untracked:", name)
    if mismatched or missing:
        print("AUDIT FAILED")
        return 1
    print("AUDIT PASSED: every manifest entry matched its recorded SHA-256 digest")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="generate the manifest instead of verifying it")
    args = parser.parse_args()
    if args.write:
        build()
        return 0
    return verify()


if __name__ == "__main__":
    sys.exit(main())
