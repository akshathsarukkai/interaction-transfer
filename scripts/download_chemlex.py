#!/usr/bin/env python
"""Fetch the ChemLex acid-amine deposit into ``data/raw/chemlex2025/``.

    python scripts/download_chemlex.py                # the version we model
    python scripts/download_chemlex.py --all-versions # + the two older ones
    python scripts/download_chemlex.py --compare      # and diff them column by column

Idempotent and digest-verified: rerunning it re-downloads nothing, and a file
whose sha256 does not match the recorded one is refetched rather than trusted.
The raw directory is gitignored -- the deposit is CC BY-NC 4.0, so it is *not*
ours to redistribute, and a reproducible fetch plus recorded digests is a better
record than a vendored copy that can silently drift from its source.

``--compare`` re-derives the claim in ``acquire``'s docstring that the three
published versions differ only in the ``Reagents`` column. It is the check that
justifies modelling the newest file rather than the one that was current when the
paper appeared.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from intervention_algebra.real_data.chemlex.acquire import (
    ACQUIRED, CHEMLEX_CODE, CHEMLEX_DOI, CHEMLEX_LICENSE, CHEMLEX_PAPER,
    CURRENT, DEFAULT_RAW_DIR, VERSIONS, download_raw, verify_raw,
)


def compare_versions(dest: Path) -> dict[str, object]:
    """Diff every column of every version against the one we model."""
    import pandas as pd

    frames = {}
    for key, ver in VERSIONS.items():
        name = ver.filename if key == "2025-11" else f"{key}_{ver.filename}"
        path = dest / name
        if not path.exists():
            continue
        frames[key] = pd.read_excel(path, sheet_name=0)

    ref_key = "2025-11"
    if ref_key not in frames:
        raise FileNotFoundError("the current version is not on disk")
    ref = frames[ref_key]
    out: dict[str, object] = {"reference": ref_key, "shape": list(ref.shape),
                              "columns": list(ref.columns), "against": {}}
    for key, frame in sorted(frames.items()):
        if key == ref_key:
            continue
        if list(frame.columns) != list(ref.columns) or len(frame) != len(ref):
            out["against"][key] = {"comparable": False,
                                   "shape": list(frame.shape)}
            continue
        diffs = {}
        for col in ref.columns:
            a, b = frame[col], ref[col]
            if a.dtype.kind == "f":
                n = int((~((a == b) | (a.isna() & b.isna()))).sum())
            else:
                n = int((a.fillna("\x00") != b.fillna("\x00")).sum())
            diffs[col] = n
        out["against"][key] = {"comparable": True, "cells_differing": diffs,
                               "columns_that_changed":
                                   sorted(c for c, n in diffs.items() if n)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=DEFAULT_RAW_DIR)
    ap.add_argument("--force", action="store_true",
                    help="refetch even if the local digest already matches")
    ap.add_argument("--all-versions", action="store_true",
                    help="also fetch the 2024-07 and 2025-05 versions")
    ap.add_argument("--compare", action="store_true",
                    help="diff the fetched versions column by column "
                         "(implies --all-versions)")
    args = ap.parse_args()

    keys = ("2025-11",)
    if args.all_versions or args.compare:
        keys = tuple(sorted(VERSIONS))

    print(f"paper    doi:{CHEMLEX_PAPER}")
    print(f"data     doi:{CHEMLEX_DOI}  ({CHEMLEX_LICENSE}, digests recorded "
          f"{ACQUIRED})")
    print(f"code     {CHEMLEX_CODE}")
    print(f"modelled record {CURRENT.record}, published {CURRENT.published}")
    paths = download_raw(args.dest, force=args.force, versions=keys)
    for key in sorted(paths):
        p = paths[key]
        print(f"  ok  {key}  {p}  ({p.stat().st_size} bytes)")

    print("\nre-deriving the digest of the file we model (do not skip -- "
          "PROVENANCE.json is written by this script and proves nothing on "
          "its own):")
    got = verify_raw(args.dest)
    print(json.dumps(got, indent=2))
    if not got.get("matches_record"):
        print("\nERROR: the file on disk is not the recorded record.",
              file=sys.stderr)
        return 1

    if args.compare:
        print("\ncomparing the published versions column by column:")
        cmp = compare_versions(args.dest)
        print(json.dumps(cmp, indent=2))
        changed = {k: v.get("columns_that_changed")
                   for k, v in cmp["against"].items()}
        if any(set(c or []) - {"Reagents"} for c in changed.values()):
            print("\nWARNING: a version differs from the modelled one outside "
                  "the Reagents column. docs/phase4_chemlex_dataset.md claims "
                  "the modelling columns are stable across versions; that claim "
                  "is now false and must be corrected before modelling.",
                  file=sys.stderr)
            return 1
        print("\nonly the Reagents column ever changed, as documented.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
