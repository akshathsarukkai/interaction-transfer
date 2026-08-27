"""Fetch the Koplev et al. (2017) deposit into ``data/raw/koplev2017/``.

    python scripts/download_koplev.py

Idempotent and digest-verified: rerunning it re-downloads nothing, and a file
whose sha256 does not match the recorded one is refetched rather than trusted.
The raw directory is gitignored -- the deposit is CC BY 4.0 and could be
redistributed, but a reproducible fetch plus recorded digests is a better record
than a vendored copy that can silently drift from its source.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from intervention_algebra.real_data.koplev import (
    ACQUIRED, DEFAULT_RAW_DIR, KOPLEV_DOI, KOPLEV_LICENSE, KOPLEV_PAPER,
    download_raw, reproduce_publication_stats, verify_raw,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=DEFAULT_RAW_DIR)
    ap.add_argument("--force", action="store_true",
                    help="refetch even if the local digest already matches")
    args = ap.parse_args()

    print(f"paper   doi:{KOPLEV_PAPER}")
    print(f"data    doi:{KOPLEV_DOI}  ({KOPLEV_LICENSE}, digests recorded {ACQUIRED})")
    paths = download_raw(args.dest, force=args.force)
    for key, path in sorted(paths.items()):
        print(f"  ok  {path}  ({path.stat().st_size} bytes)")

    print("\nre-deriving which tables are usable (do not skip -- the deposit's "
          "own descriptions are wrong about Table 4):")
    print(json.dumps(verify_raw(args.dest), indent=2))

    print("\nreproducing the publication's own counts:")
    stats = reproduce_publication_stats(args.dest)
    print(json.dumps(stats, indent=2))
    if not stats["all_counts_match"]:
        print("\nWARNING: the paper's significance counts did not reproduce. "
              "The table-to-cell-line assignment rests on them; investigate "
              "before modelling.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
