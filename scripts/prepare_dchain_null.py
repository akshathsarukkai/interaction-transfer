"""Fetch, patch, compile and verify the published d-chain sampler.

    python scripts/prepare_dchain_null.py

Downloads the four files of ``skoplev/d-chain`` at the pinned commit, checks
every digest, applies the three patches listed in
``dchain_null.dchain.PATCHES``, compiles, and then **refuses the build unless
the patched program reproduces the unpatched one byte for byte** on the
deposited example data at the published RNG seed.

The source is not vendored: ``third_party/`` is gitignored, exactly as
``data/raw/`` is for the Mendeley deposit. It is GPL-3.0 and this project is
not, and a fetched-and-verified copy is a stronger provenance claim than a
copied one.

Takes about ten seconds. Requires a C++17 compiler and a network connection.
See ``docs/dchain_reconstruction.md`` for what the patches change and why none
of it is the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intervention_algebra.real_data import koplev
from intervention_algebra.real_data.dchain_null import dchain


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=dchain.DEFAULT_DIR)
    ap.add_argument("--raw", type=Path, default=koplev.DEFAULT_RAW_DIR)
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the byte-equivalence check (do not)")
    args = ap.parse_args()

    print(f"fetching {dchain.DCHAIN_REPO} @ {dchain.DCHAIN_COMMIT[:7]} "
          f"-> {args.dir}", flush=True)
    binary = dchain.build(args.dir, verify=not args.no_verify)
    print(f"built {binary}")
    if not args.no_verify:
        print("byte-equivalence against the unpatched sampler: OK")

    if args.raw.exists():
        ref = dchain.deposited_reference(args.raw)
        print("\ndeposit identities (the fidelity benchmark for the "
              "measurement layer):")
        for label, d in ref.items():
            print(f"  {label}: {json.dumps(d)}")
        bad = [k for k, d in ref.items()
               if not (d["lambda_is_multiple_of_1_over_n"]
                       and d["zero_lambda_implies_zero_synergy"]
                       and d["abs_synergy_le_abs_lambda"])]
        if bad:
            print(f"\nWARNING: the deposit no longer satisfies the identities "
                  f"the reconstruction rests on: {bad}")
            return 1
    else:
        print(f"\n{args.raw} not present; run scripts/download_koplev.py to "
              f"enable the deposit-identity check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
