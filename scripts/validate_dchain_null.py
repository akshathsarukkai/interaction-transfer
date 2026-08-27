"""Fidelity checks on the reconstruction, run against real upstream artifacts.

    python scripts/validate_dchain_null.py

Writes ``results/dchain_null/summary/validation.json``. Three checks, in
increasing order of what they establish:

1. **The patched sampler is the published sampler.** Already enforced at build
   time by ``scripts/prepare_dchain_null.py``, which refuses a build whose
   output is not byte-identical to the unpatched program's; re-asserted here.
2. **The deposit satisfies the four identities the reconstruction of
   ``synergy_measure`` implies** — the 1/1999 quantisation that pins the MCMC
   settings, ``lambda == 0 => synergy == 0`` exactly, ``|synergy| <= |lambda|``,
   and the paper's own significance counts.
3. **The whole pipeline runs on the authors' own input data** — the 66-row
   example shipped with ``d-chain`` — and its output satisfies the same four
   identities.

What none of them establish, and cannot: numerical agreement with a published
``synergy_measure``. The Mendeley deposit contains the modelled tables and
nothing else; no posterior samples and no raw viability data were ever
deposited, so the inputs to the synergy formula do not exist publicly for any
published value. See ``docs/dchain_reconstruction.md`` §6.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from intervention_algebra.real_data import koplev
from intervention_algebra.real_data.dchain_null import dchain

OUT = Path("results/dchain_null/summary/validation.json")


def paper_counts(raw_dir: Path) -> dict:
    """Reproduce the paper's significance counts from p = 1 - |lambda| < 0.05."""
    want = {"A375": {"synergistic": 707, "antagonistic": 1845},
            "PANC1": {"synergistic": 551, "antagonistic": 1464}}
    out = {}
    for label, table in (("A375", "Data Table 1.csv"), ("PANC1", "Data Table 2.csv")):
        lam = pd.read_csv(raw_dir / table)["lambda"].to_numpy()
        sig = (1 - np.abs(lam)) < 0.05
        got = {"synergistic": int((sig & (lam > 0)).sum()),
               "antagonistic": int((sig & (lam < 0)).sum())}
        out[label] = {"paper": want[label], "reproduced": got,
                      "exact": got == want[label]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=koplev.DEFAULT_RAW_DIR)
    ap.add_argument("--dir", type=Path, default=dchain.DEFAULT_DIR)
    ap.add_argument("--iterations", type=int, default=200_000)
    args = ap.parse_args()

    binary = args.dir / "build" / "dchain"
    if not binary.exists():
        print(f"{binary} is missing. Run scripts/prepare_dchain_null.py first.")
        return 1

    report: dict = {
        "source": {"repo": dchain.DCHAIN_REPO, "commit": dchain.DCHAIN_COMMIT,
                   "license": dchain.DCHAIN_LICENSE,
                   "files": {k: v[0] for k, v in dchain.DCHAIN_FILES.items()}},
        "patch": {"hunks": [p[0] for p in dchain.PATCHES],
                  "sufficient_statistic_call_sites": dchain._SUFSTAT_CALL_N,
                  "byte_equivalence_enforced_at_build": True},
    }

    print("1. patched source still matches upstream exactly ...", flush=True)
    src = (args.dir / "dchain.cpp").read_text()
    assert dchain.sha256_of(src.encode()) == dchain.DCHAIN_FILES["dchain.cpp"][0]
    dchain.patch_source(src)                     # raises if any hunk drifted
    print("   ok")

    if args.raw.exists():
        print("2. deposit identities and the paper's counts ...", flush=True)
        report["deposit_identities"] = dchain.deposited_reference(args.raw)
        report["paper_counts"] = paper_counts(args.raw)
        ok = all(v["exact"] for v in report["paper_counts"].values())
        print(f"   paper counts reproduced exactly: {ok}")
    else:
        print(f"2. skipped: {args.raw} not present")

    print(f"3. end-to-end on the authors' own example data "
          f"({args.iterations:,} iterations) ...", flush=True)
    report["example_run"] = dchain.validate_on_deposited_example(
        binary, args.dir, iterations=args.iterations)
    print(f"   identities hold: {report['example_run']['identities_hold']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
