"""Run the d-chain null ensemble: simulate, estimate, and run Phase 2R on it.

    python scripts/prepare_dchain_null.py            # once
    python scripts/run_dchain_null.py --part oracle       # Control A, seconds
    python scripts/run_dchain_null.py --part unshared     # Control C, minutes
    python scripts/run_dchain_null.py --part primary      # the experiment, hours
    python scripts/report_dchain_null.py

Parts, and what each is for, are documented on the grid functions in
``dchain_null.grids``; run counts come from ``grids.part_counts()`` and are
printed by ``--counts`` rather than written down anywhere.

The ``joint`` parts run the published sampler at its own compiled-in defaults
(500,000 iterations), which is about 100 minutes per simulated screen at 100
drugs on one core. ``--part all`` is therefore of order ten hours on eight
cores. The free controls come first so a malformed null is caught before any of
that. ``--limit`` truncates a part for a pipeline check.

Rows are appended to ``results/dchain_null/metrics.jsonl`` as they finish, so an
interrupted run keeps what it had. Re-running a part rewrites only that part's
rows (matched on tag, seed and estimator).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from intervention_algebra.real_data import koplev
from intervention_algebra.real_data.dchain_null import dchain, grids
from intervention_algebra.real_data.dchain_null.experiment import (
    NullRunConfig, run_null_condition)

OUT_DIR = Path("results/dchain_null")
METRICS = OUT_DIR / "metrics.jsonl"
SIMS = OUT_DIR / "simulations"
CONFIGS = OUT_DIR / "configs"


def _key(row: dict) -> tuple:
    return (row.get("tag"), row.get("estimator"), row.get("sim_seed"),
            row.get("est_seed"))


def _run_one(cfg: NullRunConfig, binary: str | None,
             raw_dir: str | None = None) -> dict:
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:                                    # pragma: no cover
        pass
    work = SIMS / f"{cfg.null.tag}_s{cfg.null.sim_seed}_e{cfg.est_seed}"
    try:
        return run_null_condition(cfg, work,
                                  Path(binary) if binary else None,
                                  Path(raw_dir) if raw_dir else None)
    except Exception:                                    # noqa: BLE001
        return {"tag": cfg.null.tag, "estimator": cfg.estimator,
                "sim_seed": cfg.null.sim_seed, "est_seed": cfg.est_seed,
                "variant": cfg.null.variant, "sigma_obs": cfg.null.sigma_obs,
                "error": traceback.format_exc()}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # Not `choices=`: a comma-separated list of parts is accepted, so the
    # remaining blocks can be run as one pool instead of one after another.
    # grids.part_jobs raises on anything it does not recognise.
    ap.add_argument("--part", default="all",
                    help="one of " + ", ".join(tuple(grids.ALL_PARTS)
                                               + ("all", "smoke"))
                         + ", or a comma-separated list")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N conditions of the part")
    ap.add_argument("--counts", action="store_true")
    ap.add_argument("--dchain-dir", type=Path, default=dchain.DEFAULT_DIR)
    ap.add_argument("--raw-dir", type=Path, default=koplev.DEFAULT_RAW_DIR,
                    help="the Koplev deposit; point at tests/fixtures/koplev_tiny "
                         "to run the pipeline check without it")
    args = ap.parse_args()

    if args.counts:
        for part, n in grids.part_counts().items():
            print(f"{part:12s} {n:4d}")
        return 0

    try:
        specs = grids.part_jobs(args.part)
    except ValueError as e:
        print(e)
        return 2
    # `is not None`, not truthiness: --limit 0 is a legitimate way to ask for a
    # dry run, and under a truthiness test it silently meant "no limit" and
    # started the whole block.
    if args.limit is not None:
        specs = specs[:args.limit]

    needs_binary = any(s.estimator == "joint" for s in specs)
    binary = args.dchain_dir / "build" / "dchain"
    if needs_binary and not binary.exists():
        print(f"{binary} is missing. Run scripts/prepare_dchain_null.py first.")
        return 1

    # A pipeline check is not a result and never shares a file with one.
    metrics = (OUT_DIR / "smoke.jsonl") if args.part == "smoke" else METRICS
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS.mkdir(parents=True, exist_ok=True)
    SIMS.mkdir(parents=True, exist_ok=True)
    (CONFIGS / f"{args.part.replace(',', '+')}.json").write_text(json.dumps(
        [json.loads(json.dumps(asdict(s), default=str)) for s in specs],
        indent=2) + "\n")

    # Keep every row from other parts; replace this part's.
    kept: list[dict] = []
    if metrics.exists():
        drop = {_key({"tag": s.null.tag, "estimator": s.estimator,
                      "sim_seed": s.null.sim_seed, "est_seed": s.est_seed})
                for s in specs}
        kept = [r for r in (json.loads(l) for l in open(metrics) if l.strip())
                if _key(r) not in drop]

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    print(f"part={args.part} conditions={len(specs)} workers={workers} "
          f"-> {metrics}", flush=True)
    t0 = time.time()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_one, c, str(binary) if needs_binary else None,
                          str(args.raw_dir)): c
                for c in specs}
        for k, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            rows.append(row)
            # Written atomically. The naive open(..., "w") truncates first, so
            # anything reading the file during a long run -- the report script,
            # a monitor -- sees an empty or half-written file and silently
            # reports a smaller ensemble than exists.
            tmp = metrics.with_suffix(".jsonl.tmp")
            with open(tmp, "w") as fh:
                for r in kept + rows:
                    fh.write(json.dumps(r) + "\n")
            os.replace(tmp, metrics)
            nerr = sum(1 for r in rows if "error" in r)
            print(f"  {k}/{len(specs)} done ({nerr} failed) "
                  f"{time.time() - t0:.0f}s", flush=True)

    failed = [r for r in rows if "error" in r]
    print(f"part={args.part}: {len(rows)} rows, {len(failed)} failed, "
          f"{time.time() - t0:.0f}s")
    if failed:
        print(failed[0]["error"][-1500:])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
