"""Parallel execution of many :func:`run_experiment` calls.

Runs are independent and CPU-bound, so a process pool is the whole story. Each
finished run is appended to a JSONL file as a flat record, which is the
machine-readable artifact the analysis reads. A run that raises is recorded with
an ``error`` field rather than killing the sweep -- a partial sweep with visible
failures is far more useful than no sweep.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, as_completed, wait
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _worker(spec: dict) -> dict:
    """Run one experiment. ``spec`` is a plain dict so it pickles cheaply."""
    import torch

    torch.set_num_threads(1)
    from .experiment import RunConfig, run_experiment, seeded
    from .generator import SystemConfig
    from .models import ModelConfig
    from .splits import SplitConfig
    from .train import TrainConfig

    meta = dict(spec.get("meta", {}))
    try:
        cfg = RunConfig(
            family=spec["family"],
            system=SystemConfig(**spec.get("system", {})),
            split=SplitConfig(**spec.get("split", {})),
            model=ModelConfig(**spec.get("model", {})),
            train=TrainConfig(**spec.get("train", {})),
            capacity_match=spec.get("capacity_match", True),
            tag=spec.get("tag", ""),
        )
        cfg = seeded(cfg, spec["seed"])
        # seeded() overwrites sub-seeds from the run seed; re-apply any explicit
        # split override that must not be seed-derived (none by default).
        result = run_experiment(cfg)
        result.pop("config", None)          # keep JSONL rows small
        result["config"] = cfg.to_dict()
        result.update(meta)
        return result
    except Exception as exc:  # noqa: BLE001 - deliberately broad, recorded not raised
        import traceback

        return {"error": repr(exc), "traceback": traceback.format_exc(),
                "family": spec.get("family"), "seed": spec.get("seed"), **meta}


def run_sweep(specs: list[dict], out_path: str | Path | None = None,
              workers: int | None = None, verbose: bool = True,
              chunk: int = 48, max_tasks_per_child: int = 8) -> list[dict]:
    """Execute ``specs`` in parallel, streaming results to ``out_path`` (JSONL).

    **Callers must guard their entry point with ``if __name__ == "__main__":``.**
    macOS starts workers with ``spawn``, so each child re-imports the driver
    module; without the guard the child re-runs the sweep and forks again, and the
    process tree multiplies until the machine dies. The failure surfaces as a
    ``RuntimeError`` about bootstrapping, which the serial fallback below then
    dutifully retries -- forever. ``scripts/run_phase1.py`` is guarded correctly.

    Two robustness measures, both learned the hard way on a memory-tight machine:
    workers are recycled every ``max_tasks_per_child`` runs, and the sweep is
    submitted in ``chunk``-sized batches with a fresh pool per batch. A torch
    worker that lives for hundreds of runs grows until the OS kills it, which
    surfaces as ``BrokenProcessPool`` with no diagnostics and no results. If a
    whole batch dies anyway it is retried serially so one bad run cannot cost the
    other 47.
    """
    workers = workers or max(1, (os.cpu_count() or 4) - 2)
    t0 = time.time()
    rows: list[dict] = []

    fh = None
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fh = out_path.open("w")

    def _emit(row: dict) -> None:
        rows.append(row)
        if fh is not None:
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
        if verbose and (len(rows) % 25 == 0 or len(rows) == len(specs)):
            n_err = sum("error" in r for r in rows)
            print(f"  [{len(rows)}/{len(specs)}] {time.time()-t0:6.1f}s "
                  f"errors={n_err}", file=sys.stderr, flush=True)

    try:
        for start in range(0, len(specs), chunk):
            batch = specs[start:start + chunk]
            pending = list(batch)
            try:
                with ProcessPoolExecutor(
                        max_workers=workers,
                        max_tasks_per_child=max_tasks_per_child) as ex:
                    # as_completed, not ex.map: map yields in *submission* order,
                    # so a single slow run head-of-line-blocks the progress file
                    # and a long sweep looks stalled when it is not.
                    futs = {ex.submit(_worker, spec): spec for spec in batch}
                    for fut in as_completed(futs):
                        _emit(fut.result())
                        pending.remove(futs[fut])
            except Exception as exc:  # noqa: BLE001 - pool died; fall back to serial
                if verbose:
                    print(f"!! pool failed ({exc!r}); retrying {len(pending)} "
                          f"runs serially", file=sys.stderr, flush=True)
                for spec in pending:
                    _emit(_worker(spec))
    finally:
        if fh is not None:
            fh.close()

    errs = [r for r in rows if "error" in r]
    if errs and verbose:
        print(f"!! {len(errs)} runs failed; first: {errs[0]['error']}",
              file=sys.stderr)
    return rows


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as fh:
        return [json.loads(line) for line in fh if line.strip()]
