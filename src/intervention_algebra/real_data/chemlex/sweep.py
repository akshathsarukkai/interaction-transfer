"""What Phase 4 runs, defined once so no document can drift from it.

:data:`PART_GRIDS` is the single definition of every condition. The runner builds
its jobs from it and ``--counts`` prints the totals, so the reproduction
instructions in the documents are generated rather than retyped -- the failure
Phase 2R shipped and had to come back and close.
"""

from __future__ import annotations

import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

from .dataset import DEFAULT_RAW_DIR, SCREENS
from .experiment import (AUTH_SEED, CONTROL_LADDER, K_FOLDS, LADDER,
                         N_PARTITIONS, Spec)
from .features import REPRESENTATIONS

ALL_FOLDS = tuple((p, f) for p in range(N_PARTITIONS) for f in range(K_FOLDS))
#: Controls and secondary arms run on partition 0 only. They are checks on the
#: primary result's validity, not second headlines, and running them at full
#: replication would triple the compute for numbers that decide something only
#: when they are far from their threshold.
P0_FOLDS = tuple((0, f) for f in range(K_FOLDS))

ENDPOINTS = ("yield", "feasible")
CONTROL_REPS = tuple(r for r in REPRESENTATIONS if r != "ecfp4")

#: Planted-interaction sizes for the positive control, as a multiple of the
#: additive part's scale. Three rather than one because the number that bounds a
#: negative result is not "the pipeline can find a huge planted effect" but "the
#: smallest planted effect the pipeline can find".
POSITIVE_SCALES = (0.25, 0.5, 1.0)


def primary_grid() -> list[Spec]:
    """The whole ladder, both screens, both endpoints, all 15 authoritative folds."""
    return [Spec(block="primary", screen=s, endpoint=e, partition=p, fold=f,
                 ladder=LADDER)
            for s in SCREENS for e in ENDPOINTS for (p, f) in ALL_FOLDS]


def control_grid() -> list[Spec]:
    """Shuffled acids, shuffled amines, both shuffled, random features.

    All four must collapse **measured as an increment over the additive
    baseline**. Phase 3's registered gate read skill-against-zero instead, and
    fired on a control containing no chemistry: +0.204 that way, -0.0007 as an
    increment. Which statistic the gate reads is the whole difference between a
    verdict and a wrong verdict.
    """
    return [Spec(block="control", screen=s, endpoint="yield", representation=r,
                 partition=p, fold=f, ladder=CONTROL_LADDER, tag=r)
            for r in CONTROL_REPS for s in SCREENS for (p, f) in P0_FOLDS]


def positive_grid() -> list[Spec]:
    """A planted rank-3 interaction on the exact ChemLex entity graph.

    Distinguishes "there is no transferable pair signal here" from "the
    entity-OOD machinery cannot detect one". Without it a null result is
    unfalsifiable. Run at three planted sizes and against the both-shuffled
    control, so the report can state the smallest effect the pipeline resolves.
    """
    return [Spec(block="positive", screen="all", endpoint="yield",
                 representation=r, partition=p, fold=f,
                 ladder=CONTROL_LADDER, synthetic_scale=sc,
                 tag=f"scale{sc:g}_{r}")
            for sc in POSITIVE_SCALES for r in ("ecfp4", "shuffled_both")
            for (p, f) in P0_FOLDS]


def transductive_grid() -> list[Spec]:
    """The ceiling. Pairs held out, entities not. Never an entity-OOD result."""
    return [Spec(block="transductive", screen=s, endpoint=e, partition=0,
                 fold=f, tag="ceiling")
            for s in SCREENS for e in ENDPOINTS for f in range(K_FOLDS)]


def sensitivity_grid() -> list[Spec]:
    """Four of the five registered sensitivities, each one flag away from the primary.

    The fifth — incremental pair skill among rows with ``Conversion > 0`` only —
    was registered and **never implemented**. It is recorded here rather than
    quietly dropped, and it is the one registered analysis this phase does not
    report. It was registered as outcome-conditioned and therefore as a labelled
    diagnostic that could never be a headline, so its absence changes no verdict;
    that is a reason it was not missed, not a reason it is fine.
    """
    out: list[Spec] = []
    common = dict(block="sensitivity", screen="all", endpoint="yield",
                  ladder=CONTROL_LADDER)
    for p, f in P0_FOLDS:
        out.append(Spec(**common, partition=p, fold=f, encoding="protocol",
                        tag="encoding_protocol"))
        out.append(Spec(**common, partition=p, fold=f, aggregate_cells=True,
                        tag="aggregate_cells"))
        out.append(Spec(**common, partition=p, fold=f,
                        classical_amines_only=True, tag="classical_amines"))
    for f in range(8):
        out.append(Spec(**common, partition=0, fold=f, k=8, tag="k8"))
    return out


def smoke_grid() -> list[Spec]:
    """A pipeline check, not a result. Two folds, one screen, no grid to speak of."""
    return [Spec(block="primary", screen="hatu", endpoint="yield", partition=0,
                 fold=f, ladder=CONTROL_LADDER, max_epochs=40, n_restarts=1,
                 tag="smoke")
            for f in (0, 1)]


PART_GRIDS: dict[str, tuple[tuple[str, str], ...]] = {
    "primary": (("primary", "primary_grid"),),
    "controls": (("controls", "control_grid"), ("positive", "positive_grid")),
    "transductive": (("transductive", "transductive_grid"),),
    "sensitivity": (("sensitivity", "sensitivity_grid"),),
    "smoke": (("smoke", "smoke_grid"),),
}
#: ``--part all`` runs everything except the smoke check, which is a pipeline
#: test and writes to its own gitignored file.
ALL_PARTS: tuple[str, ...] = tuple(p for p in PART_GRIDS if p != "smoke")


def part_jobs(part: str) -> list[tuple[str, list[Spec]]]:
    if part == "all":
        return [job for p in ALL_PARTS for job in part_jobs(p)]
    if part not in PART_GRIDS:
        raise SystemExit(f"unknown part {part!r}; choose from "
                         f"{sorted(PART_GRIDS) + ['all']}")
    return [(name, globals()[fn]()) for name, fn in PART_GRIDS[part]]


def part_counts() -> dict[str, int]:
    counts = {p: sum(len(s) for _, s in part_jobs(p)) for p in PART_GRIDS}
    counts["all"] = sum(counts[p] for p in ALL_PARTS)
    return counts


def _pin_threads() -> None:
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


_RAW = {}


def _cached_raw(raw_dir: Path):
    if "df" not in _RAW:
        from .dataset import load_raw
        _RAW["df"] = load_raw(raw_dir)
    return _RAW["df"]


def _run(spec: Spec, raw_dir: str) -> dict:
    try:
        _pin_threads()
        from .experiment import run
        return run(spec, raw=_cached_raw(Path(raw_dir)))
    except Exception:                                   # noqa: BLE001
        return {"key": spec.key, "block": spec.block, "screen": spec.screen,
                "endpoint": spec.endpoint,
                "representation": spec.representation,
                "partition": spec.partition, "fold": spec.fold,
                "tag": spec.tag, "error": traceback.format_exc()}


@contextmanager
def _exclusive(outdir: Path):
    """Refuse to start while another sweep is writing to the same directory.

    Two concurrent sweeps interleave lines into one ``.partial`` file and then
    race on ``os.replace``; the surviving file is a mixture of two runs with no
    way to tell which row came from which. Phase 3 hit exactly that, and the
    second run was worse than corrupt -- it had been launched before a structure
    was corrected, so half its rows used a mapping the audit had already
    rejected, and nothing in the output said so.

    A PID lock, not a flag: a stale lock from a killed run is cleared rather
    than blocking every future attempt.
    """
    lock = outdir / ".sweep.lock"
    outdir.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        try:
            pid = int(lock.read_text().strip())
            os.kill(pid, 0)
        except (ValueError, ProcessLookupError):
            lock.unlink(missing_ok=True)
        except PermissionError:
            pass
        if lock.exists():
            raise SystemExit(
                f"another sweep is already writing to {outdir} (pid "
                f"{lock.read_text().strip()}). Stop it first, or use --outdir.")
    lock.write_text(str(os.getpid()))
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def run_sweep(specs: list[Spec], out: Path, raw_dir: Path = DEFAULT_RAW_DIR,
              workers: int | None = None, verbose: bool = True,
              per_entity_out: Path | None = None) -> list[dict]:
    """Run every spec, writing one JSON row each and the per-entity table beside it.

    The per-entity records are split into their own CSV rather than left nested
    in the JSONL, because they are the inferential unit and every downstream
    statistic reads them; leaving them inside a result row would mean every
    analysis re-parsing 60 MB of JSON to get at them.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    rows: list[dict] = []
    entity_rows: list[dict] = []
    tmp = out.with_suffix(out.suffix + ".partial")
    with _exclusive(out.parent), open(tmp, "w") as fh, \
            ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run, s, str(raw_dir)): s for s in specs}
        for k, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            per = row.pop("per_entity", [])
            spec = futs[fut]
            for r in per:
                r.update({"key": row.get("key"), "block": row.get("block"),
                          "screen": row.get("screen"),
                          "endpoint": row.get("endpoint"),
                          "representation": row.get("representation"),
                          "partition": spec.partition, "fold": spec.fold,
                          "tag": spec.tag})
            entity_rows.extend(per)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            rows.append(row)
            if verbose and (k % 5 == 0 or k == len(specs)):
                nerr = sum(1 for r in rows if "error" in r)
                print(f"  {k}/{len(specs)} done ({nerr} failed)", flush=True)
    # Atomic: a reader must never see a half-written results file. Phase 2N hit
    # this twice with a plain open(..., "w"), which truncates before the writer
    # has anything to put back.
    os.replace(tmp, out)
    if per_entity_out is not None and entity_rows:
        import pandas as pd
        etmp = per_entity_out.with_suffix(per_entity_out.suffix + ".partial")
        # The `smiles` column is dropped on the way out. Across the committed
        # blocks it would carry 497 of the deposit's 503 distinct reactant
        # structures -- effectively its whole substrate inventory -- and the
        # deposit is CC BY-NC 4.0, so it is not ours to redistribute. Nothing
        # downstream reads it: `role` plus `entity` is the join key every table
        # uses, and a reader who has fetched the deposit can re-attach the
        # structures from it. See THIRD_PARTY_DATA.md.
        pd.DataFrame(entity_rows).drop(columns=["smiles"], errors="ignore"
                                       ).to_csv(etmp, index=False)
        os.replace(etmp, per_entity_out)
    return rows
