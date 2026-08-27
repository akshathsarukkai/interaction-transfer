"""What Phase 3 runs, defined once so no document can drift from it.

``PART_GRIDS`` is the single definition of every condition. The runner builds its
jobs from it and ``--counts`` prints the totals, so the reproduction instructions
in the documents are generated rather than retyped -- the failure Phase 2R shipped
and had to come back and close.
"""

from __future__ import annotations

import json
import os
import traceback
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from .. import koplev
from .experiment import EntityConfig, run_entity_condition
from .models import LADDER_ORDER

SCREENS = ("A375", "PANC1")
N_PARTITIONS = 3
N_FOLDS = 10
ALL_FOLDS = tuple((p, f) for p in range(N_PARTITIONS) for f in range(N_FOLDS))
#: Controls and secondary arms run on partition 0 only. They are checks on the
#: primary result's validity, not second headlines, and running them at full
#: replication would triple the compute for numbers that are decisive only when
#: they are far from their threshold.
P0_FOLDS = tuple((0, f) for f in range(N_FOLDS))

#: The rungs the decision rule reads. ``antisym_mlp`` and ``pair_only`` are
#: reported but never decide anything.
CORE_RUNGS = ("zero", "potential", "lowrank")


def primary_grid() -> list[EntityConfig]:
    """Every rung, both screens, all 30 entity folds, molecular fingerprints."""
    return [EntityConfig(screen=s, model=m, partition=p, fold=f, tag="primary")
            for s in SCREENS for m in LADDER_ORDER for (p, f) in ALL_FOLDS]


def control_grid() -> list[EntityConfig]:
    """Controls A and B: random entity features, and shuffled fingerprints.

    Both must collapse. If features that cannot contain the answer predict it,
    the entity split is leaking and nothing else in the experiment means
    anything -- which is why these are a validity gate in the decision rule
    rather than a table at the back.
    """
    return [EntityConfig(screen=s, model=m, partition=p, fold=f,
                         representation=rep, tag=f"control_{rep}")
            for rep in ("random", "shuffled")
            for s in SCREENS for m in CORE_RUNGS for (p, f) in P0_FOLDS]


def positive_control_grid() -> list[EntityConfig]:
    """A target that IS a feature potential plus a known rank-2 form.

    Distinguishes "there is no transferable pair signal" from "the entity-OOD
    machinery cannot detect one". Without it a null result is unfalsifiable.
    """
    return [EntityConfig(screen=s, model=m, partition=p, fold=f,
                         synthetic_target=True, tag="positive_control")
            for s in SCREENS[:1] for m in CORE_RUNGS for (p, f) in P0_FOLDS]


def target_grid() -> list[EntityConfig]:
    """The secondary biological representation, and Control D, its shuffle."""
    return [EntityConfig(screen=s, model=m, partition=p, fold=f,
                         representation=rep, tag=f"rep_{rep}")
            for rep in ("targets", "targets_shuffled")
            for s in SCREENS for m in CORE_RUNGS for (p, f) in P0_FOLDS]


def coverage_grid() -> list[EntityConfig]:
    """Sparser pair coverage **among training entities only**.

    The test entities are excluded at every coverage; this varies how much of the
    training graph the model sees, not who is held out.
    """
    return [EntityConfig(screen=s, model=m, partition=p, fold=f, coverage=c,
                         tag=f"coverage_{c:.2f}")
            for c in (0.20, 0.40, 0.70)
            for s in SCREENS for m in ("potential", "lowrank") for (p, f) in P0_FOLDS]


def smoke_grid() -> list[EntityConfig]:
    """A pipeline check, not a result. Two folds, three rungs, one screen."""
    return [EntityConfig(screen="A375", model=m, partition=0, fold=f, tag="smoke")
            for m in CORE_RUNGS for f in (0, 1)]


PART_GRIDS: dict[str, tuple[tuple[str, str], ...]] = {
    "primary": (("primary", "primary_grid"),),
    "controls": (("controls", "control_grid"), ("positive", "positive_control_grid")),
    "targets": (("targets", "target_grid"),),
    "coverage": (("coverage", "coverage_grid"),),
    "smoke": (("smoke", "smoke_grid"),),
}
#: ``--part all`` runs everything except the smoke check, which is a pipeline
#: test and writes to its own file.
ALL_PARTS: tuple[str, ...] = tuple(p for p in PART_GRIDS if p != "smoke")


def part_jobs(part: str) -> list[tuple[str, list[EntityConfig]]]:
    if part == "all":
        return [job for p in ALL_PARTS for job in part_jobs(p)]
    if part not in PART_GRIDS:
        raise SystemExit(f"unknown part {part!r}; choose from {sorted(PART_GRIDS) + ['all']}")
    return [(name, globals()[fn]()) for name, fn in PART_GRIDS[part]]


def part_counts() -> dict[str, int]:
    counts = {p: sum(len(specs) for _, specs in part_jobs(p)) for p in PART_GRIDS}
    counts["all"] = sum(counts[p] for p in ALL_PARTS)
    return counts


def _pin_threads() -> None:
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


_SCREENS: dict[str, koplev.Screen] = {}
_MAPPING = {}


def _cached(screen: str, raw_dir: Path, mapping_path: Path):
    import pandas as pd

    if screen not in _SCREENS:
        _SCREENS[screen] = koplev.load_screen(screen, raw_dir)
    if "m" not in _MAPPING:
        _MAPPING["m"] = pd.read_csv(mapping_path)
    return _SCREENS[screen], _MAPPING["m"]


def _run(cfg: EntityConfig, raw_dir: str, mapping_path: str) -> dict:
    try:
        _pin_threads()
        screen, mapping = _cached(cfg.screen, Path(raw_dir), Path(mapping_path))
        return run_entity_condition(cfg, Path(raw_dir), Path(mapping_path),
                                    screen=screen, mapping=mapping)
    except Exception:                                   # noqa: BLE001
        return {"tag": cfg.tag, "screen": cfg.screen, "model": cfg.model,
                "representation": cfg.representation, "partition": cfg.partition,
                "fold": cfg.fold, "coverage": cfg.coverage,
                "error": traceback.format_exc()}


@contextmanager
def _exclusive(outdir: Path):
    """Refuse to start while another sweep is writing to the same directory.

    Two concurrent sweeps interleave lines into the same ``.partial`` file and
    then race on ``os.replace``, and the surviving file is a mixture of two runs
    with no way to tell which row came from which. That happened once here, and
    the second run was worse than corrupt: it had been launched before a drug
    structure was corrected, so half the rows would have been computed against a
    mapping the audit had already rejected -- and nothing in the output would
    have said so.

    A PID lock, not a flag: a stale lock from a killed run is detected and
    cleared rather than blocking the next attempt forever.
    """
    lock = outdir / ".sweep.lock"
    outdir.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        try:
            pid = int(lock.read_text().strip())
            os.kill(pid, 0)
        except (ValueError, ProcessLookupError):
            lock.unlink(missing_ok=True)          # stale
        except PermissionError:
            pass                                   # exists, owned by someone else
        if lock.exists():
            raise SystemExit(
                f"another sweep is already writing to {outdir} (pid "
                f"{lock.read_text().strip()}). Stop it first, or use --outdir.")
    lock.write_text(str(os.getpid()))
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def run_entity_sweep(specs: list[EntityConfig], out: Path,
                     raw_dir: Path = koplev.DEFAULT_RAW_DIR,
                     mapping_path: Path | None = None,
                     workers: int | None = None, verbose: bool = True) -> list[dict]:
    from .experiment import DEFAULT_MAPPING

    mapping_path = mapping_path or DEFAULT_MAPPING
    out.parent.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    rows: list[dict] = []
    tmp = out.with_suffix(out.suffix + ".partial")
    with _exclusive(out.parent), open(tmp, "w") as fh, \
            ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run, c, str(raw_dir), str(mapping_path)): c for c in specs}
        for k, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            rows.append(row)
            if verbose and (k % 10 == 0 or k == len(specs)):
                nerr = sum(1 for r in rows if "error" in r)
                print(f"  {k}/{len(specs)} done ({nerr} failed)", flush=True)
    # Atomic: a reader must never see a half-written results file. Phase 2N hit
    # this twice with a plain open(..., "w"), which truncates before the writer
    # has anything to put back.
    os.replace(tmp, out)
    return rows
