"""Present a simulated synergy matrix to Phase 2R as if it were a screen.

The point of this module is that it is the *only* new code between the null
simulation and the existing residual-directionality pipeline. Phase 2R is not
reimplemented, reconfigured or re-tuned for simulated data: it is called through
``residual_experiment.run_residual_condition``, which already accepts a
:class:`koplev.Screen` and never reads the raw CSVs when one is supplied. So the
null goes through the same split logic, the same train-only additive fit, the
same residualisation, the same rungs, the same hyperparameter grid, the same
metrics and the same coverage grid as the real screens did.

Two things this module deliberately does *not* do:

* **It does not symmetrise, rescale or centre anything.** The matrix handed over
  is the estimator's output. Rescaling would change nothing about ``cal_skill``
  (a ratio) but it would break the comparison of absolute spreads, which is one
  of the honesty checks in the report.
* **It does not invent a screen label.** Phase 2R looks up exactly one
  per-screen constant, the ``threshold_2sd_D`` used for the *exploratory* sign
  accuracy, and that lookup is keyed by ``"A375"`` / ``"PANC1"``. A simulated
  screen therefore borrows one of the two labels, and which one it borrows
  affects only that one exploratory metric -- never ``cal_skill``, ``pearson`` or
  ``spearman``, which are the numbers the verdict rests on.
  ``test_screen_label_affects_only_sign_accuracy`` pins that.

Orientation
-----------
``koplev.load_screen`` builds ``frame`` with ``i`` = first drug, ``j`` = second,
``y`` = ``synergy_measure``. The simulated matrix is indexed ``[first, second]``
throughout -- the R parser's own ``(Sample, First, Second)`` convention -- so
``y[i, j]`` goes to the row with ``i`` first. Getting this backwards would
transpose every directional target and silently negate ``D``, which is why
``test_the_adapter_preserves_pair_orientation`` reconstructs the matrix from the
frame and requires equality, not correlation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import koplev


def as_screen(synergy: np.ndarray, label: str = "A375",
              lam_ab: np.ndarray | None = None,
              drug_names: tuple[str, ...] | None = None) -> koplev.Screen:
    """Wrap an ``[first, second]`` synergy matrix as a :class:`koplev.Screen`.

    The diagonal is dropped, exactly as ``load_screen`` drops the deposited
    ``a -> a`` rows: a drug scheduled against itself has no ordering, and Phase
    2R's pair algebra is defined on ``i != j``.
    """
    S = np.asarray(synergy, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"synergy must be a square matrix, got {S.shape}")
    if label not in koplev.SCREEN_TABLES:
        raise ValueError(
            f"label must be one of {sorted(koplev.SCREEN_TABLES)} so Phase 2R's "
            f"per-screen sign threshold resolves; got {label!r}")
    n = S.shape[0]
    if not np.isfinite(S).all():
        raise ValueError(f"{int((~np.isfinite(S)).sum())} non-finite synergy values")

    names = drug_names or tuple(f"Drug{k:03d}" for k in range(n))
    if len(names) != n:
        raise ValueError(f"{len(names)} names for {n} drugs")
    if list(names) != sorted(names):
        raise ValueError(
            "drug names must already be in the sorted order load_screen would "
            "impose, or the integer encoding will not match the matrix")

    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    keep = i != j
    i, j = i[keep], j[keep]
    lam = (np.zeros(len(i)) if lam_ab is None
           else np.asarray(lam_ab, dtype=float)[keep])
    frame = pd.DataFrame({
        "first": np.asarray(names)[i],
        "second": np.asarray(names)[j],
        "i": i, "j": j,
        "pair": list(zip(np.minimum(i, j), np.maximum(i, j))),
        "y": S[keep],
        "lam": lam,
    }).sort_values(["i", "j"]).reset_index(drop=True)

    return koplev.Screen(
        label=label,
        # Not a deposit table. Named so that anything trying to read the real
        # CSV for this screen fails loudly instead of quietly loading A375.
        table_key="simulated_dchain_null",
        drugs=tuple(names), frame=frame,
        n_raw_rows=int(n * n), n_self_rows=int(n), n_missing=0)


def matrix_from_screen(screen: koplev.Screen) -> np.ndarray:
    """Inverse of :func:`as_screen` on the off-diagonal. Used by the tests."""
    M = np.zeros((screen.n_drugs, screen.n_drugs))
    M[screen.frame["i"].to_numpy(), screen.frame["j"].to_numpy()] = \
        screen.frame["y"].to_numpy()
    return M
