"""Leakage-safe removal of the per-drug ordering potential, and what is left.

The question this module exists to serve
----------------------------------------
Phase 2 established that the Koplev screen's *apparent* order dependence is
largely a per-drug tendency. A two-way additive fit

    y(i -> j) = mu + a_i + b_j

has directional component

    D_add(i, j) = (a_i + b_j) - (a_j + b_i) = g_i - g_j,   g_i := a_i - b_i

which is a **potential**: drug ``i`` is simply better-going-first than drug ``j``
is, by an amount that does not depend on which partner it is paired with. That
is not a pair-specific order interaction, and no model that predicts it has
demonstrated one. The quantity that would demonstrate one is the residual

    D_res(i, j) = [y(i->j) - y(j->i)] - [g_i - g_j]

and the question is whether ``D_res`` for an entirely unseen unordered pair can
be predicted from the other pairs.

Why the fit is closed-form and not the Phase 2 ``Additive`` family
-----------------------------------------------------------------
Phase 2's ``additive`` family is the same statistical model fitted by Adam under
the shared training budget. Here the fit is a ridge solve, for three reasons:
it is deterministic (the residual target must not depend on an optimiser seed,
or every downstream metric inherits that noise); it is exact, so "the additive
part has been removed" is a statement about the model and not about how long it
trained; and it is fast enough to redo inside every one of the ~1,000 runs
rather than being cached and silently reused across splits.
``test_closed_form_additive_matches_trained_additive`` pins the two estimators
to each other on real splits so the change of estimator cannot smuggle in a
different baseline.

Identifiability
---------------
``mu + a_i + b_j`` is not identified: ``(a_i + c, b_j - c)`` gives the same fit
for any ``c``, and shifting ``mu`` against both does too. Two things make this a
non-issue rather than a choice to defend.

* The ridge penalty (applied to ``a`` and ``b``, never to ``mu``) makes the
  solution unique -- it selects the minimum-norm representative of the family.
* **The quantity actually used is gauge-invariant anyway.** Under
  ``a_i -> a_i + c``, ``b_i -> b_i - c`` we get ``g_i -> g_i + 2c``, so
  ``g_i - g_j`` is unchanged, and ``D_add`` -- hence ``D_res`` -- does not depend
  on the convention at all. ``test_D_add_is_gauge_invariant`` asserts it.

So the penalty affects ``D_res`` only through *shrinkage*, which is a real
modelling decision and is therefore selected on validation rows, never on test.

The one failure mode that matters, and how it is detected
---------------------------------------------------------
Over-shrinkage leaves part of the potential *in* the residual. A pair-specific
model would then score well on ``D_res`` by re-learning ``g``, and the result
would be reported as pair-specific structure when it is nothing of the kind.
This is why the model ladder contains a ``potential`` rung (``c_i - c_j``, one
free scalar per drug, no pair term): it can fit leftover potential and nothing
else. If ``potential`` has skill, the additive removal was incomplete and every
richer model's skill must be read against *it*, not against zero.

Leakage
-------
The additive fit is the one place a held-out pair can leak in without touching
the split logic, because it is fitted *before* the residual target exists. Both
orientations of a test pair must be absent from the design matrix. That is
enforced by :func:`assert_rows_are_train_only`, called inside
:func:`fit_additive`, so no caller can skip it by constructing the frame itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .splits import CoverageSplit

#: Ridge penalties searched for the additive fit. Selected on validation rows.
#: Spans from "essentially OLS" -- which at coverage 0.05 fits 201 parameters to
#: ~420 rows and is badly ill-conditioned -- to heavy shrinkage.
RIDGE_LAMBDAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)


@dataclass(frozen=True)
class AdditiveFit:
    """A two-way additive fit and the provenance needed to trust it."""

    mu: float
    a: np.ndarray = field(repr=False)
    b: np.ndarray = field(repr=False)
    lam: float
    objective: str
    #: Validation score the penalty was chosen by. Never a test quantity.
    val_score: float
    n_fit_rows: int
    n_fit_pairs: int

    @property
    def g(self) -> np.ndarray:
        """The per-drug ordering potential ``g_i = a_i - b_i``.

        ``D_add(i, j) = g_i - g_j``. Invariant to the ``(a + c, b - c)`` gauge
        up to a common shift, which cancels in the difference.
        """
        return self.a - self.b

    def predict(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        return self.mu + self.a[np.asarray(i)] + self.b[np.asarray(j)]

    def d_add(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        g = self.g
        return g[np.asarray(i)] - g[np.asarray(j)]


def assert_rows_are_train_only(rows: pd.DataFrame, split: CoverageSplit) -> None:
    """Refuse to fit the additive baseline on anything but training pairs.

    The check is on the *ordered rows handed to the solver*, not on a pair list
    the caller promises corresponds to them. A frame is the only thing the fit
    actually sees, so it is the only thing worth checking.
    """
    forbidden = set(split.val_pairs) | set(split.test_pairs)
    present = set(rows["pair"])
    bad = present & forbidden
    if bad:
        n_test = len(present & set(split.test_pairs))
        raise AssertionError(
            f"additive fit was handed {len(bad)} non-training pairs "
            f"({n_test} of them TEST pairs), e.g. {sorted(bad)[:3]}. Fitting "
            f"a_i or b_j on either orientation of a held-out pair leaks that "
            f"pair into its own residual target.")
    allowed = set(split.train_pairs)
    if not present <= allowed:                          # pragma: no cover
        raise AssertionError("frame contains pairs outside the split entirely")


def _solve(i: np.ndarray, j: np.ndarray, y: np.ndarray, n_drugs: int,
           lam: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Ridge solve for ``mu + a_i + b_j``. The intercept is never penalised.

    Built from normal equations directly rather than a dense design matrix: the
    matrix is ``n_rows x 201`` and would be 99% zeros, and this is called inside
    every run for every penalty in the grid.
    """
    p = 1 + 2 * n_drugs
    xtx = np.zeros((p, p))
    xty = np.zeros(p)
    n = len(y)

    ca, cb = np.bincount(i, minlength=n_drugs), np.bincount(j, minlength=n_drugs)
    cross = np.zeros((n_drugs, n_drugs))
    np.add.at(cross, (i, j), 1.0)

    xtx[0, 0] = n
    xtx[0, 1:1 + n_drugs] = ca
    xtx[1:1 + n_drugs, 0] = ca
    xtx[0, 1 + n_drugs:] = cb
    xtx[1 + n_drugs:, 0] = cb
    xtx[1:1 + n_drugs, 1:1 + n_drugs] = np.diag(ca)
    xtx[1 + n_drugs:, 1 + n_drugs:] = np.diag(cb)
    xtx[1:1 + n_drugs, 1 + n_drugs:] = cross
    xtx[1 + n_drugs:, 1:1 + n_drugs] = cross.T

    xty[0] = y.sum()
    np.add.at(xty, 1 + i, y)
    np.add.at(xty, 1 + n_drugs + j, y)

    pen = np.full(p, lam)
    pen[0] = 0.0
    beta = np.linalg.solve(xtx + np.diag(pen), xty)
    return float(beta[0]), beta[1:1 + n_drugs], beta[1 + n_drugs:]


def fit_additive(split: CoverageSplit, frame: pd.DataFrame, n_drugs: int,
                 lambdas: tuple[float, ...] = RIDGE_LAMBDAS,
                 objective: str = "y",
                 _contaminate: bool = False) -> AdditiveFit:
    """Fit ``mu + a_i + b_j`` on TRAINING pairs, penalty chosen on validation.

    ``objective`` decides what the validation score is:

    ``"y"``
        held-out ordered MSE. The pre-specified primary: it defines the additive
        baseline as *the best additive predictor of the response*, with no
        reference to the directional quantity under study.
    ``"D"``
        held-out ``mean(D_res**2)`` on validation pairs. Chooses the penalty that
        removes as much directional structure as possible, so the leftover
        residual is a conservative floor rather than a point estimate. Reported
        as a sensitivity, never as the headline.

    Both are computed on validation pairs, which are carved out of the training
    pool and are disjoint from the evaluation pool. Test rows are not touched.
    """
    if objective not in ("y", "D"):
        raise ValueError(f"objective must be 'y' or 'D', got {objective!r}")
    if _contaminate:
        # Control C only, and reachable only through
        # ``ResidualConfig.contaminate_additive_fit``. The guard is bypassed
        # here rather than worked around by handing in a doctored split,
        # because a doctored split would also make the guard *look* satisfied --
        # and a guard that can be satisfied by lying to it is not a guard. Rows
        # produced this way are stamped ``contaminated: true`` and are written
        # to their own file.
        train_rows = frame
    else:
        train_rows = split.rows(frame, "train")
        assert_rows_are_train_only(train_rows, split)
    val_rows = split.rows(frame, "val")
    if val_rows.empty:                                   # pragma: no cover
        raise ValueError("no validation rows; the penalty would be unselectable")

    ti = train_rows["i"].to_numpy()
    tj = train_rows["j"].to_numpy()
    ty = train_rows["y"].to_numpy()
    vi = val_rows["i"].to_numpy()
    vj = val_rows["j"].to_numpy()
    vy = val_rows["y"].to_numpy()
    if objective == "D":
        v_dir = directional_pairs(val_rows)

    best: tuple[float, tuple] | None = None
    for lam in lambdas:
        mu, a, b = _solve(ti, tj, ty, n_drugs, lam)
        if objective == "y":
            score = float((((mu + a[vi] + b[vj]) - vy) ** 2).mean())
        else:
            g = a - b
            res = v_dir["D_true"].to_numpy() - (g[v_dir["i"].to_numpy()]
                                                - g[v_dir["j"].to_numpy()])
            score = float((res ** 2).mean())
        if best is None or score < best[0]:
            best = (score, (lam, mu, a, b))
    score, (lam, mu, a, b) = best                        # type: ignore[misc]
    n_fit_pairs = (len(set(train_rows["pair"])) if _contaminate
                   else len(split.train_pairs))
    return AdditiveFit(mu=mu, a=a, b=b, lam=float(lam), objective=objective,
                       val_score=score, n_fit_rows=len(train_rows),
                       n_fit_pairs=n_fit_pairs)


def directional_pairs(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per unordered pair in canonical orientation ``i < j``.

    The canonical orientation is defined here and nowhere else. ``D_true`` is
    ``y(i->j) - y(j->i)`` with ``i < j``, so reversing the orientation negates
    the target by construction -- there is no separate ``(j, i)`` example, and
    the statistical unit stays the unordered pair.
    """
    fwd = rows[rows["i"] < rows["j"]].set_index(["i", "j"])
    rev = rows[rows["i"] > rows["j"]].set_index(["j", "i"])
    rev.index.names = ["i", "j"]
    joined = fwd.join(rev, how="inner", lsuffix="_f", rsuffix="_r")
    if len(joined) != len(fwd) or len(joined) != len(rev):
        raise AssertionError(
            f"pairs are not complete in both orientations: {len(fwd)} forward, "
            f"{len(rev)} reverse, {len(joined)} matched -- D would be "
            f"half-observed for the unmatched ones")
    return pd.DataFrame({
        "i": joined.index.get_level_values(0).to_numpy(),
        "j": joined.index.get_level_values(1).to_numpy(),
        "y_f": joined["y_f"].to_numpy(),
        "y_r": joined["y_r"].to_numpy(),
        "D_true": joined["y_f"].to_numpy() - joined["y_r"].to_numpy(),
    })


def residual_targets(rows: pd.DataFrame, fit: AdditiveFit) -> pd.DataFrame:
    """Attach ``D_add`` and ``D_res`` to the canonical-orientation pair frame.

    ``D_res = D_true - D_add`` where ``D_add = g_i - g_j`` comes from a fit that
    has never seen these rows when ``rows`` are validation or test rows.
    """
    d = directional_pairs(rows)
    d["D_add"] = fit.d_add(d["i"].to_numpy(), d["j"].to_numpy())
    d["D_res"] = d["D_true"].to_numpy() - d["D_add"].to_numpy()
    return d


def ordered_residuals(rows: pd.DataFrame, fit: AdditiveFit) -> pd.DataFrame:
    """Ordered rows with the additive prediction subtracted: ``r_ij``.

    Used only by the ordered-residual model rung, which fits ``r_ij`` directly
    and derives ``D_res`` as ``rhat_ij - rhat_ji``. Note the identity
    ``r_ij - r_ji == D_res(i, j)`` for ``i < j`` -- the two formulations share a
    target, and differ only in what else the loss is asked to fit.
    """
    out = rows.copy()
    out["r"] = (out["y"].to_numpy()
                - fit.predict(out["i"].to_numpy(), out["j"].to_numpy()))
    return out


def decomposition(test_rows: pd.DataFrame, fit: AdditiveFit) -> dict:
    """How much of the held-out directional signal the potential accounts for.

    Every number is out-of-sample: the fit comes from training pairs only and
    these rows are the evaluation pool. Mean squares rather than centred
    variances are primary, because the null being compared against is
    ``Dhat = 0`` and its error is ``mean(D_res**2)``; the centred versions are
    reported too so a reader can see the two are not materially different.
    """
    d = residual_targets(test_rows, fit)
    D, Dadd, Dres = (d["D_true"].to_numpy(), d["D_add"].to_numpy(),
                     d["D_res"].to_numpy())
    y = test_rows["y"].to_numpy()
    pred = fit.predict(test_rows["i"].to_numpy(), test_rows["j"].to_numpy())
    ss = float(((y - y.mean()) ** 2).sum())
    ms_D, ms_Dres = float((D ** 2).mean()), float((Dres ** 2).mean())
    return {
        "n_test_pairs": int(len(d)),
        "n_test_rows": int(len(test_rows)),
        "additive_lambda": fit.lam,
        "additive_objective": fit.objective,
        "additive_val_score": fit.val_score,
        # --- the response itself
        "y_var": float(y.var()),
        "additive_test_mse_y": float(((pred - y) ** 2).mean()),
        "additive_test_r2_y": float(1.0 - ((y - pred) ** 2).sum() / ss)
        if ss > 0 else float("nan"),
        # --- the directional effect
        "D_mean_square": ms_D,
        "D_var": float(D.var()),
        "D_std": float(D.std()),
        "D_add_mean_square": float((Dadd ** 2).mean()),
        "D_res_mean_square": ms_Dres,
        "D_res_var": float(Dres.var()),
        "D_res_std": float(Dres.std()),
        "D_res_mean_abs": float(np.abs(Dres).mean()),
        "D_res_mean": float(Dres.mean()),
        # 1 - ms(D_res)/ms(D). Not a correlation: it is the fraction of the
        # zero-predictor's error that the potential alone removes, and it can
        # go negative if the fitted potential is worse than useless out of
        # sample -- which is exactly what should be reported if it happens.
        "frac_D_removed_by_potential": float(1.0 - ms_Dres / ms_D)
        if ms_D > 0 else float("nan"),
        "D_pearson_add": float(np.corrcoef(D, Dadd)[0, 1])
        if D.std() > 0 and Dadd.std() > 0 else float("nan"),
        # Quantiles of |D_res|, so "is the residual large" (Q2) has an answer
        # that is not a single moment.
        "D_res_abs_q50": float(np.quantile(np.abs(Dres), 0.5)),
        "D_res_abs_q90": float(np.quantile(np.abs(Dres), 0.9)),
        "D_abs_q50": float(np.quantile(np.abs(D), 0.5)),
    }


def hodge_decomposition(frame: pd.DataFrame, n_drugs: int) -> dict:
    """Split the *measured* directional effect into a potential part and a cyclic part.

    This is the model-free version of the question the whole experiment asks, and
    it needs no fit, no split and no learning at all. On a complete graph every
    antisymmetric edge function ``D`` decomposes **uniquely and orthogonally**
    into

        D = grad + curl,     grad(i,j) = g_i - g_j  with  g_i = mean_j D(i,j)

    The gradient part is exactly the class the ``potential`` rung spans -- a
    per-drug "better first than second" tendency, and the best such
    approximation to ``D`` in the least-squares sense. The curl part is what no
    per-drug potential can ever express, for any ``g``: it is the component that
    survives because ``D`` has nonzero sums around cycles. A potential is
    curl-free by construction, so a nonzero curl is not evidence *about* a
    model, it is a property of the data.

    Two things this pins down that a held-out experiment cannot:

    * an exact answer to "how much of the apparent order dependence is a per-drug
      tendency" -- no estimator, no shrinkage, no split;
    * a ceiling. Every rung above ``potential`` is competing for the curl energy
      and nothing else, so ``curl_fraction`` bounds what any of them could
      explain even with unlimited data.

    Reported in-sample on the complete 100x100 matrix and labelled as such. It is
    a description of the screen, not a generalisation claim, and it is never
    mixed with the held-out numbers.
    """
    y = np.full((n_drugs, n_drugs), np.nan)
    y[frame["i"].to_numpy(), frame["j"].to_numpy()] = frame["y"].to_numpy()
    d = y - y.T
    # The diagonal is not measured (a drug is never scheduled against itself) and
    # is set to zero rather than dropped. Zero is the correct value for an
    # antisymmetric function there, and keeping it is what makes the
    # decomposition exactly orthogonal -- computing the row means over n-1
    # entries and the energies over the off-diagonal leaves a ~1% residual in
    # ``grad + curl == D`` that looks like a bug and is only a bookkeeping
    # mismatch.
    np.fill_diagonal(d, 0.0)
    n_missing = int(np.isnan(d).sum())
    if n_missing:
        raise ValueError(
            f"{n_missing} ordered entries are missing; the Hodge decomposition "
            f"on a complete graph is only defined when every ordered pair is "
            f"observed, and this screen is documented as a complete matrix")

    g = d.mean(axis=1)
    grad = g[:, None] - g[None, :]
    curl = d - grad
    e = lambda m: float((m ** 2).mean())                       # noqa: E731
    total = e(d)
    off = ~np.eye(n_drugs, dtype=bool)
    sv = np.linalg.svd(curl, compute_uv=False)
    energy = float((sv ** 2).sum())
    return {
        "n_drugs": int(n_drugs),
        "D_mean_square": total,
        "grad_mean_square": e(grad),
        "curl_mean_square": e(curl),
        "grad_fraction": e(grad) / total,
        "curl_fraction": e(curl) / total,
        # Orthogonality is a property of the decomposition, not an assumption;
        # reported so a reader can see it rather than take it on faith.
        "grad_curl_inner_product": float((grad * curl).mean()),
        "D_std_offdiag": float(d[off].std()),
        "curl_std_offdiag": float(curl[off].std()),
        "potential_g_std": float(g.std()),
        # How concentrated the cyclic part is. A curl that lived in a handful of
        # singular directions would be easy for a low-rank model; one spread
        # across many is not, and the observed spread is why the rank the grid
        # selects is a lower bound on the structure present rather than a
        # measurement of it.
        "curl_rank_energy": {str(k): float((sv[:k] ** 2).sum() / energy)
                             for k in (1, 2, 4, 8, 16, 32, 64)},
    }
