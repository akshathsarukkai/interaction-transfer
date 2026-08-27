"""The secondary representation: curated drug -> protein-target annotations.

The question this arm answers
-----------------------------
If molecular structure transfers and target annotations do not, the reusable
geometry is tied to chemistry rather than to annotated mechanism. If targets
transfer and structure does not, it is tied to pharmacology. If both work, they
may or may not be carrying the same information, which the error correlation can
say something about. None of that is a claim about mechanism *causing* anything
-- prediction from an annotation is not evidence of a mechanism -- and the
document says so.

Source and filters, fixed before any model was fitted
-----------------------------------------------------
ChEMBL's ``mechanism`` table: manually curated direct mechanisms of action for
approved drugs, drawn from regulatory labels and primary literature. It is the
right table rather than the bioactivity table because it needs no arbitrary
potency cutoff -- inclusion is already an editorial judgement that the target is
*the* mechanism, not that a molecule was active in some assay at some
concentration.

* queried by ``parent_molecule_chembl_id``, not ``molecule_chembl_id``. This
  matters more than it looks: ChEMBL registers mechanisms against the *drug*
  record, which for a salt is the salt. Querying by ``molecule_chembl_id``
  silently returns nothing for erlotinib, imatinib, lapatinib, tamoxifen,
  doxorubicin, vinblastine and eighteen others -- 75/100 coverage that looks
  plausible until you notice the missing drugs are the famous ones. The correct
  field gives 97/100;
* targets restricted to ``Homo sapiens`` where ChEMBL records an organism;
* one binary column per distinct ``target_chembl_id``. No weighting, no potency
  threshold, no tuning of any kind against Koplev performance.

The limitation, stated up front
-------------------------------
This representation cannot transfer for a drug whose targets appear in no
training drug -- its vector is orthogonal to everything the model has seen, and
the model can do no better than predict zero for it. Seventeen of the hundred
drugs are in that position against the *full* set, and more will be against any
given fold's 80 training drugs. That is a property of the annotation's
granularity, not a defect in the experiment, and it is why this arm is secondary:
a null here is much weaker evidence than a null from fingerprints, and the per-
drug table reports each held-out drug's target coverage so the two can be told
apart.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import DrugFeatures

DEFAULT_TARGETS = Path(__file__).resolve().parents[4] / "data" / "external" / "koplev_drug_targets.csv"


def load_target_table(path: Path = DEFAULT_TARGETS) -> pd.DataFrame:
    return pd.read_csv(path)


def target_features(mapping: pd.DataFrame, shuffled: bool = False, seed: int = 0,
                    path: Path = DEFAULT_TARGETS) -> DrugFeatures:
    """Binary ``(n_drugs, n_targets)`` matrix in ``drug_index`` order.

    ``shuffled=True`` is Control D: the same target vectors permuted among drugs,
    destroying the drug-mechanism correspondence while preserving the annotation
    distribution exactly.
    """
    table = load_target_table(path)
    frame = mapping.sort_values("drug_index").reset_index(drop=True)
    targets = sorted(table["target_chembl_id"].unique())
    index = {t: k for k, t in enumerate(targets)}
    x = np.zeros((len(frame), len(targets)), dtype=np.float32)
    by_drug = table.groupby("drug_index")["target_chembl_id"].apply(list).to_dict()
    for k in range(len(frame)):
        for t in by_drug.get(k, []):
            x[k, index[t]] = 1.0
    kind = "targets"
    notes = f"{len(targets)} ChEMBL mechanism targets; " \
            f"{int((x.sum(axis=1) > 0).sum())}/{len(frame)} drugs annotated"
    if shuffled:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(frame))
        x = x[perm].copy()
        kind, notes = "targets-shuffled", notes + f"; permuted, seed={seed}"
    return DrugFeatures(kind=kind, x=x, labels=tuple(frame["label"]), dim=len(targets),
                        bits_set=x.sum(axis=1).astype(int), notes=notes)
