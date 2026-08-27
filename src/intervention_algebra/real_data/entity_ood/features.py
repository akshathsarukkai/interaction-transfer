"""Drug representations, and the controls that prove they are doing the work.

Phase 3's whole content is that the model's only knowledge of a held-out drug is
its representation. That makes this module the place where the experiment can be
quietly ruined, in two directions.

*Ruined towards a false positive*: if a feature encoded anything derived from the
screen -- a response, an ordering, even a curated "known combination" annotation
-- the model would recover the answer without generalising at all. Morgan
fingerprints make this essentially impossible: they are a function of the
molecular graph alone, computed locally by RDKit from a SMILES string, with no
path to an outcome. That is the main reason the primary representation is a
fingerprint and not a learned embedding from a pretrained chemical model, whose
training corpus nobody in this repository can audit.

*Ruined towards a false negative*: if the representation is degenerate the
experiment measures RDKit rather than chemistry. Four of the hundred compounds
are coordination complexes or inorganic salts, and a Morgan fingerprint of
``[O-2].[O-2].[O-2].[As+3].[As+3]`` carries almost nothing. Their bit counts are
recorded by :func:`fingerprint_matrix` and reported, and the primary result is
computed both with and without them under a rule fixed in advance.

The representation is deliberately plain -- ECFP4, 2048 bits, no chirality --
because the question being asked is "is transferable entity information present
at all?", and a negative answer from a simple representation is a much weaker
claim than a negative from an elaborate one. If simple structure works, richer
representations become worth trying; if it does not, that is what the next phase
has to reckon with.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: The primary representation, frozen before any model was fitted.
#: ``useChirality=False`` because Phase 3's target is a directional *interaction*
#: between two whole molecules and there is no reason to expect enantiomer-level
#: resolution to matter at this sample size; recording it as a decision rather
#: than a default so that turning it on later is visible as a change.
FP_RADIUS = 2
FP_BITS = 2048
FP_CHIRALITY = False


@dataclass(frozen=True)
class DrugFeatures:
    """A frozen ``(n_drugs, dim)`` matrix and the audit trail for it.

    ``kind`` names the provenance -- ``ecfp4``, ``random``, ``shuffled``,
    ``targets`` -- and is written into every result row, so no analysis can
    silently compare a real representation against a control one.
    """

    kind: str
    x: np.ndarray
    labels: tuple[str, ...]
    dim: int
    bits_set: np.ndarray
    notes: str = ""

    def __post_init__(self) -> None:
        if self.x.shape != (len(self.labels), self.dim):
            raise ValueError(f"feature matrix {self.x.shape} does not match "
                             f"{len(self.labels)} drugs x {self.dim} dims")
        if not np.isfinite(self.x).all():
            raise ValueError(f"{self.kind} features contain non-finite values")


def fingerprint_matrix(mapping: pd.DataFrame, radius: int = FP_RADIUS,
                       n_bits: int = FP_BITS,
                       use_chirality: bool = FP_CHIRALITY) -> DrugFeatures:
    """ECFP4 bit vectors for every drug, in ``drug_index`` order.

    ``mapping`` is ``data/external/koplev_drug_mapping.csv``. The row order is
    forced to ``drug_index``, which is the same integer the screen's ingestion
    assigns, so ``x[i]`` is the fingerprint of the drug the pair frame calls
    ``i``. Getting that wrong would permute the representation against the
    outcomes -- which is precisely Control B, and would be indistinguishable
    from it in the results.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    frame = mapping.sort_values("drug_index").reset_index(drop=True)
    if list(frame["drug_index"]) != list(range(len(frame))):
        raise ValueError("drug_index must be 0..n-1 with no gaps")

    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits, includeChirality=use_chirality)
    x = np.zeros((len(frame), n_bits), dtype=np.float32)
    for k, smi in enumerate(frame["smiles"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"RDKit cannot parse row {k}: {smi!r}")
        fp = gen.GetFingerprintAsNumPy(mol)
        x[k] = fp.astype(np.float32)
    return DrugFeatures(
        kind=f"ecfp{2 * radius}", x=x, labels=tuple(frame["label"]), dim=n_bits,
        bits_set=x.sum(axis=1).astype(int),
        notes=f"radius={radius} bits={n_bits} chirality={use_chirality}")


def random_features(base: DrugFeatures, seed: int) -> DrugFeatures:
    """Control A: a fixed random vector per drug, matched in dimension and density.

    Each drug gets an independent Bernoulli vector whose success probability is
    that drug's own bit density in ``base``, so the control matches the real
    representation in sparsity *per drug* and differs only in carrying no
    chemistry. Drawn once from ``seed`` and frozen -- generated before any split
    exists, so it cannot be redrawn per fold and cannot encode fold structure.

    Under a genuine entity-level split this must not transfer. If it does, the
    split is leaking and no other number in the experiment means anything.
    """
    rng = np.random.default_rng(seed)
    p = np.clip(base.bits_set / base.dim, 1e-6, 1 - 1e-6)
    x = (rng.random((len(base.labels), base.dim)) < p[:, None]).astype(np.float32)
    return DrugFeatures(kind=f"random-{base.kind}", x=x, labels=base.labels,
                        dim=base.dim, bits_set=x.sum(axis=1).astype(int),
                        notes=f"Bernoulli, per-drug density matched, seed={seed}")


def shuffled_features(base: DrugFeatures, seed: int) -> DrugFeatures:
    """Control B: the real fingerprints, permuted among drugs.

    Keeps the feature *distribution* exactly -- the same 100 fingerprints, the
    same similarity structure among them -- and destroys only the correspondence
    between a drug and its own structure. It is the sharper of the two negative
    controls: random features could in principle fail for a reason having
    nothing to do with leakage (wrong sparsity, wrong geometry), while this one
    differs from the real experiment in exactly one bit of information.

    Permuted before splitting, so the same permutation is used for every fold.
    A per-fold permutation would let the control differ from the real run in a
    second way.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(base.labels))
    if np.all(perm == np.arange(len(perm))):        # astronomically unlikely, but
        raise ValueError("the shuffle is the identity; the control would be vacuous")
    x = base.x[perm].copy()
    return DrugFeatures(kind=f"shuffled-{base.kind}", x=x, labels=base.labels,
                        dim=base.dim, bits_set=x.sum(axis=1).astype(int),
                        notes=f"identity permutation of {base.kind}, seed={seed}, "
                              f"n_fixed_points={int((perm == np.arange(len(perm))).sum())}")


def tanimoto_matrix(features: DrugFeatures) -> np.ndarray:
    """All-pairs Tanimoto over binary features, diagonal set to NaN.

    Used only to describe the chemical geometry -- which drugs have close
    analogues -- never as a model input. The diagonal is NaN rather than 1.0 so
    that ``nanmax`` over a row cannot silently return the drug's similarity to
    itself.
    """
    x = (features.x > 0).astype(np.float64)
    inter = x @ x.T
    size = x.sum(axis=1)
    union = size[:, None] + size[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, inter / union, 0.0)
    np.fill_diagonal(sim, np.nan)
    return sim
