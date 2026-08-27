"""Reactant representations, the controls that prove they do the work, and the
blinding that proves the *unseen* one does.

Why fingerprints and nothing else
---------------------------------
Phase 4's whole content is that a held-out reactant is known to the model only
by its representation. Two opposite ways to ruin that.

*Ruined towards a false positive*: a feature that encoded anything derived from
the screen -- a yield, a success rate, a curated "this substrate is difficult"
annotation -- would let the model recover the answer without generalising.
Morgan fingerprints make this essentially impossible: they are a function of the
molecular graph alone, computed locally by RDKit from a SMILES string, with no
path to an outcome. It is also why no pretrained chemical language model appears
here. ChemBERTa and MolFormer were trained on PubChem/ZINC corpora that nobody in
this repository can audit for overlap with a 2024 HTE deposit, and a positive
result from an unauditable representation would be worth less than a negative
from an auditable one.

*Ruined towards a false negative*: a degenerate representation measures RDKit
rather than chemistry. Bit counts are recorded per entity and reported.

Chirality is ON, and that is a change from Phase 3
--------------------------------------------------
Phase 3 set ``useChirality=False``: its entities were 100 marketed drugs, its
target a directional interaction between whole molecules, and enantiomer-level
resolution had no plausible role at that sample size. Here it does. Ignoring
stereochemistry merges **11 of the 272 acids and 5 of the 231 amines** into
partners they are only stereoisomers of -- for example the racemic and (S)
forms of Boc-Asp(OBzl)-OH are separate entries in the deposit, deliberately, and
an amide coupling to a stereodefined amine is exactly where a difference could
show. Merging them would be a leak: the "unseen" acid would have its own
enantiomer sitting in training. So chirality is included, and the entity index
is keyed on the *isomeric* canonical SMILES.

The blind representation, and why it is not a zero vector
---------------------------------------------------------
Phase 3's adversarial review killed a headline claim over exactly this. Replacing
a held-out entity's feature row with zeros does not say "I know nothing about
this molecule"; it says "this molecule has no substructures at all", which is a
point no real molecule occupies, and every head is systematically biased there.
Against that baseline a *random-feature control containing no chemistry* scored a
significant "attributable to the unseen drug" effect.

:func:`blind_features` therefore substitutes the **training-role marginal**: the
mean feature vector over the entities of that role that the model actually
trained on. That is an on-distribution, information-free stand-in -- it is where
a model should sit if it knows only "some acid from this collection" -- and the
same substitution is applied to the baseline and the pair model, so the contrast
is within-pair and does not depend on where the baseline sits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The primary representation, frozen before any model was fitted.
FP_RADIUS = 2
FP_BITS = 2048
FP_CHIRALITY = True

#: Every representation the sweep may run, primary first. Written into every
#: result row so no analysis can silently compare a real representation against
#: a control one.
REPRESENTATIONS: tuple[str, ...] = (
    "ecfp4", "shuffled_acid", "shuffled_amine", "shuffled_both", "random")


@dataclass(frozen=True)
class RoleFeatures:
    """A frozen ``(n_entities, dim)`` matrix for one reaction role."""

    role: str                     # "acid" or "amine"
    kind: str                     # "ecfp4", "random", "shuffled", ...
    x: np.ndarray
    labels: tuple[str, ...]
    bits_set: np.ndarray
    notes: str = ""

    @property
    def dim(self) -> int:
        return int(self.x.shape[1])

    def __post_init__(self) -> None:
        if self.x.ndim != 2 or self.x.shape[0] != len(self.labels):
            raise ValueError(f"{self.role} feature matrix {self.x.shape} does not "
                             f"match {len(self.labels)} entities")
        if not np.isfinite(self.x).all():
            raise ValueError(f"{self.role}/{self.kind} features are not finite")


@dataclass(frozen=True)
class Representation:
    """The acid and amine feature matrices used by one run, as a unit.

    Bundled rather than passed separately so that a control can never be applied
    to one role and forgotten on the other: ``kind`` names the pair, and every
    result row carries it.
    """

    kind: str
    acid: RoleFeatures
    amine: RoleFeatures

    def __post_init__(self) -> None:
        if self.acid.dim != self.amine.dim:
            raise ValueError("acid and amine features must share a dimension")

    @property
    def dim(self) -> int:
        return self.acid.dim


def fingerprints(smiles: tuple[str, ...], role: str, radius: int = FP_RADIUS,
                 n_bits: int = FP_BITS,
                 use_chirality: bool = FP_CHIRALITY) -> RoleFeatures:
    """ECFP4 bit vectors, in the given order.

    ``smiles`` must already be canonical and in entity-index order: ``x[i]`` is
    the fingerprint of the entity the screen frame calls ``i``. Getting that
    wrong would permute the representation against the outcomes, which is
    precisely the shuffled control and would be indistinguishable from it.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits, includeChirality=use_chirality)
    x = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for k, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"RDKit cannot parse {role} {k}: {smi!r}")
        x[k] = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
    return RoleFeatures(role=role, kind=f"ecfp{2 * radius}", x=x, labels=smiles,
                        bits_set=x.sum(axis=1).astype(int),
                        notes=f"radius={radius} bits={n_bits} "
                              f"chirality={use_chirality}")


def shuffled(base: RoleFeatures, seed: int) -> RoleFeatures:
    """The real fingerprints, permuted among the entities of one role.

    Keeps the feature *distribution* exactly -- the same molecules, the same
    similarity structure among them -- and destroys only the correspondence
    between an entity and its own structure. Sharper than random features, which
    could fail for a reason unrelated to leakage (wrong sparsity, wrong
    geometry): this differs from the real run in exactly one bit of information.

    Permuted once, before any split exists, so every fold sees the same
    permutation. A per-fold permutation would let the control differ from the
    real run in a second way.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(base.labels))
    fixed = int((perm == np.arange(len(perm))).sum())
    if fixed == len(perm):
        raise ValueError("the shuffle is the identity; the control would be vacuous")
    return RoleFeatures(role=base.role, kind=f"shuffled-{base.kind}",
                        x=base.x[perm].copy(), labels=base.labels,
                        bits_set=base.bits_set[perm].copy(),
                        notes=f"permutation of {base.kind}, seed={seed}, "
                              f"n_fixed_points={fixed}")


def random_like(base: RoleFeatures, seed: int) -> RoleFeatures:
    """A fixed random vector per entity, matched in dimension and per-entity density.

    Drawn once from ``seed`` and frozen -- generated before any split exists, so
    it cannot be redrawn per fold and cannot encode fold structure. Under a
    genuine entity-level split this must not transfer; if it does, the split is
    leaking and no other number in the experiment means anything.
    """
    rng = np.random.default_rng(seed)
    p = np.clip(base.bits_set / base.dim, 1e-6, 1 - 1e-6)
    x = (rng.random((len(base.labels), base.dim)) < p[:, None]).astype(np.float32)
    return RoleFeatures(role=base.role, kind=f"random-{base.kind}", x=x,
                        labels=base.labels, bits_set=x.sum(axis=1).astype(int),
                        notes=f"Bernoulli, per-entity density matched, seed={seed}")


def build_representation(kind: str, acid: RoleFeatures, amine: RoleFeatures,
                         seed: int) -> Representation:
    """Assemble one of :data:`REPRESENTATIONS` from the real fingerprints.

    The two roles get *different* seeds under ``shuffled_both`` so that the two
    permutations are independent; using one seed for both would permute acids
    and amines by the same map, which on unequal-length lists is not even
    well-defined and on equal-length ones would preserve a spurious alignment.
    """
    if kind == "ecfp4":
        return Representation(kind, acid, amine)
    if kind == "shuffled_acid":
        return Representation(kind, shuffled(acid, seed), amine)
    if kind == "shuffled_amine":
        return Representation(kind, acid, shuffled(amine, seed + 1))
    if kind == "shuffled_both":
        return Representation(kind, shuffled(acid, seed), shuffled(amine, seed + 1))
    if kind == "random":
        return Representation(kind, random_like(acid, seed + 2),
                              random_like(amine, seed + 3))
    raise ValueError(f"unknown representation {kind!r}; "
                     f"expected one of {REPRESENTATIONS}")


def blind_features(base: RoleFeatures, train_entities: np.ndarray,
                   blind_entities: np.ndarray) -> RoleFeatures:
    """Replace ``blind_entities``' rows with the training-role marginal.

    The marginal is the **mean over the entities the model actually trained on**,
    for that role. It is on-distribution by construction and carries no
    information about which held-out entity it stands for, which is exactly what
    "the model knows this is some acid from this collection and nothing more"
    should mean.

    Explicitly not a zero vector. See the module docstring: Phase 3 used zeros,
    and a random-feature control -- containing no chemistry at all -- scored a
    significant effect against that baseline, because zeros sit far outside the
    data and any real vector beats them.
    """
    train_entities = np.asarray(train_entities, dtype=np.int64)
    if train_entities.size == 0:
        raise ValueError("cannot form a training marginal from no entities")
    overlap = np.intersect1d(train_entities, np.asarray(blind_entities))
    if overlap.size:
        raise ValueError(f"{base.role}s {overlap.tolist()} are both trained on and "
                         f"blinded; the marginal would leak them into themselves")
    marginal = base.x[train_entities].mean(axis=0)
    x = base.x.copy()
    x[np.asarray(blind_entities, dtype=np.int64)] = marginal
    return RoleFeatures(role=base.role, kind=f"blind-{base.kind}", x=x,
                        labels=base.labels, bits_set=x.sum(axis=1).astype(int),
                        notes=f"{len(blind_entities)} entities replaced by the mean "
                              f"of {len(train_entities)} training {base.role}s")


def tanimoto(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """All-pairs Tanimoto between two binary matrices, ``(len(x), len(y))``.

    Used only to describe the chemical geometry -- which held-out entities have
    close training analogues -- never as a model input.
    """
    a = (np.asarray(x) > 0).astype(np.float64)
    b = (np.asarray(y) > 0).astype(np.float64)
    inter = a @ b.T
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def max_similarity_to(base: RoleFeatures, query: np.ndarray,
                      reference: np.ndarray) -> np.ndarray:
    """For each entity in ``query``, its greatest Tanimoto to any of ``reference``.

    Self-matches are impossible because the two sets are required to be disjoint
    -- a held-out entity compared against training entities. Asserted rather than
    assumed: a silent self-match returns 1.0 and would put every held-out entity
    in the top similarity stratum.
    """
    query = np.asarray(query, dtype=np.int64)
    reference = np.asarray(reference, dtype=np.int64)
    both = np.intersect1d(query, reference)
    if both.size:
        raise ValueError(f"{base.role}s {both.tolist()} appear in both the query "
                         f"and the reference set; the similarity would be 1.0 "
                         f"against themselves")
    if reference.size == 0 or query.size == 0:
        return np.zeros(len(query))
    return tanimoto(base.x[query], base.x[reference]).max(axis=1)
