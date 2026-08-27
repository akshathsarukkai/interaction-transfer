"""What the learned interaction geometry correlates with. Exploratory, and last.

Strictly downstream of the predictive result and strictly correlational. A latent
coordinate that lines up with molecular weight does not mean the model learned
about sterics; it means the axis and the weight vary together across 272
molecules, which is a much weaker statement and the only one on offer.

The descriptors are **pre-specified and cheap** -- RDKit built-ins plus a handful
of substructure counts -- rather than chosen after looking at the axes, which is
what would turn an exploratory correlation into a fishing expedition. Nothing
here feeds a decision rule, and nothing here is a validity gate.

The axes themselves are not identifiable. ``z_A^T W z_N`` is invariant under
``z_A -> M z_A``, ``W -> M^-T W``, so any statement about "the first latent axis"
is a statement about one arbitrary basis. What *is* invariant is the subspace and
the pairwise geometry, so the axes are rotated to the singular basis of ``W``
before anything is reported -- the left and right singular vectors of the
bilinear form are the directions along which the interaction actually varies, and
they are unique up to sign and degeneracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: (name, callable) pairs, fixed in advance. Acid and amine share the general
#: descriptors; the role-specific ones are appended below.
GENERAL_DESCRIPTORS: tuple[str, ...] = (
    "MolWt", "TPSA", "MolLogP", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "RingCount", "NumAromaticRings", "FractionCSP3",
    "HeavyAtomCount", "NHOHCount", "NOCount")

#: Substructure counts, as SMARTS. Chosen for chemical meaning in an amide
#: coupling, not for correlation with anything.
SUBSTRUCTURES: dict[str, str] = {
    "aromatic_carbon": "c",
    "halogen": "[F,Cl,Br,I]",
    "nitro": "[N+](=O)[O-]",
    "sulfonyl": "S(=O)(=O)",
    "basic_nitrogen": "[NX3;!$(N[C,S,P]=[O,S,N]);!$(N=*);!$([n])]",
    "hydroxyl": "[OX2H]",
    "carbonyl": "[CX3]=[OX1]",
    "quaternary_carbon": "[CX4]([#6])([#6])([#6])[#6]",
    "heteroatom": "[!#6;!#1]",
}

#: Steric bulk at the reacting centre, per role. The one descriptor that is a
#: mechanistic hypothesis rather than a bulk property: in an amide coupling the
#: substitution immediately around the carboxyl carbon and around the
#: nucleophilic nitrogen is what a chemist would name first.
ALPHA_ACID = "[CX3](=O)[OX2H1]"
ALPHA_AMINE = "[NX3;H1,H2;!$(N[C,S,P]=[O,S,N]);!$(N=*);!$([N+])]"


def descriptors(smiles: tuple[str, ...], role: str) -> pd.DataFrame:
    """A frozen descriptor table, one row per entity, in entity-index order."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")

    from .dataset import nucleophile_class

    alpha = Chem.MolFromSmarts(ALPHA_ACID if role == "acid" else ALPHA_AMINE)
    subs = {k: Chem.MolFromSmarts(v) for k, v in SUBSTRUCTURES.items()}
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"RDKit cannot parse {smi!r}")
        rec = {"smiles": smi}
        for name in GENERAL_DESCRIPTORS:
            rec[name] = float(getattr(Descriptors, name)(mol))
        for name, patt in subs.items():
            rec[f"n_{name}"] = len(mol.GetSubstructMatches(patt))
        match = mol.GetSubstructMatches(alpha)
        if match:
            centre = mol.GetAtomWithIdx(match[0][0])
            neighbours = [a for a in centre.GetNeighbors()
                          if a.GetSymbol() != "O" or role == "amine"]
            rec["alpha_heavy_degree"] = float(centre.GetDegree())
            rec["alpha_branching"] = float(
                max((a.GetDegree() for a in neighbours), default=0))
            rec["alpha_is_aromatic"] = float(
                any(a.GetIsAromatic() for a in neighbours))
        else:
            rec["alpha_heavy_degree"] = np.nan
            rec["alpha_branching"] = np.nan
            rec["alpha_is_aromatic"] = np.nan
        if role == "amine":
            rec["nucleophile_class"] = nucleophile_class(smi)
        rows.append(rec)
    return pd.DataFrame(rows)


def interaction_axes(model) -> dict[str, np.ndarray]:
    """Rotate the pair term into the singular basis of ``W``.

    ``z_A^T W z_N`` is invariant under ``z_A -> M z_A, W -> M^-T W``, so raw
    encoder coordinates are an arbitrary basis and any statement about "axis 1"
    made in them is meaningless. The singular vectors of ``W`` are the directions
    along which the bilinear form actually varies and are unique up to sign and
    degeneracy, so those are what is reported.

    Returns the per-entity coordinates in that basis and the singular values,
    which say how much of the interaction each axis carries.
    """
    import torch

    if not hasattr(model, "W"):
        raise ValueError(f"{type(model).__name__} has no bilinear pair term")
    with torch.no_grad():
        za = model.za(model.XA).numpy()
        zn = model.zn(model.XN).numpy()
        W = model.W.numpy()
    u, s, vt = np.linalg.svd(W)
    return {"acid_axes": za @ u, "amine_axes": zn @ vt.T,
            "singular_values": s,
            "explained": s / s.sum() if s.sum() > 0 else s}


def axis_correlations(axes: np.ndarray, desc: pd.DataFrame, role: str,
                      n_axes: int = 2) -> pd.DataFrame:
    """Spearman correlation of each latent axis with each pre-specified descriptor.

    Reported with a Benjamini-Hochberg q-value over the whole table, because this
    is a descriptor sweep and a table of raw p-values from twenty-odd descriptors
    times a few axes is exactly the shape that produces a spurious "the axis is
    sterics" sentence.
    """
    from scipy import stats

    numeric = [c for c in desc.columns
               if c not in ("smiles", "nucleophile_class")]
    rows = []
    for k in range(min(n_axes, axes.shape[1])):
        for name in numeric:
            x, y = axes[:, k], desc[name].to_numpy(dtype=np.float64)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 10 or np.std(y[ok]) == 0:
                continue
            r = stats.spearmanr(x[ok], y[ok])
            rows.append({"role": role, "axis": k + 1, "descriptor": name,
                         "n": int(ok.sum()), "rho": float(r.statistic),
                         "p": float(r.pvalue)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = np.argsort(out["p"].to_numpy())
    m = len(out)
    q = np.empty(m)
    running = 1.0
    for rank, idx in enumerate(order[::-1]):
        running = min(running, out["p"].to_numpy()[idx] * m / (m - rank))
        q[idx] = running
    out["q"] = q
    return out.sort_values("p").reset_index(drop=True)
