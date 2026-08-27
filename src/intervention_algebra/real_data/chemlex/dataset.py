"""Ingestion and canonicalisation of the ChemLex acid-amine screen.

One row is one reaction well
----------------------------
Every one of the 11,669 rows is a **physically executed wet-lab reaction**. This
was the first thing checked, because the paper's own description invites the
opposite reading: it says the dataset was enlarged by "an additional 5600
reactions introduced by chemist-designed rules", which sounds like rule-generated
labels. It is not. The rules acted on *which acid-amine pairs to run* -- SI 1.1
describes filtering purchasable substrates by amine nitrogen partial charge and
by steric hindrance to "form the infeasible reactions in chemical sense" -- and
the selected pairs were then run on the platform. Three things settle it: SI
Table S1 reports measured yields of 0.00 / 32.32 / 91.26 across the partial-charge
tiers, the main text highlights two rule-predicted negatives that came back at
62.12 % and 73.97 %, and the deposit is named ``..._Wetlab_Data.xlsx`` under a
data-availability statement reading "the HTE dataset generated in this study".

The consequence for Phase 4 is not contamination but **design**: roughly 5,600 of
the 11,669 rows were deliberately enriched for predicted failure, and no column
identifies which. The screen is a non-random sample of acid-amine space, and
every result here is conditional on that sample.

The endpoint
------------
``Conversion`` is an **uncalibrated LC-MS UV peak-area ratio at 254 nm**, defined
in SI 2.1.8 as

    Conversion Yield = Product area(%) / [1 - Coupling reagent area(%)
                                            - Acid or Amine area(%)] x 100 %

with neither an internal nor an external standard, because the building blocks
differ from well to well. Three limitations follow and all three are load-bearing:

* it is **not** an isolated yield and not calibrated per compound, so a molar
  comparison between two different products is not licensed;
* the 7,100 exact zeros (60.8 %) are **left-censored non-detections** at an
  integration threshold that is stated nowhere in the paper or its 51-page SI --
  searching both for LOD, LOQ and "detection limit" returns nothing;
* the authors themselves declined to regress on it, reformulating the task as
  binary classification because "regression models may struggle to accurately
  capture the underlying patterns of the data or may even overfit to the noise"
  (SI 2.2.1).

That last point is why both endpoints are carried. The continuous one is primary
because the quantity under test is an incremental *ratio* of two models' errors,
in which a shared noise floor attenuates the effect towards zero rather than
manufacturing one, and because a 20 % threshold discards exactly the information
a pair term is most likely to hold -- how well a pair couples, not merely
whether. The binary one is reported alongside, using the authors' own documented
rule (``Conversion >= 20``), which appears in SI 2.2.1, in the Fig. 5a caption
and in their ``train.py`` as ``labels = (df["Conversion"] >= 20).astype(int)``.

The measurement's own reliability is estimated here rather than assumed. 505
(acid, amine, protocol) cells were measured more than once -- SI 2.2.15 says 486
reactions were repeated 2-3 times -- and across the 492 two-row cells the two
readings correlate at Pearson 0.56 with a within-cell standard deviation of about
22 conversion points against a between-row standard deviation of 31. That implies
an R2 ceiling near 0.47 for *any* deterministic predictor. It is reported next to
every skill in this phase.

Conditions
----------
The ``Reagents`` column is a dot-separated mixture SMILES. Decomposed, the file
contains exactly the paper's **6 condensation reagents** (HATU, TCFH, PyBOP,
PyBrOP, BOP, EDC), **2 bases** (DIPEA, NMI) and **1 solvent** (DMF) -- but only
once two things are fixed.

*HATU appears under two depictions.* ``CN(C)C(On1nnc2cccnc21)=[N+](C)C`` and
``CN(C)C(n1n[n+]([O-])c2ncccc21)=[N+](C)C`` have the same formula, C10H15N6O+,
the same molecular weight, and are the uronium (O-) and guanidinium N-oxide (N-)
drawings of one reagent. Counting them separately gives 7 condensation reagents
and contradicts the paper's 6.

The tempting reason to keep them apart anyway is that they are confounded with
experimental campaign: the O-form spans 272 acids x 154 amines at mean conversion
20.3, the N-form 15 acids x 85 amines at mean 10.8, and the two occupy disjoint
blocks of the file. But the marginal difference is composition, not condition.
On the **37 pairs measured under both**, the paired difference is -0.05 points
(t = -0.017, p = 0.987). The same inversion appears elsewhere in the screen and
is a textbook Simpson's paradox: HATU beats TCFH/NMI by 12.1 points marginally
and *loses* to it by 2.5 points (p = 1.4e-4) on the 932 pairs both were run on.
So the primary encoding merges them, and the condition variable is the reaction
condition as actually run: 7 (reagent, base) combinations in one solvent.

The unmerged 8-level ``"protocol"`` encoding is available and is run as a
registered sensitivity. It is the more *conservative* of the two for this
phase's claim -- more condition levels means a strictly more flexible additive
baseline, and therefore a harder bar for the pair term -- which is why it is
reported rather than dropped.

*Two "reagents" are substrate counterions.* 114 rows carry a leading ``Cl.`` and
15 a leading ``O=S(=O)(O)O.`` on the HATU-N string. These are the hydrochloride
and sulfate salt forms of the amine folded into the wrong column: all 10 of the
``Cl.`` amines also appear without it, and the single ``O=S(=O)(O)O.`` amine
appears nowhere else. They are stripped. Leaving them in would put *substrate
identity* into the condition channel -- and would make the sulfate level's
intercept unestimable in exactly the folds where its one amine is held out.

Molecular identity
------------------
Entities are keyed on the **RDKit isomeric canonical SMILES**, not the raw string.
Two raw amine SMILES canonicalise to one molecule and are merged; without that,
an entity-level split could hold out one spelling while training on the other.

Stereochemistry is kept. Ignoring it would merge 11 of the 272 acids and 5 of the
231 amines into stereoisomers of themselves -- the deposit deliberately carries,
for instance, both the unspecified and the (S) form of Boc-Asp(OBzl)-OH -- and a
held-out acid whose own enantiomer sat in training would not be held out at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .acquire import DEFAULT_RAW_DIR, raw_path

#: The authors' documented feasibility rule. SI 2.2.1, Fig. 5a caption, and
#: ``train.py:95``. Not a threshold of ours, and deliberately not re-tuned.
FEASIBLE_AT = 20.0

#: Continuous endpoint scale. Conversion is reported in percent; the models see
#: it in [0, 1] so that an MSE reads as a fraction-squared.
YIELD_SCALE = 100.0

#: Substrate counterions folded into the ``Reagents`` column. Stripped from the
#: front of the string; see the module docstring. Order matters only in that
#: the longer prefix is tried first.
SALT_PREFIXES: tuple[str, ...] = ("O=S(=O)(O)O.", "Cl.")

#: Component SMILES -> (role, name). Curated once from the ten distinct
#: ``Reagents`` strings and asserted on every load: an unrecognised component
#: raises rather than being silently swept into an "other" bucket, because a
#: revised deposit that introduces a new reagent must not be modelled as though
#: it were one of these.
COMPONENTS: dict[str, tuple[str, str]] = {
    "CCN(C(C)C)C(C)C": ("base", "DIPEA"),
    "Cn1ccnc1": ("base", "NMI"),
    "F[P-](F)(F)(F)(F)F": ("counterion", "PF6-"),
    "Cl": ("counterion", "HCl"),
    "CN(C)C(On1nnc2cccnc21)=[N+](C)C": ("reagent", "HATU"),
    "CN(C)C(n1n[n+]([O-])c2ncccc21)=[N+](C)C": ("reagent", "HATU"),
    "CN(C)C(Cl)=[N+](C)C": ("reagent", "TCFH"),
    "c1ccc2c(c1)nnn2O[P+](N1CCCC1)(N1CCCC1)N1CCCC1": ("reagent", "PyBOP"),
    "Br[P+](N1CCCC1)(N1CCCC1)N1CCCC1": ("reagent", "PyBrOP"),
    "CN(C)[P+](On1nnc2ccccc21)(N(C)C)N(C)C": ("reagent", "BOP"),
    "CCN=C=NCCCN(C)C": ("reagent", "EDC"),
}

#: Human-readable protocol names, keyed on the salt-stripped ``Reagents`` string.
#: Generated by :func:`decode_conditions` and checked against this table, so a
#: name in a document always corresponds to a string in the file.
PROTOCOL_NAMES: dict[str, str] = {
    "CCN(C(C)C)C(C)C.CN(C)C(On1nnc2cccnc21)=[N+](C)C.F[P-](F)(F)(F)(F)F":
        "HATU/DIPEA (O-form)",
    "CCN(C(C)C)C(C)C.CN(C)C(n1n[n+]([O-])c2ncccc21)=[N+](C)C.F[P-](F)(F)(F)(F)F":
        "HATU/DIPEA (N-form)",
    "Cn1ccnc1.CN(C)C(Cl)=[N+](C)C.F[P-](F)(F)(F)(F)F": "TCFH/NMI",
    "CCN(C(C)C)C(C)C.F[P-](F)(F)(F)(F)F.c1ccc2c(c1)nnn2O[P+](N1CCCC1)(N1CCCC1)N1CCCC1":
        "PyBOP/DIPEA",
    "CCN=C=NCCCN(C)C.Cl.CCN(C(C)C)C(C)C": "EDC/DIPEA",
    "CCN(C(C)C)C(C)C.CN(C)C(Cl)=[N+](C)C.F[P-](F)(F)(F)(F)F": "TCFH/DIPEA",
    "CCN(C(C)C)C(C)C.Br[P+](N1CCCC1)(N1CCCC1)N1CCCC1.F[P-](F)(F)(F)(F)F":
        "PyBrOP/DIPEA",
    "CCN(C(C)C)C(C)C.CN(C)[P+](On1nnc2ccccc21)(N(C)C)N(C)C.F[P-](F)(F)(F)(F)F":
        "BOP/DIPEA",
}

#: How the condition covariate is built. ``chemistry`` is primary: it is the
#: reaction condition as actually run, 7 (reagent, base) pairs in one solvent.
#: ``protocol`` keeps the two HATU depictions apart and is the registered
#: sensitivity. See the module docstring.
CONDITION_ENCODINGS: tuple[str, ...] = ("chemistry", "protocol")

#: The screens the sweep may run. ``hatu`` is the single dominant protocol, in
#: which no condition confounding is possible because there is only one
#: condition; ``all`` is the whole table with the condition as a covariate.
#: They are **nested**, not independent replicates, and the reports say so.
SCREENS: tuple[str, ...] = ("hatu", "all")

#: The condition that defines the ``hatu`` screen, under the primary
#: ``chemistry`` encoding. It is the first-pass campaign: it covers all 272
#: acids and 7,124 of the 8,064 pairs, and unlike the rarer conditions its
#: membership is not conditioned on a reaction having already failed elsewhere.
#: That last point matters -- the screen's condition assignment is *adaptive*, a
#: pair having been retried under a second reagent because it failed under the
#: first (mean HATU conversion is 21.7 for pairs tried under one reagent, 3.3
#: for pairs tried under three, 0.0 for pairs tried under five). Restricting to
#: this condition removes both the confounding and the adaptivity.
HATU_CONDITION = "HATU/DIPEA"


def canonical(smiles: str) -> str:
    """RDKit isomeric canonical SMILES. Raises on anything unparseable."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse {smiles!r}")
    return Chem.MolToSmiles(mol)


def strip_salt_prefix(reagents: str) -> tuple[str, str | None]:
    """Split a ``Reagents`` string into (protocol, substrate salt or None)."""
    for prefix in SALT_PREFIXES:
        if reagents.startswith(prefix):
            return reagents[len(prefix):], prefix[:-1]
    return reagents, None


def decode_conditions(reagents: pd.Series) -> pd.DataFrame:
    """Decompose every distinct ``Reagents`` string into its named components.

    Returns one row per distinct raw string with the stripped protocol, the
    substrate salt if any, and the reagent and base names. Raises on a component
    that is not in :data:`COMPONENTS`: a deposit revision that introduces a new
    reagent must fail loudly rather than be modelled as one of the known ones.
    """
    rows = []
    for raw in sorted(reagents.unique()):
        protocol, salt = strip_salt_prefix(raw)
        parts = protocol.split(".")
        roles: dict[str, list[str]] = {"reagent": [], "base": [], "counterion": []}
        for p in parts:
            if p not in COMPONENTS:
                raise ValueError(
                    f"unrecognised reagent component {p!r} in {raw!r}. The "
                    f"deposit may have been revised; add it to COMPONENTS with "
                    f"its identity established, do not guess.")
            role, name = COMPONENTS[p]
            roles[role].append(name)
        if len(set(roles["reagent"])) != 1:
            raise ValueError(f"{raw!r} decodes to reagents {roles['reagent']}, "
                             f"expected exactly one")
        if len(set(roles["base"])) != 1:
            raise ValueError(f"{raw!r} decodes to bases {roles['base']}, "
                             f"expected exactly one")
        rows.append({"reagents_raw": raw, "protocol": protocol,
                     "substrate_salt": salt, "reagent": roles["reagent"][0],
                     "base": roles["base"][0],
                     "counterions": "+".join(sorted(set(roles["counterion"]))),
                     "protocol_name": PROTOCOL_NAMES.get(protocol, protocol)})
    out = pd.DataFrame(rows)
    unnamed = sorted(set(out["protocol"]) - set(PROTOCOL_NAMES))
    if unnamed:
        raise ValueError(f"protocols with no name in PROTOCOL_NAMES: {unnamed}")
    return out


@dataclass(frozen=True)
class Screen:
    """One analysis table, in canonical form.

    ``frame`` has one row per measured reaction with columns

        ``acid``, ``amine``       integer entity indices into ``acids`` / ``amines``
        ``cond``                  integer condition index into ``conditions``
        ``y``                     Conversion / 100, in [0, 1]
        ``feasible``              Conversion >= 20, the authors' rule
        ``acid_smiles``, ``amine_smiles``, ``protocol``, ``conversion``

    Entity indices come from the sorted canonical SMILES lists, and the frame is
    sorted by ``(acid, amine, cond)``, so two loads on two machines produce the
    same integer encoding.
    """

    name: str
    encoding: str
    frame: pd.DataFrame
    acids: tuple[str, ...]
    amines: tuple[str, ...]
    conditions: tuple[str, ...]
    condition_names: tuple[str, ...]
    n_raw_rows: int
    notes: dict

    @property
    def n_rows(self) -> int:
        return len(self.frame)

    @property
    def n_acids(self) -> int:
        return len(self.acids)

    @property
    def n_amines(self) -> int:
        return len(self.amines)

    @property
    def n_conditions(self) -> int:
        return len(self.conditions)


def load_raw(raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    path = raw_path(raw_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python scripts/download_chemlex.py` first.")
    return pd.read_excel(path, sheet_name=0)


def load_screen(name: str = "all", encoding: str = "chemistry",
                raw_dir: Path = DEFAULT_RAW_DIR,
                raw: pd.DataFrame | None = None) -> Screen:
    """Build one analysis table. Deterministic and side-effect free."""
    if name not in SCREENS:
        raise ValueError(f"unknown screen {name!r}; expected one of {SCREENS}")
    if encoding not in CONDITION_ENCODINGS:
        raise ValueError(f"unknown encoding {encoding!r}; "
                         f"expected one of {CONDITION_ENCODINGS}")
    df = load_raw(raw_dir) if raw is None else raw.copy()
    n_raw = len(df)
    required = {"Acid", "Amine", "Reagents", "Solvent", "Conversion"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"the deposit is missing columns {sorted(missing)}")
    if df[sorted(required)].isna().any().any():
        raise ValueError("the deposit has missing values in a modelled column; "
                         "this was not true of record 17596563 and must be "
                         "investigated, not imputed")
    if not ((df["Conversion"] >= 0) & (df["Conversion"] <= 100)).all():
        raise ValueError("Conversion outside [0, 100]")

    decoded = decode_conditions(df["Reagents"]).set_index("reagents_raw")
    df["protocol"] = df["Reagents"].map(decoded["protocol"])
    df["reagent"] = df["Reagents"].map(decoded["reagent"])
    df["base"] = df["Reagents"].map(decoded["base"])
    df["substrate_salt"] = df["Reagents"].map(decoded["substrate_salt"])

    if name == "hatu":
        df = df[(df["reagent"] == "HATU") & (df["base"] == "DIPEA")
                ].reset_index(drop=True)

    df["acid_smiles"] = [canonical(s) for s in df["Acid"]]
    df["amine_smiles"] = [canonical(s) for s in df["Amine"]]

    acids = tuple(sorted(df["acid_smiles"].unique()))
    amines = tuple(sorted(df["amine_smiles"].unique()))
    ai = {s: i for i, s in enumerate(acids)}
    ni = {s: i for i, s in enumerate(amines)}

    if encoding == "protocol":
        key = df["protocol"]
        names = {p: decoded.loc[decoded["protocol"] == p, "protocol_name"].iloc[0]
                 for p in key.unique()}
    else:
        key = df["reagent"] + "/" + df["base"]
        names = {k: k for k in key.unique()}
    conditions = tuple(sorted(key.unique()))
    ci = {c: i for i, c in enumerate(conditions)}

    frame = pd.DataFrame({
        "acid": df["acid_smiles"].map(ai).astype(np.int64),
        "amine": df["amine_smiles"].map(ni).astype(np.int64),
        "cond": key.map(ci).astype(np.int64),
        "y": df["Conversion"].to_numpy(dtype=np.float64) / YIELD_SCALE,
        "feasible": (df["Conversion"] >= FEASIBLE_AT).astype(np.int64),
        "conversion": df["Conversion"].to_numpy(dtype=np.float64),
        "acid_smiles": df["acid_smiles"], "amine_smiles": df["amine_smiles"],
        "protocol": df["protocol"], "reagent": df["reagent"], "base": df["base"],
        "raw_acid": df["Acid"], "raw_amine": df["Amine"],
    }).sort_values(["acid", "amine", "cond"], kind="stable").reset_index(drop=True)

    merged_acids = int(df["Acid"].nunique() - len(acids))
    merged_amines = int(df["Amine"].nunique() - len(amines))
    notes = {
        "n_raw_acid_smiles": int(df["Acid"].nunique()),
        "n_raw_amine_smiles": int(df["Amine"].nunique()),
        "n_canonical_acids": len(acids), "n_canonical_amines": len(amines),
        "acids_merged_by_canonicalisation": merged_acids,
        "amines_merged_by_canonicalisation": merged_amines,
        "n_raw_reagent_strings": int(df["Reagents"].nunique()),
        "n_protocols": int(df["protocol"].nunique()),
        "n_reagents": int(df["reagent"].nunique()),
        "n_bases": int(df["base"].nunique()),
        "n_solvents": int(df["Solvent"].nunique()),
        "n_salt_annotated_rows": int(df["substrate_salt"].notna().sum()),
        "n_pairs": int(frame.groupby(["acid", "amine"]).ngroups),
        "zero_fraction": float((frame["conversion"] == 0).mean()),
        "feasible_fraction": float(frame["feasible"].mean()),
    }
    return Screen(name=name, encoding=encoding, frame=frame, acids=acids,
                  amines=amines, conditions=conditions,
                  condition_names=tuple(names[c] for c in conditions),
                  n_raw_rows=n_raw, notes=notes)


def replicate_noise(screen: Screen) -> dict:
    """The measurement noise floor, from cells measured more than once.

    A "cell" is (acid, amine, condition). SI 2.2.15 says 486 reactions were
    repeated two or three times; the file holds 505 such cells. Two readings of
    one cell differ by ``d``; under an additive-noise model ``Var(noise) =
    E[d^2] / 2``, and ``1 - Var(noise) / Var(y)`` is the R2 no deterministic
    predictor of ``(acid, amine, condition)`` can exceed.

    Reported next to every skill in this phase, because a reader comparing an
    incremental pair skill of a few points against an implied ceiling near 0.5
    is reading the number correctly and one comparing it against 1.0 is not.
    """
    f = screen.frame
    g = f.groupby(["acid", "amine", "cond"])["conversion"]
    sizes = g.size()
    two = sizes[sizes == 2].index
    if len(two) < 3:
        return {"n_cells_repeated": int((sizes > 1).sum()), "n_pairs_used": 0}
    vals = f.set_index(["acid", "amine", "cond"]).loc[two, "conversion"]
    arr = vals.to_numpy().reshape(-1, 2)
    d = arr[:, 0] - arr[:, 1]
    noise_var = float((d ** 2).mean() / 2)
    total_var = float(f["conversion"].var())
    both_zero = int(((arr[:, 0] == 0) & (arr[:, 1] == 0)).sum())
    from scipy import stats
    r = float(stats.pearsonr(arr[:, 0], arr[:, 1])[0])
    lab = (arr >= FEASIBLE_AT)
    # The ceiling is an estimate from a few hundred pairs, and it is used to
    # judge whether a model has run out of headroom -- so its own uncertainty
    # has to travel with it. A bootstrap over cells, not over rows: the two
    # readings of one cell are the unit.
    rng = np.random.default_rng(20260904)
    boot = np.empty(2000)
    for b in range(boot.size):
        pick = rng.integers(0, len(d), len(d))
        boot[b] = 1.0 - ((d[pick] ** 2).mean() / 2) / total_var
    # Are the repeated cells a fair sample of the screen? SI 2.2.15 says they
    # were randomly selected; if they were in fact enriched for hard or
    # ambiguous reactions the ceiling would be biased downwards, which is the
    # direction that would make a model look closer to saturation than it is.
    rep_mask = f.set_index(["acid", "amine", "cond"]).index.isin(two)
    return {
        "n_cells_repeated": int((sizes > 1).sum()),
        "n_pairs_used": int(len(arr)),
        "pearson": r,
        "sd_of_difference": float(d.std(ddof=1)),
        "within_cell_sd": float(np.sqrt(noise_var)),
        "between_row_sd": float(f["conversion"].std()),
        "noise_var": noise_var, "total_var": total_var,
        "r2_ceiling": 1.0 - noise_var / total_var,
        "r2_ceiling_ci_lo": float(np.percentile(boot, 2.5)),
        "r2_ceiling_ci_hi": float(np.percentile(boot, 97.5)),
        "repeated_mean_conversion": float(f.loc[rep_mask, "conversion"].mean()),
        "unrepeated_mean_conversion": float(f.loc[~rep_mask, "conversion"].mean()),
        "repeated_zero_fraction": float((f.loc[rep_mask, "conversion"] == 0).mean()),
        "unrepeated_zero_fraction": float((f.loc[~rep_mask, "conversion"] == 0).mean()),
        "both_exactly_zero": both_zero,
        "binary_label_disagreement": float((lab[:, 0] != lab[:, 1]).mean()),
        "binary_accuracy_ceiling": 1.0 - float((lab[:, 0] != lab[:, 1]).mean()) / 2,
    }


def audit(raw_dir: Path = DEFAULT_RAW_DIR,
          raw: pd.DataFrame | None = None) -> dict:
    """Everything a document or a test needs to know about the deposit.

    Generated, never hand-copied: ``docs/phase4_chemlex_dataset.md`` and
    ``docs/phase4_chemlex_mapping.md`` are written from this and CI diffs them.
    """
    df = load_raw(raw_dir) if raw is None else raw.copy()
    decoded = decode_conditions(df["Reagents"])
    out: dict = {
        "n_rows": len(df), "columns": list(df.columns),
        "conditions": decoded.to_dict(orient="records"),
        "n_reagents": int(decoded["reagent"].nunique()),
        "n_bases": int(decoded["base"].nunique()),
        "reagents": sorted(decoded["reagent"].unique()),
        "bases": sorted(decoded["base"].unique()),
        "n_solvents": int(df["Solvent"].nunique()),
        "solvents": sorted(df["Solvent"].unique()),
        "author_splits": {c: df[c].value_counts(dropna=False).to_dict()
                          for c in df.columns if "Split" in c},
        "screens": {},
    }
    for name in SCREENS:
        s = load_screen(name, raw=df)
        out["screens"][name] = {
            **s.notes, "n_rows": s.n_rows, "n_acids": s.n_acids,
            "n_amines": s.n_amines, "n_conditions": s.n_conditions,
            "condition_names": list(s.condition_names),
            "possible_pairs": s.n_acids * s.n_amines,
            "observed_pair_fraction": s.notes["n_pairs"] / (s.n_acids * s.n_amines),
            "replicate_noise": replicate_noise(s),
        }
    return out


def write_audit(path: Path, raw_dir: Path = DEFAULT_RAW_DIR) -> dict:
    a = audit(raw_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(a, indent=2, default=str) + "\n")
    return a


#: How the amine's nucleophilic N-H is classified, most specific first. The
#: screen is called an "acid-amine coupling", but 47 of the 230 amine entities
#: (2,471 rows, 21.2 % of the table) are not amines: sulfonamides, amidines and
#: guanidines, hydroxylamines, thioamides and thioureas, hydrazides, and
#: phosphoramides. Their recorded products acylate them correctly, so these are
#: not mislabels -- they are a deliberately broader nucleophile panel. But they
#: are not amide couplings, and pooling them assumes a single reactivity scale
#: across seven mechanisms. Reported, and the primary contrast is repeated on
#: the classical-amine subset as a registered sensitivity.
NUCLEOPHILE_CLASSES: tuple[tuple[str, str], ...] = (
    ("hydroxylamine", "[NX3;H1,H2]-[OX2]"),
    ("hydrazide_hydrazine", "[NX3;H1,H2][NX3]"),
    ("sulfonamide", "[NX3;H1,H2][SX4](=O)=O"),
    ("phosphoramide", "[NX3;H1,H2][PX4]"),
    ("thioamide_thiourea", "[NX3;H1,H2][CX3]=[SX1]"),
    ("amidine_guanidine", "[NX3;H1,H2][CX3]=[NX2]"),
    ("amide", "[NX3;H1,H2][CX3]=[OX1]"),
    ("amine", "[NX3;H1,H2;!$(N[C,S,P]=[O,S,N]);!$(N=*);!$([N+])]"),
)

#: The strict "this N-H is a plain amine" test. 195 of the 230 amine entities
#: match it; the remaining 35 offer only a sulfonamide, amidine, hydroxylamine,
#: thioamide, hydrazide, phosphoramide or amide N-H. The primary contrast is
#: repeated on the matching subset as a registered sensitivity.
CLASSICAL_AMINE = "[NX3;H1,H2;!$(N[C,S,P]=[O,S,N]);!$(N=*);!$([N+])]"


def is_classical_amine(smiles: str) -> bool:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse {smiles!r}")
    return mol.HasSubstructMatch(Chem.MolFromSmarts(CLASSICAL_AMINE))


def nucleophile_class(smiles: str) -> str:
    """Which kind of N-H this "amine" actually offers.

    First match wins, in :data:`NUCLEOPHILE_CLASSES` order, which runs from most
    specific to least. Anything matching nothing is ``"other"`` rather than
    being silently called an amine.
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse {smiles!r}")
    for name, smarts in NUCLEOPHILE_CLASSES:
        if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
            return name
    return "other"


def role_check(screen: Screen) -> dict:
    """Assert the screen really is what its column names claim.

    Every acid must carry exactly one carboxylic acid and every amine at least
    one N-H, because that is what makes the reaction site unambiguous and the
    bipartite decomposition meaningful. Both hold exactly on record 17596563 --
    272/272 acids with exactly one COOH, no amine with more than one
    nucleophilic N-H -- and a deposit revision that broke either would change
    what the interaction term is a term *of*.
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    cooh = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
    carboxylate = Chem.MolFromSmarts("[CX3](=O)[OX1-]")
    nh = Chem.MolFromSmarts("[NX3;H1,H2]")
    nucleophilic = Chem.MolFromSmarts(
        "[NX3;H1,H2;!$(N[C,S,P]=[O,S,N]);!$(N=*);!$([N+])]")

    n_cooh, bad_acids = [], []
    for s in screen.acids:
        m = Chem.MolFromSmiles(s)
        k = len(m.GetSubstructMatches(cooh)) + len(m.GetSubstructMatches(carboxylate))
        n_cooh.append(k)
        if k != 1:
            bad_acids.append({"smiles": s, "n_cooh": k})
    n_nh, n_nuc, bad_amines, classes = [], [], [], {}
    for s in screen.amines:
        m = Chem.MolFromSmiles(s)
        n_nh.append(len(m.GetSubstructMatches(nh)))
        n_nuc.append(len(m.GetSubstructMatches(nucleophilic)))
        if n_nh[-1] == 0:
            bad_amines.append({"smiles": s, "n_nh": 0})
        classes[s] = nucleophile_class(s)
    rows = screen.frame["amine_smiles"].map(classes)
    return {
        "n_acids": screen.n_acids, "n_amines": screen.n_amines,
        "acid_cooh_counts": {int(k): int(v) for k, v in
                             pd.Series(n_cooh).value_counts().sort_index().items()},
        "acids_failing_role": bad_acids,
        "amine_nh_counts": {int(k): int(v) for k, v in
                            pd.Series(n_nh).value_counts().sort_index().items()},
        "amine_nucleophilic_nh_counts": {
            int(k): int(v) for k, v in
            pd.Series(n_nuc).value_counts().sort_index().items()},
        "amines_failing_role": bad_amines,
        "nucleophile_class_entities": {
            k: int(v) for k, v in
            pd.Series(list(classes.values())).value_counts().items()},
        "nucleophile_class_rows": {k: int(v) for k, v in
                                   rows.value_counts().items()},
        "role_collisions": sorted(set(screen.acids) & set(screen.amines)),
    }
