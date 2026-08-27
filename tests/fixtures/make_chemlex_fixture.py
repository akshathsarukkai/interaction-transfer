"""Regenerate the tiny ChemLex-schema fixture used by the Phase 4 tests.

    python tests/fixtures/make_chemlex_fixture.py

A 14-acid x 12-amine sparse screen in the deposit's exact nine-column schema, so
ingestion, split, feature, train, evaluate and report logic can be exercised in
CI without the 550 kB Zenodo file. It is **generated, not sampled**: the deposit
is CC BY-NC 4.0 and is not ours to redistribute, and a fixture that were a subset
of the real screen would make a test failure ambiguous between "the code broke"
and "the deposit changed".

The molecules are real, small and commercially trivial (acetic acid, benzoic
acid, aniline, morpholine ...) so RDKit exercises real chemistry rather than
toy strings, and the awkward cases the real deposit contains are reproduced on
purpose:

* a stereoisomer pair among the acids, so the split-group logic has something to
  merge;
* two amine SMILES that canonicalise identically, so the entity de-duplication
  fires;
* a substrate-salt prefix on the reagent string, so the stripping is exercised;
* both HATU depictions, so the merge is exercised;
* a replicate cell measured twice with different conversions, so the
  replicate-noise estimate has input;
* explicit unseen-acid and unseen-amine cases, which is the point of the phase.
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent / "chemlex_tiny"
NAME = "Chemlex_Acidamine_Wetlab_Data.xlsx"

ACIDS = [
    "CC(=O)O",                       # acetic acid
    "O=C(O)c1ccccc1",                # benzoic acid
    "O=C(O)c1ccc(Cl)cc1",            # 4-chlorobenzoic
    "O=C(O)c1ccc(OC)cc1",            # 4-methoxybenzoic
    "CCCC(=O)O",                     # butyric
    "OC(=O)C1CCCCC1",                # cyclohexanecarboxylic
    "O=C(O)Cc1ccccc1",               # phenylacetic
    "O=C(O)c1cccnc1",                # nicotinic
    "O=C(O)C(C)Cc1ccccc1",           # 2-methyl-3-phenylpropanoic, no stereo
    "O=C(O)[C@@H](C)Cc1ccccc1",      # ... and its (R) form: a split-group merge
    "O=C(O)CCC(=O)OC",               # mono-methyl succinate
    "O=C(O)c1ccc(C(F)(F)F)cc1",      # 4-CF3-benzoic
    "O=C(O)c1cc(Br)ccc1O",           # 5-bromosalicylic
    "O=C(O)CCCCC",                   # hexanoic
]
AMINES = [
    "Nc1ccccc1",                     # aniline
    "C1COCCN1",                      # morpholine
    "CCCCN",                         # butylamine
    "NC1CCCCC1",                     # cyclohexylamine
    "Nc1ccc(Cl)cc1",                 # 4-chloroaniline
    "CN(C)CCN",                      # N,N-dimethylethylenediamine
    "Nc1ccccn1",                     # 2-aminopyridine
    "C1CCNCC1",                      # piperidine
    "NCc1ccccc1",                    # benzylamine
    "OCCN",                          # ethanolamine
    "CC(N)C(=O)OC",                  # methyl alaninate, no stereo
    "N[C@@H](C)C(=O)OC",             # ... and its (S) form: a split-group merge
]
# One amine written a second way that canonicalises to the same molecule, so
# the entity de-duplication in load_screen has something to collapse.
AMINE_ALIASES = {"C1COCCN1": "O1CCNCC1"}

DIPEA = "CCN(C(C)C)C(C)C"
PF6 = "F[P-](F)(F)(F)(F)F"
HATU_O = "CN(C)C(On1nnc2cccnc21)=[N+](C)C"
HATU_N = "CN(C)C(n1n[n+]([O-])c2ncccc21)=[N+](C)C"
TCFH = "CN(C)C(Cl)=[N+](C)C"
NMI = "Cn1ccnc1"
DMF = "CN(C)C=O"

REAGENTS = [
    f"{DIPEA}.{HATU_O}.{PF6}",          # the dominant first-pass condition
    f"{DIPEA}.{HATU_N}.{PF6}",          # the same reagent, drawn the other way
    f"Cl.{DIPEA}.{HATU_N}.{PF6}",       # a substrate counterion, to be stripped
    f"{NMI}.{TCFH}.{PF6}",              # a second condition, second base
]


def _product(acid: str, amine: str) -> str:
    """Formal amide condensation, built with RDKit so the formula check passes."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    rxn = AllChem.ReactionFromSmarts(
        "[C:1](=[O:2])[OX2H1].[N;!$([N]=*);H1,H2:3]>>[C:1](=[O:2])[N:3]")
    prods = rxn.RunReactants((Chem.MolFromSmiles(acid), Chem.MolFromSmiles(amine)))
    mol = prods[0][0]
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol)


def build(seed: int = 0) -> None:
    import pandas as pd
    rng = np.random.default_rng(seed)
    HERE.mkdir(parents=True, exist_ok=True)

    # A latent additive-plus-interaction structure, so the fixture is not pure
    # noise and a smoke run produces finite, non-degenerate metrics.
    a_eff = rng.normal(size=len(ACIDS))
    n_eff = rng.normal(size=len(AMINES))
    za = rng.normal(size=(len(ACIDS), 2))
    zn = rng.normal(size=(len(AMINES), 2))
    c_eff = rng.normal(size=len(REAGENTS)) * 0.3

    rows = []
    for i, acid in enumerate(ACIDS):
        for j, amine in enumerate(AMINES):
            # Sparse: about half the cells, plus every acid and amine guaranteed
            # at least three partners so no entity is isolated.
            if rng.random() > 0.55 and (i + j) % 4:
                continue
            r = int(rng.integers(0, len(REAGENTS)))
            latent = a_eff[i] + n_eff[j] + c_eff[r] + float(za[i] @ zn[j])
            conv = float(np.clip(50 + 22 * latent + rng.normal(scale=8), 0, 100))
            if rng.random() < 0.3:
                conv = 0.0                       # the deposit's zero mass
            rows.append({
                "Acid": acid,
                "Amine": AMINE_ALIASES.get(amine, amine) if (i + j) % 7 == 0
                         else amine,
                "Reagents": REAGENTS[r], "Solvent": DMF,
                "Products": _product(acid, amine),
                "Conversion": round(conv, 2),
            })
    frame = pd.DataFrame(rows)
    # One replicate cell, measured twice with different conversions, so the
    # replicate-noise estimator has something to estimate.
    dup = frame.iloc[0].copy()
    dup["Conversion"] = round(float(dup["Conversion"]) + 17.5, 2)
    frame = pd.concat([frame, dup.to_frame().T], ignore_index=True)

    # The deposit's three split columns, so the ingestion sanity check has them.
    n = len(frame)
    rs = rng.random(n)
    frame["Random_Split"] = np.where(rs < 0.7, "train", "test")
    acid_code = frame["Acid"].map({a: k for k, a in enumerate(ACIDS)})
    frame["Stratified_Split_One_Unseen"] = np.where(acid_code < 10, "train", "test")
    amine_code = frame["Amine"].map({a: k for k, a in enumerate(AMINES)}).fillna(1)
    both = (acid_code >= 10) & (amine_code >= 8)
    neither = (acid_code < 10) & (amine_code < 8)
    frame["Stratified_Split_Both_Unseen"] = np.where(
        both, "test", np.where(neither, "train", None))

    frame.to_excel(HERE / NAME, index=False, sheet_name="Sheet1")
    print(f"{HERE / NAME}: {len(frame)} rows, "
          f"{frame['Acid'].nunique()} acids, {frame['Amine'].nunique()} amines, "
          f"{frame['Reagents'].nunique()} reagent strings")


def check() -> int:
    """Rebuild in a temporary directory and compare **content**, not bytes.

    An .xlsx is a zip archive and openpyxl stamps a creation time into it, so
    two builds of the identical table differ in a byte or two and never in a
    cell. CI's job here is to catch the fixture and its generator drifting
    apart; a byte comparison catches the clock instead, fails on every run, and
    trains everyone to ignore it.
    """
    import shutil
    import tempfile

    import pandas as pd
    global HERE
    committed = pd.read_excel(HERE / NAME, sheet_name=0)
    original, tmp = HERE, Path(tempfile.mkdtemp())
    try:
        HERE = tmp
        build()
        rebuilt = pd.read_excel(tmp / NAME, sheet_name=0)
    finally:
        HERE = original
        shutil.rmtree(tmp, ignore_errors=True)

    if list(committed.columns) != list(rebuilt.columns):
        print(f"columns differ:\n  committed {list(committed.columns)}"
              f"\n  rebuilt   {list(rebuilt.columns)}")
        return 1
    if len(committed) != len(rebuilt):
        print(f"row count differs: committed {len(committed)}, "
              f"rebuilt {len(rebuilt)}")
        return 1
    bad = []
    for col in committed.columns:
        a, b = committed[col], rebuilt[col]
        same = (a == b) | (a.isna() & b.isna())
        if not same.all():
            bad.append((col, int((~same).sum())))
    if bad:
        print("cells differ: " + ", ".join(f"{c} ({n})" for c, n in bad))
        return 1
    print(f"{NAME}: {len(committed)} rows, {len(committed.columns)} columns, "
          f"identical in content to a fresh build")
    return 0


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        raise SystemExit(check())
    build()
