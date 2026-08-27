"""Regenerate the tiny Koplev-schema fixture used by the Phase 2 tests.

    python tests/fixtures/make_fixture.py

The fixture is a complete ordered matrix over 12 synthetic drugs in the deposit's
exact column schema, so preprocessing, splitting and the model invariants can be
exercised in CI without downloading the real 1.7 MB deposit. It is generated,
not sampled from the real data: the real files are CC BY 4.0 and could be
redistributed, but a fixture that is a subset of the real screen would make a
test failure ambiguous between "the code broke" and "the data changed".
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent / "koplev_tiny"
N = 12


def build(seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    names = [f"Drugene {k:02d}" for k in range(N)]
    # Two whitespace oddities matching the real deposit, so the canonicaliser is
    # exercised: a double space and a semicolon-separated synonym.
    names[3] = "Drugene;  Ara-X"
    names[7] = "drugene  07"
    HERE.mkdir(parents=True, exist_ok=True)
    for table, tag in ((1, "A"), (2, "B")):
        u = rng.normal(size=(N, 4))
        S = 0.3 * (u @ u.T)                       # symmetric part
        K = rng.normal(size=(4, 4)); K = K - K.T
        A = 0.3 * (u @ K @ u.T)                   # antisymmetric part
        Y = S + A + rng.normal(scale=0.02, size=(N, N))
        lam = np.clip(rng.normal(size=(N, N)), -1, 1)
        rows = ["first_compound,second_compound,lambda,synergy_measure"]
        for i in range(N):
            for j in range(N):
                rows.append(f"{names[i]},{names[j]},{lam[i, j]:.9f},{Y[i, j]:.9f}")
        (HERE / f"Data Table {table}.csv").write_text("\n".join(rows) + "\n")
    # Table 4 stands in for the deposit's mislabelled duplicate: same
    # synergy_measure as table 2, plus a posterior SD column.
    src = (HERE / "Data Table 2.csv").read_text().strip().splitlines()
    out = [src[0] + ",synergy_sd"]
    for line in src[1:]:
        out.append(f"{line},{abs(rng.normal(scale=0.03)):.9f}")
    (HERE / "Data Table 4.csv").write_text("\n".join(out) + "\n")
    print(f"wrote fixture to {HERE}")


#: Twelve real, small, unambiguous molecules -- one per fixture drug. Real SMILES
#: rather than random strings so RDKit's parser, the fingerprint generator and
#: the Tanimoto geometry are all genuinely exercised in CI; small and famous so
#: nobody has to look them up to see whether the fixture is sane. They are NOT a
#: subset of the Koplev compounds, for the same reason the response matrix is
#: generated: a fixture that overlaps the real data makes a failure ambiguous.
FIXTURE_SMILES = [
    "CCO",                      # ethanol
    "c1ccccc1",                 # benzene
    "CC(=O)Oc1ccccc1C(=O)O",    # aspirin
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",   # caffeine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",     # ibuprofen
    "OCC1OC(O)C(O)C(O)C1O",     # glucose
    "c1ccc2ccccc2c1",           # naphthalene
    "CC(N)C(=O)O",              # alanine
    "Oc1ccccc1",                # phenol
    "ClCCl",                    # dichloromethane
    "CC(=O)Nc1ccc(O)cc1",       # paracetamol
    "C1CCCCC1",                 # cyclohexane
]


def build_mapping() -> None:
    """A Phase 3 drug mapping for the fixture, in the real CSV's schema.

    Lets the entity-OOD pipeline -- map, split entities, featurise, train,
    evaluate, report -- run end to end in CI with no network and no external
    database.
    """
    import csv

    names = _fixture_names()
    rows = []
    for k, (name, smi) in enumerate(zip(names, FIXTURE_SMILES)):
        rows.append({
            "drug_index": k, "label": name, "normalised_name": name.split(";")[0].strip(),
            "pubchem_cid": 1000 + k, "pubchem_parent_cid": 1000 + k,
            "pubchem_title": f"Fixture {k}", "chembl_id": f"CHEMBLFIX{k}",
            "chembl_parent_id": f"CHEMBLFIX{k}", "inchikey": f"FIXTURE{k:07d}-AAAAAAAAA-N",
            "chembl_inchikey": f"FIXTURE{k:07d}-AAAAAAAAA-N",
            "molecular_formula": "", "smiles": smi, "deposited_smiles": smi,
            "chembl_smiles": smi, "n_name_hits": 1, "n_fragments": 1,
            "discarded_fragments": "", "contains_metal": False,
            "name_in_synonyms": True, "databases_agree": True, "flags": "",
            "notes": "generated fixture",
        })
    out = HERE / "drug_mapping.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote fixture mapping to {out}")


def _fixture_names() -> list[str]:
    """The canonical drug order the ingestion assigns to the fixture.

    Must match ``koplev.load_screen`` exactly -- sorted, whitespace-collapsed --
    or ``drug_index`` in the mapping would point at the wrong molecule, which is
    Control B by accident.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from intervention_algebra.real_data import koplev

    return list(koplev.load_screen("A375", HERE).drugs)


if __name__ == "__main__":
    build()
    build_mapping()
