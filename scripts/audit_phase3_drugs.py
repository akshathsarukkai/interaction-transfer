#!/usr/bin/env python
"""Regenerate the mapping-audit statistics in ``docs/phase3_drug_mapping.md``.

Every number in that document comes from here, computed from the committed
mapping. None is transcribed from an auditor's report -- an independent audit is
evidence about the mapping, not a source of numbers to copy, and the mapping has
changed since the audit ran (three structures were overridden as a result of it).

    python scripts/audit_phase3_drugs.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from intervention_algebra.real_data.entity_ood import drugs as dm
from intervention_algebra.real_data.entity_ood import features as feat
from intervention_algebra.real_data.entity_ood import report

ROOT = Path(__file__).resolve().parents[1]
MAPPING = ROOT / "data" / "external" / "koplev_drug_mapping.csv"
TARGETS = ROOT / "data" / "external" / "koplev_drug_targets.csv"
DOC = ROOT / "docs" / "phase3_drug_mapping.md"
OUT = ROOT / "results" / "phase3_entity_ood" / "summary"


def stats() -> dict:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    m = pd.read_csv(MAPPING).sort_values("drug_index").reset_index(drop=True)
    flags = [str(f) if isinstance(f, str) else "" for f in m["flags"]]
    f = feat.fingerprint_matrix(m)
    sim = feat.tanimoto_matrix(f)
    tri = sim[np.triu_indices(len(m), 1)]
    maxsim = np.nanmax(sim, axis=1)

    # Every recorded InChIKey recomputed rather than trusted.
    recomputed = [dm.inchikey_of(s) for s in m["smiles"]]
    key_ok = sum(1 for a, b in zip(recomputed, m["inchikey"]) if a == b)

    desalted = [(r["label"], r["discarded_fragments"]) for _, r in m.iterrows()
                if isinstance(r["discarded_fragments"], str) and r["discarded_fragments"]]
    ratios = []
    for _, r in m.iterrows():
        d = r["discarded_fragments"]
        if not isinstance(d, str) or not d:
            continue
        kept = Chem.MolFromSmiles(r["smiles"]).GetNumHeavyAtoms()
        for frag in d.split("."):
            mol = Chem.MolFromSmiles(frag)
            if mol is not None and mol.GetNumHeavyAtoms():
                ratios.append(kept / mol.GetNumHeavyAtoms())

    pairs_high = []
    for a in range(len(m)):
        for b in range(a + 1, len(m)):
            if sim[a, b] >= 0.7:
                pairs_high.append((float(sim[a, b]), m["label"][a], m["label"][b]))
    pairs_high.sort(reverse=True)

    out = {
        "n_drugs": len(m),
        "n_unique_connectivity": int(m["inchikey"].str.slice(0, 14).nunique()),
        "n_unique_full_key": int(m["inchikey"].nunique()),
        "inchikey_recomputed_matches": key_ok,
        "n_name_hits_all_unique": int((m["n_name_hits"] == 1).sum()),
        "n_label_exact_synonym": int(m["name_in_synonyms"].sum()),
        "n_desalted": len(desalted),
        "min_fragment_ratio_observed": round(min(ratios), 2) if ratios else None,
        "n_metal": int(m["contains_metal"].sum()),
        "n_overridden": sum(1 for f in flags if "audited:structure-overridden" in f),
        "overridden": [r["label"] for _, r in m.iterrows()
                       if "audited:structure-overridden" in str(r["flags"])],
        "flag_census": dict(Counter(x for f in flags for x in f.split(";") if x)),
        "bits_min": int(f.bits_set.min()), "bits_median": float(np.median(f.bits_set)),
        "bits_mean": round(float(f.bits_set.mean()), 1), "bits_max": int(f.bits_set.max()),
        "lowest_bits": sorted(zip(f.bits_set.tolist(), f.labels))[:8],
        "n_identical_fingerprints": int(sum(
            1 for a in range(len(m)) for b in range(a + 1, len(m))
            if np.array_equal(f.x[a], f.x[b]))),
        "tanimoto_median": round(float(np.median(tri)), 4),
        "tanimoto_p95": round(float(np.quantile(tri, 0.95)), 4),
        "tanimoto_max": round(float(tri.max()), 4),
        "n_pairs_ge_0.7": int((tri >= 0.7).sum()),
        "n_pairs_ge_0.9": int((tri >= 0.9).sum()),
        "n_pairs_exactly_zero": int((tri == 0).sum()),
        "pairs_ge_0.7": [(round(s, 4), a, b) for s, a, b in pairs_high],
        "maxsim_min": round(float(maxsim.min()), 4),
        "maxsim_median": round(float(np.median(maxsim)), 4),
        "maxsim_max": round(float(maxsim.max()), 4),
        "n_drugs_maxsim_ge_0.7": int((maxsim >= 0.7).sum()),
        "n_drugs_maxsim_ge_0.9": int((maxsim >= 0.9).sum()),
        "maxsim_q33_current": round(float(np.quantile(maxsim, 1 / 3)), 4),
        "maxsim_q66_current": round(float(np.quantile(maxsim, 2 / 3)), 4),
    }
    if TARGETS.exists():
        t = pd.read_csv(TARGETS)
        counts = Counter(t["target_chembl_id"])
        shared = {k for k, c in counts.items() if c >= 2}
        annotated = set(t["drug_index"])
        orphan = sorted(set(m["label"]) - {r["label"] for _, r in t.iterrows()
                                           if r["target_chembl_id"] in shared})
        out["targets"] = {
            "n_rows": len(t), "n_drugs_annotated": len(annotated),
            "n_targets": len(counts), "n_targets_shared": len(shared),
            "n_drugs_uninformative": len(orphan), "uninformative": orphan,
        }
    return out


def main() -> int:
    s = stats()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mapping_stats.json").write_text(json.dumps(s, indent=2, default=str) + "\n")

    blocks = {
        "mapping_summary": _summary_table(s),
        "analogues": _analogue_table(s),
        "flags": _flag_table(s),
    }
    replaced = report.inject_blocks(DOC, blocks)
    missing = sorted(set(blocks) - set(replaced))
    if DOC.exists() and missing:
        raise SystemExit(f"{DOC.name} has no marker for generated blocks: {missing}")
    print(json.dumps({k: v for k, v in s.items()
                      if not isinstance(v, (list, dict))}, indent=1))
    return 0


def _summary_table(s: dict) -> str:
    rows = [
        ("labels resolved", f"{s['n_drugs']}/100"),
        ("distinct InChIKey connectivity blocks", f"{s['n_unique_connectivity']}/100"),
        ("recorded InChIKey equals RDKit's for the stored SMILES",
         f"{s['inchikey_recomputed_matches']}/100"),
        ("PubChem name search returned exactly one CID", f"{s['n_name_hits_all_unique']}/100"),
        ("label is an exact PubChem synonym of the resolved record",
         f"{s['n_label_exact_synonym']}/100"),
        ("rows desalted", f"{s['n_desalted']}/100"),
        ("smallest kept:discarded heavy-atom ratio observed",
         f"{s['min_fragment_ratio_observed']}x (rule: >= {dm.MIN_FRAGMENT_RATIO}x)"),
        ("metal / metalloid compounds kept whole", str(s["n_metal"])),
        ("structures overridden by the audit",
         f"{s['n_overridden']} ({', '.join(s['overridden'])})"),
        ("ECFP4 bits set: min / median / max",
         f"{s['bits_min']} / {s['bits_median']:.1f} / {s['bits_max']}"),
        ("pairs of drugs with identical fingerprints", str(s["n_identical_fingerprints"])),
        ("pairwise Tanimoto: median / 95th / max",
         f"{s['tanimoto_median']} / {s['tanimoto_p95']} / {s['tanimoto_max']}"),
        ("drugs whose nearest neighbour is >= 0.7 / >= 0.9",
         f"{s['n_drugs_maxsim_ge_0.7']} / {s['n_drugs_maxsim_ge_0.9']}"),
    ]
    return "\n".join(["| check | value |", "|---|---|"]
                     + [f"| {a} | {b} |" for a, b in rows])


def _analogue_table(s: dict) -> str:
    rows = [f"| {a} | {b} | {t:.3f} |" for t, a, b in s["pairs_ge_0.7"]]
    return "\n".join(["| drug | drug | ECFP4 Tanimoto |", "|---|---|---:|"] + rows)


def _flag_table(s: dict) -> str:
    rows = [f"| `{k}` | {v} |" for k, v in sorted(s["flag_census"].items())]
    return "\n".join(["| flag | rows |", "|---|---:|"] + rows)


if __name__ == "__main__":
    raise SystemExit(main())
