#!/usr/bin/env python
"""Resolve the 100 Koplev drug labels against PubChem and ChEMBL.

Writes ``data/external/koplev_drug_mapping.csv`` plus a provenance file
recording the query date, the API base URLs, the number of cached responses and
a digest over the cache, so the mapping is reproducible from the repository even
if a database re-curates a record.

Every raw response is cached, so a second run is offline and free. Delete
``data/external/cache`` to force a refresh -- and expect the provenance digest
to change if the databases have moved.

    python scripts/prepare_phase3_drugs.py
    python scripts/prepare_phase3_drugs.py --refresh   # ignore the cache
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import requests

from intervention_algebra.real_data import koplev
from intervention_algebra.real_data.entity_ood import drugs as dm

#: PubChem and ChEMBL both ask for a contact in the User-Agent so they can reach
#: whoever is hammering their service. Read from the environment rather than
#: hard-coded: the value that used to sit here was a personal email address, and
#: a public repository is the wrong place to publish one. Set
#: ``INTERACTION_TRANSFER_CONTACT`` to your own address before a large run.
CONTACT = os.environ.get("INTERACTION_TRANSFER_CONTACT", "")
USER_AGENT = ("interaction-transfer/phase3 (research"
              + (f"; {CONTACT}" if CONTACT else "") + ")")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external"
MAPPING = OUT / "koplev_drug_mapping.csv"
PROVENANCE = OUT / "PROVENANCE_PHASE3.json"

FIELDS = [
    "drug_index", "label", "normalised_name", "pubchem_cid", "pubchem_parent_cid",
    "pubchem_title", "chembl_id", "chembl_parent_id", "inchikey", "chembl_inchikey",
    "molecular_formula", "smiles", "deposited_smiles", "chembl_smiles",
    "n_name_hits", "n_fragments", "discarded_fragments", "contains_metal",
    "name_in_synonyms", "databases_agree", "flags", "notes",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="delete the response cache first")
    args = ap.parse_args()

    cache_root = OUT / "cache"
    if args.refresh and cache_root.exists():
        shutil.rmtree(cache_root)

    # Drug order is the screen's own canonical order -- alphabetical over
    # normalised names -- so drug_index here is the same integer the Phase 2/2R
    # code assigns. Anything else would make the two phases silently disagree
    # about which drug is which.
    screen = koplev.load_screen("A375")
    labels = list(screen.drugs)
    other = list(koplev.load_screen("PANC1").drugs)
    if labels != other:
        raise SystemExit("the two screens do not share a drug list; mapping would be ambiguous")

    cache = dm.Cache(cache_root / "pubchem")
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    records = []
    for k, label in enumerate(labels):
        rec = dm.DrugRecord(label=label, normalised_name=dm.normalise_label(label))
        dm.resolve_pubchem(rec, cache, session)
        dm.resolve_chembl(rec, cache, session)
        dm.cross_check(rec)
        dm.apply_overrides(rec)
        records.append((k, rec))
        print(f"[{k:3d}] {label:28s} -> CID {rec.pubchem_cid} "
              f"{rec.inchikey} {'|'.join(rec.flags)}", flush=True)

    # Two labels resolving to one molecule would mean the screen is not 100
    # distinct drugs, and would put the "same" entity on both sides of an
    # entity-level split.
    seen: dict[str, str] = {}
    for _, rec in records:
        key = dm.connectivity(rec.inchikey)
        if key is None:
            continue
        if key in seen:
            rec.flags.append(f"blocker:duplicate-structure-with:{seen[key]}")
        else:
            seen[key] = rec.label

    OUT.mkdir(parents=True, exist_ok=True)
    tmp = MAPPING.with_suffix(".tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for k, rec in records:
            row = asdict(rec)
            row["drug_index"] = k
            row["flags"] = ";".join(rec.flags)
            w.writerow({f: row.get(f, "") for f in FIELDS})
    tmp.replace(MAPPING)

    digest = hashlib.sha256()
    for p in sorted(cache_root.rglob("*.json")):
        digest.update(p.name.encode())
        digest.update(p.read_bytes())

    PROVENANCE.write_text(json.dumps({
        "generated": date.today().isoformat(),
        "generator": "scripts/prepare_phase3_drugs.py",
        "sources": {
            "pubchem_pug_rest": dm.PUBCHEM,
            "chembl_web_services": dm.CHEMBL,
        },
        "salt_policy": "largest covalent fragment (RDKit), except that molecules "
                       "containing a metal or metalloid are kept whole; covalent "
                       "esters and prodrugs are one fragment and survive intact",
        "min_fragment_ratio": dm.MIN_FRAGMENT_RATIO,
        "n_labels": len(labels),
        "n_with_structure": sum(1 for _, r in records if r.smiles),
        "n_flagged": sum(1 for _, r in records if r.flags),
        "n_audit_overrides": sum(1 for _, r in records
                                 if "audited:structure-overridden" in r.flags),
        "audit_overrides": sorted(dm.AUDIT_OVERRIDES),
        "n_cached_responses": len(list(cache_root.rglob("*.json"))),
        "cache_sha256": digest.hexdigest(),
        "mapping_sha256": hashlib.sha256(MAPPING.read_bytes()).hexdigest(),
    }, indent=2) + "\n")

    flagged = [(r.label, r.flags) for _, r in records if r.flags]
    print(f"\n{len(records)} labels, {sum(1 for _, r in records if r.smiles)} with a structure, "
          f"{len(flagged)} flagged, cache hits {cache.hits} misses {cache.misses}")
    for label, flags in flagged:
        print(f"  {label:30s} {'; '.join(flags)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
