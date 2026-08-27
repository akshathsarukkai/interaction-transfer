#!/usr/bin/env python
"""Fetch curated ChEMBL mechanism targets for the mapped Koplev drugs.

Writes ``data/external/koplev_drug_targets.csv``: one row per (drug, target),
with the mechanism-of-action text and organism kept for audit. Cached like the
structure mapping, so a second run is offline.

    python scripts/prepare_phase3_targets.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import requests

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
TARGETS = OUT / "koplev_drug_targets.csv"
PROVENANCE = OUT / "PROVENANCE_PHASE3_TARGETS.json"

FIELDS = ["drug_index", "label", "parent_chembl_id", "target_chembl_id",
          "target_name", "organism", "mechanism_of_action", "action_type"]


def main() -> int:
    import pandas as pd

    mapping = pd.read_csv(OUT / "koplev_drug_mapping.csv").sort_values("drug_index")
    cache = dm.Cache(OUT / "cache" / "chembl_mechanism")
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    rows, skipped = [], []
    for _, r in mapping.iterrows():
        cid = r["chembl_parent_id"] if isinstance(r["chembl_parent_id"], str) else r["chembl_id"]
        if not isinstance(cid, str):
            skipped.append((r["label"], "no ChEMBL id"))
            continue
        body = cache.get(
            f"{dm.CHEMBL}/mechanism.json?parent_molecule_chembl_id={cid}&limit=200", session)
        mechs = (body or {}).get("mechanisms", [])
        kept = 0
        for mech in mechs:
            tid = mech.get("target_chembl_id")
            if not tid:
                continue
            tbody = cache.get(f"{dm.CHEMBL}/target/{tid}.json", session) or {}
            organism = tbody.get("organism")
            # Human targets only. A record with no organism at all is kept and
            # flagged rather than dropped -- ChEMBL leaves the field empty for
            # non-protein targets like DNA, which is the annotated mechanism of
            # a large fraction of this deposit's cytotoxics.
            if organism not in (None, "", "Homo sapiens"):
                continue
            rows.append({
                "drug_index": int(r["drug_index"]), "label": r["label"],
                "parent_chembl_id": cid, "target_chembl_id": tid,
                "target_name": tbody.get("pref_name"), "organism": organism,
                "mechanism_of_action": mech.get("mechanism_of_action"),
                "action_type": mech.get("action_type"),
            })
            kept += 1
        if not kept:
            skipped.append((r["label"], f"{len(mechs)} mechanisms, none usable"))

    tmp = TARGETS.with_suffix(".tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(TARGETS)

    n_drugs = len({r["drug_index"] for r in rows})
    n_targets = len({r["target_chembl_id"] for r in rows})
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["target_chembl_id"]] = counts.get(r["target_chembl_id"], 0) + 1
    shared = {t for t, c in counts.items() if c >= 2}
    orphans = sorted({r["label"] for r in rows} -
                     {r["label"] for r in rows if r["target_chembl_id"] in shared})
    unannotated = [lbl for lbl, _ in skipped]

    PROVENANCE.write_text(json.dumps({
        "generated": date.today().isoformat(),
        "generator": "scripts/prepare_phase3_targets.py",
        "source": f"{dm.CHEMBL}/mechanism (curated drug mechanism of action)",
        "query_field": "parent_molecule_chembl_id",
        "organism_filter": "Homo sapiens, or unset (non-protein targets such as DNA)",
        "n_rows": len(rows), "n_drugs_annotated": n_drugs, "n_targets": n_targets,
        "n_targets_shared_by_2plus_drugs": len(shared),
        "drugs_with_no_shared_target": orphans,
        "drugs_with_no_annotation": unannotated,
        "targets_sha256": hashlib.sha256(TARGETS.read_bytes()).hexdigest(),
    }, indent=2) + "\n")

    print(f"{len(rows)} rows, {n_drugs}/100 drugs annotated, {n_targets} distinct targets, "
          f"{len(shared)} shared by 2+ drugs")
    print(f"{len(orphans)} annotated drugs have no target shared with another drug")
    print(f"{len(unannotated)} drugs have no usable annotation: {unannotated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
