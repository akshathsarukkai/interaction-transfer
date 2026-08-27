"""Resolution of the 100 Koplev drug labels to authoritative chemical identities.

Why this module is the highest-risk part of Phase 3
---------------------------------------------------
Phase 3 asks whether a drug's *structure* predicts its interaction behaviour. If
a label is mapped to the wrong molecule, the experiment does not become noisy in
a way that shows up as a wide confidence interval -- it silently answers a
different question, and a null result becomes unfalsifiable ("maybe the mapping
was bad"). So every mapping decision here is made by a rule stated in advance and
applied uniformly, never by inspection of the compound, and the two databases are
queried independently so that agreement is evidence rather than assumption.

The deposit's labels are the compound names of the **NCI DTP Approved Oncology
Drugs Set IV** (Developmental Therapeutics Program, NCI/DCTD), which is what
Koplev et al. state they screened and what ``docs/phase2_dataset.md`` already
records; several strings carry a vendor-style salt suffix -- ``Erlotinib HCl``,
``Imatinib Mesylate``, ``Cytarabine;  Ara-C``. An earlier version of this
docstring called them Selleck catalogue names, which pointed a reader at the
wrong reference list. Three things have to be decided.

**1. Synonym annotations.** Four labels carry a second name after a semicolon or
in parentheses (``Cytarabine;  Ara-C``, ``Lomustine;  CCNU``,
``Mitotane;  o;p'-DDD``, ``Fluorouracil  (5-FU)``, ``Sirolimus (Rapamycin)``).
The rule: take the text before the first ``;``, delete parenthesised groups,
collapse whitespace. Nothing else is removed. This is
:func:`normalise_label`, and it is the *only* string surgery performed.

**2. Salts.** Roughly a fifth of the labels name a salt. The pharmacologically
active species is the parent, the counterion carries no information about the
drug's biology, and a fingerprint computed on the salt would encode "this drug
came as a hydrochloride" as though it were a structural feature -- a formulation
fact that correlates with drug class in ways that could manufacture apparent
transfer.

The salt is removed by **keeping the largest covalently connected fragment**
(RDKit's ``LargestFragmentChooser``, by heavy-atom count), with one exception
stated below. Two things this is *not*:

*It is not a regex on the label.* ``Pemetrexed Disodium`` is a salt and must lose
its sodiums, while ``Fludarabine Phosphate``, ``Estramustine phosphate`` and
``Megestrol acetate`` are covalent esters and prodrugs whose "salt-looking"
suffix must be **kept** -- the screen dosed those molecules, and they are
different compounds from their alcohols. A fragment rule gets this right for free:
a covalent ester is one fragment and survives intact.

*It is not PubChem's ``cids_type=parent`` relation*, which was tried first and
fails in two ways that a fingerprint experiment cannot tolerate. On
**carboplatin** it returns 1,1-cyclobutanedicarboxylic acid -- it discards the
platinum and hands back the ligand, so the drug would have entered the experiment
as an unrelated small diacid. On **tamoxifen citrate** it returns the citrate
salt unchanged, because that record has no registered parent, so the fingerprint
would have carried citric acid. Both failures are silent. The fragment rule is
used instead precisely because its failure mode is inspectable: the discarded
fragments are recorded, and :func:`largest_fragment` refuses to discard a
fragment that is not small relative to what it keeps.

**2b. Metals.** The exception. Four compounds -- cisplatin, carboplatin,
oxaliplatin and arsenic trioxide -- are coordination complexes or inorganic
salts, and in these the "counterion" is the pharmacology. RDKit writes cisplatin
as ``N.N.Cl[Pt]Cl``: three fragments, none of which is the drug. So **no fragment
is ever discarded from a molecule containing a metal or metalloid**; the
deposited record is used whole. This keeps the four compounds chemically correct,
but it does not make a Morgan fingerprint a good description of them, and that is
a separate problem recorded as a representation caveat rather than papered over:
their entity-OOD results are reported both in and out of the primary set, under a
rule fixed before any model was fitted.

**3. Stereochemistry.** The isomeric SMILES is recorded and used. Where PubChem's
parent is a defined stereoisomer, that is what is fingerprinted; where the drug
is genuinely racemic or undefined (thalidomide, lenalidomide) the parent record
is undefined too and the fingerprint is the flat structure. Morgan fingerprints
with ``useChirality=False`` -- the setting used here -- ignore the distinction
either way, so this affects the audit trail rather than the features. It is
recorded so the choice is visible rather than accidental.

Verification, not trust
-----------------------
Each label is resolved twice, independently:

* PubChem, by name, then by parent relation, then properties and synonyms;
* ChEMBL, by preferred name, then by parent relation via ``molecule_hierarchy``.

The two are compared on the InChIKey **connectivity block** (the first 14
characters, which encode the heavy-atom skeleton and ignore protonation and
stereochemistry). Agreement on that block from two independently curated
databases is strong evidence the label was understood the same way by both.
Disagreement is recorded as a flag and audited by hand, never auto-resolved.

Three further automatic checks run on every mapping:

* the normalised label must appear in the PubChem synonym list for the resolved
  parent CID (case-insensitively), which catches a name search that drifted to a
  relevance-ranked near-match;
* the name query must not have been ambiguous at the top -- if PubChem returns
  several CIDs the count is recorded;
* two different labels must not resolve to the same InChIKey, which would mean
  the screen's 100 "drugs" are not 100 distinct molecules.

What is deliberately *not* done
-------------------------------
No fuzzy string matching, no vendor catalogue pages, no manual override of a
database answer to make a compound "look right". Where the databases cannot
settle a label, the drug is flagged and its fate is decided in
``docs/phase3_drug_mapping.md`` before any model is fitted.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"

#: PubChem asks for no more than 5 requests/second. One request per 250 ms is
#: comfortably inside that and the whole run is ~100 drugs, so politeness costs
#: under a minute.
REQUEST_INTERVAL_S = 0.25

_PAREN = re.compile(r"\([^)]*\)")
_WS = re.compile(r"\s+")


def normalise_label(label: str) -> str:
    """The one string transformation applied to a deposit label.

    Text before the first ``;`` (the deposit's synonym separator), with
    parenthesised groups deleted and whitespace collapsed. Salt words are
    **not** removed -- see the module docstring for why that is delegated to
    PubChem's parent relation instead.

    >>> normalise_label("Cytarabine;  Ara-C")
    'Cytarabine'
    >>> normalise_label("Fluorouracil  (5-FU)")
    'Fluorouracil'
    >>> normalise_label("Erlotinib HCl")
    'Erlotinib HCl'
    """
    head = label.split(";", 1)[0]
    head = _PAREN.sub(" ", head)
    return _WS.sub(" ", head).strip()


def connectivity(inchikey: str | None) -> str | None:
    """The skeleton block of an InChIKey: heavy-atom connectivity only.

    The first 14 characters. The second block encodes stereochemistry and
    isotopes, the final character protonation state; comparing only the first
    block is what makes "PubChem and ChEMBL agree about which molecule this is"
    a statement about the molecule rather than about two curators' salt
    conventions.
    """
    if not inchikey:
        return None
    return inchikey.split("-")[0]


@dataclass
class DrugRecord:
    """One resolved label. Every field is either from a database or a flag."""

    label: str
    normalised_name: str
    pubchem_cid: int | None = None
    pubchem_parent_cid: int | None = None
    pubchem_title: str | None = None
    deposited_smiles: str | None = None
    smiles: str | None = None
    inchikey: str | None = None
    molecular_formula: str | None = None
    chembl_id: str | None = None
    chembl_parent_id: str | None = None
    chembl_smiles: str | None = None
    chembl_inchikey: str | None = None
    n_name_hits: int = 0
    n_fragments: int = 0
    discarded_fragments: str = ""
    contains_metal: bool = False
    name_in_synonyms: bool = False
    databases_agree: bool = False
    flags: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def usable(self) -> bool:
        """A record enters the experiment only with a structure and no blocker.

        ``blocker:`` flags are set by the audit, not by the resolver; the
        resolver only records what it found.
        """
        return self.smiles is not None and not any(
            f.startswith("blocker:") for f in self.flags
        )


class Cache:
    """On-disk cache of every raw API response, keyed by URL digest.

    External databases are the one input to this project that can change under
    it without any commit. Caching the raw JSON means the mapping is
    reproducible from the repository even if PubChem re-curates a record, and
    the digest of the cache directory is recorded in the provenance file so a
    silent change is detectable rather than invisible.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, url: str) -> Path:
        return self.root / (hashlib.sha256(url.encode()).hexdigest()[:32] + ".json")

    def get(self, url: str, session: Any, timeout: int = 30) -> Any:
        """Return the parsed body for ``url``, fetching only on a cache miss.

        A 404 from PubChem means "no such name", which is an answer and is
        cached as ``None``; anything else is raised so a transient outage is not
        silently recorded as a missing drug.
        """
        p = self._path(url)
        if p.exists():
            self.hits += 1
            payload = json.loads(p.read_text())
            return payload["body"]
        self.misses += 1
        time.sleep(REQUEST_INTERVAL_S)
        resp = session.get(url, timeout=timeout, headers={"Accept": "application/json"})
        if resp.status_code == 404:
            body = None
        elif resp.status_code == 200:
            body = resp.json()
        else:
            resp.raise_for_status()
            body = None
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"url": url, "status": resp.status_code, "body": body}, indent=1))
        tmp.replace(p)
        return body


def _pubchem_properties(cid: int, cache: Cache, session: Any) -> dict[str, Any]:
    url = (
        f"{PUBCHEM}/compound/cid/{cid}/property/"
        "SMILES,ConnectivitySMILES,InChIKey,MolecularFormula,Title/JSON"
    )
    body = cache.get(url, session)
    if not body:
        return {}
    props = body.get("PropertyTable", {}).get("Properties", [])
    return props[0] if props else {}


def resolve_pubchem(rec: DrugRecord, cache: Cache, session: Any) -> None:
    """Fill the PubChem half of ``rec`` in place.

    Name -> CID -> properties for **that** CID -> desalt by largest fragment ->
    synonyms. PubChem's own ``parent`` CID is fetched and recorded as a
    cross-check, but is deliberately not the structure used; see the module
    docstring for the two ways it fails.
    """
    name = rec.normalised_name
    body = cache.get(f"{PUBCHEM}/compound/name/{_quote(name)}/cids/JSON", session)
    cids = (body or {}).get("IdentifierList", {}).get("CID", [])
    if not cids:
        rec.flags.append("blocker:pubchem-name-not-found")
        return
    rec.n_name_hits = len(cids)
    if len(cids) > 1:
        rec.flags.append(f"ambiguous:pubchem-returned-{len(cids)}-cids")
    rec.pubchem_cid = int(cids[0])

    parent_body = cache.get(
        f"{PUBCHEM}/compound/cid/{rec.pubchem_cid}/cids/JSON?cids_type=parent", session
    )
    parents = (parent_body or {}).get("IdentifierList", {}).get("CID", [])
    rec.pubchem_parent_cid = int(parents[0]) if parents else None

    props = _pubchem_properties(rec.pubchem_cid, cache, session)
    rec.deposited_smiles = props.get("SMILES")
    rec.molecular_formula = props.get("MolecularFormula")
    rec.pubchem_title = props.get("Title")
    if rec.deposited_smiles is None:
        rec.flags.append("blocker:pubchem-no-structure")
        return

    kept, discarded, meta = largest_fragment(rec.deposited_smiles)
    rec.smiles = kept
    rec.n_fragments = meta["n_fragments"]
    rec.contains_metal = meta["contains_metal"]
    rec.discarded_fragments = ".".join(discarded)
    rec.flags.extend(meta["flags"])
    rec.inchikey = inchikey_of(kept) if kept else None
    if rec.inchikey is None:
        rec.flags.append("blocker:rdkit-cannot-parse-structure")

    syn_body = cache.get(f"{PUBCHEM}/compound/cid/{rec.pubchem_cid}/synonyms/JSON", session)
    syns = (syn_body or {}).get("InformationList", {}).get("Information", [{}])
    synonyms = [s.lower() for s in (syns[0].get("Synonym", []) if syns else [])]
    probe = name.lower()
    rec.name_in_synonyms = any(probe == s for s in synonyms[:600])
    if not rec.name_in_synonyms:
        rec.flags.append("check:label-not-an-exact-pubchem-synonym")


_SALT_WORDS = (
    "hcl", "hydrochloride", "mesylate", "malate", "citrate", "tartrate",
    "sulfate", "sulphate", "ditosylate", "tosylate", "disodium", "sodium",
    "maleate", "succinate", "besylate", "fumarate", "dihydrochloride",
)


def _salt_probe(name: str) -> str:
    """``name`` with one trailing salt word removed, for synonym comparison only.

    Never used to build a query or to choose a structure -- those go through
    PubChem's parent relation. This exists because "Erlotinib HCl" is correctly
    *not* a synonym of parent CID 176870, and without it every salt would raise
    a spurious flag and drown the real ones.
    """
    parts = name.split()
    if len(parts) > 1 and parts[-1].lower().strip(".") in _SALT_WORDS:
        return " ".join(parts[:-1])
    return name


def resolve_chembl(rec: DrugRecord, cache: Cache, session: Any) -> None:
    """Fill the ChEMBL half of ``rec`` in place, independently of PubChem.

    ChEMBL is queried by *preferred name*, which is the INN, so the salt word is
    dropped for the query (ChEMBL's preferred names are parent INNs) and the
    ``molecule_hierarchy`` parent relation is then followed -- mirroring the
    desalting without borrowing PubChem's answer. The same
    :func:`largest_fragment` rule is applied to ChEMBL's structure, so the two
    databases are compared after identical treatment and a difference is a
    difference about the molecule.
    """
    probe = _salt_probe(rec.normalised_name)
    body = cache.get(
        f"{CHEMBL}/molecule.json?pref_name__iexact={_quote(probe.upper())}&limit=5", session
    )
    mols = (body or {}).get("molecules", [])
    if not mols:
        rec.flags.append("check:chembl-name-not-found")
        return
    mol = mols[0]
    rec.chembl_id = mol.get("molecule_chembl_id")
    hierarchy = mol.get("molecule_hierarchy") or {}
    rec.chembl_parent_id = hierarchy.get("parent_chembl_id") or rec.chembl_id
    if rec.chembl_parent_id and rec.chembl_parent_id != rec.chembl_id:
        pbody = cache.get(f"{CHEMBL}/molecule/{rec.chembl_parent_id}.json", session)
        mol = pbody or mol
    structures = mol.get("molecule_structures") or {}
    smi = structures.get("canonical_smiles")
    if not smi:
        rec.flags.append("check:no-chembl-structure")
        return
    kept, _, _ = largest_fragment(smi)
    rec.chembl_smiles = kept
    rec.chembl_inchikey = inchikey_of(kept) if kept else structures.get("standard_inchi_key")


def cross_check(rec: DrugRecord) -> None:
    """Compare the two independent resolutions on heavy-atom connectivity.

    A disagreement is a flag, never an auto-resolution: two curated databases
    holding different skeletons for one drug name is exactly the situation where
    a program guessing which is right would install a silent error.
    """
    a, b = connectivity(rec.inchikey), connectivity(rec.chembl_inchikey)
    if a and b:
        rec.databases_agree = a == b
        if not rec.databases_agree:
            rec.flags.append("check:pubchem-chembl-disagree")


def _quote(s: str) -> str:
    from urllib.parse import quote

    return quote(s, safe="")


#: Elements whose presence means the record is a coordination complex or an
#: inorganic salt, and that fragment-based desalting must therefore not touch.
#: Cisplatin is written ``N.N.Cl[Pt]Cl``; every one of its three fragments is a
#: counterion by the usual heuristics, and the drug is none of them. Arsenic is
#: included -- arsenic trioxide is ``[O-2].[O-2].[O-2].[As+3].[As+3]``, five
#: fragments and no organic part at all.
METAL_SYMBOLS = frozenset({
    "Pt", "As", "Au", "Ag", "Zn", "Fe", "Cu", "Ru", "Gd", "Tc", "Ga", "Sb",
    "Bi", "Ti", "V", "Cr", "Mn", "Co", "Ni", "Cd", "Hg", "Pb", "Sn", "Al",
    "Ba", "Sr", "La", "Lu", "Y", "Re", "Rh", "Pd", "Ir", "Os",
})

#: A fragment may be discarded only if the kept fragment is at least this many
#: times larger by heavy-atom count. Every real counterion in this deposit is
#: far below the line (citrate is 13 heavy atoms against tamoxifen's 32, ratio
#: 2.5); anything near 1 would mean the record is a genuine two-component
#: molecule and "largest fragment" is not a salt rule but a coin flip.
MIN_FRAGMENT_RATIO = 2.0


def _mol(smiles: str):
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    return Chem.MolFromSmiles(smiles)


def largest_fragment(smiles: str) -> tuple[str | None, list[str], dict[str, Any]]:
    """Desalt ``smiles`` by keeping its largest covalent fragment.

    Returns ``(kept_smiles, discarded_smiles, meta)``. ``meta`` carries the
    fragment count, whether a metal was seen, and any flags raised.

    Two rules stop this from being a blunt instrument:

    * a molecule containing a metal or metalloid is returned **whole**, because
      in a coordination complex the separate fragments are the drug;
    * a fragment is discarded only if what is kept is at least
      :data:`MIN_FRAGMENT_RATIO` times larger by heavy-atom count. Below that,
      the record is not a salt and the choice would be arbitrary, so the
      molecule is returned whole with a flag.

    >>> kept, gone, meta = largest_fragment("Cl.COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC")
    >>> gone
    ['Cl']
    >>> meta["contains_metal"]
    False
    >>> kept, gone, meta = largest_fragment("N.N.Cl[Pt]Cl")
    >>> gone, meta["contains_metal"]
    ([], True)
    """
    from rdkit import Chem

    flags: list[str] = []
    mol = _mol(smiles)
    if mol is None:
        return None, [], {"n_fragments": 0, "contains_metal": False,
                          "flags": ["blocker:rdkit-cannot-parse-structure"]}

    symbols = {a.GetSymbol() for a in mol.GetAtoms()}
    contains_metal = bool(symbols & METAL_SYMBOLS)
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    meta = {"n_fragments": len(frags), "contains_metal": contains_metal, "flags": flags}

    if contains_metal:
        if len(frags) > 1:
            flags.append("caveat:metal-complex-kept-whole")
        else:
            flags.append("caveat:contains-metal")
        return Chem.MolToSmiles(mol), [], meta
    if len(frags) == 1:
        return _neutralise(mol), [], meta

    order = sorted(frags, key=lambda f: (f.GetNumHeavyAtoms(), Chem.MolToSmiles(f)), reverse=True)
    kept, rest = order[0], order[1:]
    biggest_discarded = max(f.GetNumHeavyAtoms() for f in rest)
    if kept.GetNumHeavyAtoms() < MIN_FRAGMENT_RATIO * biggest_discarded:
        flags.append("check:fragments-too-similar-in-size-kept-whole")
        return _neutralise(mol), [], meta
    return _neutralise(kept), [Chem.MolToSmiles(f) for f in rest], meta


def _neutralise(mol) -> str:
    """Return ``mol`` as SMILES with ionisable groups in their neutral form.

    Deposited records disagree about protonation: pemetrexed's is the dianion
    (it is deposited as the disodium salt) and bleomycin's is a cation, while
    the other 98 are neutral. A Morgan fingerprint distinguishes ``C(=O)[O-]``
    from ``C(=O)O``, so leaving two records charged would give those two drugs
    features that encode a deposit convention rather than a property of the
    molecule. RDKit's standard ``Uncharger`` makes the convention uniform.

    Metal complexes are exempt: neutralising ``[Pt+2]`` would be a claim about
    oxidation state, not a formatting choice.
    """
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    if {a.GetSymbol() for a in mol.GetAtoms()} & METAL_SYMBOLS:
        return Chem.MolToSmiles(mol)
    try:
        return Chem.MolToSmiles(rdMolStandardize.Uncharger().uncharge(Chem.Mol(mol)))
    except Exception:
        return Chem.MolToSmiles(mol)


def inchikey_of(smiles: str) -> str | None:
    """RDKit's InChIKey for ``smiles``, or None if it cannot be computed.

    Computed locally rather than taken from either database, so the two
    resolutions are compared through one implementation and a difference cannot
    be an artefact of two different InChI versions.
    """
    from rdkit import Chem

    mol = _mol(smiles)
    if mol is None:
        return None
    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    return key or None


# ---------------------------------------------------------------------------
# Corrections from the independent audit
# ---------------------------------------------------------------------------

#: Structures the independent audit found wrong in the database record that the
#: automated resolution selected, together with the authoritative deposited
#: record that replaces them.
#:
#: **The rule for what may go in here: an override may only point at another
#: *deposited* record in an authoritative database.** Never a hand-drawn
#: structure, however confident the reasoning. That distinction is what stops
#: this table from becoming a place where the mapping gets adjusted until it
#: looks right. Oxaliplatin is the case that proves the rule bites: the audit
#: found the same defect in its record that it found in carboplatin's, and no
#: database holds a corrected depiction, so oxaliplatin is *not* overridden --
#: the defect is recorded in ``docs/phase3_drug_mapping.md`` and answered by the
#: pre-registered metal-excluded arm instead.
#:
#: Each entry carries the reason and the corroborating identifiers, because an
#: override is a place where a human overruled two databases and the reader is
#: entitled to check the work.
AUDIT_OVERRIDES: dict[str, dict] = {
    "Carboplatin": {
        "pubchem_cid": 10339178,
        "smiles": "C1CC(C1)(C(=O)[O-])C(=O)[O-].N.N.[Pt+2]",
        "reason": (
            "PubChem's name-indexed carboplatin record, CID 426756 -- the only hit "
            "for the literal name -- mis-depicts the two ammine (NH3) ligands as "
            "azanide anions [NH2-] and the chelating cyclobutane-1,1-dicarboxylate "
            "dianion as the neutral free diacid. The molecular formula is "
            "coincidentally identical (C6H12N2O4Pt) so the formula check cannot see "
            "it, and CHEMBL1351 carries no structure at all so the cross-check "
            "could not fire. Formal charge and hydrogen count are both Morgan atom "
            "invariants, so this reaches the features: ECFP4 Tanimoto between the "
            "two depictions is 0.50, and the mis-depicted one is 0.875 similar to "
            "bare 1,1-cyclobutanedicarboxylic acid. That is the exact failure the "
            "fragment rule was written to prevent, arriving by a different route."),
        "corroboration": "CID 10339178; DrugBank DB00958; ChEBI:31355; FDA UNII BG3F62OND5",
    },
    "Pentostatin": {
        "smiles": "OC[C@H]1O[C@@H](n2cnc3c2N=CNC[C@H]3O)C[C@@H]1O",
        "reason": (
            "PubChem CID 439693 deposits the 4H tautomer of the diazepine ring; the "
            "FDA label, CHEMBL1580 and PDB component DCF all give the 3,6,7,8-"
            "tetrahydro (6H) form. Standard InChI treats the proton as mobile, so "
            "both give FPVKHBSQESCIEP-JQCXWYLXSA-N and the connectivity cross-check "
            "passed -- the disagreement is invisible to every check in the "
            "pipeline, but not to a Morgan fingerprint."),
        "corroboration": "CHEMBL1580 canonical_smiles; PDB chem comp DCF",
    },
    "Vincristine Sulfate": {
        "pubchem_cid": 5388993,
        "smiles": ("CC[C@]1(O)C[C@@H]2CN(CCc3c([nH]c4ccccc34)[C@@](C(=O)OC)"
                   "(c3cc4c(cc3OC)N(C=O)[C@H]3[C@@](O)(C(=O)OC)[C@H](OC(C)=O)"
                   "[C@]5(CC)C=CCN6CC[C@]43[C@@H]65)C2)C1"),
        "reason": (
            "The resolved record is the C-15 (C-4') epimer of vincristine. The "
            "proof is internal to this deposit rather than a matter of database "
            "preference: vincristine is N1'-desmethyl-N1'-formyl-vinblastine and "
            "differs from vinblastine at no other atom, so applying that single "
            "substitution to the mapping's own vinblastine structure must reproduce "
            "vincristine -- and it yields OGWKCGZFUXNPDA-CFWMRBGOSA-N, not the "
            "recorded -XQKSVPLYSA-N. Both databases carry the same legacy "
            "structure, and the cross-check compares only the connectivity block, "
            "which is identical for two epimers. Note this correction does NOT "
            "reach the features, because the primary fingerprint is computed with "
            "useChirality=False; it is applied so the stored structure is right."),
        "corroboration": "CID 5388993; CHEMBL499458; DrugBank DB00541; FDA UNII 5J49Q6B70F",
    },
}


def apply_overrides(rec: DrugRecord) -> None:
    """Replace an audited-wrong structure with its authoritative deposited one.

    Runs after both databases have been consulted, so the record still carries
    what the automated resolution found -- ``deposited_smiles`` is untouched --
    and the override is visible as a flag rather than as an unexplained value.
    """
    fix = AUDIT_OVERRIDES.get(rec.label)
    if not fix:
        return
    if "pubchem_cid" in fix:
        rec.pubchem_cid = fix["pubchem_cid"]
    rec.smiles = fix["smiles"]
    rec.inchikey = inchikey_of(fix["smiles"])
    rec.flags = [f for f in rec.flags
                 if not f.startswith(("check:pubchem-chembl-disagree",
                                      "check:label-not-an-exact"))]
    rec.flags.append("audited:structure-overridden")
    rec.notes = f"AUDIT OVERRIDE. {fix['reason']} Corroboration: {fix['corroboration']}"
