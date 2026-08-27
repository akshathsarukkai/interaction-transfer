"""Acquisition and provenance for the ChemLex acid-amine coupling screen.

Provenance
----------
Paper   Zhong H, et al. "Towards global reaction feasibility and robustness
        prediction with high throughput data and bayesian deep learning."
        Nature Communications 16, 4522 (2025). doi:10.1038/s41467-025-59812-0
Data    Zenodo, doi:10.5281/zenodo.17596563 (record 17596563), the third and
        current version of concept doi:10.5281/zenodo.12920293.
        Licence CC BY-NC 4.0 -- non-commercial, which is why the file is fetched
        and digest-verified rather than vendored.
Code    https://github.com/Chemlex-AI/bayesian-reactivity-prediction

Why the *current* version and not the one that was current when the paper
appeared: because the difference was checked rather than assumed. All three
published versions were downloaded and compared cell by cell.

    2024-07-26  record 12920294  sha256 29bd4b27...  568672 bytes
    2025-05-14  record 15401035  sha256 30796ad3...  573981 bytes  (paper-time)
    2025-11     record 17596563  sha256 a744e340...  562875 bytes  (used here)

Every version has the same 11,669 rows and the same nine columns, and

    Acid, Amine, Products, Conversion, Random_Split,
    Stratified_Split_One_Unseen, Stratified_Split_Both_Unseen

are **byte-identical across all three**. The only column that ever changed is
``Reagents``:

* 2024 -> 2025-05 moved 129 rows out of the plain DIPEA/HATU-N-oxide condition
  into two salt-annotated variants, 114 with a leading ``Cl.`` and 15 with a
  leading ``O=S(=O)(O)O.``;
* 2025-05 -> 2025-11 changed exactly 7 cells, correcting
  ``F[P+](F)(F)(F)(F)F`` to ``F[P-](F)(F)(F)(F)F`` in the PyBroP condition.

That last edit matters: hexafluorophosphate written as a *cation* is not a
molecule RDKit will accept, so the earlier versions carry a condition string that
cannot be parsed. Using the newest version therefore costs nothing on the
modelling columns and fixes a defect on the condition column.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

CHEMLEX_PAPER = "10.1038/s41467-025-59812-0"
CHEMLEX_DOI = "10.5281/zenodo.17596563"
CHEMLEX_CONCEPT_DOI = "10.5281/zenodo.12920293"
CHEMLEX_RECORD = "17596563"
CHEMLEX_LICENSE = "CC BY-NC 4.0"
CHEMLEX_CODE = "https://github.com/Chemlex-AI/bayesian-reactivity-prediction"
#: Date the file below was fetched and its digest recorded.
ACQUIRED = "2026-08-26"

DEFAULT_RAW_DIR = Path("data/raw/chemlex2025")
FILENAME = "Chemlex_Acidamine_Wetlab_Data.xlsx"


@dataclass(frozen=True)
class Version:
    """One published version of the deposit."""

    record: str
    doi: str
    published: str
    filename: str
    sha256: str
    md5: str
    size: int

    @property
    def url(self) -> str:
        return (f"https://zenodo.org/api/records/{self.record}"
                f"/files/{self.filename}/content")


#: Every published version, with the digests observed on the acquisition date.
#: All three are recorded even though only one is used, so that a future reader
#: can re-derive the version comparison in the module docstring without guessing
#: which records existed.
VERSIONS: dict[str, Version] = {
    "2024-07-26": Version(
        "12920294", "10.5281/zenodo.12920294", "2024-07-26", FILENAME,
        "29bd4b27267faaea3104fa2b1c7d2424b7ebdac28536c5dbd19af2e1e908fd47",
        "0361f6ebc9dd4bc7e4d2e7333d950a0d", 568672),
    "2025-05-14": Version(
        "15401035", "10.5281/zenodo.15401035", "2025-05-14", FILENAME,
        "30796ad345e6e1b526e44c65d9433bcea6a35bdbb84cc3698dc2c6fbcf724d73",
        "df42568e8f3b6b7eb50591db2d9b44ce", 573981),
    "2025-11": Version(
        "17596563", "10.5281/zenodo.17596563", "2025-11", FILENAME,
        "a744e3404140cd198d4b4beb85890545763a23e6cca140d21d93e91c8bcf9773",
        "a6812ee4be7157bbc1163a5585c65c73", 562875),
}

#: The version this project models. See the module docstring for why.
CURRENT = VERSIONS["2025-11"]

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_path(dest: Path = DEFAULT_RAW_DIR) -> Path:
    return dest / CURRENT.filename


def download_raw(dest: Path = DEFAULT_RAW_DIR, force: bool = False,
                 versions: tuple[str, ...] = ("2025-11",)) -> dict[str, Path]:
    """Fetch the named versions into ``dest``, verifying each digest.

    Idempotent: a file already present with the right digest is not refetched.
    A file present with the *wrong* digest is refetched rather than trusted -- a
    partial download from an interrupted run must not become an input. Only the
    current version is fetched by default; passing every key reproduces the
    version comparison recorded in the module docstring.
    """
    dest.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for key in versions:
        ver = VERSIONS[key]
        name = ver.filename if key == "2025-11" else f"{key}_{ver.filename}"
        path = dest / name
        if path.exists() and not force and sha256_of(path) == ver.sha256:
            out[key] = path
            continue
        req = urllib.request.Request(ver.url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = resp.read()
        got = hashlib.sha256(payload).hexdigest()
        if got != ver.sha256:
            raise RuntimeError(
                f"{name}: sha256 mismatch\n  expected {ver.sha256}\n"
                f"  got      {got}\nZenodo records are immutable, so this means "
                f"the fetch was corrupted or the URL no longer resolves to the "
                f"record; do not proceed with data whose provenance no longer "
                f"matches.")
        path.write_bytes(payload)
        out[key] = path
    write_provenance(dest)
    return out


def write_provenance(dest: Path = DEFAULT_RAW_DIR) -> Path:
    path = dest / "PROVENANCE.json"
    path.write_text(json.dumps({
        "paper_doi": CHEMLEX_PAPER,
        "data_doi": CHEMLEX_DOI,
        "concept_doi": CHEMLEX_CONCEPT_DOI,
        "code": CHEMLEX_CODE,
        "license": CHEMLEX_LICENSE,
        "acquired": ACQUIRED,
        "used": CURRENT.record,
        "versions": {k: {"record": v.record, "doi": v.doi,
                         "published": v.published, "filename": v.filename,
                         "sha256": v.sha256, "md5": v.md5, "size": v.size,
                         "url": v.url}
                     for k, v in VERSIONS.items()},
    }, indent=2) + "\n")
    return path


def verify_raw(dest: Path = DEFAULT_RAW_DIR) -> dict[str, object]:
    """Re-derive the digest of the file on disk. Never trusts PROVENANCE.json."""
    path = raw_path(dest)
    if not path.exists():
        return {"present": False, "path": str(path)}
    got = sha256_of(path)
    return {"present": True, "path": str(path), "sha256": got,
            "size": path.stat().st_size, "matches_record": got == CURRENT.sha256,
            "record": CURRENT.record, "doi": CURRENT.doi}
