"""Acquisition, ingestion and audit of the Koplev et al. (2017) sequential screen.

Provenance
----------
Paper   Koplev S, Longden J, Ferkinghoff-Borg J, Blicher Bjerregard M, Cox TR,
        Erler JT, Pedersen JT, Voellmy F, Sommer MOA, Linding R.
        "Dynamic Rearrangement of Cell States Detected by Systematic Screening
        of Sequential Anticancer Treatments."
        Cell Reports 20(12):2784-2791, 2017. doi:10.1016/j.celrep.2017.08.095
Data    Mendeley Data, doi:10.17632/wgybvcvjwf.1, version 1, CC BY 4.0.

The deposit holds five CSVs. Only two of them are usable for the Phase 2
question, and establishing that took a direct comparison of the files rather
than a reading of their descriptions -- see :func:`verify_raw` and
``docs/phase2_dataset.md``:

``Data Table 1.csv``  100x100 complete ordered matrix, labelled A375.  USED
``Data Table 2.csv``  100x100 complete ordered matrix, labelled PANC1. USED
``Data Table 3.csv``  193 ordered combinations x 4 pancreatic lines. UNUSABLE --
                      contains *zero* reverse pairs, so no directional target
                      can be formed.
``Data Table 4.csv``  labelled "A375 validation screen", but it is PANC1 data:
                      its ``synergy_measure`` column is byte-identical to
                      Table 2 on all 9,900 shared ordered pairs. NOT an
                      independent screen. Its ``synergy_sd`` column is used only
                      as a posterior-uncertainty reference, never as a second
                      observation.
``Data Table 5.csv``  labelled "PANC1 validation"; a 190-row subset of Table 4,
                      likewise byte-identical to Table 2. Zero reverse pairs.

The table identities were settled against the paper rather than against the
deposit's descriptions, by two independent fingerprints:

* the paper quotes specific lambda values -- A375 bortezomib->vemurafenib
  lambda = 0.926 and erlotinib->vemurafenib lambda = 0.725; PANC1
  gemcitabine->erlotinib lambda = 0.996 and cisplatin->gemcitabine
  lambda = 0.805. Table 1 matches the A375 pair, Tables 2/4/5 match the PANC1
  ones;
* the paper's significance counts (707 synergistic / 1,845 antagonistic in A375;
  551 / 1,464 in PANC1, all at p < 0.05 i.e. |lambda| > 0.95) reproduce
  **exactly** from Table 1 and Table 2 respectively.

So Table 1 = A375 and Table 2 = PANC1 as labelled; Table 4 is a sign-corrected
re-export of the PANC1 fit (its ``lambda`` differs from Table 2's on 860 rows,
identical in magnitude and flipped in sign, and agrees with
``sign(synergy_measure)`` everywhere); and there is **no A375 validation table in
the deposit at all**. Treating Tables 4/5 as additional data would have silently
duplicated PANC1 at roughly 2x weight and mislabelled it as the other cell line.
The duplication check in :func:`verify_raw` runs on every ingestion so this
cannot regress.

Why the "validation" tables are not independent estimates: the paper states that
"these validation results were used to further refine the model so that the
updated synergy measures, presented throughout this study, take into account
both the primary screen and validation data". Tables 2, 4 and 5 are three views
of one post-validation joint posterior for PANC1. There is no validation-only
estimate anywhere in the deposit.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KOPLEV_DOI = "10.17632/wgybvcvjwf.1"
KOPLEV_PAPER = "10.1016/j.celrep.2017.08.095"
KOPLEV_LICENSE = "CC BY 4.0"
#: Date the files below were fetched and their digests recorded.
ACQUIRED = "2026-08-17"

_BASE = "https://data.mendeley.com/public-files/datasets/wgybvcvjwf/files"


@dataclass(frozen=True)
class Source:
    filename: str
    file_id: str
    sha256: str
    size: int
    deposit_description: str

    @property
    def url(self) -> str:
        return f"{_BASE}/{self.file_id}/file_downloaded"


#: Every file in the deposit, with the digest observed on the acquisition date.
#: Recorded for all five even though three are unused, so that a future reader
#: can re-derive the duplication finding without guessing which files existed.
SOURCES: dict[str, Source] = {
    "table1": Source(
        "Data Table 1.csv", "83d1dee8-5649-4151-a19c-1c2462d1296d",
        "b47d22cf16c5522ec718dd3d374c222c68a0cb0df91d2d5b979da1e7eb5a6046", 511264,
        "Modeled data from the A375 combinatorial screen."),
    "table2": Source(
        "Data Table 2.csv", "d467d5e5-1256-465a-b8cf-f786a71ba247",
        "6d534e0d4edbc2de5c3966dd86cfb442f1c8242d0b5e64eb2b16ac49a165158c", 519228,
        "Modeled data from the PANC1 combinatorial screen."),
    "table3": Source(
        "Data Table 3.csv", "f1f67bfc-c0fa-420f-a26a-a89c45c9d756",
        "cf93179edfbd89f827935f54b1b48f332521eea7221e3b151cd3b2891f2575e0", 20852,
        "Modeled data from the pancreatic cancer cell line screen."),
    "table4": Source(
        "Data Table 4.csv", "19788445-e84a-4328-b489-276c37b4057a",
        "0606dca7b37b30e5a908bf9fd4047418cbd2969bc018fdbfeb7600971ae33437", 630071,
        "Modeled data from the A375 validation screen."),
    "table5": Source(
        "Data Table 5.csv", "81f39fd0-66de-48db-b032-dd5bbef85196",
        "120d6462adb8ebbde6c603e34f33fe92d1ba18d62c9b5431e71e642836542385", 11925,
        "Modeled data from the PANC1 validation screen."),
}

#: The two tables that carry a complete ordered matrix. ``label`` is the cell
#: line as stated by the deposit; see the module docstring for why the labels
#: are reported rather than relied upon.
SCREEN_TABLES = {"A375": "table1", "PANC1": "table2"}

DEFAULT_RAW_DIR = Path("data/raw/koplev2017")

# A browser User-Agent is required: Mendeley's file host answers the default
# urllib agent with 403.
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_raw(dest: Path = DEFAULT_RAW_DIR, force: bool = False) -> dict[str, Path]:
    """Fetch the deposit into ``dest``, verifying each digest.

    Idempotent: a file already present with the right digest is not refetched.
    A file present with the *wrong* digest is refetched rather than trusted --
    a partial download from an interrupted run must not become an input.
    """
    dest.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for key, src in SOURCES.items():
        path = dest / src.filename
        if path.exists() and not force and sha256_of(path) == src.sha256:
            out[key] = path
            continue
        req = urllib.request.Request(src.url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = resp.read()
        got = hashlib.sha256(payload).hexdigest()
        if got != src.sha256:
            raise RuntimeError(
                f"{src.filename}: sha256 mismatch\n  expected {src.sha256}\n"
                f"  got      {got}\nThe deposit may have been revised; do not "
                f"proceed with data whose provenance no longer matches.")
        path.write_bytes(payload)
        out[key] = path
    (dest / "PROVENANCE.json").write_text(json.dumps({
        "paper_doi": KOPLEV_PAPER, "data_doi": KOPLEV_DOI,
        "license": KOPLEV_LICENSE, "acquired": ACQUIRED,
        "files": {k: {"filename": s.filename, "sha256": s.sha256, "size": s.size,
                      "deposit_description": s.deposit_description,
                      "url": s.url}
                  for k, s in SOURCES.items()},
    }, indent=2) + "\n")
    return out


def normalise_drug(name: str) -> str:
    """Canonical drug identifier.

    Collapses runs of whitespace and strips -- four deposit names carry a double
    space (``'Cytarabine;  Ara-C'``). Verified on the full 100-drug list not to
    merge two distinct compounds; :func:`load_screen` re-checks that invariant on
    every load rather than trusting the one-off verification.
    """
    return re.sub(r"\s+", " ", name.strip())


@dataclass(frozen=True)
class Screen:
    """One complete ordered screen, in canonical form.

    ``frame`` has one row per ordered pair ``(i, j)``, ``i != j``, with columns

        ``first``, ``second``   canonical drug names
        ``i``, ``j``            integer drug indices into ``drugs``
        ``pair``                canonical unordered pair id, ``(min, max)``
        ``y``                   ``synergy_measure`` for that ordered schedule
        ``lam``                 ``lambda``, the signed confidence

    The diagonal ``i -> i`` rows are removed (see :attr:`n_self_rows`): a drug
    followed by itself has no meaningful ordering, contributes ``A(i,i) = 0`` by
    definition, and would hand the structured model a constraint that is true by
    construction rather than learned.
    """

    label: str
    table_key: str
    drugs: tuple[str, ...]
    frame: pd.DataFrame
    n_raw_rows: int
    n_self_rows: int
    n_missing: int

    @property
    def n_drugs(self) -> int:
        return len(self.drugs)

    def matrix(self, column: str = "y") -> np.ndarray:
        """Dense ``n_drugs x n_drugs`` view; diagonal is NaN."""
        M = np.full((self.n_drugs, self.n_drugs), np.nan)
        M[self.frame["i"].to_numpy(), self.frame["j"].to_numpy()] = \
            self.frame[column].to_numpy()
        return M


def load_screen(label: str, raw_dir: Path = DEFAULT_RAW_DIR) -> Screen:
    """Load and canonicalise one of the two complete ordered screens.

    Deterministic: drug indices come from the sorted canonical name list and the
    frame is sorted by ``(i, j)``, so two loads on two machines produce the same
    integer encoding.
    """
    if label not in SCREEN_TABLES:
        raise ValueError(f"unknown screen {label!r}; expected one of "
                         f"{sorted(SCREEN_TABLES)}")
    key = SCREEN_TABLES[label]
    path = raw_dir / SOURCES[key].filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python scripts/download_koplev.py` first.")
    raw = pd.read_csv(path)
    n_raw = len(raw)
    n_missing = int(raw[["first_compound", "second_compound", "synergy_measure",
                         "lambda"]].isna().sum().sum())

    df = raw.copy()
    df["first"] = df["first_compound"].map(normalise_drug)
    df["second"] = df["second_compound"].map(normalise_drug)

    # Normalisation must not merge two distinct compounds into one id.
    originals = set(df["first_compound"]) | set(df["second_compound"])
    canonical = {normalise_drug(x) for x in originals}
    if len(canonical) != len(originals):
        collapsed = {}
        for x in originals:
            collapsed.setdefault(normalise_drug(x), []).append(x)
        raise RuntimeError("drug-name normalisation collapsed distinct names: "
                           f"{ {k: v for k, v in collapsed.items() if len(v) > 1} }")

    drugs = tuple(sorted(canonical))
    index = {d: k for k, d in enumerate(drugs)}
    df["i"] = df["first"].map(index)
    df["j"] = df["second"].map(index)

    n_self = int((df["i"] == df["j"]).sum())
    df = df[df["i"] != df["j"]].copy()
    df = df.dropna(subset=["synergy_measure"])

    df["y"] = df["synergy_measure"].astype(float)
    df["lam"] = df["lambda"].astype(float)
    df["pair"] = list(zip(np.minimum(df["i"], df["j"]), np.maximum(df["i"], df["j"])))

    dup = df.duplicated(["i", "j"]).sum()
    if dup:
        raise RuntimeError(f"{label}: {dup} duplicated ordered pairs in the raw table")

    df = df.sort_values(["i", "j"]).reset_index(drop=True)
    return Screen(label=label, table_key=key, drugs=drugs,
                  frame=df[["first", "second", "i", "j", "pair", "y", "lam"]],
                  n_raw_rows=n_raw, n_self_rows=n_self, n_missing=n_missing)


def verify_raw(raw_dir: Path = DEFAULT_RAW_DIR) -> dict:
    """Re-derive the duplication finding that decides which tables are usable.

    This is not decoration. The deposit's own descriptions say Table 4 is an
    independent A375 validation screen; the bytes say otherwise. Anyone who
    believed the descriptions would build a two-context experiment in which one
    context is a copy of the other. Run on every ingestion.
    """
    def rd(key: str) -> pd.DataFrame:
        return pd.read_csv(raw_dir / SOURCES[key].filename)

    t1, t2, t4, t5 = rd("table1"), rd("table2"), rd("table4"), rd("table5")
    t3 = rd("table3")

    def keyed(df: pd.DataFrame) -> pd.Series:
        return df.set_index(["first_compound", "second_compound"])["synergy_measure"]

    k1, k2, k4, k5 = keyed(t1), keyed(t2), keyed(t4), keyed(t5)

    def maxdiff(a: pd.Series, b: pd.Series) -> tuple[int, float]:
        common = a.index.intersection(b.index)
        if len(common) == 0:
            return 0, float("nan")
        return len(common), float(np.abs(a.loc[common] - b.loc[common]).max())

    n42, d42 = maxdiff(k4, k2)
    n52, d52 = maxdiff(k5, k2)
    n41, d41 = maxdiff(k4, k1)
    n12, d12 = maxdiff(k1, k2)

    def reverse_coverage(pairs: set[tuple[str, str]]) -> int:
        return sum(1 for a, b in pairs if a != b and (b, a) in pairs)

    p3 = t3["pretreatment_treatment"].str.split("_", n=1, expand=True)
    s3 = set(zip(p3[0], p3[1]))
    s5 = set(zip(t5["first_compound"], t5["second_compound"]))

    report = {
        "table4_vs_table2": {"shared_rows": n42, "max_abs_diff": d42,
                             "identical": d42 == 0.0},
        "table5_vs_table2": {"shared_rows": n52, "max_abs_diff": d52,
                             "identical": d52 == 0.0},
        "table4_vs_table1": {"shared_rows": n41, "max_abs_diff": d41,
                             "identical": d41 == 0.0},
        "table1_vs_table2": {"shared_rows": n12, "max_abs_diff": d12,
                             "identical": d12 == 0.0},
        "table3_ordered_combos": len(s3),
        "table3_with_both_directions": reverse_coverage(s3),
        "table5_ordered_combos": len(s5),
        "table5_with_both_directions": reverse_coverage(s5),
        "table4_synergy_sd_median": float(t4["synergy_sd"].median()),
        "table4_synergy_sd_mean": float(t4["synergy_sd"].mean()),
    }
    if not report["table4_vs_table2"]["identical"]:
        raise RuntimeError(
            "Table 4 is no longer byte-identical to Table 2. The deposit has "
            "changed, and docs/phase2_dataset.md's account of which tables are "
            "usable must be re-derived before any result is trusted.")
    if report["table1_vs_table2"]["identical"]:
        raise RuntimeError("Table 1 and Table 2 are identical; they were treated "
                           "as two independent screens. Stop and re-derive.")
    return report


def measurement_noise_sd(raw_dir: Path = DEFAULT_RAW_DIR) -> dict:
    """Uncertainty reference derived from the ``synergy_sd`` column.

    Read this carefully before using it as a "noise floor". From the study's own
    code (``skoplev/d-chain``, ``post/interpretMCMC.R``),
    ``synergy_index_sd[a,b] = sd(synergy_index[,a,b])`` -- the standard deviation
    across retained MCMC posterior samples. It is **posterior uncertainty of a
    modelled quantity**, not the spread of experimental replicates. The wet-lab
    triplicates were collapsed into sufficient statistics before the model was
    fitted and are not deposited, so a true replicate SD is not recoverable from
    this release.

    Table 4 carries this SD for each of Table 2's (PANC1) values. There is no
    such column for A375, so its scale is transferred by the ratio of the two
    screens' overall spread rather than assumed equal -- A375's values are ~1.4x
    more dispersed. Fixed *before* any model runs and never re-derived from test
    performance; the ordering-accuracy curve is additionally reported at every
    threshold so no conclusion rests on this one number.
    """
    t4 = pd.read_csv(raw_dir / SOURCES["table4"].filename)
    sd = float(t4["synergy_sd"].median())
    t1 = pd.read_csv(raw_dir / SOURCES["table1"].filename)
    t2 = pd.read_csv(raw_dir / SOURCES["table2"].filename)
    s1 = float(t1["synergy_measure"].std())
    s2 = float(t2["synergy_measure"].std())
    # D = y_ij - y_ji, so sd(D) = sqrt(2) * sd(y) for independent errors.
    return {
        "sd_reference_screen": "PANC1",
        "median_synergy_sd": sd,
        "spread_A375": s1,
        "spread_PANC1": s2,
        "sd_D": {"PANC1": float(np.sqrt(2) * sd),
                 "A375": float(np.sqrt(2) * sd * s1 / s2)},
        "threshold_2sd_D": {"PANC1": float(2 * np.sqrt(2) * sd),
                            "A375": float(2 * np.sqrt(2) * sd * s1 / s2)},
    }


def audit_screen(screen: Screen) -> dict:
    """Countable description of one screen, emitted before any model is fitted."""
    f = screen.frame
    pairs = set(f["pair"])
    ordered = set(zip(f["i"], f["j"]))
    both = sum(1 for (a, b) in pairs if (a, b) in ordered and (b, a) in ordered)
    deg = np.zeros(screen.n_drugs, dtype=int)
    for a, b in pairs:
        deg[a] += 1
        deg[b] += 1
    M = screen.matrix("y")
    off = ~np.eye(screen.n_drugs, dtype=bool)
    y = M[off]
    Sm, Am = (M + M.T) / 2, (M - M.T) / 2
    return {
        "screen": screen.label,
        "table": SOURCES[screen.table_key].filename,
        "rows_raw": screen.n_raw_rows,
        "rows_self_pairs_removed": screen.n_self_rows,
        "rows_missing_removed": screen.n_raw_rows - screen.n_self_rows - len(f),
        "rows_used": len(f),
        "missing_values_in_raw": screen.n_missing,
        "n_drugs": screen.n_drugs,
        "n_unordered_pairs": len(pairs),
        "n_ordered_pairs": len(ordered),
        "n_pairs_with_both_directions": both,
        "pair_matrix_completeness": both / len(pairs),
        # The wet-lab work was done in triplicate, but replicates were reduced
        # to sufficient statistics before the Bayesian fit; the deposit carries
        # one modelled value per ordered combination and no replicate rows.
        "replicate_rows_per_ordered_pair": 1,
        "wetlab_replicates_reported_in_paper": 3,
        "doses": "first drug at a single 1 uM dose; second drug in 4-point dose "
                 "response (10 uM / 1 uM / 100 nM / 10 nM). Collapsed by the "
                 "source model into an area-based measure over 10 concentrations "
                 "in [0.01, 10] uM; no per-dose rows in the deposit.",
        "train_degree_min": int(deg.min()),
        "train_degree_max": int(deg.max()),
        "y_mean": float(np.nanmean(y)), "y_std": float(np.nanstd(y)),
        "y_min": float(np.nanmin(y)), "y_max": float(np.nanmax(y)),
        "var_symmetric_share": float(np.nanvar(Sm[off]) / np.nanvar(y)),
        "var_antisymmetric_share": float(np.nanvar(Am[off]) / np.nanvar(y)),
        "directional_corr_forward_reverse": float(
            np.corrcoef(M[np.triu(off, 1)], M.T[np.triu(off, 1)])[0, 1]),
        "abs_D_mean": float(np.nanmean(np.abs(2 * Am[np.triu(off, 1)]))),
        "abs_D_max": float(np.nanmax(np.abs(2 * Am[np.triu(off, 1)]))),
        "high_confidence_frac_abs_lambda_gt_0.95": float((f["lam"].abs() > 0.95).mean()),
    }


#: Counts the paper states for high-confidence interactions, at p < 0.05, i.e.
#: |lambda| > 0.95: "1,258 synergistic drug combinations (p < 0.05; 551 in PANC1
#: and 707 in A375) and 3,309 antagonistic combinations (p < 0.05; 1,464 in
#: PANC1 and 1,845 in A375); in total, approximately 23% of all combinations
#: tested". Denominator is 20,000 = the two full 100x100 matrices.
PUBLISHED_COUNTS = {
    "A375": {"synergistic": 707, "antagonistic": 1845},
    "PANC1": {"synergistic": 551, "antagonistic": 1464},
}
#: "For all reversed combinations, synergy measures were only weakly correlated
#: for both A375 (r = 0.25) and PANC1 (r = 0.23)."
PUBLISHED_REVERSE_CORR = {"A375": 0.25, "PANC1": 0.23}


def reproduce_publication_stats(raw_dir: Path = DEFAULT_RAW_DIR) -> dict:
    """Re-derive the paper's own headline counts from the deposited tables.

    This runs before any model is fitted and is the check that the pipeline is
    reading the experimental design the way the authors did. Getting the
    significance counts right *exactly* is what pins Table 1 to A375 and Table 2
    to PANC1 -- the deposit's descriptions cannot be trusted for that (Table 4 is
    labelled A375 and is not).

    Note the counts are computed on the **full** 100x100 matrix including the
    100 measured self-combinations, because that is the paper's denominator
    (10,000 per cell line). The modelling pipeline excludes the diagonal; this
    check deliberately does not, or it would not reproduce.
    """
    out: dict = {"threshold": "|lambda| > 0.95  (p = 1 - |lambda| < 0.05)"}
    total_hits = 0
    for label, key in SCREEN_TABLES.items():
        df = pd.read_csv(raw_dir / SOURCES[key].filename)
        syn = int((df["lambda"] > 0.95).sum())
        ant = int((df["lambda"] < -0.95).sum())
        total_hits += syn + ant
        exp = PUBLISHED_COUNTS[label]

        m = df.set_index(["first_compound", "second_compound"])["synergy_measure"]
        fwd, rev = [], []
        for (a, b), v in m.items():
            if a < b and (b, a) in m.index:
                fwd.append(v)
                rev.append(m[(b, a)])
        r = float(np.corrcoef(fwd, rev)[0, 1])

        out[label] = {
            "table": SOURCES[key].filename,
            "n_combinations": int(len(df)),
            "synergistic": syn, "synergistic_published": exp["synergistic"],
            "antagonistic": ant, "antagonistic_published": exp["antagonistic"],
            "counts_match_publication": syn == exp["synergistic"]
                                        and ant == exp["antagonistic"],
            "reverse_direction_pearson": r,
            "reverse_direction_pearson_published": PUBLISHED_REVERSE_CORR[label],
            "n_unordered_pairs_used_for_r": len(fwd),
        }
    out["total_high_confidence"] = total_hits
    out["total_high_confidence_published"] = 1258 + 3309
    out["fraction_of_all_combinations"] = total_hits / 20000
    out["fraction_published"] = 0.23
    out["all_counts_match"] = all(
        out[label]["counts_match_publication"] for label in SCREEN_TABLES)
    return out
