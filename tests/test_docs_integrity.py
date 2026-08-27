"""A detector for the one failure mode in this project that never had one.

Five conclusions in Phase 1 reversed under powering, and every one of them
announced itself: a number moved and someone noticed. Documentation drift never
announced itself. It was the first defect found and the last, eight instances in
all, none of which changed a number and all of which would have changed what a
reader concluded from the numbers. The commit whose *message* was about
documentation drift broke three references pointing at retraction evidence.

The mechanical part of that failure mode -- a document citing a file, a results
artifact or a test that no longer exists under that name -- is checkable, so it
is checked here rather than left to a second reader. What is *not* checkable is
prose that has quietly stopped matching its data; nothing in this file addresses
that, and it remains the residual risk.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SEARCHED = ["README.md", "docs", "src", "scripts", "tests"]

#: Repo-relative paths cited in prose or code. Restricted to the directories
#: whose contents are evidence, so a stray word in a sentence cannot fail CI.
_PATH_RE = re.compile(
    r"\b((?:results|configs|docs|tests|scripts|src)/[A-Za-z0-9_./-]*"
    r"[A-Za-z0-9_-]\.(?:jsonl|json|md|png|csv|py|yml|yaml))")


def _files_to_scan() -> list[Path]:
    out: list[Path] = []
    for name in SEARCHED:
        p = ROOT / name
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out += [q for q in p.rglob("*")
                    if q.is_file() and q.suffix in {".md", ".py", ".yml", ".yaml"}
                    and "__pycache__" not in q.parts]
    return out


#: A path given as an output argument names a file the command *creates*. It is
#: not a citation and must not be required to exist.
_OUTPUT_ARG_RE = re.compile(r"--out(?:dir)?[= ]+([A-Za-z0-9_./-]+)")


#: Paths under a directory that is fetched rather than committed. A document may
#: legitimately name ``data/raw/koplev2017/Data Table 1.csv`` on a machine that
#: has not run ``scripts/download_koplev.py``; requiring it to exist made this
#: check fail in CI for a reason that has nothing to do with documentation.
_FETCHED_PREFIXES = ("data/raw/", "data/external/cache/", "third_party/")

#: Outputs of a `--smoke` pipeline check: gitignored, regenerated on demand, and
#: absent from a clean checkout. Naming one is not a dangling citation. The same
#: file is exempted from the results index for the same reason.
_GENERATED_BASENAMES = ("smoke.jsonl",)


def test_every_cited_path_resolves():
    """A dangling `results/...` path costs the proof that a retraction was warranted."""
    dangling: dict[str, list[str]] = {}
    for src in _files_to_scan():
        text = src.read_text(errors="ignore")
        produced = set(_OUTPUT_ARG_RE.findall(text))
        for match in set(_PATH_RE.findall(text)) - produced:
            if match.startswith(_FETCHED_PREFIXES):
                continue
            if match.rsplit("/", 1)[-1] in _GENERATED_BASENAMES:
                continue
            if not (ROOT / match).exists():
                dangling.setdefault(match, []).append(
                    str(src.relative_to(ROOT)))
    assert not dangling, (
        "documents cite paths that do not exist:\n"
        + "\n".join(f"  {p}  <- cited by {', '.join(sorted(where))}"
                    for p, where in sorted(dangling.items())))


def test_every_results_file_is_listed_in_the_index():
    """Sixteen of twenty-three result files were once unlisted, including every
    powered replication the headline rests on. An unlisted artifact is one a
    reader cannot tell is authoritative, superseded, or an input to selection.
    """
    indices = sorted((ROOT / "results").rglob("README_*.md"))
    if not indices:
        pytest.skip("no results index")
    # Each phase keeps its own index beside its own results, and either that one
    # or the top-level one may list a file; what is not allowed is a result
    # listed in neither.
    #
    # The match is scoped to the directory the index lives in, not global by
    # basename. Phase 3 introduced results/phase3_entity_ood/controls.jsonl while
    # results/phase2_residual/controls.jsonl already existed, and a global
    # basename match passed the new file on the strength of the old one's entry
    # -- so the check silently stopped covering a whole phase. A full relative
    # path is always accepted, from any index.
    by_dir: dict[Path, str] = {}
    everything = ""
    for p in indices:
        d = p.parent.relative_to(ROOT / "results")
        by_dir[d] = by_dir.get(d, "") + "\n" + p.read_text()
        everything += "\n" + p.read_text()

    # smoke.jsonl is the output of a `--smoke` pipeline check, regenerated on
    # demand and gitignored. It is not a result and has nothing to index.
    missing = []
    for p in sorted((ROOT / "results").rglob("*.jsonl")):
        rel = p.relative_to(ROOT / "results")
        if p.name == "smoke.jsonl":
            continue
        if str(rel) in everything:
            continue
        scope = "".join(by_dir.get(d, "") for d in (rel.parent, Path(".")))
        if p.name in scope:
            continue
        missing.append(str(rel))
    assert not missing, (
        f"result files absent from results/README_RESULTS.md: {missing}")


def test_cited_test_names_exist():
    """`identifiability.md` once cited two tests by name that did not exist.

    Scoped deliberately. ``test_`` is also this project's *split* prefix -- every
    metric column is ``test_mse``, ``test_S_pearson``, ``test_topo_base_rate`` --
    so a bare regex over ``test_\\w+`` flags metrics as missing tests. Only
    citations that are unambiguously test references are checked: a pytest node
    id (of the form ``<test module>::<test function>``), or a backticked
    ``test_`` identifier in a document that is not a recorded metric name.
    """
    defined = set()
    for p in (ROOT / "tests").glob("test_*.py"):
        defined |= set(re.findall(r"^def (test_\w+)", p.read_text(), re.M))

    metric_names = _recorded_metric_names() | _split_bucket_names()
    cited: dict[str, list[str]] = {}
    for src in _files_to_scan():
        if src.parent.name == "tests":
            continue
        text = src.read_text(errors="ignore")
        names = set(re.findall(r"::(test_[a-z0-9_]+)", text))
        if src.suffix == ".md":
            names |= {n for n in re.findall(r"`(test_[a-z0-9_]+)`", text)
                      if n not in metric_names}
        for name in names:
            cited.setdefault(name, []).append(str(src.relative_to(ROOT)))

    unknown = {n: w for n, w in cited.items() if n not in defined}
    assert not unknown, (
        "documents cite tests that do not exist:\n"
        + "\n".join(f"  {n}  <- cited by {', '.join(sorted(w))}"
                    for n, w in sorted(unknown.items())))


def _split_bucket_names() -> set[str]:
    """Split-bucket names, which share the ``test_`` prefix by coincidence.

    ``test_e1`` and ``test_e2`` are two of the four buckets an unordered pair can
    land in in Phase 3, and ``test_e1a``, ``test_e1n``, ``test_e2`` and
    ``test_e2_mixed`` are four of the nine a reaction row can land in in Phase 4.
    None of them is a pytest function. Read from the code rather than listed
    here, so renaming a bucket cannot leave this test asserting against a name
    nothing uses -- and so a new phase's buckets arrive automatically instead of
    presenting as three missing tests, which is how this failed once.
    """
    from intervention_algebra.real_data.chemlex.splits import (
        BUCKETS as CHEMLEX_BUCKETS)
    from intervention_algebra.real_data.entity_ood.splits import BUCKETS

    return set(BUCKETS) | set(CHEMLEX_BUCKETS)


def _recorded_metric_names() -> set[str]:
    """Column names actually present in the results, so metrics are not
    mistaken for test functions."""
    results = ROOT / "results" / "phase1.jsonl"
    if not results.exists():
        return set()
    import json

    with results.open() as fh:
        first = json.loads(fh.readline())
    return {k for k in first if k.startswith("test_")}
