# Public release checklist

Every gate that had to pass before this repository's visibility was changed from
private to public, with the evidence for each. Recorded so the claim "this was
checked" is itself checkable.

Audited snapshot: the single root commit of this repository, which is its
entire public history. Source: a checkpoint in a private research repository
that remains private and whose history is deliberately not published here. That
repository's identifiers, commits and tags are intentionally not named.

---

## Provenance and history

| ✅ | Gate | Evidence |
|---|---|---|
| ✅ | New Git history only | `git log --oneline \| wc -l` → **1**. `git rev-list --max-parents=0 HEAD \| wc -l` → **1**: that one commit *is* the root, with no parent. The history was rewritten as a single fresh root commit. It carries the tree as it stood after the three superseded public commits — the initial release, this checklist, and a later README revision made through the GitHub web editor — so no published content was dropped by the rewrite; only the commit metadata and messages changed. Those three commits are unreachable and were replaced on the remote. |
| ✅ | No private `.git` ancestry | The tree was assembled by copying an allow-list of *tracked* files out of the private repository and initialising a new repository afterwards. `.git` was never copied; the public tree was created empty. A fresh `git clone` of this repository reports one commit. |
| ✅ | Private source repository is still private | Its visibility was re-checked as `PRIVATE` before release. It was not touched by this repository's history rewrite: it was never rewritten and never force-pushed. It is not named here, so that this document does not itself disclose it. |
| ✅ | No stray refs published | After the rewrite, `git ls-remote origin` lists exactly `HEAD`, `refs/heads/main` and `refs/tags/v1.0.0` (with its peeled `^{}`) — no other branches, no backup refs, no `refs/original/*`, and the GitHub API reports `main` as the only branch and no pull requests. Pushed with explicit refspecs under `--force-with-lease`, never `--all`. |

## Secrets and personal data

| ✅ | Gate | Evidence |
|---|---|---|
| ✅ | No credentials or secrets | Two scans. (1) The tracked tree. (2) **Every blob reachable from every ref** of this public repository, extracted with `git rev-list --objects --all` and `git cat-file` and pattern-scanned as one ~68 MB stream. The patterns covered the documented key prefixes for OpenAI and Anthropic API keys, GitHub personal-access and OAuth tokens, AWS access-key IDs, Hugging Face tokens, Weights & Biases keys, Slack tokens, Google API keys and GitLab tokens; PEM private-key and OpenSSH public-key headers; HTTP bearer-authorization headers; presigned-URL signature and credential query parameters; and credential keywords in assignment position. A filename sweep covered dotenv files, PEM and key files, SSH private keys, `credentials`-named files and package-manager auth files. **Zero true positives** — the only hits anywhere were this document's own descriptions of the scan, which is why the prefixes are named in prose here rather than quoted literally. No rotation required, and nothing in this repository authenticates to any credentialed service. |
| ✅ | No personal or machine paths | A grep over the whole tree and over every reachable blob for macOS, Linux and Windows home-directory prefixes, for the checkout's own parent-directory name, and for the platform temporary-directory prefixes → no matches. The only absolute paths in the repository are the literal `/tmp/dchain_null_*` scratch paths inside `tests/test_dchain_null.py`, which are fixed test constants and identify no machine or user. |
| ✅ | No personal email addresses in content **or metadata** | Two scripts had a personal address hard-coded into their outbound `User-Agent`; both now read `INTERACTION_TRANSFER_CONTACT` from the environment. Content: a grep for the institutional domain that was previously exposed, plus a general RFC-5322-shaped address sweep, over the tree and over every reachable blob → no matches. Metadata: `git log --all --format='%an <%ae>%n%cn <%ce>'` and the annotated tag's tagger line → the author, committer and tagger addresses are the owner's GitHub `users.noreply.github.com` address only. The institutional address that the pre-rewrite commits carried is gone from every reachable object. |
| ✅ | No private references | `grep -rIn 'research_log\|review_findings\|docs/private'` → no matches. Every reference was repointed at a document that exists here, or reworded where none does. |
| ✅ | No private repository URLs or identifiers | The private repository's `owner/name` slug, the source commit SHA and the source tag name appear nowhere in the tree or in any reachable blob or commit message; the provenance statement above is deliberately generic. The distribution and import names `intervention-algebra` / `intervention_algebra` are this project's own package names, explained in the README, and are unrelated to any repository slug. |
| ✅ | No assistant session identifiers | The two pre-rewrite commit messages each ended with a trailer naming an assistant session URL. Those commits are gone. `git log --all --format=%B`, the annotated tag body, and a scan of every reachable blob for that trailer key, for the session-URL host and path, and for the opaque session-handle format → **no matches**. This document deliberately paraphrases those patterns rather than quoting them, so that the verification grep stays clean. Ordinary `Co-Authored-By:` attribution to the assistant is retained: it is authorship, not a session handle. |

## Third-party data and licensing

| ✅ | Gate | Evidence |
|---|---|---|
| ✅ | No raw third-party dataset published | `data/raw/` and `third_party/` are gitignored and were never committed in any of the source repository's 132 commits. Only fetch scripts and provenance metadata are here. |
| ✅ | No data with unclear redistribution rights | Koplev (CC BY 4.0) and ChemLex (CC BY-NC 4.0) are fetched, not vendored; d-chain (GPL-3.0) is fetched, patched and byte-equivalence-verified at build time. PubChem- and ChEMBL-derived tables are small factual mappings, regenerable from committed scripts with SHA-256 provenance, and their upstream terms are pointed at rather than asserted. |
| ✅ | ChemLex substrate inventory removed | Five per-entity CSVs carried a `smiles` column holding 497 of the deposit's 503 distinct reactant structures. Removed; both writers drop the column; a CI step fails the build if it returns. No numeric result changed. |
| ✅ | Test fixtures are generated, not sampled | Both generators say so, both are regenerated and diffed by CI, and the ChemLex fixture uses commercially trivial reagents chosen for RDKit rather than drawn from the deposit. |
| ✅ | `LICENSE` exists and is accurate | Apache-2.0 for this project's own code. `NOTICE` and `THIRD_PARTY_DATA.md` state that external datasets keep their own licences and that the GPL-3.0 fragments quoted in two files are not relicensed; both files carry an upstream copyright notice in their module docstrings. |
| ✅ | `THIRD_PARTY_DATA.md` exists | Per-resource table: purpose, source, DOI, licence as stated in committed evidence, whether it is redistributed, and how to reproduce it. Where the repository states no licence, it says "see original source" rather than guessing. |

## Documentation

| ✅ | Gate | Evidence |
|---|---|---|
| ✅ | README is concise and accurate | 2,429 words — a landing page, not a lab notebook. Every headline number traced to a committed artifact. |
| ✅ | `CITATION.cff` exists | Valid CFF 1.2.0, version 1.0.0, **no DOI** — none has been minted, and inventing one would be worse than not having one. `ZENODO_RELEASE.md` gives the exact steps to mint and wire one in. |
| ✅ | Formal vs corrected verdict distinction preserved | Both frozen `INCONCLUSIVE` verdicts (Phase 3 and Phase 4) are reported as registered, beside their corrected post-hoc readings, with the single changed statistic named. In README, `docs/SCIENTIFIC_RECORD.md`, `docs/CHEMLEX.md`, `docs/LIMITATIONS.md`, `docs/phase4_chemlex_interactions.md`, `results/phase4_chemlex/README_PHASE4.md` and the generated `summary/verdict.md`. |
| ✅ | Withdrawn claims preserved | The low-rank-versus-capacity claim and the misstated detection floor are reported as withdrawn in the README, the scientific record, the limitations, the changelog and — newly — in the generated Phase 4 document that tabulates the dead rung 45 times and previously never mentioned it. |
| ✅ | Pre-registrations published | All four registered decision rules, verbatim, in `docs/PREREGISTRATIONS.md`. A frozen verdict is worth nothing if the rule that produced it cannot be read. |
| ✅ | Public docs contain no private paths | Link check over the tree: **73 relative links, all resolve.** Path-citation test (`test_every_cited_path_resolves`) passes. |
| ✅ | Repository metadata no longer says "Phase 1" only | `pyproject.toml` description is now the project's actual scope; GitHub description and topics set to match. |

## Results

| ✅ | Gate | Evidence |
|---|---|---|
| ✅ | Authoritative artifacts clearly indexed | `results/README_RESULTS.md` plus a per-phase index in each phase directory. `test_every_results_file_is_listed_in_the_index` fails if an artifact is unlisted. |
| ✅ | Superseded artifacts clearly named | Three `SUPERSEDED_*.jsonl` files, retained deliberately as evidence of retracted work, with the retraction explained in the index. |
| ✅ | No duplicate-aggregation hazards | The index warns explicitly against globbing `results/*.jsonl`. One live hazard was found and fixed: the Phase 4 index counted every `.jsonl` in its directory, so two regenerable diagnostics would have silently rewritten its headline from 173 conditions to 285. Replaced with an explicit allow-list. |
| ✅ | Generated documents match committed results | Every phase report regenerated from committed artifacts before release; all summary tables reproduced **byte-identically** on numpy 2.5 / scipy 1.18 / pandas 3.0. CI fails on drift. |
| ✅ | No result references a private-only path | Verified by the grep sweep above and by the path-citation test. |

## Reproducibility

| ✅ | Gate | Evidence |
|---|---|---|
| ✅ | Installation works on a clean environment | Re-verified after the rewrite: fresh `git clone` from GitHub into an empty directory, new venv, `pip install -e ".[dev]"` → `intervention-algebra 1.0.0` installed, all dependencies resolved. |
| ✅ | Test suite passes | Re-run against the sanitized tree. In the working tree: **599 passed, 2 skipped**. In a fresh `git clone` of the rewritten public repository, in its own new venv, with no deposits and no network: **574 passed, 27 skipped** — the extra skips are the deposit-dependent tests declining themselves, which is the designed behaviour. |
| ✅ | Smoke tests pass | Phase 1, Phase 2N, Phase 3 and Phase 4 pipeline checks all run end to end on generated fixtures with no network. Both fixture generators reproduce their committed fixtures. |
| ✅ | No dependency on the private sibling repository | `grep -rIn '\.\./intervention'` → no matches. The fresh clone was installed and tested in a temporary directory with no sibling present. |
| ✅ | `git diff --check` clean | Clean. Three digest-pinned CSVs carry CRLF line endings deliberately — the SHA-256 in their provenance file is taken over those bytes — so `.gitattributes` disables line-ending conversion tree-wide and exempts them from the trailing-whitespace check rather than "fixing" the bytes a digest depends on. |
| ✅ | GitHub Actions green | Re-run after the history rewrite, on the sanitized root commit: run [`33802625353`](https://github.com/akshathsarukkai/interaction-transfer/actions/runs/33802625353), **all 7 jobs green** — `test (3.11)`, `test (3.12)`, `smoke`, `phase2-fixture`, `dchain-null`, `phase3-entity-ood`, `phase4-chemlex`. The only change made after that run was to this document; CI runs again on every push to `main`, so the current badge state is the live check. |

---

## Defects found and fixed while preparing this release

Four were found in documents that were about to be published. None changed a
number; all are recorded in `CHANGELOG.md`.

1. **The ChemLex substrate inventory was being redistributed** in five committed
   CSVs, contradicting the repository's own stated policy in six places.
2. **The Phase 4 results index had a live duplicate-aggregation hazard** that
   would have inflated its headline condition count on the next regeneration.
3. **The generated Phase 4 document never mentioned the flexible comparator's
   withdrawal**, while tabulating that dead rung 45 times.
4. **A hand-typed condition gradient survived in a document whose header says no
   number is typed**, and it contradicted the generated table beside it by
   reaching monotonicity through an omitted stratum.

Two further corrections: the split-grouping prose described three relations
where the code applies four (the fourth having been added by one of the three
defects that forced the corrected re-run), and a registered sensitivity that was
never implemented is now named rather than silently absent.

## Known and accepted

- **The GPL-3.0 quotation boundary.** Two files quote upstream d-chain source
  verbatim — ~10 lines of R and ~27 lines of C++ — so that a port and a patch set
  can be verified line by line. Both carry an upstream copyright and licence
  notice; `THIRD_PARTY_DATA.md` states that those fragments are not relicensed.
  That this is compatible with an Apache-2.0 release of the project's own code is
  a judgement, and it is documented publicly as a judgement rather than presented
  as settled.
- **Commit authorship uses a GitHub `users.noreply.github.com` address.** The
  first public push inherited the author's institutional email address from the
  global Git config; so did a later commit made through the GitHub web editor,
  and two of the pre-rewrite commit messages each also carried a trailer
  pointing at an assistant session URL. Both have since been removed by
  rewriting the public history into a single fresh root commit and force-updating
  `main` and `v1.0.0`. The old objects are unreachable here, but see the
  limitation below.

## Limitations of this audit

- **Scanning was pattern-based, not entropy-based.** Neither `gitleaks` nor
  `trufflehog` was installed on the machine that prepared this release and
  neither was installed for it; no tool is claimed here that was not run. What
  was run is the two-part scan described above — the tracked tree and every
  object reachable from every ref — using explicit provider-token, private-key,
  credential-filename, personal-address and machine-path patterns. A secret in a
  format matching none of those patterns, and carrying no keyword, would not have
  been caught. Nothing in this repository is generated by, or authenticates to,
  any credentialed service, which is why the residual risk is judged low rather
  than merely asserted to be zero.
- **Binary artifacts were scanned as bytes.** The six committed PNG figures were
  included in the reachable-object scan as raw bytes; no text extraction or
  metadata (EXIF) inspection was performed on them. They are matplotlib output
  from committed result files.
- **Rewritten history survives on GitHub until it garbage-collects.** This was
  checked rather than assumed: immediately after the force-update, the three
  superseded commit SHAs were still resolvable through the GitHub API, and so are
  still reachable by anyone who knows or guesses a full SHA. They are unreachable
  from every ref, they will not appear in a clone, a fetch or the commit list, and
  they are removed when GitHub next garbage-collects the repository — which its
  support team can be asked to do on request. The repository has **0 forks and a
  network size of 0**, so no fork network is holding the old objects open.
  Separately, anyone who cloned between the first push and this rewrite still
  holds the old metadata; a rewrite cannot retract copies already taken. The
  exposed values were an institutional email address and assistant session URLs —
  neither is a credential, so nothing required rotation.
