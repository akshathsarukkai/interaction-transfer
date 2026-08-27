# Archiving this release on Zenodo

This repository ships `.zenodo.json` so that a Zenodo archive picks up the right
title, authors, licence, keywords and related dataset DOIs automatically.

**There is no DOI yet, and none is claimed anywhere in this repository.**
`CITATION.cff` deliberately has no `doi:` field and the README's citation block
has no DOI. Inventing one would be worse than not having one. The steps below
mint a real DOI and then wire it in.

Zenodo needs an account action that cannot be performed from this repository, so
these steps are for a human.

## 1. Connect GitHub to Zenodo

1. Go to <https://zenodo.org> and sign in — "Log in with GitHub" is the simplest
   route and creates the link in one step.
2. Open <https://zenodo.org/account/settings/github/>.
3. Authorise Zenodo for your GitHub account if prompted. Zenodo needs
   `admin:repo_hook` and `read:org` to install the release webhook.

## 2. Enable the repository

1. On that same page, find `akshathsarukkai/interaction-transfer` in the list.
2. Flip its toggle to **ON**.

If it is not listed, click **Sync now**. A repository must be **public** for
Zenodo to archive it, and the list only refreshes on sync.

Zenodo archives releases created **after** the toggle is switched on. Enabling it
does not retroactively archive `v1.0.0`.

## 3. Create — or recreate — the v1.0.0 release

If `v1.0.0` was published **before** you enabled the toggle, Zenodo has not seen
it. Recreate it so the webhook fires:

```bash
gh release delete v1.0.0 --repo akshathsarukkai/interaction-transfer --yes
git push --delete origin v1.0.0
git push origin v1.0.0
gh release create v1.0.0 \
  --repo akshathsarukkai/interaction-transfer \
  --title "v1.0.0 — Interaction transfer through Phase 4" \
  --notes-file RELEASE_NOTES_v1.0.0.md
```

Deleting and re-pushing an annotated tag that points at the same commit changes
nothing about the code; it only re-fires the webhook. If the release was created
after the toggle was on, skip this step entirely.

## 4. Let Zenodo archive it

Archiving usually completes within a few minutes. Watch
<https://zenodo.org/account/settings/github/> — the repository row gains a DOI
badge when the deposit is published.

You will get **two** DOIs:

- a **concept DOI**, which always resolves to the newest version — cite this one
  in `CITATION.cff` and the README, so the citation does not go stale;
- a **version DOI** for `v1.0.0` specifically — cite this when a reader must land
  on exactly this snapshot.

## 5. Retrieve the DOI

From the badge on the Zenodo settings page, or from the deposit page itself. The
concept DOI is shown on the record under "Cite all versions".

## 6. Wire the DOI in

Three places, in this order:

**`CITATION.cff`** — add above `version:`:

```yaml
doi: 10.5281/zenodo.XXXXXXX
identifiers:
  - type: doi
    value: 10.5281/zenodo.XXXXXXX
    description: Concept DOI, resolves to the latest version
```

**`README.md`** — add the badge under the existing ones:

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```

and add `doi = {10.5281/zenodo.XXXXXXX}` to the BibTeX block.

**`.zenodo.json`** — leave it alone. Zenodo owns the DOI; writing it back here
would be a hand-maintained copy of a field Zenodo already controls.

## 7. Only if a source change is needed

If wiring the DOI in is the *only* change, it does not need a new release — the
archive of `v1.0.0` is the snapshot people cite, and a README badge added
afterwards does not alter the science.

Cut `v1.0.1` only if something in the code or the results changes. If you do,
Zenodo will archive it automatically and the concept DOI will start resolving to
it, so nothing in `CITATION.cff` needs updating.

## What not to do

- Do not add a DOI to `CITATION.cff` or the README before Zenodo has minted one.
- Do not fabricate a DOI for a placeholder, a draft, or a test deposit.
- Do not use the **version** DOI in `CITATION.cff` where the concept DOI belongs;
  it pins readers to a snapshot that later versions supersede.
