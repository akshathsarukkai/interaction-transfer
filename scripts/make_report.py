"""Turn the Phase 1 results JSONL into the markdown that goes in the README.

    python scripts/make_report.py --results results/phase1.jsonl \
        --outdir results/summary --inject README.md

``--inject`` replaces everything between the ``<!-- RESULTS -->`` marker and the
next ``##`` heading in the given file, so the README's reported numbers can never
drift from the results file they claim to summarise.

The table-building logic lives in :mod:`intervention_algebra.report` so that it
is importable and tested (``tests/test_report.py``); this file is the CLI only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from intervention_algebra import analysis, report


BEGIN = "<!-- RESULTS -->"
END = "<!-- END RESULTS -->"


def inject(path: Path, body: str, begin: str = BEGIN, end: str = END) -> None:
    """Replace the text between ``begin`` and ``end``, and nothing else.

    The first version of this function replaced everything from the marker to
    the next ``##`` heading. That is fine on a document where the marker is
    immediately followed by the next section, and catastrophic on this one: the
    whole of section 8's hand-written analysis -- the coverage-by-coverage
    reading, the controls, the misspecification sweep, the capacity retraction,
    the multiplicity accounting -- sat between the marker and section 9, so the
    command advertised as "regenerates the reported tables" deleted several
    hundred lines of argument and replaced them with tables. It had simply never
    been run against a README that had prose there.

    An explicit end marker makes the destroyed region visible in the document
    itself, and a missing end marker is now an error rather than a licence to
    consume the rest of the section.
    """
    text = path.read_text()
    if begin not in text:
        raise SystemExit(f"marker {begin!r} not found in {path}")
    if end not in text:
        raise SystemExit(
            f"marker {end!r} not found in {path}. Refusing to guess where the "
            f"generated block ends -- the previous behaviour (delete to the next "
            f"'##' heading) silently destroyed hand-written analysis. Add "
            f"{end!r} immediately after the generated block.")
    head, rest = text.split(begin, 1)
    _, tail = rest.split(end, 1)
    path.write_text(f"{head}{begin}\n\n{body}\n{end}{tail}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/phase1.jsonl")
    ap.add_argument("--outdir", default="results/summary")
    ap.add_argument("--inject", default=None)
    args = ap.parse_args(argv)

    rows, errors = analysis.load(args.results, return_errors=True)
    if errors:
        print(f"WARNING: {len(errors)} failed runs in {args.results}",
              file=sys.stderr)
    df = analysis.to_frame(rows)
    if df.empty:
        raise SystemExit("no successful runs")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    body = report.build_report(df)
    (outdir / "report.md").write_text(body)
    df.to_csv(outdir / "runs.csv", index=False)

    # Figures are drawn from the *powered* cells for the same reason the tables
    # are: a figure built from tag="main" plots five seeds under a caption that
    # claims seventeen. `tag=None` is safe here only because `powered_frame` has
    # already restricted the rows to one condition per (family, coverage).
    powered = report.add_powered_skill(report.powered_frame(df))
    figs = analysis.make_figures(powered, outdir, tag=None)
    print(f"wrote {outdir/'report.md'}, {outdir/'runs.csv'}, {len(figs)} figures",
          file=sys.stderr)

    if args.inject:
        inject(Path(args.inject), body)
        print(f"injected results into {args.inject}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
