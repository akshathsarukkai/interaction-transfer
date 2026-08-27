"""Shared test configuration.

Two jobs:

1. **Import the source tree in this repo, not a stale editable install.**
   ``<repo>/src`` is inserted at the *front* of ``sys.path`` so that
   ``import intervention_algebra`` resolves here even if a ``.pth`` from an old
   ``pip install -e`` elsewhere on the machine points at a different checkout.
   Getting this wrong is silent: the suite would pass against code nobody is
   editing.

2. **Pin torch to a single thread.**  Every system in this suite is tiny; the
   thread pool costs more than it saves and, more importantly, multi-threaded
   reductions are not bit-reproducible, which
   ``tests/test_reproducibility.py`` asserts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"

# Front of the path, and remove any earlier occurrence so re-entry is idempotent.
_src = str(SRC)
while _src in sys.path:
    sys.path.remove(_src)
sys.path.insert(0, _src)

# The repository root too, so `import scripts.run_phase3_entity_ood` resolves.
# Tests that exercise a runner's `main()` need the runner importable, and a bare
# `pytest` from the repo root happened to supply this via the working directory
# on some machines and not in CI -- where every such test failed with
# ModuleNotFoundError: No module named 'scripts'.
_root = str(REPO_ROOT)
while _root in sys.path:
    sys.path.remove(_root)
sys.path.insert(1, _root)

import torch  # noqa: E402  (must follow the sys.path fix-up)

torch.set_num_threads(1)


def pytest_report_header(config):  # pragma: no cover - diagnostic only
    import intervention_algebra as ia

    return f"intervention_algebra from: {ia.__file__}"


#: The Mendeley deposit is 1.7 MB of third-party data, is gitignored, and is
#: fetched by `scripts/download_koplev.py`. Tests that genuinely need it must
#: skip without it: a green build must never require a third-party host to be
#: up, and "the data is missing" must not be reported as "the code is broken".
#: One definition, so a new test cannot invent a different notion of present.
DEPOSIT = REPO_ROOT / "data" / "raw" / "koplev2017"


def deposit_available() -> bool:
    return (DEPOSIT / "Data Table 1.csv").exists()


requires_deposit = pytest.mark.skipif(
    not deposit_available(),
    reason="the Koplev deposit is absent; run scripts/download_koplev.py")
