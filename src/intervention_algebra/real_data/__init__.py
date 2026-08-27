"""Phase 2: the intervention algebra on real sequential-intervention data.

Deliberately kept separate from the Phase 1 synthetic package. Phase 1 is frozen
at the ``phase1-final`` tag and nothing in here imports from it, so a change on
this side cannot alter a Phase 1 number.

The Phase 1 benchmark supplied four row types per unordered pair (two singles, a
simultaneous row, and two ordered rows). The Koplev screen supplies only the two
ordered rows. The models here therefore encode strictly less structure than the
Phase 1 ``AlgebraModel`` does -- see :mod:`.models`.
"""

from .koplev import (  # noqa: F401
    SOURCES,
    KOPLEV_DOI,
    KOPLEV_PAPER,
    Screen,
    audit_screen,
    download_raw,
    load_screen,
    verify_raw,
)

__all__ = [
    "SOURCES",
    "KOPLEV_DOI",
    "KOPLEV_PAPER",
    "Screen",
    "audit_screen",
    "download_raw",
    "load_screen",
    "verify_raw",
]
