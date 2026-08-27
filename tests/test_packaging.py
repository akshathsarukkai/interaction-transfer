"""Every third-party module the code imports is declared as a dependency.

This repository has now shipped the same defect twice. Phase 3 imported `rdkit`
and `requests` without declaring them, and CI -- which installs only what
`pyproject.toml` says -- failed sixteen tests with `ModuleNotFoundError` while
the local suite was green. Phase 4 did it again with `openpyxl`, the Excel engine
pandas needs to read the deposit, and forty tests failed the same way.

It is the worst kind of defect: correct code that nobody else can run, invisible
on the machine where it was written, and it presents as a mystery CI break rather
than as a missing line. So it becomes a test.

The check is deliberately crude -- a regex for top-level imports over the source
tree, a small module-to-distribution map, and a set difference. A crude check
that runs on every commit beats an accurate one that nobody wrote.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
SCRIPTS = REPO / "scripts"

#: Import name -> distribution name, where they differ.
_DISTRIBUTION = {
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "PIL": "pillow",
}

#: Modules that are part of the standard library on every supported Python, plus
#: this project's own package. `sys.stdlib_module_names` covers the first.
_LOCAL = {"intervention_algebra", "scripts", "tests", "conftest"}


def _imported_top_level(root: Path) -> set[str]:
    names: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                # `from . import x` and `from ..y import z` are relative.
                if node.level == 0 and node.module:
                    names.add(node.module.split(".")[0])
    return names


def _declared() -> set[str]:
    text = (REPO / "pyproject.toml").read_text()
    block = text[text.index("dependencies = ["):]
    block = block[:block.index("]")]
    out = set()
    for line in block.splitlines():
        line = line.split("#")[0].strip().strip(",").strip('"')
        if not line or line.startswith("dependencies"):
            continue
        name = re.split(r"[<>=!~\[]", line)[0].strip()
        if name:
            out.add(name.lower())
    return out


def test_every_imported_third_party_module_is_declared():
    imported = _imported_top_level(SRC) | _imported_top_level(SCRIPTS)
    third_party = {
        n for n in imported
        if n not in sys.stdlib_module_names and n not in _LOCAL
    }
    declared = _declared()
    missing = sorted(
        n for n in third_party
        if _DISTRIBUTION.get(n, n).lower() not in declared)
    assert not missing, (
        f"imported but not declared in pyproject.toml: {missing}. "
        f"This is the defect that made CI fail while the local suite was green, "
        f"twice. Add it to [project].dependencies.")


def test_the_excel_engine_pandas_needs_is_declared():
    """Named separately because the import is inside pandas, not inside us.

    Nothing in this repository writes `import openpyxl`; `pd.read_excel` does,
    at call time, and raises an ImportError that the generic check above cannot
    see. The Phase 4 deposit is an .xlsx, so this is load-bearing.
    """
    assert "openpyxl" in _declared()


def test_the_declared_dependencies_are_importable_here():
    """A cheap smoke test that the current environment matches the declaration."""
    import importlib
    for dist in sorted(_declared()):
        module = {"scikit-learn": "sklearn", "pyyaml": "yaml"}.get(dist, dist)
        importlib.import_module(module.replace("-", "_"))
