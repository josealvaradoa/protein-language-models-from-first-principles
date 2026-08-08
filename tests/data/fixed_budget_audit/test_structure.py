"""Independent filesystem and import contracts for the audit package."""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[3]
DATA_ROOT = PROJECT_ROOT / "src/protein_lm/data"
PACKAGE_ROOT = DATA_ROOT / "fixed_budget_audit"
EXPECTED_MODULES = (
    "__init__",
    "config",
    "diagnostic_reporting",
    "diagnostic_workflow",
    "errors",
    "evidence",
    "execution",
    "provenance",
    "reporting",
    "search",
    "source",
    "tracks",
    "validation",
    "workflow",
)


def test_fixed_budget_audit_has_exact_python_module_inventory() -> None:
    discovered = tuple(
        sorted(path.stem for path in PACKAGE_ROOT.glob("*.py") if path.is_file())
    )

    assert discovered == EXPECTED_MODULES


def test_runtime_has_no_legacy_task7_modules() -> None:
    assert (
        tuple(sorted(path for path in DATA_ROOT.rglob("task7_*.py") if path.is_file()))
        == ()
    )


def test_runtime_has_no_legacy_task7_imports() -> None:
    runtime_paths = (
        *sorted((PROJECT_ROOT / "src").rglob("*.py")),
        *sorted((PROJECT_ROOT / "scripts").rglob("*.py")),
    )
    legacy_imports = {
        (path.relative_to(PROJECT_ROOT).as_posix(), imported)
        for path in runtime_paths
        for imported in _imported_modules(path)
        if imported.startswith("protein_lm.data.task7_")
    }

    assert legacy_imports == set()


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
            imported.update(
                f"{node.module}.{alias.name}"
                for alias in node.names
                if alias.name != "*"
            )
    return imported
