"""Fresh-process import boundaries for the two disjoint reporting owners."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[3]
PACKAGE_ROOT = PROJECT_ROOT / "src/protein_lm/data/fixed_budget_audit"
A004_MODULE = "protein_lm.data.fixed_budget_audit.reporting"
DIAGNOSTIC_MODULE = "protein_lm.data.fixed_budget_audit.diagnostic_reporting"
VALIDATION_MODULE = "protein_lm.data.fixed_budget_audit.validation"
A004_WORKFLOW_MODULE = "protein_lm.data.fixed_budget_audit.workflow"
DIAGNOSTIC_WORKFLOW_MODULE = "protein_lm.data.fixed_budget_audit.diagnostic_workflow"


def test_a004_reporting_import_does_not_load_diagnostic_dependencies() -> None:
    _assert_fresh_import_excludes(
        A004_MODULE,
        {
            "protein_lm.data.task5_report",
            "protein_lm.data.random_split",
            DIAGNOSTIC_MODULE,
            "protein_lm.data.fixed_budget_audit.tracks",
        },
    )


def test_diagnostic_reporting_import_does_not_load_a004_dependencies() -> None:
    _assert_fresh_import_excludes(
        DIAGNOSTIC_MODULE,
        {
            A004_MODULE,
            "protein_lm.data.fixed_budget_audit.tracks",
        },
    )


def test_reporting_owners_do_not_import_each_other() -> None:
    a004_imports = _source_imports(PACKAGE_ROOT / "reporting.py")
    diagnostic_imports = _source_imports(PACKAGE_ROOT / "diagnostic_reporting.py")

    assert DIAGNOSTIC_MODULE not in a004_imports
    assert A004_MODULE not in diagnostic_imports


def test_validation_import_does_not_load_workflow() -> None:
    _assert_fresh_import_excludes(
        VALIDATION_MODULE,
        {A004_WORKFLOW_MODULE},
    )


def test_validation_has_no_legacy_task7_imports() -> None:
    imported = _source_imports(PACKAGE_ROOT / "validation.py")

    assert not {
        module for module in imported if module.startswith("protein_lm.data.task7")
    }


def test_diagnostic_workflow_import_does_not_load_a004_workflow_dependencies() -> None:
    _assert_fresh_import_excludes(
        DIAGNOSTIC_WORKFLOW_MODULE,
        {
            A004_WORKFLOW_MODULE,
            A004_MODULE,
            "protein_lm.data.fixed_budget_audit.tracks",
        },
    )


def test_a004_workflow_import_does_not_load_diagnostic_workflow() -> None:
    _assert_fresh_import_excludes(
        A004_WORKFLOW_MODULE,
        {DIAGNOSTIC_WORKFLOW_MODULE, DIAGNOSTIC_MODULE},
    )


def test_workflow_siblings_are_isolated_without_a_direct_import_cycle() -> None:
    a004_imports = _source_imports(PACKAGE_ROOT / "workflow.py")
    diagnostic_imports = _source_imports(PACKAGE_ROOT / "diagnostic_workflow.py")

    assert DIAGNOSTIC_WORKFLOW_MODULE not in a004_imports
    assert A004_WORKFLOW_MODULE not in diagnostic_imports
    assert A004_MODULE not in diagnostic_imports
    assert "protein_lm.data.fixed_budget_audit.tracks" not in diagnostic_imports


def test_legacy_workflow_module_files_are_absent() -> None:
    data_root = PACKAGE_ROOT.parent

    assert not (data_root / "task7_a004_workflow.py").exists()
    assert not (data_root / "task7_workflow.py").exists()


def test_runtime_sources_exclude_legacy_workflow_names_imports_and_flags() -> None:
    runtime_paths = [
        *sorted((PROJECT_ROOT / "src").rglob("*.py")),
        *sorted((PROJECT_ROOT / "scripts").glob("*.py")),
        PROJECT_ROOT / "scripts/README.md",
    ]
    runtime_text = "\n".join(path.read_text(encoding="utf-8") for path in runtime_paths)

    for forbidden in (
        "protein_lm.data.task7_a004_workflow",
        "protein_lm.data.task7_workflow",
        "A004WorkflowResult",
        "run_a004_fixed_budget_audit",
        "--execute-fixed-budget-audit",
        "--execute-diagnostic-audit",
    ):
        assert forbidden not in runtime_text


def test_fixed_budget_package_import_graph_is_acyclic() -> None:
    package = "protein_lm.data.fixed_budget_audit"
    module_paths = {
        package if path.name == "__init__.py" else f"{package}.{path.stem}": path
        for path in PACKAGE_ROOT.glob("*.py")
    }
    graph = {
        module: _source_imports(path) & module_paths.keys()
        for module, path in module_paths.items()
    }
    visited: set[str] = set()
    active: list[str] = []

    def visit(module: str) -> None:
        if module in active:
            cycle = " -> ".join((*active[active.index(module) :], module))
            raise AssertionError(f"fixed-budget import cycle: {cycle}")
        if module in visited:
            return
        active.append(module)
        for dependency in sorted(graph[module]):
            visit(dependency)
        active.pop()
        visited.add(module)

    for module in sorted(graph):
        visit(module)


def _assert_fresh_import_excludes(module: str, forbidden: set[str]) -> None:
    code = (
        "import sys; "
        f"import {module}; "
        f"forbidden = {sorted(forbidden)!r}; "
        "loaded = sorted(set(forbidden) & set(sys.modules)); "
        "assert not loaded, loaded"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def _source_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return imported
