"""No-argument CLI dry-run characterization."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

REPOSITORY = Path(__file__).parents[3]
SCRIPT = REPOSITORY / "scripts/run_read_only_fixed_budget_audit.py"


def test_main_without_arguments_prints_exact_plan_and_writes_nothing(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    module = _load_script()
    project_root = tmp_path / "project"
    config_path = project_root / "experiments/week_01/a004.toml"
    source_workspace = project_root / "synthetic-a003"
    workspace = project_root / "synthetic-a004"
    validation_calls = []
    workflow_calls = []

    def validate(**kwargs):
        validation_calls.append(kwargs)
        return SimpleNamespace(
            paths={
                "source_workspace": source_workspace,
                "workspace": workspace,
            }
        )

    def forbidden_workflow(**kwargs):
        workflow_calls.append(kwargs)
        raise AssertionError("main([]) must not invoke the workflow")

    monkeypatch.setattr(module, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(module, "validate_a004_configuration", validate)
    monkeypatch.setattr(module, "run_a004_fixed_budget_audit", forbidden_workflow)

    result = module.main([])

    expected = "\n".join(
        (
            "A-004 fixed-budget stage plan:",
            "- imported_a003: random/validation/residual at all-query caps 1,000 "
            "and 10,000; 100,000 only for changed queries",
            "- executed_a004: random/validation/enforcement at all-query caps "
            "1,000 and 10,000; 100,000 only for changed queries",
            "- executed_a004: random/test/enforcement at all-query caps 1,000 "
            "and 10,000; 100,000 only for changed queries",
            "- executed_a004: random/test/residual at all-query caps 1,000 and "
            "10,000; 100,000 only for changed queries",
            "- executed_a004: group_aware/validation/enforcement at all-query "
            "caps 1,000 and 10,000; 100,000 only for changed queries",
            "- executed_a004: group_aware/validation/residual at all-query caps "
            "1,000 and 10,000; 100,000 only for changed queries",
            "- executed_a004: group_aware/test/enforcement at all-query caps "
            "1,000 and 10,000; 100,000 only for changed queries",
            "- executed_a004: group_aware/test/residual at all-query caps 1,000 "
            "and 10,000; 100,000 only for changed queries",
            f"- source workspace: {source_workspace}",
            f"- A-004 workspace: {workspace}",
            "Configuration valid. No database, search, or evidence output was created.",
            "Network requests made: none",
            "",
        )
    )
    assert result == 0
    assert capsys.readouterr().out == expected
    assert validation_calls == [
        {"project_root": project_root, "config_path": config_path}
    ]
    assert workflow_calls == []
    assert not tmp_path.exists() or list(tmp_path.rglob("*")) == []


def _load_script():
    spec = importlib.util.spec_from_file_location("a004_characterization_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
