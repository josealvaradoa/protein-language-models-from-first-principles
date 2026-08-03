import importlib.util
import json
from pathlib import Path

import pytest

import protein_lm.data.task7_a004_workflow as workflow_module
from protein_lm.data.similarity_audit_policy import SimilarityAuditError
from protein_lm.data.task7_a004_report_payload import (
    COMMON_RESULT,
    STAGED_RESULT,
    validate_report_payload,
)
from protein_lm.data.task7_a004_plan import fixed_budget_stage_plan
from protein_lm.data.task7_a004_workflow import (
    run_a004_fixed_budget_audit,
    validate_a004_configuration,
)
from protein_lm.data.task7_checkpoints import file_identity
from a004_workflow_test_support import install_synthetic_workflow

PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "week_01"
    / "read_only_similarity_audit_a004.toml"
)
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_read_only_fixed_budget_audit.py"


def test_plan_imports_one_residual_pass_and_fresh_runs_the_other_seven() -> None:
    plan = fixed_budget_stage_plan()

    assert len(plan) == 8
    assert [item.origin for item in plan].count("imported_a003") == 1
    assert [item.origin for item in plan].count("executed_a004") == 7
    assert plan[0].strategy == "random"
    assert plan[0].partition == "validation"
    assert plan[0].pass_name == "residual"
    assert ("random", "validation", "enforcement") in {
        (item.strategy, item.partition, item.pass_name)
        for item in plan
        if item.origin == "executed_a004"
    }


def test_configuration_validation_keeps_a004_workspace_separate() -> None:
    configuration = validate_a004_configuration(
        project_root=PROJECT_ROOT, config_path=CONFIG_PATH
    )

    assert configuration.policy.model_use == "prohibited"
    assert configuration.policy.task8_membership_use_authorized is False
    assert configuration.paths["workspace"] != configuration.paths["source_workspace"]
    assert configuration.source_policy.workspace_relative_path == (
        configuration.policy.source_workspace_relative_path
    )


def test_cli_requires_an_explicit_execution_flag() -> None:
    module = _load_script()

    assert module.parse_args([]).execute_fixed_budget_audit is False
    assert module.parse_args(["--execute-fixed-budget-audit"]).execute_fixed_budget_audit is True


def test_synthetic_top_level_publishes_report_receipt_and_completion(
    monkeypatch,
    tmp_path: Path,
) -> None:
    synthetic = install_synthetic_workflow(monkeypatch, tmp_path)

    result = run_a004_fixed_budget_audit(
        project_root=synthetic.project_root,
        config_path=synthetic.config_path,
        search_runner=synthetic.search_runner,
        database_runner=synthetic.database_runner,
        hardware=synthetic.hardware,
    )

    assert len(synthetic.database_calls) == 2
    assert len(synthetic.search_calls) == 14
    assert result.completion_path.is_file()
    report_path = synthetic.workspace / "evidence/report/a004_report.json"
    markdown_path = synthetic.workspace / "evidence/report/a004_report.md"
    report = json.loads(report_path.read_text())
    receipt = json.loads(result.receipt_path.read_text())
    serialized = json.dumps(report, sort_keys=True)
    assert report["hardware"] == synthetic.hardware
    assert report["diagnostic_assignments_unchanged"] is True
    assert report["result_semantics"]["common_result_name"] == COMMON_RESULT
    assert report["result_semantics"]["staged_result_name"] == STAGED_RESULT
    assert "all_query_100000" not in serialized
    assert {track["source_label"] for track in report["tracks"]} == {
        "imported_a003",
        "executed_a004",
    }
    assert all("cap_sensitivity" in track for track in report["tracks"])
    assert all(
        result_record[COMMON_RESULT]["rate"]["denominator"] == 1
        and result_record[STAGED_RESULT]["rate"]["denominator"] == 1
        for result_record in report["partition_results"]
    )
    assert receipt["hardware"] == synthetic.hardware
    assert receipt["diagnostic_assignments"]["unchanged"] is True
    assert receipt["report"]["json"] == file_identity(report_path)
    assert receipt["report"]["markdown"] == file_identity(markdown_path)
    assert all(
        "staged_union_with_changed_query_100000" in record
        for record in receipt["pair_unions"].values()
    )
    assert "## Cap sensitivity" in markdown_path.read_text()
    invalid_report = {**report, "diagnostic_assignments_unchanged": False}
    with pytest.raises(SimilarityAuditError, match="authority or schema drifted"):
        validate_report_payload(invalid_report, fingerprint=result.fingerprint)


def test_final_gate_rejects_canonical_tamper_before_completion(
    monkeypatch,
    tmp_path: Path,
) -> None:
    synthetic = install_synthetic_workflow(monkeypatch, tmp_path)
    publish_receipt = workflow_module.publish_receipt

    def publish_then_tamper(**kwargs):
        publication = publish_receipt(**kwargs)
        executed = next(
            track
            for track in kwargs["tracks"].values()
            if track.plan.origin == "executed_a004"
        )
        canonical = executed.stages[0].canonical_path
        canonical.write_bytes(canonical.read_bytes() + b"tampered\n")
        return publication

    monkeypatch.setattr(workflow_module, "publish_receipt", publish_then_tamper)

    with pytest.raises(SimilarityAuditError, match="checksum drifted"):
        run_a004_fixed_budget_audit(
            project_root=synthetic.project_root,
            config_path=synthetic.config_path,
            search_runner=synthetic.search_runner,
            database_runner=synthetic.database_runner,
            hardware=synthetic.hardware,
        )

    assert not (synthetic.workspace / "a004_complete.json").exists()
    assert (synthetic.workspace / "a004_import_receipt.json").is_file()


def _load_script():
    spec = importlib.util.spec_from_file_location("a004_cli", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
