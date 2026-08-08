"""Top-level fixed-budget audit workflow and final-gate contracts."""

import importlib.util
import json
from pathlib import Path

import pytest

import protein_lm.data.fixed_budget_audit.workflow as workflow_module
from protein_lm.data.artifacts import file_identity
from protein_lm.data.fixed_budget_audit.errors import (
    AuditConfigurationError,
    AuditValidationError,
    SourceEvidenceError,
)
from protein_lm.data.fixed_budget_audit.reporting import (
    COMMON_RESULT,
    STAGED_RESULT,
    validate_report_payload,
)
from protein_lm.data.fixed_budget_audit.config import TrackOrigin
from protein_lm.data.fixed_budget_audit.workflow import (
    run_fixed_budget_audit,
    validate_a004_configuration,
)
from a004_workflow_test_support import install_synthetic_workflow

PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "experiments" / "week_01" / "read_only_similarity_audit_a004.toml"
)
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_read_only_fixed_budget_audit.py"


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

    assert module.parse_args([]).execute_searches is False
    assert module.parse_args(["--execute-searches"]).execute_searches is True


def test_synthetic_top_level_publishes_report_receipt_and_completion(
    monkeypatch,
    tmp_path: Path,
) -> None:
    synthetic = install_synthetic_workflow(monkeypatch, tmp_path)

    result = run_fixed_budget_audit(
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
    with pytest.raises(AuditValidationError, match="authority or schema drifted"):
        validate_report_payload(invalid_report, fingerprint=result.fingerprint)


def test_synthetic_top_level_rejects_malformed_hardware_before_execution(
    monkeypatch,
    tmp_path: Path,
) -> None:
    synthetic = install_synthetic_workflow(monkeypatch, tmp_path)

    with pytest.raises(AuditConfigurationError, match="provenance is malformed"):
        run_fixed_budget_audit(
            project_root=synthetic.project_root,
            config_path=synthetic.config_path,
            search_runner=synthetic.search_runner,
            database_runner=synthetic.database_runner,
            hardware=object(),  # type: ignore[arg-type]
        )

    assert synthetic.database_calls == []
    assert synthetic.search_calls == []


def test_workflow_builds_final_context_before_publication_and_reuses_it(
    monkeypatch,
    tmp_path: Path,
) -> None:
    synthetic = install_synthetic_workflow(monkeypatch, tmp_path)
    build_context = workflow_module.build_final_validation_context
    publish_report = workflow_module.publish_a004_report
    publish_receipt = workflow_module.publish_receipt
    final_validation = workflow_module.revalidate_before_completion
    events: list[str] = []
    captured: dict[str, object] = {}

    def capture_context(**kwargs):
        events.append("context")
        context = build_context(**kwargs)
        captured["context"] = context
        assert kwargs["source_paths"] == synthetic.source_paths
        assert kwargs["source_policy_path"] == synthetic.source_policy_path
        return context

    def capture_report(**kwargs):
        events.append("report")
        assert "context" in captured
        return publish_report(**kwargs)

    def capture_receipt(**kwargs):
        events.append("receipt")
        return publish_receipt(**kwargs)

    def capture_final_validation(**kwargs):
        events.append("final_validation")
        assert kwargs["context"] is captured["context"]
        return final_validation(**kwargs)

    monkeypatch.setattr(
        workflow_module,
        "build_final_validation_context",
        capture_context,
    )
    monkeypatch.setattr(workflow_module, "publish_a004_report", capture_report)
    monkeypatch.setattr(workflow_module, "publish_receipt", capture_receipt)
    monkeypatch.setattr(
        workflow_module,
        "revalidate_before_completion",
        capture_final_validation,
    )

    run_fixed_budget_audit(
        project_root=synthetic.project_root,
        config_path=synthetic.config_path,
        search_runner=synthetic.search_runner,
        database_runner=synthetic.database_runner,
        hardware=synthetic.hardware,
    )

    assert events == ["context", "report", "receipt", "final_validation"]


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
            if track.plan.origin is TrackOrigin.EXECUTED_A004
        )
        canonical = executed.stages[0].canonical_path
        canonical.write_bytes(canonical.read_bytes() + b"tampered\n")
        return publication

    monkeypatch.setattr(workflow_module, "publish_receipt", publish_then_tamper)

    with pytest.raises(AuditValidationError, match="checksum drifted"):
        run_fixed_budget_audit(
            project_root=synthetic.project_root,
            config_path=synthetic.config_path,
            search_runner=synthetic.search_runner,
            database_runner=synthetic.database_runner,
            hardware=synthetic.hardware,
        )

    assert not (synthetic.workspace / "a004_complete.json").exists()
    assert (synthetic.workspace / "a004_import_receipt.json").is_file()


@pytest.mark.parametrize(
    ("source_name", "suffix", "message"),
    (
        (
            "source_policy",
            b"\n# valid post-receipt source-policy drift\n",
            "source policy changed",
        ),
        (
            "task5_local",
            b"post-receipt Task 5 mutation\n",
            "frozen task5_local checksum changed",
        ),
        (
            "task6_report",
            b"post-receipt Task 6 mutation\n",
            "frozen task6_report checksum changed",
        ),
    ),
)
def test_final_gate_rechecks_frozen_sources_after_receipt(
    monkeypatch,
    tmp_path: Path,
    source_name: str,
    suffix: bytes,
    message: str,
) -> None:
    synthetic = install_synthetic_workflow(monkeypatch, tmp_path)
    publish_receipt = workflow_module.publish_receipt
    target = (
        synthetic.source_policy_path
        if source_name == "source_policy"
        else synthetic.source_paths[source_name]
    )
    original = target.read_bytes()

    def publish_then_mutate(**kwargs):
        publication = publish_receipt(**kwargs)
        target.write_bytes(original + suffix)
        return publication

    monkeypatch.setattr(workflow_module, "publish_receipt", publish_then_mutate)

    with pytest.raises(SourceEvidenceError, match=message):
        run_fixed_budget_audit(
            project_root=synthetic.project_root,
            config_path=synthetic.config_path,
            search_runner=synthetic.search_runner,
            database_runner=synthetic.database_runner,
            hardware=synthetic.hardware,
        )

    assert target.read_bytes() == original + suffix
    assert (synthetic.workspace / "a004_import_receipt.json").is_file()
    assert not (synthetic.workspace / "a004_complete.json").exists()


def _load_script():
    spec = importlib.util.spec_from_file_location("a004_cli", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
