"""Historical diagnostic report assembly, rendering, and publication contracts."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime
from decimal import InvalidOperation
from pathlib import Path
from types import SimpleNamespace

import protein_lm.data.fixed_budget_audit.diagnostic_reporting as diagnostic_reporting_module
import protein_lm.data.fixed_budget_audit.diagnostic_workflow as diagnostic_workflow_module
import pytest

from protein_lm.data.similarity_alignment import (
    CATEGORY_30_TO_40,
    CATEGORY_40_TO_50,
    CATEGORY_CLOSEST_PROHIBITED,
    CATEGORY_GE_50_LOW_COVERAGE,
    CATEGORY_PROHIBITED,
    CATEGORY_UNDER_30_OR_NONE,
)
from protein_lm.data.fixed_budget_audit.errors import (
    AuditPublicationError,
    AuditValidationError,
)
from protein_lm.data.fixed_budget_audit.diagnostic_reporting import (
    DIAGNOSTIC_COMPLETION_FILENAME,
    DIAGNOSTIC_COMPLETION_SCOPE,
    DIAGNOSTIC_OUTPUT_STEM,
    DIAGNOSTIC_REPORT_FILENAMES,
    RenderedDiagnosticReport,
    build_diagnostic_report,
    diagnostic_public_pass_evidence,
    publish_diagnostic_report,
    render_diagnostic_report,
)
from protein_lm.data.similarity_audit_models import FileEvidence
from protein_lm.data.similarity_audit_policy import load_similarity_audit_policy
from protein_lm.data.similarity_fastas import FastaEvidence

PROJECT_ROOT = Path(__file__).parents[3]
DIAGNOSTIC_POLICY = (
    PROJECT_ROOT / "experiments/week_01/diagnostic_similarity_audit.toml"
)
EXPECTED_MARKDOWN = Path(__file__).parent / "goldens/diagnostic_report.md"
EXPECTED_JSON_BYTE_SIZE = 8_910
EXPECTED_JSON_SHA256 = (
    "9d0e27b8681cfb54d070881e3c15c83a3246f93562da165be772765f2aaa0ba0"
)
EXPECTED_MARKDOWN_BYTE_SIZE = 4_456
EXPECTED_MARKDOWN_SHA256 = (
    "ba8a2518148bf34dbf721c8cff1cc837a30e353fa167da69952bb654ae500d5f"
)
EXPECTED_JSON_FILENAME = "task_07_diagnostic_similarity_audit.json"
EXPECTED_MARKDOWN_FILENAME = "task_07_diagnostic_similarity_audit.md"
EXPECTED_SIDECAR_FILENAME = "task_07_diagnostic_similarity_audit.sha256"
EXPECTED_COMPLETION_FILENAME = "task_07_diagnostic_similarity_audit.complete.json"
EXPECTED_REPORT_FILENAMES = (
    EXPECTED_JSON_FILENAME,
    EXPECTED_MARKDOWN_FILENAME,
    EXPECTED_SIDECAR_FILENAME,
)
EXPECTED_COMPLETION_SCOPE = "week_01_task_07_public_outputs"
EXPECTED_PUBLIC_CONTRACT = [
    "DIAGNOSTIC_OUTPUT_STEM",
    "DIAGNOSTIC_REPORT_FILENAMES",
    "DIAGNOSTIC_COMPLETION_FILENAME",
    "DIAGNOSTIC_COMPLETION_SCOPE",
    "RenderedDiagnosticReport",
    "diagnostic_public_pass_evidence",
    "diagnostic_structural_membership_evidence",
    "diagnostic_overall_similarity",
    "build_diagnostic_report",
    "render_diagnostic_report",
    "publish_diagnostic_report",
]


def test_diagnostic_public_contract_uses_pinned_literals() -> None:
    assert diagnostic_reporting_module.__all__ == EXPECTED_PUBLIC_CONTRACT
    assert DIAGNOSTIC_OUTPUT_STEM == "task_07_diagnostic_similarity_audit"
    assert DIAGNOSTIC_REPORT_FILENAMES == EXPECTED_REPORT_FILENAMES
    assert DIAGNOSTIC_COMPLETION_FILENAME == EXPECTED_COMPLETION_FILENAME
    assert DIAGNOSTIC_COMPLETION_SCOPE == EXPECTED_COMPLETION_SCOPE


def test_diagnostic_rendering_matches_independent_exact_bytes() -> None:
    report = _compact_diagnostic_report()

    rendered = render_diagnostic_report(report)

    expected_json = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    assert type(rendered) is RenderedDiagnosticReport
    assert rendered.json_text.encode("utf-8") == expected_json
    assert len(expected_json) == EXPECTED_JSON_BYTE_SIZE
    assert rendered.json_sha256 == hashlib.sha256(expected_json).hexdigest()
    assert rendered.json_sha256 == EXPECTED_JSON_SHA256
    markdown_bytes = rendered.markdown_text.encode("utf-8")
    assert markdown_bytes == EXPECTED_MARKDOWN.read_bytes()
    assert len(markdown_bytes) == EXPECTED_MARKDOWN_BYTE_SIZE
    assert hashlib.sha256(markdown_bytes).hexdigest() == EXPECTED_MARKDOWN_SHA256
    assert rendered.markdown_text.endswith("\n")
    assert not rendered.markdown_text.endswith("\n\n")


def test_diagnostic_guard_and_reconciliation_use_validation_error() -> None:
    report = _compact_diagnostic_report()
    drifted_authority = dict(report, selected_split_authorized=True)

    with pytest.raises(AuditValidationError, match="authority guard"):
        render_diagnostic_report(drifted_authority)

    drifted_counts = deepcopy(report)
    drifted_counts["strategies"]["random"]["overall"]["unique_prohibited_pairs"] = 3
    with pytest.raises(AuditValidationError, match="overall pair counts"):
        render_diagnostic_report(drifted_counts)


def test_diagnostic_public_pass_evidence_strips_private_query_ids() -> None:
    marker = {
        "schema_version": 1,
        "query_count": 2,
        "stages": {"1000": {"runtime_seconds": "0.100"}},
        "convergence": {
            "escalated_query_ids": ["PRIVATE_QUERY"],
            "changed_query_count": 1,
        },
        "accepted": {"cap": 10_000},
    }

    public = diagnostic_public_pass_evidence(marker)

    assert set(public) == {"query_count", "stages", "convergence", "accepted"}
    assert public["convergence"] == {"changed_query_count": 1}
    assert marker["convergence"]["escalated_query_ids"] == ["PRIVATE_QUERY"]
    assert public["stages"] is marker["stages"]
    assert public["accepted"] == marker["accepted"]
    assert public["accepted"] is not marker["accepted"]


def test_diagnostic_assembly_preserves_runtime_timestamp_and_hardware_contract(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        diagnostic_reporting_module.platform,
        "platform",
        lambda: "platform",
    )
    monkeypatch.setattr(
        diagnostic_reporting_module.platform,
        "machine",
        lambda: "machine",
    )
    monkeypatch.setattr(
        diagnostic_reporting_module.platform,
        "processor",
        lambda: "",
    )
    monkeypatch.setattr(diagnostic_reporting_module.os, "cpu_count", lambda: None)
    started_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    completed_at = datetime(2026, 1, 2, 3, 5, 6, tzinfo=UTC)
    inputs, manifests, database_reports, strategy_reports = _assembly_evidence()

    report = build_diagnostic_report(
        policy=load_similarity_audit_policy(DIAGNOSTIC_POLICY),
        code_revision="a" * 40,
        fingerprint="b" * 64,
        mmseqs_version="18-test",
        started_at=started_at,
        completed_at=completed_at,
        runtime_seconds=1.2346,
        inputs=inputs,  # type: ignore[arg-type]
        manifests=manifests,  # type: ignore[arg-type]
        frozen_reports={"random": {"sources": {"fixture": "identity"}}},
        database_reports=database_reports,
        strategy_reports=strategy_reports,
    )

    runtime = report["runtime"]
    assert runtime == {
        "final_invocation_started_at_utc": "2026-01-02T03:04:05+00:00",
        "completed_at_utc": "2026-01-02T03:05:06+00:00",
        "final_invocation_wall_clock_seconds": "1.235",
        "completed_mmseqs_command_count": 6,
        "completed_mmseqs_command_runtime_seconds": "4.000",
        "hardware": {
            "platform": "platform",
            "machine": "machine",
            "processor": "",
            "logical_cpu_count": None,
        },
        "workspace_byte_ceiling": 214_748_364_800,
        "free_space_reserve": 322_122_547_200,
    }
    assert report["code_revision"] == "a" * 40
    assert report["run_fingerprint"] == "b" * 64
    assert report["procedure"]["staged_caps"] == [1_000, 10_000, 100_000]
    assert report["inputs"]["source_checksums"] == {"fixture": "identity"}


def test_diagnostic_assembly_rejects_invalid_runtime_evidence() -> None:
    inputs, manifests, database_reports, strategy_reports = _assembly_evidence()
    database_reports["random"]["runtime_seconds"] = "-0.001"

    with pytest.raises(AuditValidationError, match="runtime evidence is invalid"):
        build_diagnostic_report(
            policy=load_similarity_audit_policy(DIAGNOSTIC_POLICY),
            code_revision="a" * 40,
            fingerprint="b" * 64,
            mmseqs_version="18-test",
            started_at=datetime(2026, 1, 2),
            completed_at=datetime(2026, 1, 3),
            runtime_seconds=1.0,
            inputs=inputs,  # type: ignore[arg-type]
            manifests=manifests,  # type: ignore[arg-type]
            frozen_reports={"random": {"sources": {}}},
            database_reports=database_reports,
            strategy_reports=strategy_reports,
        )


def test_diagnostic_assembly_propagates_decimal_parse_error() -> None:
    inputs, manifests, database_reports, strategy_reports = _assembly_evidence()
    database_reports["random"]["runtime_seconds"] = "not-a-decimal"

    with pytest.raises(InvalidOperation):
        build_diagnostic_report(
            policy=load_similarity_audit_policy(DIAGNOSTIC_POLICY),
            code_revision="a" * 40,
            fingerprint="b" * 64,
            mmseqs_version="18-test",
            started_at=datetime(2026, 1, 2),
            completed_at=datetime(2026, 1, 3),
            runtime_seconds=1.0,
            inputs=inputs,  # type: ignore[arg-type]
            manifests=manifests,  # type: ignore[arg-type]
            frozen_reports={"random": {"sources": {}}},
            database_reports=database_reports,
            strategy_reports=strategy_reports,
        )


def test_diagnostic_publication_matches_independent_bytes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    staging = workspace / "public_report_staging"
    (staging / "old").mkdir(parents=True)
    (staging / "old/stale.txt").write_text("stale")
    report_directory = tmp_path / "reports/week_01"
    json_bytes = b'{"compact": true}\n'
    markdown_bytes = b"# Compact diagnostic\n"
    digest = hashlib.sha256(json_bytes).hexdigest()
    rendered = RenderedDiagnosticReport(
        json_text=json_bytes.decode(),
        markdown_text=markdown_bytes.decode(),
        json_sha256=digest,
    )

    publish_diagnostic_report(rendered, workspace, report_directory)

    sidecar_bytes = f"{digest}  task_07_diagnostic_similarity_audit.json\n".encode(
        "ascii"
    )
    expected_outputs = {
        EXPECTED_JSON_FILENAME: json_bytes,
        EXPECTED_MARKDOWN_FILENAME: markdown_bytes,
        EXPECTED_SIDECAR_FILENAME: sidecar_bytes,
    }
    for filename, expected in expected_outputs.items():
        assert (report_directory / filename).read_bytes() == expected
    completion_payload = {
        "schema_version": 1,
        "scope": "week_01_task_07_public_outputs",
        "complete": True,
        "artifacts": {
            f"reports/week_01/{filename}": {
                "byte_size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for filename, content in expected_outputs.items()
        },
    }
    expected_completion = (
        json.dumps(completion_payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    assert (
        report_directory / EXPECTED_COMPLETION_FILENAME
    ).read_bytes() == expected_completion
    assert not staging.exists()


def test_diagnostic_publication_propagates_failure_without_completion(
    monkeypatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    report_directory = tmp_path / "reports/week_01"
    report_directory.mkdir(parents=True)
    completion_path = report_directory / EXPECTED_COMPLETION_FILENAME
    completion_path.write_text("previous completion\n")
    rendered = RenderedDiagnosticReport(
        json_text='{"compact": true}\n',
        markdown_text="# Compact diagnostic\n",
        json_sha256=hashlib.sha256(b'{"compact": true}\n').hexdigest(),
    )
    original_replace = Path.replace

    def fail_markdown_promotion(path: Path, target: Path) -> Path:
        if path.name == "task_07_diagnostic_similarity_audit.md":
            raise OSError("synthetic promotion failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_markdown_promotion)

    with pytest.raises(OSError, match="synthetic promotion failure"):
        publish_diagnostic_report(rendered, workspace, report_directory)

    assert not completion_path.exists()


def test_historical_workflow_proves_public_paths_before_refusing_completion(
    monkeypatch,
    tmp_path: Path,
) -> None:
    report_directory = tmp_path / "reports/week_01"
    report_directory.mkdir(parents=True)
    (report_directory / EXPECTED_COMPLETION_FILENAME).write_text("completed")
    workspace = tmp_path / "private-workspace"
    public_paths = []
    monkeypatch.setattr(
        diagnostic_workflow_module,
        "load_similarity_audit_policy",
        lambda path: object(),
    )
    monkeypatch.setattr(
        diagnostic_workflow_module,
        "require_committed_execution_code",
        lambda project_root: None,
    )
    monkeypatch.setattr(
        diagnostic_workflow_module,
        "git_output",
        lambda *args: "a" * 40,
    )
    monkeypatch.setattr(
        diagnostic_workflow_module,
        "verify_mmseqs",
        lambda *args: "18-test",
    )
    monkeypatch.setattr(
        diagnostic_workflow_module,
        "policy_paths",
        lambda *args: {"workspace": workspace},
    )
    monkeypatch.setattr(
        diagnostic_workflow_module,
        "prove_path_is_ignored",
        lambda *args: None,
    )
    monkeypatch.setattr(
        diagnostic_workflow_module,
        "prove_path_is_public",
        lambda path, project_root: public_paths.append(path),
    )

    with pytest.raises(
        AuditPublicationError,
        match="completed Task 7 public report already exists",
    ):
        diagnostic_workflow_module.run_diagnostic_similarity_audit(
            project_root=tmp_path,
            config_path=DIAGNOSTIC_POLICY,
            report_directory=report_directory,
        )

    assert public_paths == [
        *(report_directory / filename for filename in EXPECTED_REPORT_FILENAMES),
        report_directory / EXPECTED_COMPLETION_FILENAME,
    ]
    assert not workspace.exists()


def _compact_diagnostic_report() -> dict[str, object]:
    closest_categories = {
        CATEGORY_CLOSEST_PROHIBITED: 1,
        CATEGORY_GE_50_LOW_COVERAGE: 0,
        CATEGORY_40_TO_50: 0,
        CATEGORY_30_TO_40: 0,
        CATEGORY_UNDER_30_OR_NONE: 1,
    }
    status_categories = {
        CATEGORY_PROHIBITED: 1,
        CATEGORY_GE_50_LOW_COVERAGE: 0,
        CATEGORY_40_TO_50: 0,
        CATEGORY_30_TO_40: 0,
        CATEGORY_UNDER_30_OR_NONE: 1,
    }
    similarity = {
        "held_out_queries_with_prohibited_match": 1,
        "held_out_query_count": 2,
        "prohibited_query_rate_percent": "50.000000",
        "unique_prohibited_pairs": 1,
        "prohibited_pair_attribution": {
            "exact_sequence_duplicate": 0,
            "same_uniref50_group": 0,
            "cross_uniref50_group": 1,
        },
        "enforcement_returned_pairs": 1,
        "residual_returned_pairs": 1,
        "unique_returned_pair_union": 1,
        "closest_residual_categories": closest_categories,
        "held_out_query_status_categories": status_categories,
    }
    balance = {
        "records": 2,
        "record_share_percent": "5.000000",
        "residues": 200,
        "residue_share_percent": "5.000000",
    }
    strategy = {
        "structural_membership": {
            "exact_sequence_hash_crossings": 1,
            "uniref50_group_crossings": 1,
            "retained_records": 100,
            "retained_residues": 10_000,
            "excluded_records": 0,
            "excluded_residues": 0,
            "largest_uniref50_group_records": 5,
            "largest_uniref50_group_residues": 500,
        },
        "partitions": {
            "training": {"balance": {}},
            "validation": {"balance": balance, "similarity": similarity},
            "test": {"balance": balance, "similarity": similarity},
        },
        "overall": {
            "held_out_queries_with_prohibited_match": 2,
            "held_out_query_count": 4,
            "prohibited_query_rate_percent": "50.000000",
            "unique_prohibited_pairs": 2,
            "prohibited_pair_attribution": {
                "exact_sequence_duplicate": 0,
                "same_uniref50_group": 0,
                "cross_uniref50_group": 2,
            },
            "enforcement_returned_pairs": 2,
            "residual_returned_pairs": 2,
            "unique_returned_pair_union": 2,
            "closest_residual_categories": {
                key: value * 2 for key, value in closest_categories.items()
            },
            "held_out_query_status_categories": {
                key: value * 2 for key, value in status_categories.items()
            },
        },
    }
    return {
        "diagnostic_only": True,
        "diagnostic_audit_authorized": True,
        "candidate_status": "failed_balance",
        "repair_authorized": False,
        "repair_performed": False,
        "selected_split_authorized": False,
        "task8_membership_use_authorized": False,
        "model_use": "prohibited",
        "post_audit_review_required": True,
        "strategies": {"random": strategy, "group_aware": strategy},
    }


def _assembly_evidence():
    file_evidence = FileEvidence(1, 2, "1" * 64)
    fasta_evidence = FastaEvidence(1, 4, 8, "2" * 64)
    inputs = SimpleNamespace(
        catalog=file_evidence,
        fastas={
            strategy: {
                partition: fasta_evidence
                for partition in ("training", "validation", "test")
            }
            for strategy in ("random", "group_aware")
        },
    )
    manifests = {
        strategy: SimpleNamespace(
            public_manifest=file_evidence,
            local_assignment=file_evidence,
        )
        for strategy in ("random", "group_aware")
    }
    database_reports = {
        "random": {"runtime_seconds": "1.000"},
        "group_aware": {"runtime_seconds": "2.000"},
    }
    stage_runtimes = {
        ("random", "validation"): "0.100",
        ("random", "test"): "0.200",
        ("group_aware", "validation"): "0.300",
        ("group_aware", "test"): "0.400",
    }
    strategy_reports = {
        strategy: {
            "partitions": {
                partition: {
                    "passes": {
                        "enforcement": {
                            "stages": {
                                "1000": {
                                    "runtime_seconds": stage_runtimes[
                                        strategy,
                                        partition,
                                    ]
                                }
                            }
                        }
                    }
                }
                for partition in ("validation", "test")
            }
        }
        for strategy in ("random", "group_aware")
    }
    return inputs, manifests, database_reports, strategy_reports
