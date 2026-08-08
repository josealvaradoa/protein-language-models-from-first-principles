"""Focused contracts for search inputs, validation, and historical cleanup."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import protein_lm.data.fixed_budget_audit.search as search_module
from protein_lm.data.fixed_budget_audit.errors import (
    AuditConfigurationError,
    AuditValidationError,
)
from protein_lm.data.fixed_budget_audit.search import (
    ensure_search_pass,
    ensure_target_database,
    query_ids_sha256,
    require_fixed_policy_caps,
    search_command,
    validate_completed_pass,
    verify_query_fasta,
)
from protein_lm.data.similarity_audit_policy import (
    SimilarityAuditError,
    load_similarity_audit_policy,
)
from protein_lm.data.similarity_fastas import FastaEvidence

PROJECT_ROOT = Path(__file__).parents[3]
POLICY_PATH = (
    PROJECT_ROOT / "experiments" / "week_01" / "diagnostic_similarity_audit.toml"
)


def test_invalid_search_pass_and_query_contracts_use_configuration_errors(
    tmp_path: Path,
) -> None:
    policy = load_similarity_audit_policy(POLICY_PATH)

    with pytest.raises(AuditConfigurationError, match="unknown search pass: repair"):
        search_command(
            policy,
            pass_name="repair",
            cap=1_000,
            query_fasta=tmp_path / "query.fasta",
            target_database=tmp_path / "target",
            raw_output=tmp_path / "raw.tsv",
            temp_directory=tmp_path / "tmp",
        )
    with pytest.raises(
        AuditConfigurationError,
        match="query identifiers must be nonempty and unique",
    ):
        query_ids_sha256(("Q1", "Q1"))
    with pytest.raises(
        AuditConfigurationError,
        match="policy must use the frozen fixed-budget caps",
    ):
        require_fixed_policy_caps(replace(policy, comparison_cap=9_999))


def test_query_contract_does_not_relabel_lower_artifact_failure(tmp_path: Path) -> None:
    path = tmp_path / "query.fasta"
    path.write_text(">Q1\nAAAA\n")
    evidence = FastaEvidence(
        record_count=1,
        residue_count=4,
        byte_size=path.stat().st_size,
        sha256="0" * 64,
    )

    with pytest.raises(SimilarityAuditError, match="checksum drifted") as error:
        verify_query_fasta(path, evidence, {"Q1": None})

    assert type(error.value) is SimilarityAuditError


def test_historical_completed_pass_cleanup_removes_only_full_alignment_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    policy = load_similarity_audit_policy(POLICY_PATH)
    workspace = tmp_path / "workspace"
    pass_directory = workspace / "tracks/random/validation/residual"
    pass_marker_path = pass_directory / "complete.json"
    stage_directory = pass_directory / "cap_1000"
    stage_marker = stage_directory / "complete.json"
    raw = stage_directory / "raw.tsv"
    canonical = stage_directory / "canonical.tsv"
    temporary = stage_directory / "mmseqs_tmp/cache"
    temporary.parent.mkdir(parents=True)
    pass_marker_path.write_text("{}")
    stage_marker.write_text("preserve marker")
    raw.write_text("retire raw")
    canonical.write_text("retire canonical")
    temporary.write_text("retire temporary")
    monkeypatch.setattr(
        search_module,
        "validate_completed_pass",
        lambda *args, **kwargs: None,
    )

    result = ensure_search_pass(
        strategy="random",
        partition="validation",
        pass_name="residual",
        query_fasta=tmp_path / "unused.fasta",
        query_fasta_evidence=FastaEvidence(1, 4, 9, "0" * 64),
        query_metadata={"Q1": object()},
        target_database=tmp_path / "target",
        target_metadata={},
        project_root=tmp_path,
        workspace=workspace,
        policy=policy,
        fingerprint="synthetic-fingerprint",
    )

    assert result == {}
    assert stage_marker.read_text() == "preserve marker"
    assert not raw.exists()
    assert not canonical.exists()
    assert not temporary.parent.exists()


def test_historical_database_rebuild_cleans_unmarked_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    policy = load_similarity_audit_policy(POLICY_PATH)
    workspace = tmp_path / "workspace"
    final_stale = workspace / "databases/random/stale.bin"
    incomplete_stale = workspace / "databases/.random.incomplete/stale.bin"
    final_stale.parent.mkdir(parents=True)
    incomplete_stale.parent.mkdir(parents=True)
    final_stale.write_bytes(b"stale final")
    incomplete_stale.write_bytes(b"stale incomplete")
    training = tmp_path / "training.fasta"
    content = b">T1\nAAAA\n"
    training.write_bytes(content)
    evidence = FastaEvidence(1, 4, len(content), "0" * 64)

    def fake_runner(command, **kwargs):
        del kwargs
        assert not final_stale.exists()
        assert not incomplete_stale.exists()
        Path(command[3]).write_bytes(b"rebuilt target")
        return "0.01"

    monkeypatch.setattr(search_module, "run_mmseqs_command", fake_runner)

    prefix, marker = ensure_target_database(
        strategy="random",
        training_fasta=training,
        training_fasta_evidence=evidence,
        project_root=tmp_path,
        workspace=workspace,
        policy=policy,
        fingerprint="synthetic-fingerprint",
    )

    assert prefix.read_bytes() == b"rebuilt target"
    assert marker["command"][1] == "createdb"
    assert not (workspace / "databases/.random.incomplete").exists()


def test_completed_pass_uses_local_validation_integer_contract(tmp_path: Path) -> None:
    policy = load_similarity_audit_policy(POLICY_PATH)
    marker = {
        "schema_version": 1,
        "stage": "completed_search_pass",
        "fingerprint": "synthetic-fingerprint",
        "strategy": "random",
        "partition": "validation",
        "pass_name": "residual",
        "query_count": 1,
        "stages": {str(policy.initial_cap): {}, str(policy.comparison_cap): {}},
        "convergence": {
            "expected_queries": 1,
            "converged_at_comparison_cap": 1,
            "escalated_queries": 0,
            "converged_at_escalation_cap": 0,
            "final_differing_queries": 0,
            "escalated_query_ids": [],
        },
        "accepted": {
            "pass_name": "residual",
            "accepted_at_comparison_cap": 1,
            "accepted_at_escalation_cap": 0,
            "accepted_rows": True,
        },
    }

    with pytest.raises(
        AuditValidationError,
        match="accepted row count must be a nonnegative integer",
    ):
        validate_completed_pass(
            marker,
            marker_path=tmp_path / "complete.json",
            fingerprint="synthetic-fingerprint",
            strategy="random",
            partition="validation",
            pass_name="residual",
            expected_query_ids=frozenset({"Q1"}),
            policy=policy,
        )
