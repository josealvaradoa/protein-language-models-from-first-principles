import hashlib
import json
from pathlib import Path

import pytest

from similarity_evidence_test_support import (
    alignment_tsv_row,
    canonicalize_rows,
    metadata,
)
from protein_lm.data.fixed_budget_audit.errors import (
    AuditConfigurationError,
    AuditExecutionError,
    AuditValidationError,
)
from protein_lm.data.fixed_budget_audit.evidence import (
    ensure_cap_summary,
    ensure_pair_union,
    verify_cap_summary,
    verify_pair_union,
)
from protein_lm.data.similarity_fastas import FastaEvidence


def test_a004_evidence_marks_imported_and_executed_sources_separately(
    tmp_path: Path,
) -> None:
    query_fasta, query_evidence = _write_fasta(tmp_path / "queries.fasta")
    queries = {"Q1": metadata("q1")}
    targets = {"T1": metadata("t1", partition="training")}
    canonical = canonicalize_rows(
        tmp_path,
        "canonical",
        (alignment_tsv_row("Q1", "T1"),),
        queries,
        targets,
    )
    identity = _evidence(canonical, 1)
    imported = ensure_cap_summary(
        source_label="imported_a003",
        cap=1_000,
        canonical_path=canonical,
        canonical_evidence=identity,
        query_fasta=query_fasta,
        query_fasta_evidence=query_evidence,
        expected_query_ids=queries,
        output_directory=tmp_path / "imported",
        fingerprint="synthetic-fingerprint",
    )
    executed = ensure_cap_summary(
        source_label="executed_a004",
        cap=1_000,
        canonical_path=canonical,
        canonical_evidence=identity,
        query_fasta=query_fasta,
        query_fasta_evidence=query_evidence,
        expected_query_ids=queries,
        output_directory=tmp_path / "executed",
        fingerprint="synthetic-fingerprint",
    )

    assert _marker(imported.directory)["source_label"] == "imported_a003"
    assert _marker(executed.directory)["source_label"] == "executed_a004"
    common = ensure_pair_union(
        label="common_all_query_10000_random_validation",
        source_paths={
            "enforcement_executed_a004_cap_1000": executed.directory
            / "prohibited_pairs.tsv",
            "residual_imported_a003_cap_1000": imported.directory
            / "prohibited_pairs.tsv",
        },
        output_directory=tmp_path / "common",
        fingerprint="synthetic-fingerprint",
    )
    staged = ensure_pair_union(
        label="staged_union_with_changed_query_100000_random_validation",
        source_paths={
            "common_all_query_10000": common.directory / "prohibited_pairs.tsv"
        },
        output_directory=tmp_path / "staged",
        fingerprint="synthetic-fingerprint",
    )

    assert _marker(common.directory)["label"].startswith("common_all_query_10000")
    assert _marker(staged.directory)["label"].startswith(
        "staged_union_with_changed_query_100000"
    )
    assert "all_query_100000" not in _marker(staged.directory)["label"]


def test_stored_evidence_contract_and_fresh_output_errors_are_typed(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        AuditConfigurationError,
        match="pair-union label is required",
    ):
        ensure_pair_union(
            label="",
            source_paths={},
            output_directory=tmp_path / "union",
            fingerprint="synthetic-fingerprint",
        )

    inputs = _cap_inputs(tmp_path)
    output = tmp_path / "existing"
    sentinel = output / "preserve.txt"
    sentinel.parent.mkdir()
    sentinel.write_text("preserve me")
    with pytest.raises(
        AuditExecutionError,
        match="cap-summary output lacks its completion marker",
    ):
        ensure_cap_summary(
            **inputs,
            output_directory=output,
            fingerprint="synthetic-fingerprint",
        )
    assert sentinel.read_text() == "preserve me"


def test_direct_stored_evidence_verification_rejects_invalid_inputs(
    tmp_path: Path,
) -> None:
    inputs = _cap_inputs(tmp_path)
    with pytest.raises(
        AuditConfigurationError,
        match="cap-summary source label is required",
    ):
        verify_cap_summary(
            **{**inputs, "source_label": ""},
            output_directory=tmp_path / "cap",
            fingerprint="synthetic-fingerprint",
        )
    with pytest.raises(
        AuditConfigurationError,
        match="outside the frozen A-004 stages",
    ):
        verify_cap_summary(
            **{**inputs, "cap": 999},
            output_directory=tmp_path / "cap",
            fingerprint="synthetic-fingerprint",
        )
    with pytest.raises(
        AuditConfigurationError,
        match="expected query identifiers must be unique",
    ):
        verify_cap_summary(
            **{**inputs, "expected_query_ids": ("Q1", "Q1")},
            output_directory=tmp_path / "cap",
            fingerprint="synthetic-fingerprint",
        )
    with pytest.raises(
        AuditConfigurationError,
        match="pair union requires named source files",
    ):
        verify_pair_union(
            label="stored_union",
            source_paths={},
            output_directory=tmp_path / "union",
            fingerprint="synthetic-fingerprint",
        )


def test_stored_cap_summary_maps_resumed_identity_drift_to_validation(
    tmp_path: Path,
) -> None:
    inputs = _cap_inputs(tmp_path)
    output = tmp_path / "stored"
    summary = ensure_cap_summary(
        **inputs,
        output_directory=output,
        fingerprint="synthetic-fingerprint",
    )
    marker = _marker(summary.directory)
    marker["source_label"] = "tampered"
    (summary.directory / "complete.json").write_text(json.dumps(marker))

    with pytest.raises(AuditValidationError, match="identity drifted"):
        ensure_cap_summary(
            **inputs,
            output_directory=output,
            fingerprint="synthetic-fingerprint",
        )


def _write_fasta(path: Path) -> tuple[Path, FastaEvidence]:
    content = ">Q1\nAAAA\n"
    path.write_text(content)
    return path, FastaEvidence(
        record_count=1,
        residue_count=4,
        byte_size=len(content.encode()),
        sha256=hashlib.sha256(content.encode()).hexdigest(),
    )


def _cap_inputs(tmp_path: Path) -> dict[str, object]:
    query_fasta, query_evidence = _write_fasta(tmp_path / "query.fasta")
    queries = {"Q1": metadata("q1")}
    canonical = canonicalize_rows(
        tmp_path,
        "stored_canonical",
        (alignment_tsv_row("Q1", "T1"),),
        queries,
        {"T1": metadata("t1", partition="training")},
    )
    return {
        "source_label": "executed_a004",
        "cap": 1_000,
        "canonical_path": canonical,
        "canonical_evidence": _evidence(canonical, 1),
        "query_fasta": query_fasta,
        "query_fasta_evidence": query_evidence,
        "expected_query_ids": queries,
    }


def _evidence(path: Path, rows: int):
    content = path.read_bytes()
    from protein_lm.data.similarity_audit_models import FileEvidence

    return FileEvidence(rows, len(content), hashlib.sha256(content).hexdigest())


def _marker(directory: Path) -> dict[str, object]:
    return json.loads((directory / "complete.json").read_text())
