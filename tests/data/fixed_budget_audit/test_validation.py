"""Frozen-context and final-validation ordering contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import protein_lm.data.fixed_budget_audit.validation as validation_module
import pytest
from protein_lm.data.artifacts import file_identity
from protein_lm.data.fixed_budget_audit.errors import (
    AuditConfigurationError,
    AuditValidationError,
    SourceEvidenceError,
)
from protein_lm.data.fixed_budget_audit.reporting import (
    ReceiptPublication,
    ReportPublication,
)
from protein_lm.data.fixed_budget_audit.source import (
    A003Import,
    DatabaseImport,
    ImportedStage,
    MarkerEvidence,
)
from protein_lm.data.fixed_budget_audit.validation import (
    FinalValidationContext,
    build_final_validation_context,
    revalidate_before_completion,
)
from protein_lm.data.similarity_audit_models import FileEvidence
from protein_lm.data.similarity_audit_policy import SimilarityAuditError
from protein_lm.data.similarity_fastas import FastaEvidence


def test_final_validation_context_freezes_baseline_mappings(tmp_path: Path) -> None:
    source_paths = {"catalog": tmp_path / "catalog.tsv"}
    assignments = {
        "task5_local": {"byte_size": 10, "sha256": "1" * 64},
    }

    context = _context(
        tmp_path,
        source_paths=source_paths,
        assignments=assignments,
    )
    source_paths["catalog"] = tmp_path / "changed.tsv"
    assignments["task5_local"]["sha256"] = "2" * 64

    assert type(context) is FinalValidationContext
    assert context.source_paths == {"catalog": tmp_path / "catalog.tsv"}
    assert context.baseline_assignment_identities == {
        "task5_local": {"byte_size": 10, "sha256": "1" * 64},
    }
    with pytest.raises(FrozenInstanceError):
        context.code_revision = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        context.source_paths["extra"] = tmp_path  # type: ignore[index]
    with pytest.raises(TypeError):
        context.baseline_assignment_identities["task5_local"]["sha256"] = "3" * 64  # type: ignore[index]


def test_final_validation_context_detaches_a003_import_and_detects_caller_drift(
    monkeypatch,
    tmp_path: Path,
) -> None:
    original_fasta = FastaEvidence(1, 4, 8, "1" * 64)
    changed_fasta = FastaEvidence(1, 4, 8, "2" * 64)
    fastas = {"random": {"training": original_fasta}}
    imported = _a003_import(tmp_path, fastas=fastas)
    context = _context(tmp_path, imported=imported)

    fastas["random"]["training"] = changed_fasta
    fastas["group_aware"] = {"training": changed_fasta}

    assert context.baseline_imported_a003 == _a003_import(tmp_path)
    assert context.baseline_imported_a003 is not imported
    assert context.baseline_imported_a003.fastas is not imported.fastas
    assert (
        context.baseline_imported_a003.fastas["random"] is not imported.fastas["random"]
    )
    assert context.baseline_imported_a003.database is not imported.database
    assert context.baseline_imported_a003.stages is not imported.stages
    assert context.baseline_imported_a003.stages[0] is not imported.stages[0]
    assert (
        context.baseline_imported_a003.stages[0].marker is not imported.stages[0].marker
    )
    assert (
        context.baseline_imported_a003.stages[0].query_fasta
        is not imported.stages[0].query_fasta
    )
    assert (
        context.baseline_imported_a003.stages[0].canonical
        is not imported.stages[0].canonical
    )
    assert (
        context.baseline_imported_a003.stages[0].command
        is not imported.stages[0].command
    )
    assert context.baseline_imported_a003.escalated_query_ids is not (
        imported.escalated_query_ids
    )

    report, receipt = _publications(tmp_path)
    monkeypatch.setattr(
        validation_module,
        "reverify_frozen_run_state",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        validation_module,
        "load_a004_policy",
        lambda path: context.a004_policy,
    )
    monkeypatch.setattr(
        validation_module,
        "verify_a003_residual_import",
        lambda **kwargs: imported,
    )

    with pytest.raises(SourceEvidenceError, match="imported evidence changed"):
        revalidate_before_completion(
            context=context,
            fingerprint="f" * 64,
            databases={},
            tracks={},
            unions={},
            report=report,
            receipt=receipt,
        )


def test_final_validation_context_a003_fastas_are_deeply_immutable(
    tmp_path: Path,
) -> None:
    imported = _a003_import(tmp_path)
    context = _context(tmp_path, imported=imported)
    changed_fasta = FastaEvidence(1, 4, 8, "2" * 64)

    with pytest.raises(TypeError):
        context.baseline_imported_a003.fastas["group_aware"] = {  # type: ignore[index]
            "training": changed_fasta
        }
    with pytest.raises(TypeError):
        context.baseline_imported_a003.fastas["random"]["training"] = (  # type: ignore[index]
            changed_fasta
        )

    imported.fastas["random"]["training"] = changed_fasta  # type: ignore[index]
    assert imported.fastas["random"]["training"] == changed_fasta
    assert context.baseline_imported_a003.fastas["random"]["training"] != changed_fasta


def test_final_validation_rechecks_sources_before_a004_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    report, receipt = _publications(tmp_path)
    calls: list[str] = []
    database = object()
    track = object()
    bundle = object()

    monkeypatch.setattr(
        validation_module,
        "reverify_frozen_run_state",
        lambda **kwargs: calls.append("source_state"),
    )
    monkeypatch.setattr(
        validation_module,
        "load_a004_policy",
        lambda path: calls.append("a004_policy") or context.a004_policy,
    )
    monkeypatch.setattr(
        validation_module,
        "verify_a003_residual_import",
        lambda **kwargs: calls.append("a003_import") or context.baseline_imported_a003,
    )
    monkeypatch.setattr(
        validation_module,
        "frozen_assignment_identities",
        lambda paths: (
            calls.append("assignments") or context.baseline_assignment_identities
        ),
    )
    monkeypatch.setattr(
        validation_module,
        "verify_a004_database",
        lambda value, **kwargs: calls.append("database"),
    )
    monkeypatch.setattr(
        validation_module,
        "_verify_track",
        lambda value, **kwargs: calls.append("track"),
    )
    monkeypatch.setattr(
        validation_module,
        "_verify_union_bundle",
        lambda **kwargs: calls.append("union"),
    )
    monkeypatch.setattr(
        validation_module,
        "verify_report_publication",
        lambda value, **kwargs: calls.append("report"),
    )
    monkeypatch.setattr(
        validation_module,
        "verify_receipt_publication",
        lambda value, **kwargs: calls.append("receipt"),
    )

    authorization = revalidate_before_completion(
        context=context,
        fingerprint="f" * 64,
        databases={"random": database},  # type: ignore[dict-item]
        tracks={("random", "test", "residual"): track},  # type: ignore[dict-item]
        unions={("random", "test"): bundle},  # type: ignore[dict-item]
        report=report,
        receipt=receipt,
    )

    assert calls == [
        "source_state",
        "a004_policy",
        "a003_import",
        "assignments",
        "database",
        "track",
        "union",
        "report",
        "receipt",
    ]
    assert authorization.receipt_identity == file_identity(receipt.path)


def test_final_validation_classifies_a004_policy_drift(
    monkeypatch,
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    report, receipt = _publications(tmp_path)
    monkeypatch.setattr(
        validation_module,
        "reverify_frozen_run_state",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(validation_module, "load_a004_policy", lambda path: object())

    with pytest.raises(AuditConfigurationError, match="policy changed"):
        revalidate_before_completion(
            context=context,
            fingerprint="f" * 64,
            databases={},
            tracks={},
            unions={},
            report=report,
            receipt=receipt,
        )


def test_final_validation_maps_lower_a003_evidence_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    report, receipt = _publications(tmp_path)
    lower = SimilarityAuditError("sentinel A-003 checksum drift")
    monkeypatch.setattr(
        validation_module,
        "reverify_frozen_run_state",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        validation_module,
        "load_a004_policy",
        lambda path: context.a004_policy,
    )
    monkeypatch.setattr(
        validation_module,
        "verify_a003_residual_import",
        lambda **kwargs: (_ for _ in ()).throw(lower),
    )

    with pytest.raises(
        SourceEvidenceError, match="sentinel A-003 checksum drift"
    ) as raised:
        revalidate_before_completion(
            context=context,
            fingerprint="f" * 64,
            databases={},
            tracks={},
            unions={},
            report=report,
            receipt=receipt,
        )

    assert raised.value.__cause__ is lower


@pytest.mark.parametrize("drift", ("a003", "assignments"))
def test_final_validation_classifies_frozen_source_drift(
    monkeypatch,
    tmp_path: Path,
    drift: str,
) -> None:
    context = _context(tmp_path)
    report, receipt = _publications(tmp_path)
    monkeypatch.setattr(
        validation_module,
        "reverify_frozen_run_state",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        validation_module,
        "load_a004_policy",
        lambda path: context.a004_policy,
    )
    monkeypatch.setattr(
        validation_module,
        "verify_a003_residual_import",
        lambda **kwargs: (
            object() if drift == "a003" else context.baseline_imported_a003
        ),
    )
    monkeypatch.setattr(
        validation_module,
        "frozen_assignment_identities",
        lambda paths: (
            {"changed": {}}
            if drift == "assignments"
            else context.baseline_assignment_identities
        ),
    )

    with pytest.raises(SourceEvidenceError):
        revalidate_before_completion(
            context=context,
            fingerprint="f" * 64,
            databases={},
            tracks={},
            unions={},
            report=report,
            receipt=receipt,
        )


def test_final_validation_maps_lower_a004_evidence_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    report, receipt = _publications(tmp_path)
    lower = SimilarityAuditError("sentinel A-004 checksum drift")
    monkeypatch.setattr(
        validation_module, "_revalidate_frozen_sources", lambda value: None
    )
    monkeypatch.setattr(
        validation_module,
        "verify_a004_database",
        lambda value, **kwargs: (_ for _ in ()).throw(lower),
    )

    with pytest.raises(
        AuditValidationError, match="sentinel A-004 checksum drift"
    ) as raised:
        revalidate_before_completion(
            context=context,
            fingerprint="f" * 64,
            databases={"random": object()},  # type: ignore[dict-item]
            tracks={},
            unions={},
            report=report,
            receipt=receipt,
        )

    assert raised.value.__cause__ is lower


def _context(
    tmp_path: Path,
    *,
    source_paths: dict[str, Path] | None = None,
    assignments: dict[str, dict[str, object]] | None = None,
    imported: A003Import | None = None,
) -> FinalValidationContext:
    return build_final_validation_context(
        source_paths=source_paths or {"catalog": tmp_path / "catalog.tsv"},
        source_policy=object(),  # type: ignore[arg-type]
        a004_policy=object(),  # type: ignore[arg-type]
        a004_config_path=tmp_path / "a004.toml",
        source_policy_path=tmp_path / "source.toml",
        project_root=tmp_path,
        code_revision="b" * 40,
        mmseqs_version="18-test",
        baseline_assignment_identities=assignments
        or {"task5_local": {"byte_size": 1, "sha256": "1" * 64}},
        baseline_imported_a003=imported or _a003_import(tmp_path),
    )


def _a003_import(
    tmp_path: Path,
    *,
    fastas: dict[str, dict[str, FastaEvidence]] | None = None,
) -> A003Import:
    fasta = FastaEvidence(1, 4, 8, "1" * 64)
    stage_marker = MarkerEvidence(3, "3" * 64)
    return A003Import(
        fingerprint="a" * 64,
        fastas=fastas or {"random": {"training": fasta}},
        database=DatabaseImport(
            marker=MarkerEvidence(2, "2" * 64),
            artifact_count=3,
        ),
        stages=(
            ImportedStage(
                cap=1_000,
                marker=stage_marker,
                query_fasta=fasta,
                canonical=FileEvidence(1, 10, "4" * 64),
                canonical_path=tmp_path / "canonical.tsv",
                command=("mmseqs", "search"),
                runtime_seconds="1.000000",
            ),
        ),
        escalated_query_ids=("Q1",),
    )


def _publications(tmp_path: Path) -> tuple[ReportPublication, ReceiptPublication]:
    report = ReportPublication(
        directory=tmp_path / "report",
        json_path=tmp_path / "report/a004_report.json",
        markdown_path=tmp_path / "report/a004_report.md",
        marker_path=tmp_path / "report/complete.json",
        payload={},
        json_identity={"byte_size": 1, "sha256": "1" * 64},
        markdown_identity={"byte_size": 2, "sha256": "2" * 64},
        marker_identity={"byte_size": 3, "sha256": "3" * 64},
    )
    report_record = {
        "directory": str(report.directory),
        "json": dict(report.json_identity),
        "markdown": dict(report.markdown_identity),
        "marker": dict(report.marker_identity),
    }
    receipt_path = tmp_path / "a004_import_receipt.json"
    receipt_path.write_bytes(b"{}\n")
    receipt = ReceiptPublication(
        path=receipt_path,
        payload={"report": report_record},
        identity=file_identity(receipt_path),
    )
    return report, receipt
