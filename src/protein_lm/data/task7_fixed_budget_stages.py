"""Atomic stage markers and resume checks for A-004 fixed-budget searches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

from protein_lm.data.similarity_audit_models import SequenceMetadata
from protein_lm.data.similarity_audit_policy import (
    SimilarityAuditError,
    SimilarityAuditPolicy,
)
from protein_lm.data.similarity_fastas import FastaEvidence, write_fasta_subset
from protein_lm.data.similarity_results import canonicalize_mmseqs_tsv
from protein_lm.data.task7_checkpoints import (
    canonical_evidence_from,
    fasta_evidence_from,
    file_identity,
    read_json,
    require_marker_identity,
    verify_file,
    write_json_atomic,
)
from protein_lm.data.task7_commands import search_command
from protein_lm.data.task7_execution import require_disk_capacity, run_mmseqs_command
from protein_lm.data.task7_fixed_budget_contract import (
    FixedBudgetStage,
    SearchRunner,
    query_ids_sha256,
    verify_query_fasta,
)


def ensure_search_stage(
    *,
    cap: int,
    strategy: str,
    partition: str,
    pass_name: str,
    query_fasta: Path,
    query_fasta_evidence: FastaEvidence,
    query_ids: tuple[str, ...],
    query_metadata: Mapping[str, SequenceMetadata],
    target_database: Path,
    target_database_identity: Mapping[str, object],
    target_metadata: Mapping[str, SequenceMetadata],
    pass_directory: Path,
    project_root: Path,
    workspace: Path,
    policy: SimilarityAuditPolicy,
    fingerprint: str,
    command_runner: SearchRunner | None,
) -> FixedBudgetStage:
    """Create or fully validate one canonical stage without deleting evidence."""

    stage_directory = pass_directory / f"cap_{cap}"
    raw_path = stage_directory / "raw.tsv"
    canonical_path = stage_directory / "canonical.tsv"
    marker_path = stage_directory / "complete.json"
    command = search_command(
        policy,
        pass_name=pass_name,
        cap=cap,
        query_fasta=query_fasta,
        target_database=target_database,
        raw_output=raw_path,
        temp_directory=stage_directory / "mmseqs_tmp",
    )
    identity = _stage_identity(
        fingerprint=fingerprint,
        strategy=strategy,
        partition=partition,
        pass_name=pass_name,
        cap=cap,
        query_fasta=query_fasta_evidence,
        query_ids=query_ids,
        target_database=target_database,
        target_database_identity=target_database_identity,
        command=command,
        canonical_path=canonical_path,
    )
    if marker_path.exists():
        return _verify_completed_stage(
            marker_path=marker_path,
            raw_path=raw_path,
            canonical_path=canonical_path,
            identity=identity,
            cap=cap,
            query_fasta=query_fasta_evidence,
            command=command,
            fingerprint=fingerprint,
        )
    if stage_directory.exists() and any(stage_directory.iterdir()):
        raise SimilarityAuditError("A-004 search stage has unmarked output")
    stage_directory.mkdir(parents=True, exist_ok=True)
    runtime = _run_search_command(
        command,
        project_root=project_root,
        workspace=workspace,
        log_path=stage_directory / "command.log",
        policy=policy,
        command_runner=command_runner,
    )
    if not raw_path.is_file():
        raise SimilarityAuditError("A-004 MMseqs2 search did not produce its TSV")
    alignment = canonicalize_mmseqs_tsv(
        raw_path,
        canonical_path,
        query_metadata=query_metadata,
        target_metadata=target_metadata,
        chunk_rows=policy.parser_chunk_rows,
        resource_guard=lambda: require_disk_capacity(workspace, policy),
        delete_raw_after_parse=True,
    )
    if raw_path.exists():
        raise SimilarityAuditError("A-004 raw search output was not retired after parsing")
    write_json_atomic(
        marker_path,
        {
            **identity,
            "runtime_seconds": runtime,
            "raw_retained": False,
            "alignment_evidence": asdict(alignment),
        },
    )
    return FixedBudgetStage(
        cap=cap,
        query_fasta=query_fasta_evidence,
        canonical=alignment.canonical,
        canonical_path=canonical_path,
        command=command,
        runtime_seconds=runtime,
        marker_path=marker_path,
    )


def ensure_escalation_fasta(
    *,
    pass_directory: Path,
    source_fasta: Path,
    source_evidence: FastaEvidence,
    source_query_ids: tuple[str, ...],
    changed_query_ids: tuple[str, ...],
    fingerprint: str,
) -> tuple[Path, FastaEvidence, Path]:
    """Publish and resume-verify the exact 100k query subset atomically."""

    path = pass_directory / "escalated_queries.fasta"
    marker_path = pass_directory / "escalated_queries.complete.json"
    identity = {
        "schema_version": 1,
        "stage": "a004_escalation_fasta",
        "fingerprint": fingerprint,
        "source_fasta": asdict(source_evidence),
        "source_query_ids_sha256": query_ids_sha256(source_query_ids),
        "query_count": len(changed_query_ids),
        "query_ids_sha256": query_ids_sha256(changed_query_ids),
    }
    if marker_path.exists():
        marker = read_json(marker_path)
        require_marker_identity(marker, fingerprint, "a004_escalation_fasta")
        fasta = fasta_evidence_from(marker.get("fasta"))
        if marker != {**identity, "fasta": asdict(fasta)}:
            raise SimilarityAuditError("A-004 escalation FASTA identity drifted")
        verify_query_fasta(path, fasta, {item: None for item in changed_query_ids})
        return path, fasta, marker_path
    if path.exists():
        raise SimilarityAuditError("A-004 escalation FASTA lacks its completion marker")
    fasta = write_fasta_subset(source_fasta, path, set(changed_query_ids))
    write_json_atomic(marker_path, {**identity, "fasta": asdict(fasta)})
    return path, fasta, marker_path


def pass_marker(
    *,
    fingerprint: str,
    strategy: str,
    partition: str,
    pass_name: str,
    query_fasta: FastaEvidence,
    query_ids: tuple[str, ...],
    target_database: Path,
    target_database_identity: Mapping[str, object],
    changed_query_ids: tuple[str, ...],
    stages: tuple[FixedBudgetStage, ...],
    escalation_fasta: FastaEvidence | None,
    escalation_marker: Path | None,
) -> dict[str, object]:
    """Build the final pass identity from already verified child stages."""

    escalation = None
    if escalation_fasta is not None and escalation_marker is not None:
        escalation = {
            "fasta": asdict(escalation_fasta),
            "marker": file_identity(escalation_marker),
        }
    return {
        "schema_version": 1,
        "stage": "a004_fixed_budget_pass",
        "fingerprint": fingerprint,
        "strategy": strategy,
        "partition": partition,
        "pass_name": pass_name,
        "query_fasta": asdict(query_fasta),
        "query_ids_sha256": query_ids_sha256(query_ids),
        "target_database": str(target_database),
        "target_database_identity": dict(target_database_identity),
        "changed_query_ids": list(changed_query_ids),
        "escalation": escalation,
        "stages": {
            str(stage.cap): {
                "marker": file_identity(stage.marker_path),
                "canonical": asdict(stage.canonical),
                "query_fasta": asdict(stage.query_fasta),
                "command": list(stage.command),
            }
            for stage in stages
        },
    }


def _verify_completed_stage(
    *,
    marker_path: Path,
    raw_path: Path,
    canonical_path: Path,
    identity: Mapping[str, object],
    cap: int,
    query_fasta: FastaEvidence,
    command: tuple[str, ...],
    fingerprint: str,
) -> FixedBudgetStage:
    marker = read_json(marker_path)
    require_marker_identity(marker, fingerprint, "a004_fixed_budget_search_stage")
    alignment = canonical_evidence_from(marker.get("alignment_evidence"))
    expected = {
        **identity,
        "runtime_seconds": marker.get("runtime_seconds"),
        "raw_retained": False,
        "alignment_evidence": asdict(alignment),
    }
    runtime = marker.get("runtime_seconds")
    if marker != expected or not isinstance(runtime, str):
        raise SimilarityAuditError("A-004 search-stage identity drifted")
    if raw_path.exists() or marker.get("raw_retained") is not False:
        raise SimilarityAuditError("A-004 search-stage raw-output state drifted")
    verify_file(canonical_path, alignment.canonical.byte_size, alignment.canonical.sha256)
    return FixedBudgetStage(
        cap=cap,
        query_fasta=query_fasta,
        canonical=alignment.canonical,
        canonical_path=canonical_path,
        command=command,
        runtime_seconds=runtime,
        marker_path=marker_path,
    )


def _stage_identity(
    *,
    fingerprint: str,
    strategy: str,
    partition: str,
    pass_name: str,
    cap: int,
    query_fasta: FastaEvidence,
    query_ids: tuple[str, ...],
    target_database: Path,
    target_database_identity: Mapping[str, object],
    command: Sequence[str],
    canonical_path: Path,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "stage": "a004_fixed_budget_search_stage",
        "fingerprint": fingerprint,
        "strategy": strategy,
        "partition": partition,
        "pass_name": pass_name,
        "cap": cap,
        "query_fasta": asdict(query_fasta),
        "query_ids_sha256": query_ids_sha256(query_ids),
        "target_database": str(target_database),
        "target_database_identity": dict(target_database_identity),
        "command": list(command),
        "canonical_path": str(canonical_path),
    }


def _run_search_command(
    command: Sequence[str],
    *,
    project_root: Path,
    workspace: Path,
    log_path: Path,
    policy: SimilarityAuditPolicy,
    command_runner: SearchRunner | None,
) -> str:
    if command_runner is None:
        return run_mmseqs_command(
            command,
            project_root=project_root,
            workspace=workspace,
            log_path=log_path,
            policy=policy,
        )
    return command_runner(command, project_root, workspace, log_path, policy)
