"""Resumable MMseqs2 search orchestration for the Task 7 audit."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from protein_lm.data.similarity_audit_models import SequenceMetadata
from protein_lm.data.similarity_audit_policy import (
    SimilarityAuditError,
    SimilarityAuditPolicy,
)
from protein_lm.data.similarity_evidence import compact_converged_results
from protein_lm.data.similarity_fastas import FastaEvidence, write_fasta_subset
from protein_lm.data.similarity_results import (
    canonicalize_mmseqs_tsv,
    compare_canonical_results,
    convergence_evidence,
)
from protein_lm.data.task7_checkpoints import (
    canonical_evidence_from,
    file_identity,
    read_json,
    require_marker_identity,
    validate_completed_pass,
    verify_artifact_index,
    verify_file,
    write_json_atomic,
)
from protein_lm.data.task7_execution import (
    require_disk_capacity,
    run_mmseqs_command,
)


def ensure_target_database(
    *,
    strategy: str,
    training_fasta: Path,
    training_fasta_evidence: FastaEvidence,
    project_root: Path,
    workspace: Path,
    policy: SimilarityAuditPolicy,
    fingerprint: str,
) -> tuple[Path, dict[str, object]]:
    """Create or verify one strategy's deterministic training database."""

    database_root = workspace / "databases"
    final_directory = database_root / strategy
    incomplete = database_root / f".{strategy}.incomplete"
    database_prefix = incomplete / "target"
    command = [
        policy.mmseqs_executable,
        "createdb",
        str(training_fasta),
        str(database_prefix),
        "--dbtype",
        "1",
        "--shuffle",
        str(policy.createdb_shuffle),
        "--createdb-mode",
        str(policy.createdb_mode),
        "--threads",
        str(policy.threads),
    ]
    marker_path = final_directory / "complete.json"
    if marker_path.exists():
        marker = read_json(marker_path)
        require_marker_identity(marker, fingerprint, "target_database")
        if (
            marker.get("strategy") != strategy
            or marker.get("training_fasta") != asdict(training_fasta_evidence)
            or marker.get("command") != command
        ):
            raise SimilarityAuditError("target database completion evidence drifted")
        verify_artifact_index(final_directory, marker.get("artifacts"))
        return final_directory / "target", marker

    if incomplete.exists():
        shutil.rmtree(incomplete)
    if final_directory.exists():
        shutil.rmtree(final_directory)
    incomplete.mkdir(parents=True, exist_ok=False)
    runtime = run_mmseqs_command(
        command,
        project_root=project_root,
        workspace=workspace,
        log_path=workspace / "logs" / f"createdb_{strategy}.log",
        policy=policy,
    )
    artifact_paths = tuple(
        sorted(path for path in incomplete.iterdir() if path.is_file())
    )
    if not artifact_paths:
        raise SimilarityAuditError("MMseqs2 createdb produced no database files")
    artifacts = {path.name: file_identity(path) for path in artifact_paths}
    incomplete.replace(final_directory)
    marker = {
        "schema_version": 1,
        "stage": "target_database",
        "fingerprint": fingerprint,
        "strategy": strategy,
        "training_fasta": asdict(training_fasta_evidence),
        "command": command,
        "runtime_seconds": runtime,
        "artifacts": artifacts,
    }
    write_json_atomic(marker_path, marker)
    return final_directory / "target", marker


def ensure_search_pass(
    *,
    strategy: str,
    partition: str,
    pass_name: str,
    query_fasta: Path,
    query_fasta_evidence: FastaEvidence,
    query_metadata: Mapping[str, SequenceMetadata],
    target_database: Path,
    target_metadata: Mapping[str, SequenceMetadata],
    project_root: Path,
    workspace: Path,
    policy: SimilarityAuditPolicy,
    fingerprint: str,
) -> dict[str, object]:
    """Run or resume all caps needed to converge one diagnostic search pass."""

    pass_directory = workspace / "tracks" / strategy / partition / pass_name
    marker_path = pass_directory / "complete.json"
    if marker_path.exists():
        marker = read_json(marker_path)
        validate_completed_pass(
            marker,
            marker_path=marker_path,
            fingerprint=fingerprint,
            strategy=strategy,
            partition=partition,
            pass_name=pass_name,
            expected_query_ids=frozenset(query_metadata),
            policy=policy,
        )
        _cleanup_full_alignment_rows(pass_directory)
        return marker

    pass_directory.mkdir(parents=True, exist_ok=True)
    stages = {}
    initial = _ensure_search_stage(
        cap=policy.initial_cap,
        query_fasta=query_fasta,
        query_fasta_evidence=query_fasta_evidence,
        query_metadata=query_metadata,
        target_database=target_database,
        target_metadata=target_metadata,
        pass_directory=pass_directory,
        pass_name=pass_name,
        project_root=project_root,
        workspace=workspace,
        policy=policy,
        fingerprint=fingerprint,
    )
    stages[str(policy.initial_cap)] = initial["evidence"]
    comparison = _ensure_search_stage(
        cap=policy.comparison_cap,
        query_fasta=query_fasta,
        query_fasta_evidence=query_fasta_evidence,
        query_metadata=query_metadata,
        target_database=target_database,
        target_metadata=target_metadata,
        pass_directory=pass_directory,
        pass_name=pass_name,
        project_root=project_root,
        workspace=workspace,
        policy=policy,
        fingerprint=fingerprint,
    )
    stages[str(policy.comparison_cap)] = comparison["evidence"]
    differing = compare_canonical_results(
        initial["canonical_path"],
        comparison["canonical_path"],
        expected_query_ids=query_metadata,
    )

    escalation_path = None
    if differing:
        escalation_fasta = pass_directory / "escalated_queries.fasta"
        escalation_fasta_evidence = write_fasta_subset(
            query_fasta,
            escalation_fasta,
            set(differing),
        )
        escalation_metadata = {
            accession: query_metadata[accession] for accession in differing
        }
        escalation = _ensure_search_stage(
            cap=policy.escalation_cap,
            query_fasta=escalation_fasta,
            query_fasta_evidence=escalation_fasta_evidence,
            query_metadata=escalation_metadata,
            target_database=target_database,
            target_metadata=target_metadata,
            pass_directory=pass_directory,
            pass_name=pass_name,
            project_root=project_root,
            workspace=workspace,
            policy=policy,
            fingerprint=fingerprint,
        )
        escalation_path = escalation["canonical_path"]
        stages[str(policy.escalation_cap)] = escalation["evidence"]

    convergence = convergence_evidence(
        expected_query_ids=query_metadata,
        initial_path=initial["canonical_path"],
        comparison_path=comparison["canonical_path"],
        escalation_path=escalation_path,
    )
    accepted = compact_converged_results(
        pass_name=pass_name,
        comparison_path=comparison["canonical_path"],
        escalation_path=escalation_path,
        convergence=convergence,
        expected_query_ids=query_metadata,
        output_directory=pass_directory / "compact",
        resource_guard=lambda: require_disk_capacity(workspace, policy),
    )
    marker = {
        "schema_version": 1,
        "stage": "completed_search_pass",
        "fingerprint": fingerprint,
        "strategy": strategy,
        "partition": partition,
        "pass_name": pass_name,
        "query_count": len(query_metadata),
        "stages": stages,
        "convergence": asdict(convergence),
        "accepted": asdict(accepted),
    }
    write_json_atomic(marker_path, marker)
    _cleanup_full_alignment_rows(pass_directory)
    return marker


def _ensure_search_stage(
    *,
    cap: int,
    query_fasta: Path,
    query_fasta_evidence: FastaEvidence,
    query_metadata: Mapping[str, SequenceMetadata],
    target_database: Path,
    target_metadata: Mapping[str, SequenceMetadata],
    pass_directory: Path,
    pass_name: str,
    project_root: Path,
    workspace: Path,
    policy: SimilarityAuditPolicy,
    fingerprint: str,
) -> dict[str, object]:
    stage_directory = pass_directory / f"cap_{cap}"
    stage_directory.mkdir(parents=True, exist_ok=True)
    raw_path = stage_directory / "raw.tsv"
    canonical_path = stage_directory / "canonical.tsv"
    marker_path = stage_directory / "complete.json"
    temp_directory = stage_directory / "mmseqs_tmp"
    command = _search_command(
        policy=policy,
        pass_name=pass_name,
        cap=cap,
        query_fasta=query_fasta,
        target_database=target_database,
        raw_output=raw_path,
        temp_directory=temp_directory,
    )
    if marker_path.exists():
        marker = read_json(marker_path)
        require_marker_identity(marker, fingerprint, "search_stage")
        if marker.get("cap") != cap or marker.get("command") != command:
            raise SimilarityAuditError("completed search command drifted")
        if marker.get("query_count") != len(query_metadata) or marker.get(
            "query_fasta"
        ) != asdict(query_fasta_evidence):
            raise SimilarityAuditError("completed search query input drifted")
        evidence = marker.get("alignment_evidence")
        if not isinstance(evidence, dict):
            raise SimilarityAuditError("search-stage evidence is malformed")
        parsed = canonical_evidence_from(evidence)
        if marker.get("raw_retained") is not False or raw_path.exists():
            raise SimilarityAuditError("completed raw-output retention state drifted")
        verify_file(
            canonical_path,
            parsed.canonical.byte_size,
            parsed.canonical.sha256,
        )
        return {"canonical_path": canonical_path, "evidence": marker}

    raw_path.unlink(missing_ok=True)
    canonical_path.unlink(missing_ok=True)
    if temp_directory.exists():
        shutil.rmtree(temp_directory)
    runtime = run_mmseqs_command(
        command,
        project_root=project_root,
        workspace=workspace,
        log_path=stage_directory / "command.log",
        policy=policy,
    )
    if not raw_path.is_file():
        raise SimilarityAuditError("MMseqs2 did not produce its requested TSV")
    if temp_directory.exists():
        shutil.rmtree(temp_directory)
    alignment_evidence = canonicalize_mmseqs_tsv(
        raw_path,
        canonical_path,
        query_metadata=query_metadata,
        target_metadata=target_metadata,
        chunk_rows=policy.parser_chunk_rows,
        resource_guard=lambda: require_disk_capacity(workspace, policy),
        delete_raw_after_parse=True,
    )
    marker = {
        "schema_version": 1,
        "stage": "search_stage",
        "fingerprint": fingerprint,
        "cap": cap,
        "query_count": len(query_metadata),
        "query_fasta": asdict(query_fasta_evidence),
        "command": command,
        "runtime_seconds": runtime,
        "raw_retained": False,
        "alignment_evidence": asdict(alignment_evidence),
    }
    write_json_atomic(marker_path, marker)
    return {"canonical_path": canonical_path, "evidence": marker}


def _search_command(
    *,
    policy: SimilarityAuditPolicy,
    pass_name: str,
    cap: int,
    query_fasta: Path,
    target_database: Path,
    raw_output: Path,
    temp_directory: Path,
) -> list[str]:
    if pass_name == "enforcement":
        pass_options = (
            policy.enforcement_min_sequence_identity,
            policy.enforcement_coverage,
            policy.enforcement_coverage_mode,
        )
    elif pass_name == "residual":
        pass_options = (
            policy.residual_min_sequence_identity,
            policy.residual_coverage,
            policy.residual_coverage_mode,
        )
    else:
        raise SimilarityAuditError(f"unknown search pass: {pass_name}")
    min_identity, coverage, coverage_mode = pass_options
    return [
        policy.mmseqs_executable,
        "easy-search",
        str(query_fasta),
        str(target_database),
        str(raw_output),
        str(temp_directory),
        "--search-type",
        str(policy.search_type),
        "--alignment-mode",
        str(policy.alignment_mode),
        "--seq-id-mode",
        str(policy.sequence_identity_mode),
        "-s",
        policy.sensitivity,
        "-e",
        policy.evalue_cutoff,
        "--mask",
        str(policy.mask),
        "--comp-bias-corr",
        str(policy.composition_bias_correction),
        "--max-seqs",
        str(cap),
        "--threads",
        str(policy.threads),
        "--format-output",
        policy.format_output,
        "--min-seq-id",
        min_identity,
        "-c",
        coverage,
        "--cov-mode",
        str(coverage_mode),
    ]


def _cleanup_full_alignment_rows(pass_directory: Path) -> None:
    for stage_directory in pass_directory.glob("cap_*"):
        if not stage_directory.is_dir():
            continue
        (stage_directory / "raw.tsv").unlink(missing_ok=True)
        (stage_directory / "canonical.tsv").unlink(missing_ok=True)
        temp_directory = stage_directory / "mmseqs_tmp"
        if temp_directory.exists():
            shutil.rmtree(temp_directory)
