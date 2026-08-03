"""Read-only checks for reusable A-003 database and search stages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath

from protein_lm.data.similarity_audit_models import FileEvidence
from protein_lm.data.similarity_audit_policy import (
    SimilarityAuditError,
    SimilarityAuditPolicy,
)
from protein_lm.data.similarity_fastas import FastaEvidence
from protein_lm.data.task7_a004_policy import A004Policy
from protein_lm.data.task7_checkpoints import (
    canonical_evidence_from,
    fasta_evidence_from,
    require_marker_identity,
    verify_database_artifacts,
    verify_file,
)
from protein_lm.data.task7_commands import createdb_command, search_command


@dataclass(frozen=True)
class MarkerEvidence:
    """Identity of one pinned historical completion marker."""

    byte_size: int
    sha256: str


@dataclass(frozen=True)
class DatabaseImport:
    """Verified A-003 target-database evidence."""

    marker: MarkerEvidence
    artifact_count: int


@dataclass(frozen=True)
class ImportedStage:
    """One verified A-003 residual search stage."""

    cap: int
    marker: MarkerEvidence
    query_fasta: FastaEvidence
    canonical: FileEvidence
    canonical_path: Path
    command: tuple[str, ...]
    runtime_seconds: str


def verify_database(
    *,
    source_workspace: Path,
    source_policy: SimilarityAuditPolicy,
    policy: A004Policy,
    fingerprint: str,
    training: FastaEvidence,
) -> DatabaseImport:
    """Verify the database marker, input, command, and current artifacts."""

    database_directory = source_workspace / "databases" / policy.import_strategy
    marker, marker_evidence = read_pinned_marker(
        database_directory / "complete.json",
        policy.source_database_marker_sha256,
    )
    require_marker_identity(marker, fingerprint, "target_database")
    if marker.get("strategy") != policy.import_strategy:
        raise SimilarityAuditError("A-003 target-database strategy drifted")
    if marker.get("training_fasta") != asdict(training):
        raise SimilarityAuditError("A-003 target-database input drifted")

    command = marker_command(marker)
    if len(command) < 4 or command != createdb_command(
        source_policy,
        training_fasta=Path(command[2]),
        database_prefix=Path(command[3]),
    ):
        raise SimilarityAuditError("A-003 target-database command drifted")
    require_command_path(
        command[2], policy.source_workspace_relative_path, "fastas/random_training.fasta"
    )
    require_command_path(
        command[3],
        policy.source_workspace_relative_path,
        "databases/.random.incomplete/target",
    )
    runtime_seconds(marker)
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, dict):
        raise SimilarityAuditError("A-003 database artifact index is malformed")
    verify_database_artifacts(database_directory, artifacts)
    return DatabaseImport(marker=marker_evidence, artifact_count=len(artifacts))


def verify_stage(
    *,
    cap: int,
    source_workspace: Path,
    source_policy: SimilarityAuditPolicy,
    policy: A004Policy,
    fingerprint: str,
    expected_query: FastaEvidence | None,
) -> ImportedStage:
    """Verify one canonical residual stage without modifying it."""

    relative_stage = (
        Path("tracks")
        / policy.import_strategy
        / policy.import_partition
        / policy.import_pass
        / f"cap_{cap}"
    )
    stage_directory = source_workspace / relative_stage
    marker, marker_evidence = read_pinned_marker(
        stage_directory / "complete.json",
        policy.stage_marker_sha256(cap),
    )
    require_marker_identity(marker, fingerprint, "search_stage")
    if marker.get("cap") != cap:
        raise SimilarityAuditError(f"A-003 cap {cap} marker identity drifted")

    query = fasta_evidence_from(marker.get("query_fasta"))
    if marker.get("query_count") != query.record_count:
        raise SimilarityAuditError(f"A-003 cap {cap} query count drifted")
    if expected_query is not None and query != expected_query:
        raise SimilarityAuditError(f"A-003 cap {cap} query input drifted")
    query_path = stage_query_path(cap, source_workspace, policy)
    verify_file(query_path, query.byte_size, query.sha256)

    command = marker_command(marker)
    verify_search_command(
        command=command,
        cap=cap,
        relative_stage=relative_stage,
        source_policy=source_policy,
        policy=policy,
    )
    evidence = canonical_evidence_from(marker.get("alignment_evidence"))
    canonical_path = stage_directory / "canonical.tsv"
    if marker.get("raw_retained") is not False or (stage_directory / "raw.tsv").exists():
        raise SimilarityAuditError(f"A-003 cap {cap} raw-retention state drifted")
    verify_file(
        canonical_path,
        evidence.canonical.byte_size,
        evidence.canonical.sha256,
    )
    return ImportedStage(
        cap=cap,
        marker=marker_evidence,
        query_fasta=query,
        canonical=evidence.canonical,
        canonical_path=canonical_path,
        command=command,
        runtime_seconds=runtime_seconds(marker),
    )


def read_pinned_marker(
    path: Path,
    expected_sha256: str,
) -> tuple[dict[str, object], MarkerEvidence]:
    """Read one marker only after its bytes match the A-004 pin."""

    if not path.is_file():
        raise SimilarityAuditError(f"A-003 completion marker is missing: {path}")
    try:
        content = path.read_bytes()
    except OSError as error:
        raise SimilarityAuditError(f"could not read A-003 completion marker: {path}") from error
    calculated_sha256 = hashlib.sha256(content).hexdigest()
    if calculated_sha256 != expected_sha256:
        raise SimilarityAuditError(f"A-003 completion marker checksum drifted: {path}")
    try:
        marker = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SimilarityAuditError(f"A-003 completion marker is malformed: {path}") from error
    if not isinstance(marker, dict):
        raise SimilarityAuditError(f"A-003 completion marker root is malformed: {path}")
    evidence = MarkerEvidence(
        byte_size=len(content),
        sha256=calculated_sha256,
    )
    return marker, evidence


def verify_search_command(
    *,
    command: tuple[str, ...],
    cap: int,
    relative_stage: Path,
    source_policy: SimilarityAuditPolicy,
    policy: A004Policy,
) -> None:
    """Match all options and the intended historical path suffixes."""

    if len(command) < 6 or command != search_command(
        source_policy,
        pass_name=policy.import_pass,
        cap=cap,
        query_fasta=Path(command[2]),
        target_database=Path(command[3]),
        raw_output=Path(command[4]),
        temp_directory=Path(command[5]),
    ):
        raise SimilarityAuditError(f"A-003 cap {cap} search command drifted")

    query_suffix = (
        "tracks/random/validation/residual/escalated_queries.fasta"
        if cap == policy.staged_escalation_cap
        else "fastas/random_validation.fasta"
    )
    require_command_path(command[2], policy.source_workspace_relative_path, query_suffix)
    require_command_path(
        command[3], policy.source_workspace_relative_path, "databases/random/target"
    )
    stage = relative_stage.as_posix()
    require_command_path(
        command[4], policy.source_workspace_relative_path, f"{stage}/raw.tsv"
    )
    require_command_path(
        command[5], policy.source_workspace_relative_path, f"{stage}/mmseqs_tmp"
    )


def stage_query_path(cap: int, source_workspace: Path, policy: A004Policy) -> Path:
    """Return the full-query or escalation FASTA for one cap."""

    if cap == policy.staged_escalation_cap:
        return (
            source_workspace
            / "tracks"
            / policy.import_strategy
            / policy.import_partition
            / policy.import_pass
            / "escalated_queries.fasta"
        )
    return source_workspace / "fastas" / "random_validation.fasta"


def marker_command(marker: dict[str, object]) -> tuple[str, ...]:
    command = marker.get("command")
    if not isinstance(command, list) or any(not isinstance(value, str) for value in command):
        raise SimilarityAuditError("A-003 MMseqs2 command is malformed")
    return tuple(command)


def require_command_path(value: str, workspace: str, suffix: str) -> None:
    path = PurePosixPath(value)
    expected = PurePosixPath(workspace) / suffix
    if not path.is_absolute() or path.parts[-len(expected.parts) :] != expected.parts:
        raise SimilarityAuditError("A-003 MMseqs2 command path drifted")


def runtime_seconds(marker: dict[str, object]) -> str:
    value = marker.get("runtime_seconds")
    if not isinstance(value, str):
        raise SimilarityAuditError("A-003 runtime is malformed")
    try:
        runtime = Decimal(value)
    except InvalidOperation as error:
        raise SimilarityAuditError("A-003 runtime is malformed") from error
    if not runtime.is_finite() or runtime < 0:
        raise SimilarityAuditError("A-003 runtime is malformed")
    return value
