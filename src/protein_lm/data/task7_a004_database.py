"""Resumable, separate target databases for the A-004 audit workspace."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from protein_lm.data.similarity_audit_policy import (
    SimilarityAuditError,
    SimilarityAuditPolicy,
)
from protein_lm.data.similarity_fastas import FastaEvidence
from protein_lm.data.task7_checkpoints import (
    file_identity,
    read_json,
    require_marker_identity,
    verify_database_artifacts,
    verify_file,
    write_json_atomic,
)
from protein_lm.data.task7_commands import createdb_command
from protein_lm.data.task7_execution import run_mmseqs_command

DatabaseRunner = Callable[
    [Sequence[str], Path, Path, Path, SimilarityAuditPolicy], str
]


@dataclass(frozen=True)
class A004Database:
    """One verified A-004 database, never reused from the A-003 workspace."""

    strategy: str
    prefix: Path
    command: tuple[str, ...]
    runtime_seconds: str
    marker_path: Path
    identity: Mapping[str, object]


def ensure_a004_target_database(
    *,
    strategy: str,
    training_fasta: Path,
    training_fasta_evidence: FastaEvidence,
    project_root: Path,
    workspace: Path,
    policy: SimilarityAuditPolicy,
    fingerprint: str,
    command_runner: DatabaseRunner | None = None,
) -> A004Database:
    """Build once or verify a fresh database inside the A-004 workspace."""

    verify_file(
        training_fasta,
        training_fasta_evidence.byte_size,
        training_fasta_evidence.sha256,
    )
    database_root = workspace / "databases"
    final_directory = database_root / strategy
    incomplete_directory = database_root / f".{strategy}.incomplete"
    marker_path = final_directory / "complete.json"
    command = createdb_command(
        policy,
        training_fasta=training_fasta,
        database_prefix=incomplete_directory / "target",
    )
    if marker_path.exists():
        return _verified_database(
            marker_path=marker_path,
            final_directory=final_directory,
            strategy=strategy,
            training_fasta_evidence=training_fasta_evidence,
            fingerprint=fingerprint,
            command=command,
        )
    if final_directory.exists():
        raise SimilarityAuditError("A-004 database directory lacks its completion marker")
    if incomplete_directory.exists():
        raise SimilarityAuditError("A-004 database has an unmarked incomplete directory")

    incomplete_directory.mkdir(parents=True)
    runtime = _run_database_command(
        command,
        project_root=project_root,
        workspace=workspace,
        log_path=workspace / "logs" / f"a004_createdb_{strategy}.log",
        policy=policy,
        command_runner=command_runner,
    )
    artifact_paths = tuple(sorted(path for path in incomplete_directory.iterdir() if path.is_file()))
    if not artifact_paths or not (incomplete_directory / "target").is_file():
        raise SimilarityAuditError("A-004 MMseqs2 createdb produced no target database")
    artifacts = {path.name: file_identity(path) for path in artifact_paths}
    marker = {
        "schema_version": 1,
        "stage": "a004_target_database",
        "fingerprint": fingerprint,
        "strategy": strategy,
        "training_fasta": asdict(training_fasta_evidence),
        "database_prefix": str(final_directory / "target"),
        "command": list(command),
        "runtime_seconds": runtime,
        "artifacts": artifacts,
    }
    write_json_atomic(incomplete_directory / "complete.json", marker)
    incomplete_directory.replace(final_directory)
    return _verified_database(
        marker_path=marker_path,
        final_directory=final_directory,
        strategy=strategy,
        training_fasta_evidence=training_fasta_evidence,
        fingerprint=fingerprint,
        command=command,
    )


def _verified_database(
    *,
    marker_path: Path,
    final_directory: Path,
    strategy: str,
    training_fasta_evidence: FastaEvidence,
    fingerprint: str,
    command: tuple[str, ...],
) -> A004Database:
    marker = read_json(marker_path)
    require_marker_identity(marker, fingerprint, "a004_target_database")
    expected = {
        "strategy": strategy,
        "training_fasta": asdict(training_fasta_evidence),
        "database_prefix": str(final_directory / "target"),
        "command": list(command),
    }
    if any(marker.get(name) != value for name, value in expected.items()):
        raise SimilarityAuditError("A-004 target-database identity drifted")
    runtime = marker.get("runtime_seconds")
    if not isinstance(runtime, str):
        raise SimilarityAuditError("A-004 target-database runtime is malformed")
    verify_database_artifacts(final_directory, marker.get("artifacts"))
    identity = {
        "marker": file_identity(marker_path),
        "training_fasta": marker["training_fasta"],
        "artifacts": marker["artifacts"],
    }
    return A004Database(
        strategy=strategy,
        prefix=final_directory / "target",
        command=command,
        runtime_seconds=runtime,
        marker_path=marker_path,
        identity=identity,
    )


def verify_a004_database(database: A004Database, *, fingerprint: str) -> None:
    """Re-read one completed database marker and its exact artifact inventory."""

    marker_identity = database.identity.get("marker")
    if not isinstance(marker_identity, dict):
        raise SimilarityAuditError("A-004 target-database marker identity is malformed")
    size = marker_identity.get("byte_size")
    digest = marker_identity.get("sha256")
    if isinstance(size, bool) or not isinstance(size, int) or not isinstance(digest, str):
        raise SimilarityAuditError("A-004 target-database marker identity is malformed")
    verify_file(database.marker_path, size, digest)
    marker = read_json(database.marker_path)
    require_marker_identity(marker, fingerprint, "a004_target_database")
    if (
        marker.get("strategy") != database.strategy
        or marker.get("database_prefix") != str(database.prefix)
        or marker.get("artifacts") != database.identity.get("artifacts")
        or marker.get("training_fasta") != database.identity.get("training_fasta")
    ):
        raise SimilarityAuditError("A-004 target-database evidence drifted")
    verify_database_artifacts(database.prefix.parent, marker.get("artifacts"))


def _run_database_command(
    command: Sequence[str],
    *,
    project_root: Path,
    workspace: Path,
    log_path: Path,
    policy: SimilarityAuditPolicy,
    command_runner: DatabaseRunner | None,
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
