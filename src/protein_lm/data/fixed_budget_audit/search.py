"""Search commands, lifecycles, records, and verification for fixed-budget audits."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from protein_lm.data.artifacts import (
    canonical_evidence_from,
    fasta_evidence_from,
    file_evidence_from,
    file_identity,
    read_json,
    require_marker_identity,
    verify_compact_file,
    verify_database_artifacts,
    verify_file,
    write_json_atomic,
)
from protein_lm.data.fixed_budget_audit.config import AuditPass, FIXED_CAPS
from protein_lm.data.fixed_budget_audit.errors import (
    AuditConfigurationError,
    AuditExecutionError,
    AuditValidationError,
)
from protein_lm.data.fixed_budget_audit.execution import (
    require_disk_capacity,
    run_mmseqs_command,
)
from protein_lm.data.similarity_audit_models import FileEvidence, SequenceMetadata
from protein_lm.data.similarity_audit_policy import SimilarityAuditPolicy
from protein_lm.data.similarity_evidence import compact_converged_results
from protein_lm.data.similarity_fastas import (
    FastaEvidence,
    iter_one_line_fasta,
    write_fasta_subset,
)
from protein_lm.data.similarity_results import (
    canonicalize_mmseqs_tsv,
    compare_canonical_results,
    convergence_evidence,
)


# Records and runner boundaries


SearchRunner = Callable[[Sequence[str], Path, Path, Path, SimilarityAuditPolicy], str]
DatabaseRunner = Callable[[Sequence[str], Path, Path, Path, SimilarityAuditPolicy], str]


@dataclass(frozen=True)
class FixedBudgetStage:
    """One retained canonical output produced at a frozen candidate cap."""

    cap: int
    query_fasta: FastaEvidence
    canonical: FileEvidence
    canonical_path: Path
    command: tuple[str, ...]
    runtime_seconds: str
    marker_path: Path


@dataclass(frozen=True)
class A004Database:
    """One verified A-004 database, never reused from the A-003 workspace."""

    strategy: str
    prefix: Path
    command: tuple[str, ...]
    runtime_seconds: str
    marker_path: Path
    identity: Mapping[str, object]


# Pure command and query contracts


def createdb_command(
    policy: SimilarityAuditPolicy,
    *,
    training_fasta: Path,
    database_prefix: Path,
) -> tuple[str, ...]:
    """Return the deterministic training-database command."""

    return (
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
    )


def search_command(
    policy: SimilarityAuditPolicy,
    *,
    pass_name: str,
    cap: int,
    query_fasta: Path,
    target_database: Path,
    raw_output: Path,
    temp_directory: Path,
) -> tuple[str, ...]:
    """Return one deterministic enforcement or residual search command."""

    if pass_name == AuditPass.ENFORCEMENT.value:
        min_identity = policy.enforcement_min_sequence_identity
        coverage = policy.enforcement_coverage
        coverage_mode = policy.enforcement_coverage_mode
    elif pass_name == AuditPass.RESIDUAL.value:
        min_identity = policy.residual_min_sequence_identity
        coverage = policy.residual_coverage
        coverage_mode = policy.residual_coverage_mode
    else:
        raise AuditConfigurationError(f"unknown search pass: {pass_name}")

    return (
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
    )


def query_ids_sha256(query_ids: Iterable[str]) -> str:
    """Return a deterministic membership identity for a query universe."""

    ordered = tuple(sorted(query_ids))
    if not ordered or len(set(ordered)) != len(ordered):
        raise AuditConfigurationError("query identifiers must be nonempty and unique")
    content = "".join(f"{value}\n" for value in ordered).encode()
    return hashlib.sha256(content).hexdigest()


def verify_query_fasta(
    path: Path,
    evidence: FastaEvidence,
    query_metadata: Mapping[str, object],
) -> tuple[str, ...]:
    """Verify a FASTA's bytes and exact accession universe."""

    verify_file(path, evidence.byte_size, evidence.sha256)
    expected = set(query_metadata)
    if not expected or evidence.record_count != len(expected):
        raise AuditConfigurationError("A-004 query FASTA evidence does not reconcile")
    observed: set[str] = set()
    for accession, _ in iter_one_line_fasta(path):
        if accession in observed or accession not in expected:
            raise AuditConfigurationError("A-004 query FASTA universe drifted")
        observed.add(accession)
    if observed != expected:
        raise AuditConfigurationError("A-004 query FASTA universe drifted")
    return tuple(sorted(observed))


def require_fixed_policy_caps(policy: SimilarityAuditPolicy) -> None:
    """Require the frozen 1k, 10k, and staged 100k candidate budget."""

    if (policy.initial_cap, policy.comparison_cap, policy.escalation_cap) != FIXED_CAPS:
        raise AuditConfigurationError(
            "A-004 policy must use the frozen fixed-budget caps"
        )


# Historical Task 7 search lifecycle


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
    command = list(
        createdb_command(
            policy,
            training_fasta=training_fasta,
            database_prefix=database_prefix,
        )
    )
    marker_path = final_directory / "complete.json"
    if marker_path.exists():
        marker = read_json(marker_path)
        require_marker_identity(marker, fingerprint, "target_database")
        if (
            marker.get("strategy") != strategy
            or marker.get("training_fasta") != asdict(training_fasta_evidence)
            or marker.get("command") != command
        ):
            raise AuditValidationError("target database completion evidence drifted")
        verify_database_artifacts(final_directory, marker.get("artifacts"))
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
        raise AuditExecutionError("MMseqs2 createdb produced no database files")
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
        _cleanup_historical_full_alignment_rows(pass_directory)
        return marker

    pass_directory.mkdir(parents=True, exist_ok=True)
    stages = {}
    initial = _ensure_historical_search_stage(
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
    comparison = _ensure_historical_search_stage(
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
        escalation = _ensure_historical_search_stage(
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
    _cleanup_historical_full_alignment_rows(pass_directory)
    return marker


def _ensure_historical_search_stage(
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
    command = list(
        search_command(
            policy,
            pass_name=pass_name,
            cap=cap,
            query_fasta=query_fasta,
            target_database=target_database,
            raw_output=raw_path,
            temp_directory=temp_directory,
        )
    )
    if marker_path.exists():
        marker = read_json(marker_path)
        require_marker_identity(marker, fingerprint, "search_stage")
        if marker.get("cap") != cap or marker.get("command") != command:
            raise AuditValidationError("completed search command drifted")
        if marker.get("query_count") != len(query_metadata) or marker.get(
            "query_fasta"
        ) != asdict(query_fasta_evidence):
            raise AuditValidationError("completed search query input drifted")
        evidence = marker.get("alignment_evidence")
        if not isinstance(evidence, dict):
            raise AuditValidationError("search-stage evidence is malformed")
        parsed = canonical_evidence_from(evidence)
        if marker.get("raw_retained") is not False or raw_path.exists():
            raise AuditValidationError("completed raw-output retention state drifted")
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
        raise AuditExecutionError("MMseqs2 did not produce its requested TSV")
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


def _cleanup_historical_full_alignment_rows(pass_directory: Path) -> None:
    for stage_directory in pass_directory.glob("cap_*"):
        if not stage_directory.is_dir():
            continue
        (stage_directory / "raw.tsv").unlink(missing_ok=True)
        (stage_directory / "canonical.tsv").unlink(missing_ok=True)
        temp_directory = stage_directory / "mmseqs_tmp"
        if temp_directory.exists():
            shutil.rmtree(temp_directory)


# A-004 database lifecycle


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
        return _verified_a004_database(
            marker_path=marker_path,
            final_directory=final_directory,
            strategy=strategy,
            training_fasta_evidence=training_fasta_evidence,
            fingerprint=fingerprint,
            command=command,
        )
    if final_directory.exists():
        raise AuditExecutionError(
            "A-004 database directory lacks its completion marker"
        )
    if incomplete_directory.exists():
        raise AuditExecutionError("A-004 database has an unmarked incomplete directory")

    incomplete_directory.mkdir(parents=True)
    runtime = _run_database_command(
        command,
        project_root=project_root,
        workspace=workspace,
        log_path=workspace / "logs" / f"a004_createdb_{strategy}.log",
        policy=policy,
        command_runner=command_runner,
    )
    artifact_paths = tuple(
        sorted(path for path in incomplete_directory.iterdir() if path.is_file())
    )
    if not artifact_paths or not (incomplete_directory / "target").is_file():
        raise AuditExecutionError("A-004 MMseqs2 createdb produced no target database")
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
    return _verified_a004_database(
        marker_path=marker_path,
        final_directory=final_directory,
        strategy=strategy,
        training_fasta_evidence=training_fasta_evidence,
        fingerprint=fingerprint,
        command=command,
    )


def verify_a004_database(database: A004Database, *, fingerprint: str) -> None:
    """Re-read one completed database marker and its exact artifact inventory."""

    marker_identity = database.identity.get("marker")
    if not isinstance(marker_identity, dict):
        raise AuditValidationError("A-004 target-database marker identity is malformed")
    size = marker_identity.get("byte_size")
    digest = marker_identity.get("sha256")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or not isinstance(digest, str)
    ):
        raise AuditValidationError("A-004 target-database marker identity is malformed")
    verify_file(database.marker_path, size, digest)
    marker = read_json(database.marker_path)
    require_marker_identity(marker, fingerprint, "a004_target_database")
    if (
        marker.get("strategy") != database.strategy
        or marker.get("database_prefix") != str(database.prefix)
        or marker.get("artifacts") != database.identity.get("artifacts")
        or marker.get("training_fasta") != database.identity.get("training_fasta")
    ):
        raise AuditValidationError("A-004 target-database evidence drifted")
    verify_database_artifacts(database.prefix.parent, marker.get("artifacts"))


def _verified_a004_database(
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
        raise AuditValidationError("A-004 target-database identity drifted")
    runtime = marker.get("runtime_seconds")
    if not isinstance(runtime, str):
        raise AuditValidationError("A-004 target-database runtime is malformed")
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


# A-004 fixed-budget stage lifecycle


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
        raise AuditExecutionError("A-004 search stage has unmarked output")
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
        raise AuditExecutionError("A-004 MMseqs2 search did not produce its TSV")
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
        raise AuditExecutionError(
            "A-004 raw search output was not retired after parsing"
        )
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
            raise AuditValidationError("A-004 escalation FASTA identity drifted")
        verify_query_fasta(path, fasta, {item: None for item in changed_query_ids})
        return path, fasta, marker_path
    if path.exists():
        raise AuditExecutionError("A-004 escalation FASTA lacks its completion marker")
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
        raise AuditValidationError("A-004 search-stage identity drifted")
    if raw_path.exists() or marker.get("raw_retained") is not False:
        raise AuditValidationError("A-004 search-stage raw-output state drifted")
    verify_file(
        canonical_path, alignment.canonical.byte_size, alignment.canonical.sha256
    )
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


# Completed historical pass verification


def validate_completed_pass(
    marker: Mapping[str, object],
    *,
    marker_path: Path,
    fingerprint: str,
    strategy: str,
    partition: str,
    pass_name: str,
    expected_query_ids: frozenset[str],
    policy: SimilarityAuditPolicy,
) -> None:
    """Prove that a resumed search pass is complete and internally consistent."""

    require_marker_identity(marker, fingerprint, "completed_search_pass")
    expected = {
        "strategy": strategy,
        "partition": partition,
        "pass_name": pass_name,
        "query_count": len(expected_query_ids),
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        raise AuditValidationError("completed search-pass identity drifted")
    convergence = marker.get("convergence")
    if (
        not isinstance(convergence, dict)
        or convergence.get("final_differing_queries") != 0
    ):
        raise AuditValidationError("completed search pass is not converged")
    escalated_ids = convergence.get("escalated_query_ids")
    if (
        not isinstance(escalated_ids, list)
        or any(not isinstance(value, str) for value in escalated_ids)
        or escalated_ids != sorted(set(escalated_ids))
        or not set(escalated_ids) <= expected_query_ids
    ):
        raise AuditValidationError("completed escalation query set is malformed")
    escalated_count = len(escalated_ids)
    expected_convergence = {
        "expected_queries": len(expected_query_ids),
        "converged_at_comparison_cap": len(expected_query_ids) - escalated_count,
        "escalated_queries": escalated_count,
        "converged_at_escalation_cap": escalated_count,
        "final_differing_queries": 0,
    }
    if any(
        convergence.get(key) != value for key, value in expected_convergence.items()
    ):
        raise AuditValidationError("completed convergence counts do not reconcile")
    expected_stage_caps = {str(policy.initial_cap), str(policy.comparison_cap)}
    if escalated_count:
        expected_stage_caps.add(str(policy.escalation_cap))
    stages = marker.get("stages")
    if not isinstance(stages, dict) or set(stages) != expected_stage_caps:
        raise AuditValidationError("completed search stages do not reconcile")
    accepted = marker.get("accepted")
    if not isinstance(accepted, dict):
        raise AuditValidationError("accepted pass evidence is malformed")
    expected_accepted = {
        "pass_name": pass_name,
        "accepted_at_comparison_cap": len(expected_query_ids) - escalated_count,
        "accepted_at_escalation_cap": escalated_count,
    }
    if any(accepted.get(key) != value for key, value in expected_accepted.items()):
        raise AuditValidationError("accepted cap distribution does not reconcile")
    accepted_rows = _nonnegative_int(
        accepted.get("accepted_rows"), "accepted row count"
    )
    compact = marker_path.parent / "compact"
    verify_compact_file(compact / "returned_pairs.tsv", accepted.get("returned_pairs"))
    verify_compact_file(
        compact / "prohibited_pairs.tsv",
        accepted.get("prohibited_pairs"),
    )
    returned = file_evidence_from(accepted["returned_pairs"])
    prohibited = file_evidence_from(accepted["prohibited_pairs"])
    if returned.row_count != accepted_rows or prohibited.row_count > accepted_rows:
        raise AuditValidationError("accepted pair counts do not reconcile")
    summaries = accepted.get("residual_summaries")
    if pass_name == AuditPass.RESIDUAL.value:
        verify_compact_file(compact / "residual_summaries.tsv", summaries)
        if file_evidence_from(summaries).row_count != len(expected_query_ids):
            raise AuditValidationError("residual summary query count drifted")
    elif summaries is not None:
        raise AuditValidationError("enforcement pass has residual summaries")


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AuditValidationError(f"{name} must be a nonnegative integer")
    return value
