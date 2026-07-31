"""Run the A-003-authorized Week 1 Task 7 diagnostic similarity audit."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from protein_lm.data.random_split import sha256_sidecar
from protein_lm.data.similarity_audit import (
    CLOSEST_RESIDUAL_CATEGORIES,
    RESIDUAL_CATEGORIES,
    CanonicalAlignmentEvidence,
    FileEvidence,
    SequenceMetadata,
    aggregate_partition_evidence,
    canonicalize_mmseqs_tsv,
    compact_converged_results,
    compare_canonical_results,
    convergence_evidence,
    verify_boundary_fixtures,
)
from protein_lm.data.similarity_audit_policy import (
    APPROVED_SIMILARITY_AUDIT_CONFIG_SHA256,
    SimilarityAuditError,
    SimilarityAuditPolicy,
    load_similarity_audit_policy,
)
from protein_lm.data.similarity_inputs import (
    PARTITIONS,
    STRATEGIES,
    FastaEvidence,
    MaterializedInputs,
    StrategyManifest,
    load_strategy_manifest,
    materialize_strategy_fastas,
    metadata_by_partition,
    write_fasta_subset,
)
from protein_lm.data.task5_report import (
    CompletedPublicArtifact,
    render_completion_index,
)
from protein_lm.data.task7_report import render_task7_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "experiments" / "week_01" / "diagnostic_similarity_audit.toml"
)
REPORT_DIRECTORY = PROJECT_ROOT / "reports" / "week_01"
OUTPUT_STEM = "task_07_diagnostic_similarity_audit"
REPORT_FILENAMES = (
    f"{OUTPUT_STEM}.json",
    f"{OUTPUT_STEM}.md",
    f"{OUTPUT_STEM}.sha256",
)
COMPLETION_FILENAME = f"{OUTPUT_STEM}.complete.json"
COMPLETION_SCOPE = "week_01_task_07_public_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run only the A-003 diagnostic MMseqs2 audit. This command never "
            "repairs, selects, or trains on a split."
        )
    )
    parser.add_argument(
        "--execute-diagnostic-audit",
        action="store_true",
        required=True,
        help="required safety acknowledgement that starts the corpus searches",
    )
    return parser.parse_args()


def main() -> int:
    parse_args()
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    try:
        policy = load_similarity_audit_policy(CONFIG_PATH)
        _require_committed_execution_code()
        code_revision = _git_output("rev-parse", "HEAD")
        mmseqs_version = _verify_mmseqs(policy)
        paths = _policy_paths(policy)
        workspace = paths["workspace"]
        _prove_path_is_ignored(workspace)
        for output_path in (
            *(REPORT_DIRECTORY / filename for filename in REPORT_FILENAMES),
            REPORT_DIRECTORY / COMPLETION_FILENAME,
        ):
            _prove_path_is_public(output_path)
        if (REPORT_DIRECTORY / COMPLETION_FILENAME).exists():
            raise SimilarityAuditError(
                "a completed Task 7 public report already exists; it will not be overwritten"
            )

        workspace.mkdir(parents=True, exist_ok=True)
        with _exclusive_lock(workspace / "audit.lock"):
            _require_disk_capacity(workspace, policy)
            verify_boundary_fixtures()
            reports = _load_and_validate_frozen_reports(paths, policy)
            manifests = _load_frozen_manifests(paths, policy)
            balances = _validate_report_populations(reports, manifests, policy)
            fingerprint = _run_fingerprint(
                policy=policy,
                code_revision=code_revision,
                mmseqs_version=mmseqs_version,
            )
            inputs = _ensure_materialized_inputs(
                workspace=workspace,
                catalog_path=paths["catalog"],
                manifests=manifests,
                policy=policy,
                fingerprint=fingerprint,
            )

            strategy_reports: dict[str, object] = {}
            database_reports: dict[str, object] = {}
            for strategy in STRATEGIES:
                print(f"preparing {strategy} training database...")
                training_fasta = _fasta_path(workspace, strategy, "training")
                database_prefix, database_evidence = _ensure_target_database(
                    strategy=strategy,
                    training_fasta=training_fasta,
                    training_fasta_evidence=inputs.fastas[strategy]["training"],
                    workspace=workspace,
                    policy=policy,
                    fingerprint=fingerprint,
                )
                database_reports[strategy] = database_evidence
                training_metadata = metadata_by_partition(
                    manifests[strategy],
                    "training",
                )
                partition_reports: dict[str, object] = {
                    "training": {"balance": balances[strategy]["training"]}
                }
                for partition in ("validation", "test"):
                    print(f"auditing {strategy} {partition} against training...")
                    query_metadata = metadata_by_partition(
                        manifests[strategy],
                        partition,
                    )
                    query_fasta = _fasta_path(workspace, strategy, partition)
                    pass_reports = {}
                    for pass_name in ("enforcement", "residual"):
                        pass_reports[pass_name] = _ensure_search_pass(
                            strategy=strategy,
                            partition=partition,
                            pass_name=pass_name,
                            query_fasta=query_fasta,
                            query_fasta_evidence=inputs.fastas[strategy][partition],
                            query_metadata=query_metadata,
                            target_database=database_prefix,
                            target_metadata=training_metadata,
                            workspace=workspace,
                            policy=policy,
                            fingerprint=fingerprint,
                        )
                    partition_directory = (
                        workspace / "tracks" / strategy / partition
                    )
                    similarity = aggregate_partition_evidence(
                        expected_query_ids=query_metadata,
                        query_metadata=query_metadata,
                        target_metadata=training_metadata,
                        enforcement_directory=partition_directory / "enforcement" / "compact",
                        residual_directory=partition_directory / "residual" / "compact",
                    )
                    partition_reports[partition] = {
                        "balance": balances[strategy][partition],
                        "similarity": similarity,
                        "passes": {
                            pass_name: _public_pass_evidence(pass_reports[pass_name])
                            for pass_name in ("enforcement", "residual")
                        },
                    }
                strategy_reports[strategy] = {
                    "stage": manifests[strategy].stage,
                    "structural_membership": _structural_membership_evidence(
                        strategy,
                        manifests[strategy],
                        reports[strategy],
                    ),
                    "partitions": partition_reports,
                    "overall": _overall_similarity(partition_reports),
                }

            _reverify_frozen_run_state(
                paths=paths,
                policy=policy,
                code_revision=code_revision,
                mmseqs_version=mmseqs_version,
            )
            completed_at = datetime.now(UTC)
            report = _build_report(
                policy=policy,
                code_revision=code_revision,
                fingerprint=fingerprint,
                mmseqs_version=mmseqs_version,
                started_at=started_at,
                completed_at=completed_at,
                runtime_seconds=time.perf_counter() - started,
                inputs=inputs,
                manifests=manifests,
                frozen_reports=reports,
                database_reports=database_reports,
                strategy_reports=strategy_reports,
            )
            rendered = render_task7_report(report)
            _publish_report(rendered, workspace)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"diagnostic similarity audit failed: {error}")
        return 1

    print(f"report JSON SHA-256: {rendered.json_sha256}")
    print(f"outputs: {REPORT_DIRECTORY}")
    print("candidate status: failed_balance")
    print("repair performed: false")
    print("selected split authorized: false")
    print("model use: prohibited")
    print("post-audit review required: true")
    print("network requests made: none")
    return 0


def _load_frozen_manifests(
    paths: Mapping[str, Path],
    policy: SimilarityAuditPolicy,
) -> dict[str, StrategyManifest]:
    random_manifest = load_strategy_manifest(
        public_path=paths["task5_public"],
        local_path=paths["task5_local"],
        strategy="random",
        stage="diagnostic",
        expected_public_sha256=policy.task5_public_manifest_sha256,
        expected_local_sha256=policy.task5_local_assignment_sha256,
    )
    candidate_manifest = load_strategy_manifest(
        public_path=paths["task6_public"],
        local_path=paths["task6_local"],
        strategy="group_aware",
        stage="pre_repair",
        expected_public_sha256=policy.task6_public_manifest_sha256,
        expected_local_sha256=policy.task6_local_assignment_sha256,
    )
    return {"random": random_manifest, "group_aware": candidate_manifest}


def _ensure_materialized_inputs(
    *,
    workspace: Path,
    catalog_path: Path,
    manifests: Mapping[str, StrategyManifest],
    policy: SimilarityAuditPolicy,
    fingerprint: str,
) -> MaterializedInputs:
    fasta_directory = workspace / "fastas"
    marker_path = fasta_directory / "complete.json"
    if marker_path.exists():
        marker = _read_json(marker_path)
        _require_marker_identity(marker, fingerprint, "materialized_inputs")
        catalog_evidence = _file_evidence_from(marker["catalog"])
        if (
            catalog_evidence.row_count != policy.task4_catalog_row_count
            or catalog_evidence.byte_size != policy.task4_catalog_byte_size
            or catalog_evidence.sha256 != policy.task4_catalog_sha256
        ):
            raise SimilarityAuditError("materialized-input catalog evidence drifted")
        _verify_file(
            catalog_path,
            catalog_evidence.byte_size,
            catalog_evidence.sha256,
        )
        fastas = {
            strategy: {
                partition: _fasta_evidence_from(marker["fastas"][strategy][partition])
                for partition in PARTITIONS
            }
            for strategy in STRATEGIES
        }
        for strategy in STRATEGIES:
            for partition in PARTITIONS:
                _verify_file(
                    _fasta_path(workspace, strategy, partition),
                    fastas[strategy][partition].byte_size,
                    fastas[strategy][partition].sha256,
                )
        return MaterializedInputs(
            catalog=catalog_evidence,
            fastas=fastas,
        )

    print("materializing six pinned FASTA inputs...")
    inputs = materialize_strategy_fastas(
        catalog_path=catalog_path,
        manifests=manifests,
        output_directory=fasta_directory,
        policy=policy,
    )
    _write_json_atomic(
        marker_path,
        {
            "schema_version": 1,
            "stage": "materialized_inputs",
            "fingerprint": fingerprint,
            "catalog": asdict(inputs.catalog),
            "fastas": {
                strategy: {
                    partition: asdict(inputs.fastas[strategy][partition])
                    for partition in PARTITIONS
                }
                for strategy in STRATEGIES
            },
        },
    )
    return inputs


def _ensure_target_database(
    *,
    strategy: str,
    training_fasta: Path,
    training_fasta_evidence: FastaEvidence,
    workspace: Path,
    policy: SimilarityAuditPolicy,
    fingerprint: str,
) -> tuple[Path, dict[str, object]]:
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
        marker = _read_json(marker_path)
        _require_marker_identity(marker, fingerprint, "target_database")
        if (
            marker.get("strategy") != strategy
            or marker.get("training_fasta") != asdict(training_fasta_evidence)
            or marker.get("command") != command
        ):
            raise SimilarityAuditError("target database completion evidence drifted")
        _verify_artifact_index(final_directory, marker.get("artifacts"))
        return final_directory / "target", marker

    if incomplete.exists():
        shutil.rmtree(incomplete)
    if final_directory.exists():
        shutil.rmtree(final_directory)
    incomplete.mkdir(parents=True, exist_ok=False)
    runtime = _run_mmseqs_command(
        command,
        workspace=workspace,
        log_path=workspace / "logs" / f"createdb_{strategy}.log",
        policy=policy,
    )
    artifact_paths = tuple(
        sorted(path for path in incomplete.iterdir() if path.is_file())
    )
    if not artifact_paths:
        raise SimilarityAuditError("MMseqs2 createdb produced no database files")
    artifacts = {
        path.name: _file_identity(path)
        for path in artifact_paths
    }
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
    _write_json_atomic(marker_path, marker)
    return final_directory / "target", marker


def _ensure_search_pass(
    *,
    strategy: str,
    partition: str,
    pass_name: str,
    query_fasta: Path,
    query_fasta_evidence: FastaEvidence,
    query_metadata: Mapping[str, SequenceMetadata],
    target_database: Path,
    target_metadata: Mapping[str, SequenceMetadata],
    workspace: Path,
    policy: SimilarityAuditPolicy,
    fingerprint: str,
) -> dict[str, object]:
    pass_directory = workspace / "tracks" / strategy / partition / pass_name
    marker_path = pass_directory / "complete.json"
    if marker_path.exists():
        marker = _read_json(marker_path)
        _validate_completed_pass(
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

    escalation = None
    escalation_path = None
    if differing:
        selected = set(differing)
        escalation_fasta = pass_directory / "escalated_queries.fasta"
        escalation_fasta_evidence = write_fasta_subset(
            query_fasta,
            escalation_fasta,
            selected,
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
    compact_directory = pass_directory / "compact"
    accepted = compact_converged_results(
        pass_name=pass_name,
        comparison_path=comparison["canonical_path"],
        escalation_path=escalation_path,
        convergence=convergence,
        expected_query_ids=query_metadata,
        output_directory=compact_directory,
        resource_guard=lambda: _require_disk_capacity(workspace, policy),
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
    _write_json_atomic(marker_path, marker)
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
        marker = _read_json(marker_path)
        _require_marker_identity(marker, fingerprint, "search_stage")
        if marker.get("cap") != cap or marker.get("command") != command:
            raise SimilarityAuditError("completed search command drifted")
        if marker.get("query_count") != len(query_metadata) or marker.get(
            "query_fasta"
        ) != asdict(query_fasta_evidence):
            raise SimilarityAuditError("completed search query input drifted")
        evidence = marker.get("alignment_evidence")
        if not isinstance(evidence, dict):
            raise SimilarityAuditError("search-stage evidence is malformed")
        parsed = _canonical_evidence_from(evidence)
        if marker.get("raw_retained") is not False or raw_path.exists():
            raise SimilarityAuditError("completed raw-output retention state drifted")
        _verify_file(
            canonical_path,
            parsed.canonical.byte_size,
            parsed.canonical.sha256,
        )
        return {
            "canonical_path": canonical_path,
            "evidence": marker,
        }

    raw_path.unlink(missing_ok=True)
    canonical_path.unlink(missing_ok=True)
    if temp_directory.exists():
        shutil.rmtree(temp_directory)
    runtime = _run_mmseqs_command(
        command,
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
        resource_guard=lambda: _require_disk_capacity(workspace, policy),
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
    _write_json_atomic(marker_path, marker)
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


def _run_mmseqs_command(
    command: Sequence[str],
    *,
    workspace: Path,
    log_path: Path,
    policy: SimilarityAuditPolicy,
) -> str:
    _require_disk_capacity(workspace, policy)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            list(command),
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        next_heartbeat = 60.0
        try:
            while process.poll() is None:
                time.sleep(policy.disk_check_interval_seconds)
                _require_disk_capacity(workspace, policy)
                elapsed = time.perf_counter() - started
                if elapsed >= next_heartbeat:
                    print(
                        f"MMseqs2 still running after {elapsed:.0f} seconds; "
                        f"log: {log_path}",
                        flush=True,
                    )
                    next_heartbeat += 60.0
        except BaseException:
            _terminate_process_group(process)
            raise
        return_code = process.returncode
    if return_code != 0:
        tail = _log_tail(log_path)
        raise SimilarityAuditError(
            f"MMseqs2 exited with status {return_code}. Log tail:\n{tail}"
        )
    return f"{time.perf_counter() - started:.3f}"


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _cleanup_full_alignment_rows(pass_directory: Path) -> None:
    for stage_directory in pass_directory.glob("cap_*"):
        if not stage_directory.is_dir():
            continue
        (stage_directory / "raw.tsv").unlink(missing_ok=True)
        (stage_directory / "canonical.tsv").unlink(missing_ok=True)
        temp_directory = stage_directory / "mmseqs_tmp"
        if temp_directory.exists():
            shutil.rmtree(temp_directory)


def _validate_completed_pass(
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
    _require_marker_identity(marker, fingerprint, "completed_search_pass")
    expected = {
        "strategy": strategy,
        "partition": partition,
        "pass_name": pass_name,
        "query_count": len(expected_query_ids),
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        raise SimilarityAuditError("completed search-pass identity drifted")
    convergence = marker.get("convergence")
    if not isinstance(convergence, dict) or convergence.get("final_differing_queries") != 0:
        raise SimilarityAuditError("completed search pass is not converged")
    escalated_ids = convergence.get("escalated_query_ids")
    if (
        not isinstance(escalated_ids, list)
        or any(not isinstance(value, str) for value in escalated_ids)
        or escalated_ids != sorted(set(escalated_ids))
        or not set(escalated_ids) <= expected_query_ids
    ):
        raise SimilarityAuditError("completed escalation query set is malformed")
    escalated_count = len(escalated_ids)
    expected_convergence = {
        "expected_queries": len(expected_query_ids),
        "converged_at_comparison_cap": len(expected_query_ids) - escalated_count,
        "escalated_queries": escalated_count,
        "converged_at_escalation_cap": escalated_count,
        "final_differing_queries": 0,
    }
    if any(convergence.get(key) != value for key, value in expected_convergence.items()):
        raise SimilarityAuditError("completed convergence counts do not reconcile")
    expected_stage_caps = {str(policy.initial_cap), str(policy.comparison_cap)}
    if escalated_count:
        expected_stage_caps.add(str(policy.escalation_cap))
    stages = marker.get("stages")
    if not isinstance(stages, dict) or set(stages) != expected_stage_caps:
        raise SimilarityAuditError("completed search stages do not reconcile")
    accepted = marker.get("accepted")
    if not isinstance(accepted, dict):
        raise SimilarityAuditError("accepted pass evidence is malformed")
    expected_accepted = {
        "pass_name": pass_name,
        "accepted_at_comparison_cap": len(expected_query_ids) - escalated_count,
        "accepted_at_escalation_cap": escalated_count,
    }
    if any(accepted.get(key) != value for key, value in expected_accepted.items()):
        raise SimilarityAuditError("accepted cap distribution does not reconcile")
    accepted_rows = _strict_int(accepted.get("accepted_rows"), "accepted row count")
    compact = marker_path.parent / "compact"
    _verify_compact_file(compact / "returned_pairs.tsv", accepted.get("returned_pairs"))
    _verify_compact_file(
        compact / "prohibited_pairs.tsv",
        accepted.get("prohibited_pairs"),
    )
    returned = _file_evidence_from(accepted["returned_pairs"])
    prohibited = _file_evidence_from(accepted["prohibited_pairs"])
    if returned.row_count != accepted_rows or prohibited.row_count > accepted_rows:
        raise SimilarityAuditError("accepted pair counts do not reconcile")
    summaries = accepted.get("residual_summaries")
    if pass_name == "residual":
        _verify_compact_file(compact / "residual_summaries.tsv", summaries)
        if _file_evidence_from(summaries).row_count != len(expected_query_ids):
            raise SimilarityAuditError("residual summary query count drifted")
    elif summaries is not None:
        raise SimilarityAuditError("enforcement pass has residual summaries")


def _public_pass_evidence(marker: Mapping[str, object]) -> dict[str, object]:
    convergence = dict(marker["convergence"])
    convergence.pop("escalated_query_ids", None)
    accepted = dict(marker["accepted"])
    return {
        "query_count": marker["query_count"],
        "stages": marker["stages"],
        "convergence": convergence,
        "accepted": accepted,
    }


def _structural_membership_evidence(
    strategy: str,
    manifest: StrategyManifest,
    frozen_report: Mapping[str, object],
) -> dict[str, object]:
    evidence = asdict(manifest.structural_audit)
    evidence["record_retention_percent"] = "100.000000"
    evidence["residue_retention_percent"] = "100.000000"
    evidence["exclusion_reasons"] = {}
    if strategy == "group_aware":
        assignment_units = frozen_report.get("assignment_units")
        if not isinstance(assignment_units, dict):
            raise SimilarityAuditError("Task 6 assignment-unit evidence is malformed")
        evidence["largest_assignment_unit_records"] = assignment_units.get(
            "largest_unit_records"
        )
        evidence["largest_assignment_unit_residues"] = assignment_units.get(
            "largest_unit_residues"
        )
        evidence["largest_assignment_unit_original_groups"] = assignment_units.get(
            "largest_unit_original_groups"
        )
    return evidence


def _overall_similarity(partitions: Mapping[str, object]) -> dict[str, object]:
    held_out = [partitions[name]["similarity"] for name in ("validation", "test")]
    numerator = sum(item["held_out_queries_with_prohibited_match"] for item in held_out)
    denominator = sum(item["held_out_query_count"] for item in held_out)
    attribution = Counter()
    closest_categories = Counter(
        {category: 0 for category in CLOSEST_RESIDUAL_CATEGORIES}
    )
    status_categories = Counter({category: 0 for category in RESIDUAL_CATEGORIES})
    for item in held_out:
        attribution.update(item["prohibited_pair_attribution"])
        closest_categories.update(item["closest_residual_categories"])
        status_categories.update(item["held_out_query_status_categories"])
    return {
        "held_out_queries_with_prohibited_match": numerator,
        "held_out_query_count": denominator,
        "prohibited_query_rate_percent": (
            f"{(Decimal(numerator) * 100 / Decimal(denominator)):.6f}"
        ),
        "unique_prohibited_pairs": sum(item["unique_prohibited_pairs"] for item in held_out),
        "prohibited_pair_attribution": dict(attribution),
        "enforcement_returned_pairs": sum(
            item["enforcement_returned_pairs"] for item in held_out
        ),
        "residual_returned_pairs": sum(
            item["residual_returned_pairs"] for item in held_out
        ),
        "unique_returned_pair_union": sum(
            item["unique_returned_pair_union"] for item in held_out
        ),
        "closest_residual_categories": dict(closest_categories),
        "held_out_query_status_categories": dict(status_categories),
    }


def _build_report(
    *,
    policy: SimilarityAuditPolicy,
    code_revision: str,
    fingerprint: str,
    mmseqs_version: str,
    started_at: datetime,
    completed_at: datetime,
    runtime_seconds: float,
    inputs: MaterializedInputs,
    manifests: Mapping[str, StrategyManifest],
    frozen_reports: Mapping[str, Mapping[str, object]],
    database_reports: Mapping[str, object],
    strategy_reports: Mapping[str, object],
) -> dict[str, object]:
    command_count, command_runtime = _completed_command_runtime(
        database_reports,
        strategy_reports,
    )
    return {
        "schema_version": policy.schema_version,
        "scope": policy.scope,
        "adjustment_id": policy.adjustment_id,
        "diagnostic_only": True,
        "diagnostic_audit_authorized": policy.diagnostic_audit_authorized,
        "diagnostic_audit_completed": True,
        "candidate_status": policy.candidate_status,
        "repair_authorized": policy.repair_authorized,
        "repair_performed": policy.repair_performed,
        "selected_split_authorized": policy.selected_split_authorized,
        "task8_membership_use_authorized": policy.task8_membership_use_authorized,
        "model_use": policy.model_use,
        "post_audit_review_required": policy.post_audit_review_required,
        "network_requests_made": False,
        "code_revision": code_revision,
        "config_sha256": APPROVED_SIMILARITY_AUDIT_CONFIG_SHA256,
        "run_fingerprint": fingerprint,
        "inputs": {
            "task4_catalog": asdict(inputs.catalog),
            "task5_public_manifest": asdict(manifests["random"].public_manifest),
            "task5_local_assignment": asdict(manifests["random"].local_assignment),
            "task5_report_sha256": policy.task5_report_sha256,
            "task6_public_manifest": asdict(manifests["group_aware"].public_manifest),
            "task6_local_assignment": asdict(manifests["group_aware"].local_assignment),
            "task6_report_sha256": policy.task6_report_sha256,
            "task6_repair_state_sha256": policy.task6_repair_state_sha256,
            "source_checksums": frozen_reports["random"]["sources"],
            "materialized_fastas": {
                strategy: {
                    partition: asdict(inputs.fastas[strategy][partition])
                    for partition in PARTITIONS
                }
                for strategy in STRATEGIES
            },
        },
        "procedure": {
            "mmseqs_executable": policy.mmseqs_executable,
            "mmseqs_version": mmseqs_version,
            "threads": policy.threads,
            "format_output": policy.format_output,
            "inclusive_prohibited_boundary": {
                "minimum_identity": policy.prohibited_min_sequence_identity,
                "minimum_query_coverage": policy.prohibited_min_query_coverage,
                "minimum_target_coverage": policy.prohibited_min_target_coverage,
            },
            "prohibited_pair_evidence": policy.prohibited_pair_evidence,
            "staged_caps": [
                policy.initial_cap,
                policy.comparison_cap,
                policy.escalation_cap,
            ],
            "boundary_fixtures_passed": True,
            "target_databases": database_reports,
        },
        "runtime": {
            "final_invocation_started_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "final_invocation_wall_clock_seconds": f"{runtime_seconds:.3f}",
            "completed_mmseqs_command_count": command_count,
            "completed_mmseqs_command_runtime_seconds": command_runtime,
            "hardware": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "logical_cpu_count": os.cpu_count(),
            },
            "workspace_byte_ceiling": policy.workspace_byte_ceiling,
            "free_space_reserve": policy.free_space_reserve,
        },
        "strategies": dict(strategy_reports),
        "limitations": [
            "Only validation-to-training and test-to-training were searched.",
            "Validation-to-test similarity was not measured.",
            "The heuristic search cannot prove the absence of homology.",
            "The failed-balance candidate has shorter held-out proteins on average.",
            "The diagnostic cannot repair, select, or authorize model use of a split.",
        ],
    }


def _completed_command_runtime(
    database_reports: Mapping[str, object],
    strategy_reports: Mapping[str, object],
) -> tuple[int, str]:
    durations = []
    for raw_database in database_reports.values():
        if not isinstance(raw_database, dict):
            raise SimilarityAuditError("database runtime evidence is malformed")
        durations.append(Decimal(str(raw_database.get("runtime_seconds"))))
    for raw_strategy in strategy_reports.values():
        if not isinstance(raw_strategy, dict):
            raise SimilarityAuditError("strategy runtime evidence is malformed")
        partitions = raw_strategy["partitions"]
        for partition in ("validation", "test"):
            for raw_pass in partitions[partition]["passes"].values():
                for raw_stage in raw_pass["stages"].values():
                    durations.append(Decimal(str(raw_stage["runtime_seconds"])))
    if any(not duration.is_finite() or duration < 0 for duration in durations):
        raise SimilarityAuditError("command runtime evidence is invalid")
    return len(durations), f"{sum(durations, Decimal('0')):.3f}"


def _publish_report(rendered, workspace: Path) -> None:
    staging = workspace / "public_report_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    outputs = {
        f"{OUTPUT_STEM}.json": rendered.json_text.encode("utf-8"),
        f"{OUTPUT_STEM}.md": rendered.markdown_text.encode("utf-8"),
        f"{OUTPUT_STEM}.sha256": sha256_sidecar(
            f"{OUTPUT_STEM}.json",
            rendered.json_sha256,
        ).encode("ascii"),
    }
    completed = []
    for filename, content in outputs.items():
        path = staging / filename
        path.write_bytes(content)
        completed.append(
            CompletedPublicArtifact(
                relative_path=f"reports/week_01/{filename}",
                byte_size=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            )
        )
    completion_text = render_completion_index(
        tuple(completed),
        scope=COMPLETION_SCOPE,
    )
    (staging / COMPLETION_FILENAME).write_text(completion_text, encoding="utf-8")

    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    completion_path = REPORT_DIRECTORY / COMPLETION_FILENAME
    completion_path.unlink(missing_ok=True)
    for filename in REPORT_FILENAMES:
        (staging / filename).replace(REPORT_DIRECTORY / filename)
    (staging / COMPLETION_FILENAME).replace(completion_path)
    staging.rmdir()


def _load_and_validate_frozen_reports(
    paths: Mapping[str, Path],
    policy: SimilarityAuditPolicy,
) -> dict[str, dict[str, object]]:
    task5 = _load_pinned_json(paths["task5_report"], policy.task5_report_sha256)
    task6 = _load_pinned_json(paths["task6_report"], policy.task6_report_sha256)
    task5_expected = {
        "scope": "week_01_task_05_random_diagnostic",
        "strategy": "random",
        "stage": "diagnostic",
        "diagnostic_only": True,
        "selected_for_training": False,
        "model_use": "prohibited",
    }
    task6_expected = {
        "scope": "week_01_task_06_group_aware_pre_repair",
        "strategy": "group_aware",
        "stage": "pre_repair",
        "candidate_status": "failed_balance",
        "task6_gates_passed": False,
        "task7_authorized": False,
        "selected_for_training": False,
        "model_use": "prohibited",
    }
    _require_report_fields(task5, task5_expected, "Task 5")
    _require_report_fields(task6, task6_expected, "Task 6")
    if task5.get("sources") != task6.get("sources"):
        raise SimilarityAuditError("Task 5 and Task 6 source evidence differs")
    repair_state = task6.get("repair_state")
    if not isinstance(repair_state, dict) or repair_state.get("sha256") != (
        policy.task6_repair_state_sha256
    ):
        raise SimilarityAuditError("Task 6 repair-state-zero digest drifted")
    return {"random": task5, "group_aware": task6}


def _validate_report_populations(
    reports: Mapping[str, Mapping[str, object]],
    manifests: Mapping[str, StrategyManifest],
    policy: SimilarityAuditPolicy,
) -> dict[str, dict[str, dict[str, object]]]:
    balances = {}
    for strategy in STRATEGIES:
        report = reports[strategy]
        population = report.get("population")
        if not isinstance(population, dict) or (
            population.get("records"),
            population.get("residues"),
        ) != (policy.expected_eligible_records, policy.expected_eligible_residues):
            raise SimilarityAuditError(f"{strategy} report population drifted")
        report_partitions = report.get("partitions")
        if not isinstance(report_partitions, dict) or set(report_partitions) != set(
            PARTITIONS
        ):
            raise SimilarityAuditError(f"{strategy} report partitions drifted")
        strategy_balances = {}
        for partition in PARTITIONS:
            raw = report_partitions[partition]
            manifest_population = manifests[strategy].partitions[partition]
            if not isinstance(raw, dict) or (
                raw.get("records"),
                raw.get("residues"),
                raw.get("unique_groups"),
            ) != (
                manifest_population.records,
                manifest_population.residues,
                manifest_population.unique_groups,
            ):
                raise SimilarityAuditError(
                    f"{strategy} {partition} report and manifest disagree"
                )
            strategy_balances[partition] = {
                key: raw[key]
                for key in (
                    "target_numerator",
                    "target_denominator",
                    "target_share_percent",
                    "records",
                    "residues",
                    "unique_groups",
                    "record_share_percent",
                    "residue_share_percent",
                    "record_deviation_percentage_points",
                    "residue_deviation_percentage_points",
                )
            }
        balances[strategy] = strategy_balances
    return balances


def _policy_paths(policy: SimilarityAuditPolicy) -> dict[str, Path]:
    configured = {
        "workspace": policy.workspace_relative_path,
        "catalog": policy.task4_catalog_relative_path,
        "task5_public": policy.task5_public_manifest_relative_path,
        "task5_local": policy.task5_local_assignment_relative_path,
        "task5_report": policy.task5_report_relative_path,
        "task6_public": policy.task6_public_manifest_relative_path,
        "task6_local": policy.task6_local_assignment_relative_path,
        "task6_report": policy.task6_report_relative_path,
    }
    resolved = {}
    root = PROJECT_ROOT.resolve()
    for name, relative in configured.items():
        path = (PROJECT_ROOT / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise SimilarityAuditError(f"configured {name} path leaves the repository") from error
        resolved[name] = path
    return resolved


def _run_fingerprint(
    *,
    policy: SimilarityAuditPolicy,
    code_revision: str,
    mmseqs_version: str,
) -> str:
    payload = {
        "code_revision": code_revision,
        "config_sha256": APPROVED_SIMILARITY_AUDIT_CONFIG_SHA256,
        "mmseqs_version": mmseqs_version,
        "task4_catalog_sha256": policy.task4_catalog_sha256,
        "task5_public_manifest_sha256": policy.task5_public_manifest_sha256,
        "task5_local_assignment_sha256": policy.task5_local_assignment_sha256,
        "task5_report_sha256": policy.task5_report_sha256,
        "task6_public_manifest_sha256": policy.task6_public_manifest_sha256,
        "task6_local_assignment_sha256": policy.task6_local_assignment_sha256,
        "task6_report_sha256": policy.task6_report_sha256,
        "task6_repair_state_sha256": policy.task6_repair_state_sha256,
    }
    content = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(content).hexdigest()


def _reverify_frozen_run_state(
    *,
    paths: Mapping[str, Path],
    policy: SimilarityAuditPolicy,
    code_revision: str,
    mmseqs_version: str,
) -> None:
    """Repeat every mutable trust check immediately before publication."""

    expected_checksums = {
        "catalog": policy.task4_catalog_sha256,
        "task5_public": policy.task5_public_manifest_sha256,
        "task5_local": policy.task5_local_assignment_sha256,
        "task5_report": policy.task5_report_sha256,
        "task6_public": policy.task6_public_manifest_sha256,
        "task6_local": policy.task6_local_assignment_sha256,
        "task6_report": policy.task6_report_sha256,
    }
    for name, expected_sha256 in expected_checksums.items():
        if _file_identity(paths[name])["sha256"] != expected_sha256:
            raise SimilarityAuditError(
                f"frozen {name} checksum changed during the audit"
            )
    load_similarity_audit_policy(CONFIG_PATH)
    _require_committed_execution_code()
    if _git_output("rev-parse", "HEAD") != code_revision:
        raise SimilarityAuditError("code revision changed during the audit")
    if _verify_mmseqs(policy) != mmseqs_version:
        raise SimilarityAuditError("MMseqs2 version changed during the audit")


def _verify_mmseqs(policy: SimilarityAuditPolicy) -> str:
    executable = Path(policy.mmseqs_executable)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise SimilarityAuditError(
            f"pinned MMseqs2 executable is unavailable: {executable}"
        )
    result = subprocess.run(
        [str(executable), "version"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    version = result.stdout.strip()
    if version != policy.mmseqs_version:
        raise SimilarityAuditError(
            f"MMseqs2 version is {version!r}, expected {policy.mmseqs_version!r}"
        )
    return version


def _require_disk_capacity(workspace: Path, policy: SimilarityAuditPolicy) -> None:
    workspace_size = 0
    for path in workspace.rglob("*"):
        try:
            if path.is_file():
                workspace_size += path.stat().st_size
        except FileNotFoundError:
            continue
    free_bytes = shutil.disk_usage(workspace).free
    if workspace_size > policy.workspace_byte_ceiling:
        raise SimilarityAuditError(
            "Task 7 workspace exceeded its fixed 200 GiB ceiling"
        )
    if free_bytes < policy.free_space_reserve:
        raise SimilarityAuditError(
            "free disk space fell below the fixed 300 GiB reserve"
        )


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SimilarityAuditError("another Task 7 audit is already running") from error
        lock.seek(0)
        lock.truncate()
        lock.write(f"pid={os.getpid()}\n")
        lock.flush()
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _require_committed_execution_code() -> None:
    status = _git_output(
        "status",
        "--porcelain",
        "--",
        "src",
        "scripts",
        "experiments",
        "pyproject.toml",
        "uv.lock",
        ".gitignore",
        ".gitattributes",
    )
    if status:
        raise SimilarityAuditError(
            "execution code has uncommitted changes; review and commit it first"
        )


def _prove_path_is_ignored(path: Path) -> None:
    relative = path.resolve().relative_to(PROJECT_ROOT.resolve())
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative.as_posix()],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise SimilarityAuditError(f"private path is not ignored by Git: {relative}")


def _prove_path_is_public(path: Path) -> None:
    relative = path.resolve().relative_to(PROJECT_ROOT.resolve())
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative.as_posix()],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if result.returncode == 0:
        raise SimilarityAuditError(f"public path is unexpectedly ignored: {relative}")
    if result.returncode != 1:
        raise SimilarityAuditError(f"could not prove public Git status: {relative}")


def _git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_pinned_json(path: Path, expected_sha256: str) -> dict[str, object]:
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise SimilarityAuditError(f"pinned report checksum drifted: {path.name}")
    try:
        parsed = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SimilarityAuditError(f"pinned report is malformed: {path.name}") from error
    if not isinstance(parsed, dict):
        raise SimilarityAuditError(f"pinned report root is not an object: {path.name}")
    return parsed


def _require_report_fields(
    report: Mapping[str, object],
    expected: Mapping[str, object],
    name: str,
) -> None:
    drift = [
        f"{key}: found {report.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if report.get(key) != value
    ]
    if drift:
        raise SimilarityAuditError(f"{name} authority drift: " + "; ".join(drift))


def _fasta_path(workspace: Path, strategy: str, partition: str) -> Path:
    return workspace / "fastas" / f"{strategy}_{partition}.fasta"


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary.write_bytes(content)
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, object]:
    try:
        parsed = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SimilarityAuditError(f"completion marker is malformed: {path}") from error
    if not isinstance(parsed, dict):
        raise SimilarityAuditError(f"completion marker root is not an object: {path}")
    return parsed


def _require_marker_identity(
    marker: Mapping[str, object],
    fingerprint: str,
    stage: str,
) -> None:
    if marker.get("schema_version") != 1 or marker.get("stage") != stage:
        raise SimilarityAuditError(f"{stage} completion marker is malformed")
    if marker.get("fingerprint") != fingerprint:
        raise SimilarityAuditError(
            f"{stage} completion marker belongs to a different frozen run"
        )


def _file_identity(path: Path) -> dict[str, object]:
    hasher = hashlib.sha256()
    byte_size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            hasher.update(chunk)
            byte_size += len(chunk)
    return {"byte_size": byte_size, "sha256": hasher.hexdigest()}


def _verify_file(path: Path, expected_size: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise SimilarityAuditError(f"completed artifact is missing: {path}")
    identity = _file_identity(path)
    if identity != {"byte_size": expected_size, "sha256": expected_sha256}:
        raise SimilarityAuditError(f"completed artifact checksum drifted: {path}")


def _verify_artifact_index(directory: Path, raw_index: object) -> None:
    if not isinstance(raw_index, dict) or not raw_index:
        raise SimilarityAuditError("database artifact index is malformed")
    for filename, evidence in raw_index.items():
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise SimilarityAuditError("database artifact filename is unsafe")
        if not isinstance(evidence, dict):
            raise SimilarityAuditError("database artifact evidence is malformed")
        _verify_file(
            directory / filename,
            _strict_int(evidence.get("byte_size"), "database byte size"),
            _strict_string(evidence.get("sha256"), "database SHA-256"),
        )


def _verify_compact_file(path: Path, raw_evidence: object) -> None:
    if not isinstance(raw_evidence, dict):
        raise SimilarityAuditError("compact artifact evidence is malformed")
    evidence = _file_evidence_from(raw_evidence)
    _verify_file(path, evidence.byte_size, evidence.sha256)


def _file_evidence_from(raw: object) -> FileEvidence:
    if not isinstance(raw, dict):
        raise SimilarityAuditError("file evidence is malformed")
    return FileEvidence(
        row_count=_strict_int(raw.get("row_count"), "row count"),
        byte_size=_strict_int(raw.get("byte_size"), "byte size"),
        sha256=_strict_string(raw.get("sha256"), "SHA-256"),
    )


def _fasta_evidence_from(raw: object) -> FastaEvidence:
    if not isinstance(raw, dict):
        raise SimilarityAuditError("FASTA evidence is malformed")
    return FastaEvidence(
        record_count=_strict_int(raw.get("record_count"), "FASTA record count"),
        residue_count=_strict_int(raw.get("residue_count"), "FASTA residue count"),
        byte_size=_strict_int(raw.get("byte_size"), "FASTA byte size"),
        sha256=_strict_string(raw.get("sha256"), "FASTA SHA-256"),
    )


def _canonical_evidence_from(raw: object) -> CanonicalAlignmentEvidence:
    if not isinstance(raw, dict):
        raise SimilarityAuditError("canonical alignment evidence is malformed")
    return CanonicalAlignmentEvidence(
        raw=_file_evidence_from(raw.get("raw")),
        canonical=_file_evidence_from(raw.get("canonical")),
    )


def _strict_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SimilarityAuditError(f"{name} must be a nonnegative integer")
    return value


def _strict_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SimilarityAuditError(f"{name} must be a nonempty string")
    return value


def _log_tail(path: Path, lines: int = 20) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "<log unavailable>"
    return "\n".join(content[-lines:])


if __name__ == "__main__":
    raise SystemExit(main())
