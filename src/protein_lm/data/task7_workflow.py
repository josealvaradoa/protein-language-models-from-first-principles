"""Readable top-level workflow for the Week 1 Task 7 diagnostic audit."""

from __future__ import annotations

import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from protein_lm.data.similarity_alignment import verify_boundary_fixtures
from protein_lm.data.similarity_audit_models import SequenceMetadata
from protein_lm.data.similarity_audit_policy import (
    SimilarityAuditError,
    SimilarityAuditPolicy,
    load_similarity_audit_policy,
)
from protein_lm.data.similarity_evidence import aggregate_partition_evidence
from protein_lm.data.similarity_fastas import MaterializedInputs
from protein_lm.data.similarity_manifests import (
    STRATEGIES,
    StrategyManifest,
    metadata_by_partition,
)
from protein_lm.data.task7_execution import (
    exclusive_lock,
    git_output,
    prove_path_is_ignored,
    prove_path_is_public,
    require_committed_execution_code,
    require_disk_capacity,
    verify_mmseqs,
)
from protein_lm.data.task7_inputs import (
    ensure_materialized_inputs,
    fasta_path,
    load_and_validate_frozen_reports,
    load_frozen_manifests,
    policy_paths,
    reverify_frozen_run_state,
    run_fingerprint,
    validate_report_populations,
)
from protein_lm.data.task7_report import RenderedTask7Report, render_task7_report
from protein_lm.data.task7_report_output import (
    COMPLETION_FILENAME,
    REPORT_FILENAMES,
    build_report,
    overall_similarity,
    public_pass_evidence,
    publish_report,
    structural_membership_evidence,
)
from protein_lm.data.task7_search import (
    ensure_search_pass,
    ensure_target_database,
)


def run_diagnostic_similarity_audit(
    *,
    project_root: Path,
    config_path: Path,
    report_directory: Path,
) -> RenderedTask7Report:
    """Run only the A-003-authorized diagnostic and publish its aggregate report."""

    started_at = datetime.now(UTC)
    started = time.perf_counter()
    policy = load_similarity_audit_policy(config_path)
    require_committed_execution_code(project_root)
    code_revision = git_output(project_root, "rev-parse", "HEAD")
    mmseqs_version = verify_mmseqs(policy, project_root)
    paths = policy_paths(policy, project_root)
    workspace = paths["workspace"]

    prove_path_is_ignored(workspace, project_root)
    for output_path in (
        *(report_directory / filename for filename in REPORT_FILENAMES),
        report_directory / COMPLETION_FILENAME,
    ):
        prove_path_is_public(output_path, project_root)
    if (report_directory / COMPLETION_FILENAME).exists():
        raise SimilarityAuditError(
            "a completed Task 7 public report already exists; it will not be overwritten"
        )

    workspace.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(workspace / "audit.lock"):
        require_disk_capacity(workspace, policy)
        verify_boundary_fixtures()
        reports = load_and_validate_frozen_reports(paths, policy)
        manifests = load_frozen_manifests(paths, policy)
        balances = validate_report_populations(reports, manifests, policy)
        fingerprint = run_fingerprint(
            policy=policy,
            code_revision=code_revision,
            mmseqs_version=mmseqs_version,
        )
        inputs = ensure_materialized_inputs(
            workspace=workspace,
            catalog_path=paths["catalog"],
            manifests=manifests,
            policy=policy,
            fingerprint=fingerprint,
        )
        strategy_reports, database_reports = _audit_strategies(
            project_root=project_root,
            workspace=workspace,
            policy=policy,
            fingerprint=fingerprint,
            manifests=manifests,
            reports=reports,
            balances=balances,
            inputs=inputs,
        )

        reverify_frozen_run_state(
            paths=paths,
            policy=policy,
            code_revision=code_revision,
            mmseqs_version=mmseqs_version,
            config_path=config_path,
            project_root=project_root,
        )
        completed_at = datetime.now(UTC)
        report = build_report(
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
        publish_report(rendered, workspace, report_directory)
    return rendered


def _audit_strategies(
    *,
    project_root: Path,
    workspace: Path,
    policy: SimilarityAuditPolicy,
    fingerprint: str,
    manifests: Mapping[str, StrategyManifest],
    reports: Mapping[str, Mapping[str, object]],
    balances: Mapping[str, Mapping[str, object]],
    inputs: MaterializedInputs,
) -> tuple[dict[str, object], dict[str, object]]:
    strategy_reports: dict[str, object] = {}
    database_reports: dict[str, object] = {}
    for strategy in STRATEGIES:
        strategy_report, database_report = _audit_strategy(
            strategy=strategy,
            project_root=project_root,
            workspace=workspace,
            policy=policy,
            fingerprint=fingerprint,
            manifest=manifests[strategy],
            frozen_report=reports[strategy],
            balances=balances[strategy],
            inputs=inputs,
        )
        strategy_reports[strategy] = strategy_report
        database_reports[strategy] = database_report
    return strategy_reports, database_reports


def _audit_strategy(
    *,
    strategy: str,
    project_root: Path,
    workspace: Path,
    policy: SimilarityAuditPolicy,
    fingerprint: str,
    manifest: StrategyManifest,
    frozen_report: Mapping[str, object],
    balances: Mapping[str, object],
    inputs: MaterializedInputs,
) -> tuple[dict[str, object], dict[str, object]]:
    print(f"preparing {strategy} training database...")
    training_fasta = fasta_path(workspace, strategy, "training")
    database_prefix, database_report = ensure_target_database(
        strategy=strategy,
        training_fasta=training_fasta,
        training_fasta_evidence=inputs.fastas[strategy]["training"],
        project_root=project_root,
        workspace=workspace,
        policy=policy,
        fingerprint=fingerprint,
    )
    training_metadata = metadata_by_partition(manifest, "training")
    partitions: dict[str, object] = {
        "training": {"balance": balances["training"]}
    }
    for partition in ("validation", "test"):
        partitions[partition] = _audit_partition(
            strategy=strategy,
            partition=partition,
            project_root=project_root,
            workspace=workspace,
            policy=policy,
            fingerprint=fingerprint,
            manifest=manifest,
            balance=balances[partition],
            inputs=inputs,
            target_database=database_prefix,
            target_metadata=training_metadata,
        )
    return (
        {
            "stage": manifest.stage,
            "structural_membership": structural_membership_evidence(
                strategy,
                manifest,
                frozen_report,
            ),
            "partitions": partitions,
            "overall": overall_similarity(partitions),
        },
        database_report,
    )


def _audit_partition(
    *,
    strategy: str,
    partition: str,
    project_root: Path,
    workspace: Path,
    policy: SimilarityAuditPolicy,
    fingerprint: str,
    manifest: StrategyManifest,
    balance: object,
    inputs: MaterializedInputs,
    target_database: Path,
    target_metadata: Mapping[str, SequenceMetadata],
) -> dict[str, object]:
    print(f"auditing {strategy} {partition} against training...")
    query_metadata = metadata_by_partition(manifest, partition)
    query_fasta = fasta_path(workspace, strategy, partition)
    pass_reports = {}
    for pass_name in ("enforcement", "residual"):
        pass_reports[pass_name] = ensure_search_pass(
            strategy=strategy,
            partition=partition,
            pass_name=pass_name,
            query_fasta=query_fasta,
            query_fasta_evidence=inputs.fastas[strategy][partition],
            query_metadata=query_metadata,
            target_database=target_database,
            target_metadata=target_metadata,
            project_root=project_root,
            workspace=workspace,
            policy=policy,
            fingerprint=fingerprint,
        )
    partition_directory = workspace / "tracks" / strategy / partition
    similarity = aggregate_partition_evidence(
        expected_query_ids=query_metadata,
        query_metadata=query_metadata,
        target_metadata=target_metadata,
        enforcement_directory=partition_directory / "enforcement" / "compact",
        residual_directory=partition_directory / "residual" / "compact",
    )
    return {
        "balance": balance,
        "similarity": similarity,
        "passes": {
            pass_name: public_pass_evidence(pass_reports[pass_name])
            for pass_name in ("enforcement", "residual")
        },
    }
