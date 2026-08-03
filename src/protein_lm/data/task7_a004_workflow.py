"""Top-level read-only A-004 fixed-budget audit orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from protein_lm.data.similarity_alignment import verify_boundary_fixtures
from protein_lm.data.similarity_audit_policy import (
    SimilarityAuditError,
    SimilarityAuditPolicy,
    load_similarity_audit_policy,
)
from protein_lm.data.similarity_fastas import MaterializedInputs
from protein_lm.data.similarity_manifests import STRATEGIES
from protein_lm.data.task7_a003_import import verify_a003_residual_import
from protein_lm.data.task7_a004_aggregation import build_pair_unions, run_tracks
from protein_lm.data.task7_a004_database import (
    A004Database,
    DatabaseRunner,
    ensure_a004_target_database,
)
from protein_lm.data.task7_a004_policy import (
    A004Policy,
    load_a004_policy,
    resolve_a004_paths,
)
from protein_lm.data.task7_a004_receipt import (
    a004_fingerprint,
    frozen_assignment_identities,
    publish_completion_marker,
    publish_receipt,
    require_same_six_fastas,
)
from protein_lm.data.task7_a004_report import publish_a004_report
from protein_lm.data.task7_a004_runtime import hardware_provenance
from protein_lm.data.task7_a004_validation import revalidate_before_completion
from protein_lm.data.task7_checkpoints import file_identity
from protein_lm.data.task7_execution import (
    exclusive_lock,
    git_output,
    prove_path_is_ignored,
    require_committed_execution_code,
    require_disk_capacity,
    verify_mmseqs,
)
from protein_lm.data.task7_fixed_budget_contract import SearchRunner
from protein_lm.data.task7_inputs import (
    ensure_materialized_inputs,
    fasta_path,
    load_and_validate_frozen_reports,
    load_frozen_manifests,
    policy_paths,
    reverify_frozen_run_state,
    validate_report_populations,
)


@dataclass(frozen=True)
class A004Configuration:
    """Byte-pinned policy and non-overlapping workspace paths."""

    policy: A004Policy
    source_policy: SimilarityAuditPolicy
    paths: Mapping[str, Path]


@dataclass(frozen=True)
class A004WorkflowResult:
    """Local ignored completion identities for the A-004 workflow."""

    receipt_path: Path
    completion_path: Path
    fingerprint: str


def validate_a004_configuration(
    *, project_root: Path, config_path: Path
) -> A004Configuration:
    """Validate only byte-pinned configuration and paths, without execution."""

    policy = load_a004_policy(config_path)
    paths = resolve_a004_paths(policy, project_root)
    if file_identity(paths["source_policy"])["sha256"] != policy.source_policy_sha256:
        raise SimilarityAuditError("A-003 source policy checksum drifted")
    source_policy = load_similarity_audit_policy(paths["source_policy"])
    expected = (
        source_policy.adjustment_id == policy.source_adjustment_id
        and source_policy.mmseqs_version == policy.source_mmseqs_version
        and source_policy.workspace_relative_path == policy.source_workspace_relative_path
        and source_policy.comparison_cap == policy.all_query_cap
        and source_policy.escalation_cap == policy.staged_escalation_cap
    )
    if not expected:
        raise SimilarityAuditError("A-004 source-policy authority drifted")
    return A004Configuration(policy=policy, source_policy=source_policy, paths=paths)


def run_a004_fixed_budget_audit(
    *,
    project_root: Path,
    config_path: Path,
    search_runner: SearchRunner | None = None,
    database_runner: DatabaseRunner | None = None,
    hardware: Mapping[str, object] | None = None,
) -> A004WorkflowResult:
    """Execute only the approved A-004 local read-only audit plan."""

    configuration = validate_a004_configuration(
        project_root=project_root, config_path=config_path
    )
    require_committed_execution_code(project_root)
    code_revision = git_output(project_root, "rev-parse", "HEAD")
    mmseqs_version = verify_mmseqs(configuration.source_policy, project_root)
    fingerprint = a004_fingerprint(
        policy=configuration.policy,
        source_policy=configuration.source_policy,
        code_revision=code_revision,
        mmseqs_version=mmseqs_version,
    )
    runtime_hardware = dict(hardware) if hardware is not None else hardware_provenance()
    workspace = configuration.paths["workspace"]
    prove_path_is_ignored(workspace, project_root)
    workspace.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(workspace / "audit.lock"):
        require_disk_capacity(workspace, configuration.source_policy)
        verify_boundary_fixtures()
        source_paths = policy_paths(configuration.source_policy, project_root)
        reports = load_and_validate_frozen_reports(source_paths, configuration.source_policy)
        manifests = load_frozen_manifests(source_paths, configuration.source_policy)
        balances = validate_report_populations(
            reports, manifests, configuration.source_policy
        )
        assignments = frozen_assignment_identities(source_paths)
        imported = verify_a003_residual_import(
            project_root=project_root, policy=configuration.policy
        )
        inputs = ensure_materialized_inputs(
            workspace=workspace,
            catalog_path=source_paths["catalog"],
            manifests=manifests,
            policy=configuration.source_policy,
            fingerprint=fingerprint,
        )
        require_same_six_fastas(inputs, imported)
        databases = _ensure_databases(
            project_root=project_root,
            workspace=workspace,
            policy=configuration.source_policy,
            fingerprint=fingerprint,
            inputs=inputs,
            command_runner=database_runner,
        )
        tracks = run_tracks(
            project_root=project_root,
            workspace=workspace,
            source_workspace=configuration.paths["source_workspace"],
            policy=configuration.source_policy,
            fingerprint=fingerprint,
            manifests=manifests,
            inputs=inputs,
            imported=imported,
            databases=databases,
            search_runner=search_runner,
        )
        unions = build_pair_unions(
            workspace=workspace, tracks=tracks, fingerprint=fingerprint
        )
        reverify_frozen_run_state(
            paths=source_paths,
            policy=configuration.source_policy,
            code_revision=code_revision,
            mmseqs_version=mmseqs_version,
            config_path=configuration.paths["source_policy"],
            project_root=project_root,
        )
        if load_a004_policy(config_path) != configuration.policy:
            raise SimilarityAuditError("A-004 policy changed during the audit")
        if verify_a003_residual_import(project_root=project_root, policy=configuration.policy) != imported:
            raise SimilarityAuditError("A-003 imported evidence changed during A-004")
        assignments_after = frozen_assignment_identities(source_paths)
        if assignments_after != assignments:
            raise SimilarityAuditError("immutable Task 5 or Task 6 assignment changed")
        report = publish_a004_report(
            workspace=workspace,
            fingerprint=fingerprint,
            policy=configuration.policy,
            hardware=runtime_hardware,
            assignment_balances=balances,
            assignments_unchanged=assignments_after == assignments,
            tracks=tracks,
            unions=unions,
        )
        receipt = publish_receipt(
            workspace=workspace,
            fingerprint=fingerprint,
            policy=configuration.policy,
            source_policy_path=configuration.paths["source_policy"],
            code_revision=code_revision,
            mmseqs_version=mmseqs_version,
            hardware=runtime_hardware,
            assignments_before=assignments,
            assignments_after=assignments_after,
            imported=imported,
            databases=databases,
            tracks=tracks,
            unions=unions,
            report=report,
        )
        authorization = revalidate_before_completion(
            fingerprint=fingerprint,
            databases=databases,
            tracks=tracks,
            unions=unions,
            report=report,
            receipt=receipt,
        )
        completion_path = publish_completion_marker(
            workspace=workspace,
            fingerprint=fingerprint,
            receipt=receipt,
            report=report,
            authorization=authorization,
        )
    return A004WorkflowResult(receipt.path, completion_path, fingerprint)


def _ensure_databases(
    *,
    project_root: Path,
    workspace: Path,
    policy: SimilarityAuditPolicy,
    fingerprint: str,
    inputs: MaterializedInputs,
    command_runner: DatabaseRunner | None,
) -> dict[str, A004Database]:
    return {
        strategy: ensure_a004_target_database(
            strategy=strategy,
            training_fasta=fasta_path(workspace, strategy, "training"),
            training_fasta_evidence=inputs.fastas[strategy]["training"],
            project_root=project_root,
            workspace=workspace,
            policy=policy,
            fingerprint=fingerprint,
            command_runner=command_runner,
        )
        for strategy in STRATEGIES
    }
