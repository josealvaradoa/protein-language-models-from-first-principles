"""A-004 orchestration of retained fixed-budget MMseqs2 search stages."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from protein_lm.data.similarity_audit_models import SequenceMetadata
from protein_lm.data.similarity_audit_policy import (
    SimilarityAuditError,
    SimilarityAuditPolicy,
)
from protein_lm.data.similarity_fastas import FastaEvidence
from protein_lm.data.similarity_results import compare_canonical_results
from protein_lm.data.task7_checkpoints import read_json, require_marker_identity, write_json_atomic
from protein_lm.data.task7_fixed_budget_contract import (
    FixedBudgetPass,
    FixedBudgetStage,
    SearchRunner,
    require_fixed_policy_caps,
    verify_query_fasta,
)
from protein_lm.data.task7_fixed_budget_stages import (
    ensure_escalation_fasta,
    ensure_search_stage,
    pass_marker,
)

__all__ = [
    "FixedBudgetPass",
    "FixedBudgetStage",
    "SearchRunner",
    "ensure_fixed_budget_pass",
    "verify_query_fasta",
]


def ensure_fixed_budget_pass(
    *,
    strategy: str,
    partition: str,
    pass_name: str,
    query_fasta: Path,
    query_fasta_evidence: FastaEvidence,
    query_metadata: Mapping[str, SequenceMetadata],
    target_database: Path,
    target_database_identity: Mapping[str, object],
    target_metadata: Mapping[str, SequenceMetadata],
    project_root: Path,
    workspace: Path,
    policy: SimilarityAuditPolicy,
    fingerprint: str,
    command_runner: SearchRunner | None = None,
) -> FixedBudgetPass:
    """Run 1k and 10k for all queries, then 100k only for changed rows.

    A-004 records any 10k-to-100k difference as cap sensitivity. It never
    requires convergence and retains every completed canonical TSV.
    """

    require_fixed_policy_caps(policy)
    all_query_ids = verify_query_fasta(
        query_fasta, query_fasta_evidence, query_metadata
    )
    pass_directory = workspace / "tracks" / strategy / partition / pass_name
    pass_directory.mkdir(parents=True, exist_ok=True)
    common_kwargs = {
        "strategy": strategy,
        "partition": partition,
        "pass_name": pass_name,
        "target_database": target_database,
        "target_database_identity": target_database_identity,
        "target_metadata": target_metadata,
        "pass_directory": pass_directory,
        "project_root": project_root,
        "workspace": workspace,
        "policy": policy,
        "fingerprint": fingerprint,
        "command_runner": command_runner,
    }
    first = ensure_search_stage(
        cap=policy.initial_cap,
        query_fasta=query_fasta,
        query_fasta_evidence=query_fasta_evidence,
        query_ids=all_query_ids,
        query_metadata=query_metadata,
        **common_kwargs,
    )
    comparison = ensure_search_stage(
        cap=policy.comparison_cap,
        query_fasta=query_fasta,
        query_fasta_evidence=query_fasta_evidence,
        query_ids=all_query_ids,
        query_metadata=query_metadata,
        **common_kwargs,
    )
    changed = compare_canonical_results(
        first.canonical_path,
        comparison.canonical_path,
        expected_query_ids=all_query_ids,
    )
    escalation_fasta: FastaEvidence | None = None
    escalation_marker: Path | None = None
    stages = [first, comparison]
    if changed:
        escalation_path, escalation_fasta, escalation_marker = ensure_escalation_fasta(
            pass_directory=pass_directory,
            source_fasta=query_fasta,
            source_evidence=query_fasta_evidence,
            source_query_ids=all_query_ids,
            changed_query_ids=changed,
            fingerprint=fingerprint,
        )
        stages.append(
            ensure_search_stage(
                cap=policy.escalation_cap,
                query_fasta=escalation_path,
                query_fasta_evidence=escalation_fasta,
                query_ids=changed,
                query_metadata={key: query_metadata[key] for key in changed},
                **common_kwargs,
            )
        )
    elif (pass_directory / "escalated_queries.fasta").exists() or (
        pass_directory / "escalated_queries.complete.json"
    ).exists():
        raise SimilarityAuditError("A-004 has an unexpected escalation FASTA")
    expected_cap_directories = {f"cap_{stage.cap}" for stage in stages}
    actual_cap_outputs = {
        path.name for path in pass_directory.iterdir() if path.name.startswith("cap_")
    }
    if actual_cap_outputs != expected_cap_directories:
        raise SimilarityAuditError("A-004 fixed-budget cap inventory drifted")
    marker_path = pass_directory / "complete.json"
    marker = pass_marker(
        fingerprint=fingerprint,
        strategy=strategy,
        partition=partition,
        pass_name=pass_name,
        query_fasta=query_fasta_evidence,
        query_ids=all_query_ids,
        target_database=target_database,
        target_database_identity=target_database_identity,
        changed_query_ids=changed,
        stages=tuple(stages),
        escalation_fasta=escalation_fasta,
        escalation_marker=escalation_marker,
    )
    if marker_path.exists():
        existing = read_json(marker_path)
        require_marker_identity(existing, fingerprint, "a004_fixed_budget_pass")
        if existing != marker:
            raise SimilarityAuditError("A-004 fixed-budget pass identity drifted")
    else:
        write_json_atomic(marker_path, marker)
    return FixedBudgetPass(
        strategy=strategy,
        partition=partition,
        pass_name=pass_name,
        all_query_ids=all_query_ids,
        changed_query_ids=changed,
        escalation_fasta=escalation_fasta,
        stages=tuple(stages),
        marker_path=marker_path,
    )
