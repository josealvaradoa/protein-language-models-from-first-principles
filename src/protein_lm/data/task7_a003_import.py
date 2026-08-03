"""Verify the A-003 residual evidence that A-004 may reuse read-only."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from protein_lm.data.similarity_audit_policy import (
    SimilarityAuditError,
    SimilarityAuditPolicy,
    load_similarity_audit_policy,
)
from protein_lm.data.similarity_fastas import FastaEvidence, iter_one_line_fasta
from protein_lm.data.similarity_manifests import PARTITIONS, STRATEGIES
from protein_lm.data.similarity_results import compare_canonical_results
from protein_lm.data.task7_a003_stages import (
    DatabaseImport,
    ImportedStage,
    read_pinned_marker,
    verify_database,
    verify_stage,
)
from protein_lm.data.task7_a004_policy import A004Policy, FIXED_CAPS, resolve_a004_paths
from protein_lm.data.task7_checkpoints import (
    fasta_evidence_from,
    file_identity,
    require_marker_identity,
    verify_file,
)
from protein_lm.data.task7_inputs import run_fingerprint


@dataclass(frozen=True)
class A003Import:
    """Verified inputs and stages available to the A-004 workflow."""

    fingerprint: str
    fastas: Mapping[str, Mapping[str, FastaEvidence]]
    database: DatabaseImport
    stages: tuple[ImportedStage, ...]
    escalated_query_ids: tuple[str, ...]

    @property
    def training_fasta(self) -> FastaEvidence:
        """Return the imported random training FASTA for the reused database."""

        return self.fasta("random", "training")

    @property
    def validation_fasta(self) -> FastaEvidence:
        """Return the imported random validation FASTA for the reused stages."""

        return self.fasta("random", "validation")

    def fasta(self, strategy: str, partition: str) -> FastaEvidence:
        """Return one checksum-verified preserved Task 7 FASTA."""

        try:
            return self.fastas[strategy][partition]
        except KeyError as error:
            raise SimilarityAuditError(
                f"A-003 FASTA is unavailable: {strategy} {partition}"
            ) from error

    def stage(self, cap: int) -> ImportedStage:
        """Return one imported stage by its fixed cap."""

        try:
            return next(stage for stage in self.stages if stage.cap == cap)
        except StopIteration as error:
            raise SimilarityAuditError(f"cap was not imported from A-003: {cap}") from error


def verify_a003_residual_import(
    *,
    project_root: Path,
    policy: A004Policy,
) -> A003Import:
    """Verify, but never copy or modify, the reusable A-003 residual stages."""

    paths = resolve_a004_paths(policy, project_root)
    source_policy = _load_source_policy(paths["source_policy"], policy)
    source_workspace = paths["source_workspace"]
    expected_workspace = (project_root / source_policy.workspace_relative_path).resolve()
    if source_workspace != expected_workspace:
        raise SimilarityAuditError("A-004 source workspace differs from A-003")

    fingerprint = run_fingerprint(
        policy=source_policy,
        code_revision=policy.source_code_revision,
        mmseqs_version=policy.source_mmseqs_version,
    )
    if fingerprint != policy.source_run_fingerprint:
        raise SimilarityAuditError("A-003 run fingerprint cannot be reconstructed")

    fastas_marker, _ = read_pinned_marker(
        source_workspace / "fastas" / "complete.json",
        policy.source_fastas_marker_sha256,
    )
    require_marker_identity(fastas_marker, fingerprint, "materialized_inputs")
    fastas = _verify_materialized_fastas(fastas_marker, source_workspace)
    validation_path = source_workspace / "fastas" / "random_validation.fasta"
    training = fastas["random"]["training"]
    validation = fastas["random"]["validation"]

    database = verify_database(
        source_workspace=source_workspace,
        source_policy=source_policy,
        policy=policy,
        fingerprint=fingerprint,
        training=training,
    )
    stages = tuple(
        verify_stage(
            cap=cap,
            source_workspace=source_workspace,
            source_policy=source_policy,
            policy=policy,
            fingerprint=fingerprint,
            expected_query=validation if cap != policy.staged_escalation_cap else None,
        )
        for cap in FIXED_CAPS
    )
    escalated = _verify_escalation_membership(
        validation_path=validation_path,
        initial=stages[0],
        comparison=stages[1],
        escalation=stages[2],
        source_workspace=source_workspace,
    )
    return A003Import(
        fingerprint=fingerprint,
        fastas=fastas,
        database=database,
        stages=stages,
        escalated_query_ids=escalated,
    )


def _load_source_policy(path: Path, policy: A004Policy) -> SimilarityAuditPolicy:
    if not path.is_file():
        raise SimilarityAuditError(f"A-003 source policy is missing: {path}")
    identity = file_identity(path)
    if identity["sha256"] != policy.source_policy_sha256:
        raise SimilarityAuditError("A-003 source policy checksum drifted")
    source = load_similarity_audit_policy(path)
    if (
        source.adjustment_id != policy.source_adjustment_id
        or source.mmseqs_version != policy.source_mmseqs_version
    ):
        raise SimilarityAuditError("A-003 source authority drifted")
    return source


def _verify_escalation_membership(
    *,
    validation_path: Path,
    initial: ImportedStage,
    comparison: ImportedStage,
    escalation: ImportedStage,
    source_workspace: Path,
) -> tuple[str, ...]:
    validation_ids = tuple(accession for accession, _ in iter_one_line_fasta(validation_path))
    if len(validation_ids) != len(set(validation_ids)):
        raise SimilarityAuditError("A-003 validation FASTA contains duplicate accessions")
    changed = compare_canonical_results(
        initial.canonical_path,
        comparison.canonical_path,
        expected_query_ids=validation_ids,
    )
    escalation_path = (
        source_workspace
        / "tracks"
        / "random"
        / "validation"
        / "residual"
        / "escalated_queries.fasta"
    )
    escalated_ids = tuple(accession for accession, _ in iter_one_line_fasta(escalation_path))
    if len(escalated_ids) != len(set(escalated_ids)):
        raise SimilarityAuditError("A-003 escalation FASTA contains duplicate accessions")
    if frozenset(escalated_ids) != frozenset(changed):
        raise SimilarityAuditError("A-003 escalation membership differs from cap changes")
    if escalation.query_fasta.record_count != len(escalated_ids):
        raise SimilarityAuditError("A-003 escalation query count drifted")
    return tuple(sorted(escalated_ids))


def _verify_materialized_fastas(
    marker: Mapping[str, object],
    source_workspace: Path,
) -> dict[str, dict[str, FastaEvidence]]:
    """Verify all six A-003 FASTAs before exposing any one to A-004."""

    fastas = marker.get("fastas")
    if not isinstance(fastas, dict) or set(fastas) != set(STRATEGIES):
        raise SimilarityAuditError("A-003 materialized FASTA marker is malformed")
    verified: dict[str, dict[str, FastaEvidence]] = {}
    for strategy in STRATEGIES:
        partitions = fastas.get(strategy)
        if not isinstance(partitions, dict) or set(partitions) != set(PARTITIONS):
            raise SimilarityAuditError("A-003 materialized FASTA marker is malformed")
        verified[strategy] = {}
        for partition in PARTITIONS:
            evidence = fasta_evidence_from(partitions[partition])
            path = source_workspace / "fastas" / f"{strategy}_{partition}.fasta"
            verify_file(path, evidence.byte_size, evidence.sha256)
            verified[strategy][partition] = evidence
    return verified
