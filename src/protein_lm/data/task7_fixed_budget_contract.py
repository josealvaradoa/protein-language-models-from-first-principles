"""Shared immutable records and input checks for A-004 fixed-budget stages."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from protein_lm.data.similarity_audit_models import FileEvidence
from protein_lm.data.similarity_audit_policy import (
    SimilarityAuditError,
    SimilarityAuditPolicy,
)
from protein_lm.data.similarity_fastas import FastaEvidence, iter_one_line_fasta
from protein_lm.data.task7_checkpoints import verify_file

FIXED_CAPS = (1_000, 10_000, 100_000)
SearchRunner = Callable[[Sequence[str], Path, Path, Path, SimilarityAuditPolicy], str]


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
class FixedBudgetPass:
    """All completed stages for one immutable strategy, partition, and pass."""

    strategy: str
    partition: str
    pass_name: str
    all_query_ids: tuple[str, ...]
    changed_query_ids: tuple[str, ...]
    escalation_fasta: FastaEvidence | None
    stages: tuple[FixedBudgetStage, ...]
    marker_path: Path

    def stage(self, cap: int) -> FixedBudgetStage:
        """Return a completed stage by its frozen cap."""

        try:
            return next(stage for stage in self.stages if stage.cap == cap)
        except StopIteration as error:
            raise SimilarityAuditError(f"fixed-budget cap is unavailable: {cap}") from error


def query_ids_sha256(query_ids: Iterable[str]) -> str:
    """Return a deterministic membership identity for a query universe."""

    ordered = tuple(sorted(query_ids))
    if not ordered or len(set(ordered)) != len(ordered):
        raise SimilarityAuditError("query identifiers must be nonempty and unique")
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
        raise SimilarityAuditError("A-004 query FASTA evidence does not reconcile")
    observed: set[str] = set()
    for accession, _ in iter_one_line_fasta(path):
        if accession in observed or accession not in expected:
            raise SimilarityAuditError("A-004 query FASTA universe drifted")
        observed.add(accession)
    if observed != expected:
        raise SimilarityAuditError("A-004 query FASTA universe drifted")
    return tuple(sorted(observed))


def require_fixed_policy_caps(policy: SimilarityAuditPolicy) -> None:
    """Require the frozen 1k, 10k, and staged 100k candidate budget."""

    if (policy.initial_cap, policy.comparison_cap, policy.escalation_cap) != FIXED_CAPS:
        raise SimilarityAuditError("A-004 policy must use the frozen fixed-budget caps")
