"""Pure builders for the MMseqs2 commands frozen by Task 7."""

from __future__ import annotations

from pathlib import Path

from protein_lm.data.similarity_audit_policy import (
    SimilarityAuditError,
    SimilarityAuditPolicy,
)


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

    if pass_name == "enforcement":
        min_identity = policy.enforcement_min_sequence_identity
        coverage = policy.enforcement_coverage
        coverage_mode = policy.enforcement_coverage_mode
    elif pass_name == "residual":
        min_identity = policy.residual_min_sequence_identity
        coverage = policy.residual_coverage
        coverage_mode = policy.residual_coverage_mode
    else:
        raise SimilarityAuditError(f"unknown search pass: {pass_name}")

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
