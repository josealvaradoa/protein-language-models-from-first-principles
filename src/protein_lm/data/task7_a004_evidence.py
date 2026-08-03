"""Resume-safe A-004 summaries and prohibited-pair union evidence."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from protein_lm.data.similarity_audit_models import FileEvidence
from protein_lm.data.similarity_audit_policy import SimilarityAuditError
from protein_lm.data.similarity_fastas import FastaEvidence
from protein_lm.data.task7_checkpoints import (
    file_evidence_from,
    file_identity,
    read_json,
    require_marker_identity,
    verify_file,
    write_json_atomic,
)
from protein_lm.data.task7_fixed_budget import CapEvidence, summarize_cap
from protein_lm.data.task7_fixed_budget_contract import (
    query_ids_sha256,
    verify_query_fasta,
)
from protein_lm.data.task7_pair_union import PairUnionEvidence, union_prohibited_pairs


@dataclass(frozen=True)
class CapSummary:
    """A local compact summary tied to a retained canonical TSV."""

    source_label: str
    directory: Path
    evidence: CapEvidence
    marker_identity: Mapping[str, object]


@dataclass(frozen=True)
class StoredPairUnion:
    """One resume-validated aggregate union of prohibited pairs."""

    label: str
    directory: Path
    evidence: PairUnionEvidence
    marker_identity: Mapping[str, object]


def ensure_cap_summary(
    *,
    source_label: str,
    cap: int,
    canonical_path: Path,
    canonical_evidence: FileEvidence,
    query_fasta: Path,
    query_fasta_evidence: FastaEvidence,
    expected_query_ids: Iterable[str],
    output_directory: Path,
    fingerprint: str,
) -> CapSummary:
    """Summarize one cap once, or verify its local compact evidence on resume."""

    query_ids = tuple(sorted(expected_query_ids))
    if not source_label:
        raise SimilarityAuditError("A-004 cap-summary source label is required")
    verify_file(canonical_path, canonical_evidence.byte_size, canonical_evidence.sha256)
    verify_query_fasta(
        query_fasta,
        query_fasta_evidence,
        {identifier: None for identifier in query_ids},
    )
    marker_path = output_directory / "complete.json"
    expected = {
        "schema_version": 1,
        "stage": "a004_cap_summary",
        "fingerprint": fingerprint,
        "source_label": source_label,
        "cap": cap,
        "canonical_path": str(canonical_path),
        "canonical": asdict(canonical_evidence),
        "query_fasta": asdict(query_fasta_evidence),
        "query_ids_sha256": query_ids_sha256(query_ids),
    }
    if marker_path.exists():
        return verify_cap_summary(
            source_label=source_label,
            cap=cap,
            canonical_path=canonical_path,
            canonical_evidence=canonical_evidence,
            query_fasta=query_fasta,
            query_fasta_evidence=query_fasta_evidence,
            expected_query_ids=query_ids,
            output_directory=output_directory,
            fingerprint=fingerprint,
        )
    if output_directory.exists():
        raise SimilarityAuditError("A-004 cap-summary output lacks its completion marker")
    evidence = summarize_cap(
        cap=cap,
        canonical_path=canonical_path,
        expected_query_ids=query_ids,
        output_directory=output_directory,
    )
    write_json_atomic(marker_path, {**expected, "evidence": asdict(evidence)})
    return CapSummary(
        source_label=source_label,
        directory=output_directory,
        evidence=evidence,
        marker_identity=file_identity(marker_path),
    )


def verify_cap_summary(
    *,
    source_label: str,
    cap: int,
    canonical_path: Path,
    canonical_evidence: FileEvidence,
    query_fasta: Path,
    query_fasta_evidence: FastaEvidence,
    expected_query_ids: Iterable[str],
    output_directory: Path,
    fingerprint: str,
) -> CapSummary:
    """Re-read one cap summary without creating or changing evidence."""

    query_ids = tuple(sorted(expected_query_ids))
    verify_file(canonical_path, canonical_evidence.byte_size, canonical_evidence.sha256)
    verify_query_fasta(
        query_fasta,
        query_fasta_evidence,
        {identifier: None for identifier in query_ids},
    )
    marker_path = output_directory / "complete.json"
    marker = read_json(marker_path)
    require_marker_identity(marker, fingerprint, "a004_cap_summary")
    evidence = _cap_evidence_from(marker.get("evidence"))
    expected = {
        "schema_version": 1,
        "stage": "a004_cap_summary",
        "fingerprint": fingerprint,
        "source_label": source_label,
        "cap": cap,
        "canonical_path": str(canonical_path),
        "canonical": asdict(canonical_evidence),
        "query_fasta": asdict(query_fasta_evidence),
        "query_ids_sha256": query_ids_sha256(query_ids),
        "evidence": asdict(evidence),
    }
    if evidence.cap != cap or marker != expected:
        raise SimilarityAuditError("A-004 cap-summary identity drifted")
    _verify_cap_evidence(output_directory, evidence, len(query_ids))
    return CapSummary(
        source_label=source_label,
        directory=output_directory,
        evidence=evidence,
        marker_identity=file_identity(marker_path),
    )


def ensure_pair_union(
    *,
    label: str,
    source_paths: Mapping[str, Path],
    output_directory: Path,
    fingerprint: str,
) -> StoredPairUnion:
    """Build one pair union once, retaining and validating it on resume."""

    if not label:
        raise SimilarityAuditError("A-004 pair-union label is required")
    source_identities = {
        source_label: file_identity(path)
        for source_label, path in sorted(source_paths.items())
    }
    marker_path = output_directory / "complete.json"
    expected = {
        "schema_version": 1,
        "stage": "a004_pair_union",
        "fingerprint": fingerprint,
        "label": label,
        "sources": source_identities,
    }
    if marker_path.exists():
        return verify_pair_union(
            label=label,
            source_paths=source_paths,
            output_directory=output_directory,
            fingerprint=fingerprint,
        )
    if output_directory.exists():
        raise SimilarityAuditError("A-004 pair-union output lacks its completion marker")
    evidence = union_prohibited_pairs(
        source_paths=source_paths,
        output_directory=output_directory,
    )
    write_json_atomic(marker_path, {**expected, "evidence": _pair_union_payload(evidence)})
    return StoredPairUnion(
        label=label,
        directory=output_directory,
        evidence=evidence,
        marker_identity=file_identity(marker_path),
    )


def verify_pair_union(
    *,
    label: str,
    source_paths: Mapping[str, Path],
    output_directory: Path,
    fingerprint: str,
) -> StoredPairUnion:
    """Re-read one pair union and all source identities without writing."""

    source_identities = {
        source_label: file_identity(path)
        for source_label, path in sorted(source_paths.items())
    }
    marker_path = output_directory / "complete.json"
    marker = read_json(marker_path)
    require_marker_identity(marker, fingerprint, "a004_pair_union")
    evidence = _pair_union_evidence_from(marker.get("evidence"))
    expected = {
        "schema_version": 1,
        "stage": "a004_pair_union",
        "fingerprint": fingerprint,
        "label": label,
        "sources": source_identities,
        "evidence": _pair_union_payload(evidence),
    }
    if marker != expected:
        raise SimilarityAuditError("A-004 pair-union identity drifted")
    verify_file(
        output_directory / "prohibited_pairs.tsv",
        evidence.prohibited_pair_file.byte_size,
        evidence.prohibited_pair_file.sha256,
    )
    if evidence.prohibited_pair_file.row_count != evidence.unique_pairs:
        raise SimilarityAuditError("A-004 pair-union counts do not reconcile")
    return StoredPairUnion(
        label=label,
        directory=output_directory,
        evidence=evidence,
        marker_identity=file_identity(marker_path),
    )


def _verify_cap_evidence(
    directory: Path,
    evidence: CapEvidence,
    expected_queries: int,
) -> None:
    if evidence.query_count != expected_queries or evidence.prohibited_queries > expected_queries:
        raise SimilarityAuditError("A-004 cap-summary counts do not reconcile")
    if evidence.prohibited_pairs > evidence.returned_rows:
        raise SimilarityAuditError("A-004 cap-summary pair counts do not reconcile")
    if sum(evidence.closest_categories.values()) != expected_queries:
        raise SimilarityAuditError("A-004 cap-summary category counts do not reconcile")
    verify_file(
        directory / "prohibited_pairs.tsv",
        evidence.prohibited_pair_file.byte_size,
        evidence.prohibited_pair_file.sha256,
    )
    verify_file(
        directory / "query_summaries.tsv",
        evidence.query_summary_file.byte_size,
        evidence.query_summary_file.sha256,
    )
    if evidence.query_summary_file.row_count != expected_queries:
        raise SimilarityAuditError("A-004 cap-summary query evidence drifted")
    if evidence.prohibited_pair_file.row_count != evidence.prohibited_pairs:
        raise SimilarityAuditError("A-004 cap-summary pair evidence drifted")


def _cap_evidence_from(raw: object) -> CapEvidence:
    if not isinstance(raw, dict):
        raise SimilarityAuditError("A-004 cap-summary evidence is malformed")
    integer_names = ("cap", "query_count", "returned_rows", "prohibited_pairs", "prohibited_queries")
    values = {name: _nonnegative_int(raw.get(name), name) for name in integer_names}
    categories = raw.get("closest_categories")
    if not isinstance(categories, dict) or any(
        not isinstance(name, str) or _nonnegative_int(count, "closest category") != count
        for name, count in categories.items()
    ):
        raise SimilarityAuditError("A-004 closest-category evidence is malformed")
    return CapEvidence(
        **values,
        closest_categories=categories,
        prohibited_pair_file=file_evidence_from(raw.get("prohibited_pair_file")),
        query_summary_file=file_evidence_from(raw.get("query_summary_file")),
    )


def _pair_union_evidence_from(raw: object) -> PairUnionEvidence:
    if not isinstance(raw, dict):
        raise SimilarityAuditError("A-004 pair-union evidence is malformed")
    labels = raw.get("source_labels")
    if not isinstance(labels, list) or not labels or any(not isinstance(label, str) for label in labels):
        raise SimilarityAuditError("A-004 pair-union labels are malformed")
    return PairUnionEvidence(
        source_labels=tuple(labels),
        unique_pairs=_nonnegative_int(raw.get("unique_pairs"), "unique pairs"),
        unique_queries=_nonnegative_int(raw.get("unique_queries"), "unique queries"),
        prohibited_pair_file=file_evidence_from(raw.get("prohibited_pair_file")),
    )


def _pair_union_payload(evidence: PairUnionEvidence) -> dict[str, object]:
    payload = asdict(evidence)
    payload["source_labels"] = list(evidence.source_labels)
    return payload


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SimilarityAuditError(f"A-004 {name} must be a nonnegative integer")
    return value
