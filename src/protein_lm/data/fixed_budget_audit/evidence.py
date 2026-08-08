"""Cap summaries, pair unions, comparisons, and resume-safe stored evidence."""

from __future__ import annotations

import heapq
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from protein_lm.data.artifacts import (
    EvidenceWriter,
    file_evidence_from,
    file_identity,
    read_json,
    require_marker_identity,
    verify_file,
    write_json_atomic,
)
from protein_lm.data.fixed_budget_audit.config import CandidateCap
from protein_lm.data.fixed_budget_audit.errors import (
    AuditConfigurationError,
    AuditExecutionError,
    AuditValidationError,
)
from protein_lm.data.fixed_budget_audit.search import (
    query_ids_sha256,
    verify_query_fasta,
)
from protein_lm.data.similarity_alignment import (
    CLOSEST_RESIDUAL_CATEGORIES,
    closest_residual_key,
    decode_strict_tsv_line,
    residual_category,
    violates_prohibited_boundary,
)
from protein_lm.data.similarity_audit_models import FileEvidence
from protein_lm.data.similarity_fastas import FastaEvidence
from protein_lm.data.similarity_results import (
    compare_canonical_results,
    iter_converged_query_rows,
)

_ALLOWED_CAPS = frozenset(cap.value for cap in CandidateCap)
_ALLOWED_CAP_TRANSITIONS = frozenset(
    (
        (CandidateCap.INITIAL.value, CandidateCap.COMPARISON.value),
        (CandidateCap.COMPARISON.value, CandidateCap.ESCALATION.value),
    )
)


# Evidence records


@dataclass(frozen=True)
class CapEvidence:
    """Compact evidence from one completed search cap."""

    cap: int
    query_count: int
    returned_rows: int
    prohibited_pairs: int
    prohibited_queries: int
    closest_categories: Mapping[str, int]
    prohibited_pair_file: FileEvidence
    query_summary_file: FileEvidence


@dataclass(frozen=True)
class CapComparison:
    """Decision-relevant changes between two completed caps."""

    baseline_cap: int
    comparison_cap: int
    compared_queries: int
    complete_row_change_query_ids: tuple[str, ...]
    complete_row_changes: int
    newly_prohibited_queries: int
    no_longer_prohibited_queries: int
    closest_category_changes: int


@dataclass(frozen=True)
class PairUnionEvidence:
    """Identity and aggregate counts for one sorted pair union."""

    source_labels: tuple[str, ...]
    unique_pairs: int
    unique_queries: int
    prohibited_pair_file: FileEvidence


@dataclass(frozen=True)
class PairUnionComparison:
    """Additional detected evidence in the staged union."""

    common_pairs: int
    staged_pairs: int
    additional_pairs: int
    common_queries: int
    staged_queries: int
    newly_prohibited_queries: int


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


# Cap summaries and comparisons


def summarize_cap(
    *,
    cap: int,
    canonical_path: Path,
    expected_query_ids: Iterable[str],
    output_directory: Path,
) -> CapEvidence:
    """Write compact pair and per-query evidence for one search cap."""

    _require_cap(cap)
    expected = _validated_query_ids(expected_query_ids)

    staging = output_directory.with_name(f".{output_directory.name}.incomplete")
    if output_directory.exists() or staging.exists():
        raise AuditExecutionError("cap evidence requires a fresh output directory")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    pair_writer = EvidenceWriter(staging / "prohibited_pairs.tsv")
    summary_writer = EvidenceWriter(staging / "query_summaries.tsv")
    closest = Counter({category: 0 for category in CLOSEST_RESIDUAL_CATEGORIES})
    returned_rows = 0
    prohibited_pairs = 0
    prohibited_queries = 0

    try:
        for query, rows in iter_converged_query_rows(
            expected,
            comparison_path=canonical_path,
            escalation_path=None,
            escalated_queries=frozenset(),
        ):
            if len(rows) > cap:
                raise AuditValidationError("query rows exceed the frozen search cap")
            query_prohibited = 0
            for row in rows:
                returned_rows += 1
                if violates_prohibited_boundary(row):
                    query_prohibited += 1
                    prohibited_pairs += 1
                    pair_writer.write(f"{query}\t{row.target}\n".encode("ascii"))

            closest_category = residual_category(
                min(rows, key=closest_residual_key, default=None)
            )
            closest[closest_category] += 1
            is_prohibited = int(query_prohibited > 0)
            prohibited_queries += is_prohibited
            summary_writer.write(
                (
                    f"{query}\t{is_prohibited}\t{closest_category}\t"
                    f"{len(rows)}\t{query_prohibited}\n"
                ).encode("ascii")
            )

        pair_evidence = pair_writer.finish()
        summary_evidence = summary_writer.finish()
        staging.replace(output_directory)
    except BaseException:
        pair_writer.abort()
        summary_writer.abort()
        staging.rmdir()
        raise

    if summary_evidence.row_count != len(expected):
        raise AuditValidationError("cap summary query count does not reconcile")
    return CapEvidence(
        cap=cap,
        query_count=len(expected),
        returned_rows=returned_rows,
        prohibited_pairs=prohibited_pairs,
        prohibited_queries=prohibited_queries,
        closest_categories=dict(closest),
        prohibited_pair_file=pair_evidence,
        query_summary_file=summary_evidence,
    )


def compare_caps(
    *,
    baseline_cap: int,
    comparison_cap: int,
    baseline_canonical_path: Path,
    comparison_canonical_path: Path,
    baseline_summary_path: Path,
    comparison_summary_path: Path,
    expected_query_ids: Iterable[str],
    baseline_contains_other_queries: bool = False,
) -> CapComparison:
    """Count row, prohibited-status, and closest-category changes."""

    if (baseline_cap, comparison_cap) not in _ALLOWED_CAP_TRANSITIONS:
        raise AuditConfigurationError(
            "cap comparison is outside the frozen A-004 stages"
        )
    expected = _validated_query_ids(expected_query_ids)

    changed = compare_canonical_results(
        baseline_canonical_path,
        comparison_canonical_path,
        expected_query_ids=expected,
        left_may_contain_other_queries=baseline_contains_other_queries,
    )
    baseline = _read_query_summaries(
        baseline_summary_path,
        expected,
        allow_other_queries=baseline_contains_other_queries,
    )
    comparison = _read_query_summaries(
        comparison_summary_path,
        expected,
        allow_other_queries=False,
    )
    newly_prohibited = 0
    no_longer_prohibited = 0
    closest_changes = 0
    for query in expected:
        before_status, before_category = baseline[query]
        after_status, after_category = comparison[query]
        newly_prohibited += int(not before_status and after_status)
        no_longer_prohibited += int(before_status and not after_status)
        closest_changes += int(before_category != after_category)

    return CapComparison(
        baseline_cap=baseline_cap,
        comparison_cap=comparison_cap,
        compared_queries=len(expected),
        complete_row_change_query_ids=changed,
        complete_row_changes=len(changed),
        newly_prohibited_queries=newly_prohibited,
        no_longer_prohibited_queries=no_longer_prohibited,
        closest_category_changes=closest_changes,
    )


def _read_query_summaries(
    path: Path,
    expected_query_ids: tuple[str, ...],
    *,
    allow_other_queries: bool,
) -> dict[str, tuple[bool, str]]:
    expected = frozenset(expected_query_ids)
    summaries = {}
    previous = ""
    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            cells = decode_strict_tsv_line(
                raw_line,
                context=f"{path.name} line {line_number}",
            ).split("\t")
            if len(cells) != 5 or cells[0] <= previous:
                raise AuditValidationError("query summary rows are malformed")
            query, status, category, returned_text, prohibited_text = cells
            previous = query
            if query not in expected:
                if allow_other_queries:
                    continue
                raise AuditValidationError("query summary contains an unknown query")
            try:
                returned_rows = int(returned_text)
                prohibited_pairs = int(prohibited_text)
            except ValueError as error:
                raise AuditValidationError(
                    "query summary counts are malformed"
                ) from error
            if (
                status not in {"0", "1"}
                or category not in CLOSEST_RESIDUAL_CATEGORIES
                or returned_rows < 0
                or prohibited_pairs < 0
                or prohibited_pairs > returned_rows
                or (status == "1") != (prohibited_pairs > 0)
            ):
                raise AuditValidationError("query summary values are malformed")
            summaries[query] = (status == "1", category)
    if set(summaries) != expected:
        raise AuditValidationError("query summary universe drifted")
    return summaries


# Prohibited-pair unions and comparisons


def union_prohibited_pairs(
    *,
    source_paths: Mapping[str, Path],
    output_directory: Path,
) -> PairUnionEvidence:
    """Stream sorted pair files into one immutable, duplicate-free union."""

    labels, paths = _validated_pair_sources(source_paths)

    staging = output_directory.with_name(f".{output_directory.name}.incomplete")
    if output_directory.exists() or staging.exists():
        raise AuditExecutionError("pair union requires a fresh output directory")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    writer = EvidenceWriter(staging / "prohibited_pairs.tsv")
    unique_pairs = 0
    unique_queries = 0
    previous_pair: tuple[str, str] | None = None
    previous_query: str | None = None

    try:
        readers = tuple(_iter_pair_file(path) for path in paths)
        for pair in heapq.merge(*readers):
            if pair == previous_pair:
                continue
            query, target = pair
            writer.write(f"{query}\t{target}\n".encode("ascii"))
            unique_pairs += 1
            if query != previous_query:
                unique_queries += 1
                previous_query = query
            previous_pair = pair
        pair_evidence = writer.finish()
        staging.replace(output_directory)
    except BaseException:
        writer.abort()
        staging.rmdir()
        raise

    if pair_evidence.row_count != unique_pairs:
        raise AuditValidationError("pair union count does not reconcile")
    return PairUnionEvidence(
        source_labels=labels,
        unique_pairs=unique_pairs,
        unique_queries=unique_queries,
        prohibited_pair_file=pair_evidence,
    )


def compare_pair_unions(
    *,
    common_path: Path,
    staged_path: Path,
) -> PairUnionComparison:
    """Require the staged union to contain the common union and count additions."""

    if common_path == staged_path:
        raise AuditConfigurationError("pair-union comparison requires distinct files")
    common_iterator = _iter_pair_file(common_path)
    staged_iterator = _iter_pair_file(staged_path)
    common = next(common_iterator, None)
    staged = next(staged_iterator, None)
    common_pairs = 0
    staged_pairs = 0
    common_queries: set[str] = set()
    staged_queries: set[str] = set()

    while common is not None and staged is not None:
        if common < staged:
            raise AuditValidationError("staged union is missing a common-cap pair")
        if staged < common:
            staged_pairs += 1
            staged_queries.add(staged[0])
            staged = next(staged_iterator, None)
            continue
        common_pairs += 1
        staged_pairs += 1
        common_queries.add(common[0])
        staged_queries.add(staged[0])
        common = next(common_iterator, None)
        staged = next(staged_iterator, None)

    if common is not None:
        raise AuditValidationError("staged union is missing a common-cap pair")
    if staged is not None:
        staged_pairs += 1
        staged_queries.add(staged[0])
    for pair in staged_iterator:
        staged_pairs += 1
        staged_queries.add(pair[0])

    return PairUnionComparison(
        common_pairs=common_pairs,
        staged_pairs=staged_pairs,
        additional_pairs=staged_pairs - common_pairs,
        common_queries=len(common_queries),
        staged_queries=len(staged_queries),
        newly_prohibited_queries=len(staged_queries - common_queries),
    )


def _iter_pair_file(path: Path) -> Iterator[tuple[str, str]]:
    previous: tuple[str, str] | None = None
    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            cells = decode_strict_tsv_line(
                raw_line,
                context=f"{path.name} line {line_number}",
            ).split("\t")
            if len(cells) != 2 or not all(cells):
                raise AuditValidationError("prohibited-pair row is malformed")
            pair = (cells[0], cells[1])
            if previous is not None and pair <= previous:
                raise AuditValidationError(
                    "prohibited-pair rows must be unique and sorted"
                )
            previous = pair
            yield pair


# Resume-safe stored evidence


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

    _require_source_label(source_label)
    _require_cap(cap)
    query_ids = _validated_query_ids(expected_query_ids)
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
        raise AuditExecutionError(
            "A-004 cap-summary output lacks its completion marker"
        )
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

    _require_source_label(source_label)
    _require_cap(cap)
    query_ids = _validated_query_ids(expected_query_ids)
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
        raise AuditValidationError("A-004 cap-summary identity drifted")
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

    _require_pair_union_label(label)
    source_labels, paths = _validated_pair_sources(source_paths)
    source_identities = {
        source_label: file_identity(path)
        for source_label, path in zip(source_labels, paths, strict=True)
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
        raise AuditExecutionError("A-004 pair-union output lacks its completion marker")
    evidence = union_prohibited_pairs(
        source_paths=source_paths,
        output_directory=output_directory,
    )
    write_json_atomic(
        marker_path, {**expected, "evidence": _pair_union_payload(evidence)}
    )
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

    _require_pair_union_label(label)
    source_labels, paths = _validated_pair_sources(source_paths)
    source_identities = {
        source_label: file_identity(path)
        for source_label, path in zip(source_labels, paths, strict=True)
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
        raise AuditValidationError("A-004 pair-union identity drifted")
    verify_file(
        output_directory / "prohibited_pairs.tsv",
        evidence.prohibited_pair_file.byte_size,
        evidence.prohibited_pair_file.sha256,
    )
    if evidence.prohibited_pair_file.row_count != evidence.unique_pairs:
        raise AuditValidationError("A-004 pair-union counts do not reconcile")
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
    if (
        evidence.query_count != expected_queries
        or evidence.prohibited_queries > expected_queries
    ):
        raise AuditValidationError("A-004 cap-summary counts do not reconcile")
    if evidence.prohibited_pairs > evidence.returned_rows:
        raise AuditValidationError("A-004 cap-summary pair counts do not reconcile")
    if sum(evidence.closest_categories.values()) != expected_queries:
        raise AuditValidationError("A-004 cap-summary category counts do not reconcile")
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
        raise AuditValidationError("A-004 cap-summary query evidence drifted")
    if evidence.prohibited_pair_file.row_count != evidence.prohibited_pairs:
        raise AuditValidationError("A-004 cap-summary pair evidence drifted")


def _cap_evidence_from(raw: object) -> CapEvidence:
    if not isinstance(raw, dict):
        raise AuditValidationError("A-004 cap-summary evidence is malformed")
    integer_names = (
        "cap",
        "query_count",
        "returned_rows",
        "prohibited_pairs",
        "prohibited_queries",
    )
    values = {name: _nonnegative_int(raw.get(name), name) for name in integer_names}
    categories = raw.get("closest_categories")
    if not isinstance(categories, dict) or any(
        not isinstance(name, str)
        or _nonnegative_int(count, "closest category") != count
        for name, count in categories.items()
    ):
        raise AuditValidationError("A-004 closest-category evidence is malformed")
    return CapEvidence(
        **values,
        closest_categories=categories,
        prohibited_pair_file=file_evidence_from(raw.get("prohibited_pair_file")),
        query_summary_file=file_evidence_from(raw.get("query_summary_file")),
    )


def _pair_union_evidence_from(raw: object) -> PairUnionEvidence:
    if not isinstance(raw, dict):
        raise AuditValidationError("A-004 pair-union evidence is malformed")
    labels = raw.get("source_labels")
    if (
        not isinstance(labels, list)
        or not labels
        or any(not isinstance(label, str) for label in labels)
    ):
        raise AuditValidationError("A-004 pair-union labels are malformed")
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
        raise AuditValidationError(f"A-004 {name} must be a nonnegative integer")
    return value


def _require_cap(cap: int) -> None:
    if cap not in _ALLOWED_CAPS:
        raise AuditConfigurationError("search cap is outside the frozen A-004 stages")


def _validated_query_ids(expected_query_ids: Iterable[str]) -> tuple[str, ...]:
    query_ids = tuple(sorted(expected_query_ids))
    if not query_ids or len(set(query_ids)) != len(query_ids):
        raise AuditConfigurationError("expected query identifiers must be unique")
    return query_ids


def _require_source_label(source_label: str) -> None:
    if not source_label:
        raise AuditConfigurationError("A-004 cap-summary source label is required")


def _require_pair_union_label(label: str) -> None:
    if not label:
        raise AuditConfigurationError("A-004 pair-union label is required")


def _validated_pair_sources(
    source_paths: Mapping[str, Path],
) -> tuple[tuple[str, ...], tuple[Path, ...]]:
    labels = tuple(sorted(source_paths))
    if not labels or any(not label for label in labels):
        raise AuditConfigurationError("pair union requires named source files")
    paths = tuple(source_paths[label] for label in labels)
    if len(set(paths)) != len(paths):
        raise AuditConfigurationError("pair union source paths must be distinct")
    return labels, paths
