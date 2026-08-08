"""Compact and aggregate Task 7 similarity evidence."""

from __future__ import annotations

import hashlib
import heapq
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from decimal import Decimal
from pathlib import Path

from protein_lm.data.similarity_alignment import (
    CATEGORY_CLOSEST_PROHIBITED,
    CATEGORY_PROHIBITED,
    CATEGORY_UNDER_30_OR_NONE,
    CLOSEST_RESIDUAL_CATEGORIES,
    RESIDUAL_CATEGORIES,
    closest_residual_key,
    decode_strict_tsv_line,
    residual_category,
    violates_prohibited_boundary,
)
from protein_lm.data.similarity_audit_models import (
    AcceptedPassEvidence,
    AlignmentRow,
    ConvergenceEvidence,
    FileEvidence,
    SequenceMetadata,
    canonical_decimal,
)
from protein_lm.data.similarity_audit_policy import SimilarityAuditError
from protein_lm.data.similarity_results import iter_converged_query_rows


def compact_converged_results(
    *,
    pass_name: str,
    comparison_path: Path,
    escalation_path: Path | None,
    convergence: ConvergenceEvidence,
    expected_query_ids: Iterable[str],
    output_directory: Path,
    resource_guard: Callable[[], None] | None = None,
) -> AcceptedPassEvidence:
    """Keep pair and closest-hit evidence, then allow full rows to be removed."""

    if pass_name not in {"enforcement", "residual"}:
        raise SimilarityAuditError(f"unknown search pass: {pass_name}")
    expected = tuple(sorted(expected_query_ids))
    escalated = frozenset(convergence.escalated_query_ids)
    output_directory.mkdir(parents=True, exist_ok=True)
    pair_path = output_directory / "returned_pairs.tsv"
    prohibited_path = output_directory / "prohibited_pairs.tsv"
    summary_path = output_directory / "residual_summaries.tsv"

    pair_writer = _AtomicEvidenceWriter(pair_path)
    prohibited_writer = _AtomicEvidenceWriter(prohibited_path)
    summary_writer = (
        _AtomicEvidenceWriter(summary_path) if pass_name == "residual" else None
    )
    accepted_rows = 0
    try:
        for query, rows in iter_converged_query_rows(
            expected,
            comparison_path=comparison_path,
            escalation_path=escalation_path,
            escalated_queries=escalated,
        ):
            closest = min(rows, key=closest_residual_key, default=None)
            for row in rows:
                accepted_rows += 1
                pair_writer.write(f"{row.query}\t{row.target}\n".encode("ascii"))
                if violates_prohibited_boundary(row):
                    prohibited_writer.write(
                        f"{row.query}\t{row.target}\n".encode("ascii")
                    )
                if resource_guard is not None and accepted_rows % 100_000 == 0:
                    resource_guard()
            if summary_writer is not None:
                summary_writer.write(_residual_summary_line(query, closest))
        if resource_guard is not None:
            resource_guard()
        pair_evidence = pair_writer.finish()
        prohibited_evidence = prohibited_writer.finish()
        summary_evidence = summary_writer.finish() if summary_writer else None
    except BaseException:
        pair_writer.abort()
        prohibited_writer.abort()
        if summary_writer:
            summary_writer.abort()
        raise

    if pair_evidence.row_count != accepted_rows:
        raise SimilarityAuditError("accepted alignment and pair counts differ")
    return AcceptedPassEvidence(
        pass_name=pass_name,
        accepted_rows=accepted_rows,
        accepted_at_comparison_cap=len(expected) - len(escalated),
        accepted_at_escalation_cap=len(escalated),
        returned_pairs=pair_evidence,
        prohibited_pairs=prohibited_evidence,
        residual_summaries=summary_evidence,
    )


def aggregate_partition_evidence(
    *,
    expected_query_ids: Iterable[str],
    query_metadata: Mapping[str, SequenceMetadata],
    target_metadata: Mapping[str, SequenceMetadata],
    enforcement_directory: Path,
    residual_directory: Path,
) -> dict[str, object]:
    """Combine both passes into aggregate-only evidence for one partition."""

    expected = tuple(sorted(expected_query_ids))
    if set(expected) != set(query_metadata):
        raise SimilarityAuditError("query metadata differs from the held-out universe")

    enforcement_pairs = enforcement_directory / "returned_pairs.tsv"
    residual_pairs = residual_directory / "returned_pairs.tsv"
    unique_returned_pairs = sum(
        1 for _ in _merge_unique_pairs(enforcement_pairs, residual_pairs)
    )

    violating_queries: set[str] = set()
    attribution = Counter(
        {
            "exact_sequence_duplicate": 0,
            "same_uniref50_group": 0,
            "cross_uniref50_group": 0,
        }
    )
    unique_prohibited_pairs = 0
    for query, target in _merge_unique_pairs(
        enforcement_directory / "prohibited_pairs.tsv",
        residual_directory / "prohibited_pairs.tsv",
    ):
        unique_prohibited_pairs += 1
        violating_queries.add(query)
        query_record = query_metadata[query]
        target_record = target_metadata[target]
        if query_record.sequence_sha256 == target_record.sequence_sha256:
            attribution["exact_sequence_duplicate"] += 1
        elif query_record.uniref50_group == target_record.uniref50_group:
            attribution["same_uniref50_group"] += 1
        else:
            attribution["cross_uniref50_group"] += 1

    if sum(attribution.values()) != unique_prohibited_pairs:
        raise SimilarityAuditError("prohibited-pair attribution does not reconcile")
    closest_categories, status_categories = _category_counts(
        residual_directory / "residual_summaries.tsv",
        expected_query_ids=expected,
        violating_queries=violating_queries,
    )
    if sum(closest_categories.values()) != len(expected):
        raise SimilarityAuditError("closest residual categories do not reconcile")
    if sum(status_categories.values()) != len(expected):
        raise SimilarityAuditError("held-out query statuses do not reconcile")

    numerator = len(violating_queries)
    query_hashes = {record.sequence_sha256 for record in query_metadata.values()}
    target_hashes = {record.sequence_sha256 for record in target_metadata.values()}
    query_groups = {record.uniref50_group for record in query_metadata.values()}
    target_groups = {record.uniref50_group for record in target_metadata.values()}
    return {
        "held_out_queries_with_prohibited_match": numerator,
        "held_out_query_count": len(expected),
        "prohibited_query_rate_percent": _percentage(numerator, len(expected)),
        "unique_prohibited_pairs": unique_prohibited_pairs,
        "prohibited_pair_attribution": dict(attribution),
        "enforcement_returned_pairs": _count_pair_rows(enforcement_pairs),
        "residual_returned_pairs": _count_pair_rows(residual_pairs),
        "unique_returned_pair_union": unique_returned_pairs,
        "exact_sequence_hash_crossings_to_training": len(query_hashes & target_hashes),
        "uniref50_group_crossings_to_training": len(query_groups & target_groups),
        "closest_residual_categories": closest_categories,
        "held_out_query_status_categories": status_categories,
    }


def _residual_summary_line(query: str, closest: AlignmentRow | None) -> bytes:
    if closest is None:
        return f"{query}\t{CATEGORY_UNDER_30_OR_NONE}\t\t\t\t\n".encode("ascii")
    return (
        f"{query}\t{residual_category(closest)}\t{closest.target}\t"
        f"{canonical_decimal(closest.fident)}\t"
        f"{canonical_decimal(closest.qcov)}\t"
        f"{canonical_decimal(closest.tcov)}\n"
    ).encode("ascii")


class _AtomicEvidenceWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.temporary_path = path.with_name(f".{path.name}.incomplete")
        self.temporary_path.unlink(missing_ok=True)
        self._output = self.temporary_path.open("wb")
        self._hasher = hashlib.sha256()
        self._bytes = 0
        self._rows = 0

    def write(self, content: bytes) -> None:
        if not content.endswith(b"\n"):
            raise SimilarityAuditError("compact evidence rows must end with LF")
        self._output.write(content)
        self._hasher.update(content)
        self._bytes += len(content)
        self._rows += 1

    def finish(self) -> FileEvidence:
        self._output.close()
        self.temporary_path.replace(self.path)
        return FileEvidence(self._rows, self._bytes, self._hasher.hexdigest())

    def abort(self) -> None:
        if not self._output.closed:
            self._output.close()
        self.temporary_path.unlink(missing_ok=True)


def _iter_pair_file(path: Path) -> Iterator[tuple[str, str]]:
    previous: tuple[str, str] | None = None
    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            cells = decode_strict_tsv_line(
                raw_line,
                context=f"{path.name} line {line_number}",
            ).split("\t")
            if len(cells) != 2:
                raise SimilarityAuditError(f"{path.name} pair row is malformed")
            pair = (cells[0], cells[1])
            if previous is not None and pair <= previous:
                raise SimilarityAuditError(
                    f"{path.name} pair rows are not unique and sorted"
                )
            previous = pair
            yield pair


def _merge_unique_pairs(first_path: Path, second_path: Path) -> Iterator[tuple[str, str]]:
    previous: tuple[str, str] | None = None
    for pair in heapq.merge(_iter_pair_file(first_path), _iter_pair_file(second_path)):
        if pair != previous:
            yield pair
            previous = pair


def _count_pair_rows(path: Path) -> int:
    return sum(1 for _ in _iter_pair_file(path))


def _category_counts(
    path: Path,
    *,
    expected_query_ids: Sequence[str],
    violating_queries: set[str],
) -> tuple[dict[str, int], dict[str, int]]:
    closest_counts = Counter(
        {category: 0 for category in CLOSEST_RESIDUAL_CATEGORIES}
    )
    status_counts = Counter({category: 0 for category in RESIDUAL_CATEGORIES})
    expected = iter(expected_query_ids)
    next_expected = next(expected, None)
    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            cells = decode_strict_tsv_line(
                raw_line,
                context=f"{path.name} line {line_number}",
            ).split("\t")
            if len(cells) != 6:
                raise SimilarityAuditError("residual summary row is malformed")
            query, category = cells[0], cells[1]
            if query != next_expected:
                raise SimilarityAuditError("residual summary query universe drifted")
            if category not in CLOSEST_RESIDUAL_CATEGORIES:
                raise SimilarityAuditError("residual summary category is unknown")
            closest_counts[category] += 1
            if query in violating_queries:
                status_counts[CATEGORY_PROHIBITED] += 1
            elif category == CATEGORY_CLOSEST_PROHIBITED:
                raise SimilarityAuditError(
                    "a closest prohibited row is absent from the violation union"
                )
            else:
                status_counts[category] += 1
            next_expected = next(expected, None)
    if next_expected is not None:
        raise SimilarityAuditError("residual summary is missing expected queries")
    return dict(closest_counts), dict(status_counts)


def _percentage(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        raise SimilarityAuditError("percentage denominator must be positive")
    return f"{(Decimal(numerator) * 100 / Decimal(denominator)):.6f}"
