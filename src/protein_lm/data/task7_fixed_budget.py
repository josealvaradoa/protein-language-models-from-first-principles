"""Summarize Task 7 search caps without requiring full-row convergence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from protein_lm.data.similarity_alignment import (
    CLOSEST_RESIDUAL_CATEGORIES,
    closest_residual_key,
    decode_strict_tsv_line,
    residual_category,
    violates_prohibited_boundary,
)
from protein_lm.data.similarity_audit_models import FileEvidence
from protein_lm.data.similarity_audit_policy import SimilarityAuditError
from protein_lm.data.similarity_results import (
    compare_canonical_results,
    iter_converged_query_rows,
)
from protein_lm.data.task7_evidence_io import EvidenceWriter

_FROZEN_CAPS = frozenset({1_000, 10_000, 100_000})
_FROZEN_TRANSITIONS = frozenset({(1_000, 10_000), (10_000, 100_000)})


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


def summarize_cap(
    *,
    cap: int,
    canonical_path: Path,
    expected_query_ids: Iterable[str],
    output_directory: Path,
) -> CapEvidence:
    """Write compact pair and per-query evidence for one search cap."""

    if cap not in _FROZEN_CAPS:
        raise SimilarityAuditError("search cap is outside the frozen A-004 stages")
    expected = tuple(sorted(expected_query_ids))
    if not expected or len(set(expected)) != len(expected):
        raise SimilarityAuditError("expected query identifiers must be unique")

    staging = output_directory.with_name(f".{output_directory.name}.incomplete")
    if output_directory.exists() or staging.exists():
        raise SimilarityAuditError("cap evidence requires a fresh output directory")
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
                raise SimilarityAuditError("query rows exceed the frozen search cap")
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
        raise SimilarityAuditError("cap summary query count does not reconcile")
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

    if (baseline_cap, comparison_cap) not in _FROZEN_TRANSITIONS:
        raise SimilarityAuditError("cap comparison is outside the frozen A-004 stages")
    expected = tuple(sorted(expected_query_ids))
    if not expected or len(set(expected)) != len(expected):
        raise SimilarityAuditError("expected query identifiers must be unique")

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
                raise SimilarityAuditError("query summary rows are malformed")
            query, status, category, returned_text, prohibited_text = cells
            previous = query
            if query not in expected:
                if allow_other_queries:
                    continue
                raise SimilarityAuditError("query summary contains an unknown query")
            try:
                returned_rows = int(returned_text)
                prohibited_pairs = int(prohibited_text)
            except ValueError as error:
                raise SimilarityAuditError("query summary counts are malformed") from error
            if (
                status not in {"0", "1"}
                or category not in CLOSEST_RESIDUAL_CATEGORIES
                or returned_rows < 0
                or prohibited_pairs < 0
                or prohibited_pairs > returned_rows
                or (status == "1") != (prohibited_pairs > 0)
            ):
                raise SimilarityAuditError("query summary values are malformed")
            summaries[query] = (status == "1", category)
    if set(summaries) != expected:
        raise SimilarityAuditError("query summary universe drifted")
    return summaries
