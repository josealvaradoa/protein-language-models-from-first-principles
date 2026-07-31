"""Render aggregate-only evidence for the Task 7 diagnostic audit."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping

from protein_lm.data.similarity_alignment import (
    CLOSEST_RESIDUAL_CATEGORIES,
    RESIDUAL_CATEGORIES,
)
from protein_lm.data.similarity_audit_policy import SimilarityAuditError


@dataclass(frozen=True)
class RenderedTask7Report:
    """Public JSON, Markdown, and the canonical JSON digest."""

    json_text: str
    markdown_text: str
    json_sha256: str


@dataclass(frozen=True)
class _SimilarityEvidence:
    prohibited_queries: int
    query_count: int
    pair_counts: Mapping[str, int]
    attribution: Mapping[str, int]
    closest_categories: Mapping[str, int]
    status_categories: Mapping[str, int]


_STRATEGIES = ("random", "group_aware")
_HELD_OUT_PARTITIONS = ("validation", "test")
_PAIR_COUNT_FIELDS = (
    "unique_prohibited_pairs",
    "enforcement_returned_pairs",
    "residual_returned_pairs",
    "unique_returned_pair_union",
)
_ATTRIBUTION_CATEGORIES = {
    "exact_sequence_duplicate",
    "same_uniref50_group",
    "cross_uniref50_group",
}
_STRUCTURAL_COUNT_FIELDS = (
    "exact_sequence_hash_crossings",
    "uniref50_group_crossings",
    "retained_records",
    "retained_residues",
    "excluded_records",
    "excluded_residues",
    "largest_uniref50_group_records",
    "largest_uniref50_group_residues",
)
_REQUIRED_AUTHORITY_GUARDS = {
    "diagnostic_only": True,
    "diagnostic_audit_authorized": True,
    "candidate_status": "failed_balance",
    "repair_authorized": False,
    "repair_performed": False,
    "selected_split_authorized": False,
    "task8_membership_use_authorized": False,
    "model_use": "prohibited",
    "post_audit_review_required": True,
}


def render_task7_report(report: Mapping[str, object]) -> RenderedTask7Report:
    """Validate the authority boundary and render the final aggregate report."""

    _validate_authority_guards(report)
    strategies = _validated_strategies(report)
    _validate_aggregate_report(strategies)

    report_dict = dict(report)
    json_text = json.dumps(report_dict, indent=2, sort_keys=True) + "\n"
    markdown_text = _render_markdown(strategies)
    return RenderedTask7Report(
        json_text=json_text,
        markdown_text=markdown_text,
        json_sha256=hashlib.sha256(json_text.encode("utf-8")).hexdigest(),
    )


def _validate_authority_guards(report: Mapping[str, object]) -> None:
    drift = [
        f"{key}: found {report.get(key)!r}, expected {expected!r}"
        for key, expected in _REQUIRED_AUTHORITY_GUARDS.items()
        if report.get(key) != expected
    ]
    if drift:
        raise SimilarityAuditError("Task 7 authority guard failed: " + "; ".join(drift))


def _validated_strategies(report: Mapping[str, object]) -> dict[str, object]:
    strategies = report.get("strategies")
    if not isinstance(strategies, dict) or set(strategies) != {"random", "group_aware"}:
        raise SimilarityAuditError("Task 7 report must contain both strategies")
    return strategies


def _render_markdown(strategies: Mapping[str, object]) -> str:
    lines = []
    lines.extend(_comparison_section(strategies))
    lines.extend(_structural_membership_section(strategies))
    lines.extend(_query_status_section(strategies))
    lines.extend(_closest_residual_section(strategies))
    lines.extend(_interpretation_section())
    return "\n".join(lines)


def _comparison_section(strategies: Mapping[str, object]) -> list[str]:
    lines = [
        "# Week 1 Task 7 Diagnostic Similarity Audit",
        "",
        "This report compares the frozen Task 5 random diagnostic with the "
        "immutable Task 6 failed-balance pre-repair candidate.",
        "",
        "It records detected sequence similarity under the pinned MMseqs2 "
        "procedure. It does not repair or select a split, authorize Task 8, "
        "authorize model use, or prove the absence of homology.",
        "",
        "| Strategy | Held-out partition | Records (share) | Residues (share) | Prohibited queries | Queries audited | Rate | Prohibited pairs |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy in _STRATEGIES:
        strategy_report = strategies[strategy]
        if not isinstance(strategy_report, dict):
            raise SimilarityAuditError("strategy report is malformed")
        partitions = strategy_report["partitions"]
        for partition in _HELD_OUT_PARTITIONS:
            lines.append(_partition_comparison_row(strategy, partition, partitions))
        lines.append(_overall_comparison_row(strategy, strategy_report, partitions))
    return lines


def _partition_comparison_row(
    strategy: str,
    partition: str,
    partitions: Mapping[str, object],
) -> str:
    comparison = partitions[partition]["similarity"]
    balance = partitions[partition]["balance"]
    return (
        "| "
        f"{strategy} | {partition} | "
        f"{balance['records']} ({balance['record_share_percent']}%) | "
        f"{balance['residues']} ({balance['residue_share_percent']}%) | "
        f"{comparison['held_out_queries_with_prohibited_match']} | "
        f"{comparison['held_out_query_count']} | "
        f"{comparison['prohibited_query_rate_percent']}% | "
        f"{comparison['unique_prohibited_pairs']} |"
    )


def _overall_comparison_row(
    strategy: str,
    strategy_report: Mapping[str, object],
    partitions: Mapping[str, object],
) -> str:
    validation_balance = partitions["validation"]["balance"]
    test_balance = partitions["test"]["balance"]
    overall = strategy_report["overall"]
    held_out_record_share = Decimal(
        validation_balance["record_share_percent"]
    ) + Decimal(test_balance["record_share_percent"])
    held_out_residue_share = Decimal(
        validation_balance["residue_share_percent"]
    ) + Decimal(test_balance["residue_share_percent"])
    return (
        "| "
        f"{strategy} | overall held-out | "
        f"{validation_balance['records'] + test_balance['records']} "
        f"({held_out_record_share:.6f}%) | "
        f"{validation_balance['residues'] + test_balance['residues']} "
        f"({held_out_residue_share:.6f}%) | "
        f"{overall['held_out_queries_with_prohibited_match']} | "
        f"{overall['held_out_query_count']} | "
        f"{overall['prohibited_query_rate_percent']}% | "
        f"{overall['unique_prohibited_pairs']} |"
    )


def _structural_membership_section(strategies: Mapping[str, object]) -> list[str]:
    lines = [
        "",
        "## Frozen membership structure",
        "",
        "| Strategy | Exact-hash crossings | UniRef50 crossings | Retained records | Retained residues | Excluded records | Largest group or unit (records) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy in _STRATEGIES:
        structural = strategies[strategy]["structural_membership"]
        largest = structural.get(
            "largest_assignment_unit_records",
            structural["largest_uniref50_group_records"],
        )
        lines.append(
            f"| {strategy} | {structural['exact_sequence_hash_crossings']} | "
            f"{structural['uniref50_group_crossings']} | "
            f"{structural['retained_records']} | {structural['retained_residues']} | "
            f"{structural['excluded_records']} | {largest} |"
        )
    return lines


def _query_status_section(strategies: Mapping[str, object]) -> list[str]:
    lines = [
        "",
        "## Held-out query status categories",
        "",
        "| Strategy | Partition | Category | Queries |",
        "| --- | --- | --- | ---: |",
    ]
    lines.extend(
        _category_rows(
            strategies,
            evidence_name="held_out_query_status_categories",
            categories=RESIDUAL_CATEGORIES,
        )
    )
    return lines


def _closest_residual_section(strategies: Mapping[str, object]) -> list[str]:
    lines = [
        "",
        "## Closest residual-match categories",
        "",
        "These categories describe the selected closest residual row itself. "
        "A query can be prohibited by a different returned pair.",
        "",
        "| Strategy | Partition | Category | Queries |",
        "| --- | --- | --- | ---: |",
    ]
    lines.extend(
        _category_rows(
            strategies,
            evidence_name="closest_residual_categories",
            categories=CLOSEST_RESIDUAL_CATEGORIES,
        )
    )
    return lines


def _category_rows(
    strategies: Mapping[str, object],
    *,
    evidence_name: str,
    categories: tuple[str, ...],
) -> list[str]:
    lines = []
    for strategy in _STRATEGIES:
        partitions = strategies[strategy]["partitions"]
        for partition in _HELD_OUT_PARTITIONS:
            category_counts = partitions[partition]["similarity"][evidence_name]
            for category in categories:
                lines.append(
                    f"| {strategy} | {partition} | {category} | "
                    f"{category_counts[category]} |"
                )
    return lines


def _interpretation_section() -> list[str]:
    return [
        "",
        "## Interpretation boundary",
        "",
        "The candidate held-out partitions contain shorter proteins on "
        "average than its training partition. The comparison therefore "
        "cannot isolate grouping as the only cause of a difference.",
        "",
        "Even zero detected prohibited matches would not cure the Task 6 "
        "balance failure. Jose must review this diagnostic before any "
        "later adjustment is considered.",
        "",
    ]


def _validate_aggregate_report(strategies: Mapping[str, object]) -> None:
    for strategy_name, raw_strategy in strategies.items():
        _validate_strategy_report(strategy_name, raw_strategy)


def _validate_strategy_report(strategy_name: str, raw_strategy: object) -> None:
    if not isinstance(raw_strategy, dict):
        raise SimilarityAuditError(f"{strategy_name} report is malformed")

    _validate_structural_evidence(
        strategy_name,
        raw_strategy.get("structural_membership"),
    )
    partitions = raw_strategy.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != {
        "training",
        "validation",
        "test",
    }:
        raise SimilarityAuditError(f"{strategy_name} partitions are malformed")

    held_out = _validate_held_out_partitions(partitions)
    _validate_overall_report(raw_strategy.get("overall"), held_out)


def _validate_structural_evidence(strategy_name: str, structural: object) -> None:
    if not isinstance(structural, dict) or any(
        isinstance(structural.get(key), bool)
        or not isinstance(structural.get(key), int)
        or structural[key] < 0
        for key in _STRUCTURAL_COUNT_FIELDS
    ):
        raise SimilarityAuditError(f"{strategy_name} structural evidence is malformed")


def _validate_held_out_partitions(
    partitions: Mapping[str, object],
) -> _SimilarityEvidence:
    total_numerator = 0
    total_denominator = 0
    summed_counts = Counter()
    summed_attribution = Counter()
    summed_closest = Counter(
        {category: 0 for category in CLOSEST_RESIDUAL_CATEGORIES}
    )
    summed_statuses = Counter({category: 0 for category in RESIDUAL_CATEGORIES})

    for partition in _HELD_OUT_PARTITIONS:
        evidence = _validate_partition_report(partitions[partition])
        total_numerator += evidence.prohibited_queries
        total_denominator += evidence.query_count
        summed_counts.update(evidence.pair_counts)
        summed_attribution.update(evidence.attribution)
        summed_closest.update(evidence.closest_categories)
        summed_statuses.update(evidence.status_categories)

    return _SimilarityEvidence(
        prohibited_queries=total_numerator,
        query_count=total_denominator,
        pair_counts=dict(summed_counts),
        attribution=dict(summed_attribution),
        closest_categories=dict(summed_closest),
        status_categories=dict(summed_statuses),
    )


def _validate_partition_report(raw_partition: object) -> _SimilarityEvidence:
    if not isinstance(raw_partition, dict):
        raise SimilarityAuditError("held-out partition report is malformed")
    similarity = raw_partition.get("similarity")
    if not isinstance(similarity, dict):
        raise SimilarityAuditError("partition similarity report is malformed")

    numerator, denominator = _validate_violation_counts(similarity)
    closest, statuses = _validate_category_counts(
        similarity,
        numerator=numerator,
        denominator=denominator,
    )
    attribution = _validate_attribution(similarity)
    pair_counts = _validate_pair_counts(similarity)
    return _SimilarityEvidence(
        prohibited_queries=numerator,
        query_count=denominator,
        pair_counts=pair_counts,
        attribution=attribution,
        closest_categories=closest,
        status_categories=statuses,
    )


def _validate_violation_counts(similarity: Mapping[str, object]) -> tuple[int, int]:
    numerator = similarity.get("held_out_queries_with_prohibited_match")
    denominator = similarity.get("held_out_query_count")
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or numerator < 0
        or denominator <= 0
        or numerator > denominator
    ):
        raise SimilarityAuditError("partition violation counts are invalid")
    expected_rate = f"{(Decimal(numerator) * 100 / denominator):.6f}"
    if similarity.get("prohibited_query_rate_percent") != expected_rate:
        raise SimilarityAuditError("partition violation rate does not reconcile")
    return numerator, denominator


def _validate_category_counts(
    similarity: Mapping[str, object],
    *,
    numerator: int,
    denominator: int,
) -> tuple[dict[str, int], dict[str, int]]:
    closest = similarity.get("closest_residual_categories")
    if not isinstance(closest, dict) or set(closest) != set(
        CLOSEST_RESIDUAL_CATEGORIES
    ):
        raise SimilarityAuditError("closest-hit categories are malformed")
    if _contains_invalid_count(closest) or sum(closest.values()) != denominator:
        raise SimilarityAuditError("closest-hit categories do not reconcile")

    statuses = similarity.get("held_out_query_status_categories")
    if not isinstance(statuses, dict) or set(statuses) != set(RESIDUAL_CATEGORIES):
        raise SimilarityAuditError("held-out query statuses are malformed")
    if _contains_invalid_count(statuses) or sum(statuses.values()) != denominator:
        raise SimilarityAuditError("held-out query statuses do not reconcile")
    if statuses["prohibited"] != numerator:
        raise SimilarityAuditError(
            "prohibited query status does not match the primary numerator"
        )
    if closest["closest_match_prohibited"] > numerator:
        raise SimilarityAuditError(
            "closest prohibited count exceeds the primary numerator"
        )
    return closest, statuses


def _validate_attribution(similarity: Mapping[str, object]) -> dict[str, int]:
    attribution = similarity.get("prohibited_pair_attribution")
    if not isinstance(attribution, dict) or set(attribution) != _ATTRIBUTION_CATEGORIES:
        raise SimilarityAuditError("prohibited attribution is malformed")
    if _contains_invalid_count(attribution):
        raise SimilarityAuditError("prohibited attribution is invalid")

    unique_pairs = similarity.get("unique_prohibited_pairs")
    if (
        isinstance(unique_pairs, bool)
        or not isinstance(unique_pairs, int)
        or unique_pairs < 0
        or sum(attribution.values()) != unique_pairs
    ):
        raise SimilarityAuditError("prohibited attribution does not reconcile")
    return attribution


def _validate_pair_counts(similarity: Mapping[str, object]) -> dict[str, int]:
    counts = {}
    for key in _PAIR_COUNT_FIELDS:
        value = similarity.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SimilarityAuditError(f"{key} is invalid")
        counts[key] = value
    return counts


def _contains_invalid_count(counts: Mapping[str, object]) -> bool:
    return any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counts.values()
    )


def _validate_overall_report(overall: object, held_out: _SimilarityEvidence) -> None:
    if not isinstance(overall, dict):
        raise SimilarityAuditError("overall strategy report is malformed")
    if (
        overall.get("held_out_queries_with_prohibited_match")
        != held_out.prohibited_queries
        or overall.get("held_out_query_count") != held_out.query_count
    ):
        raise SimilarityAuditError("overall strategy totals do not reconcile")
    expected_rate = (
        f"{(Decimal(held_out.prohibited_queries) * 100 / held_out.query_count):.6f}"
    )
    if overall.get("prohibited_query_rate_percent") != expected_rate:
        raise SimilarityAuditError("overall strategy rate does not reconcile")
    if any(overall.get(key) != value for key, value in held_out.pair_counts.items()):
        raise SimilarityAuditError("overall pair counts do not reconcile")
    if overall.get("prohibited_pair_attribution") != dict(held_out.attribution):
        raise SimilarityAuditError("overall attribution does not reconcile")
    if overall.get("closest_residual_categories") != dict(
        held_out.closest_categories
    ):
        raise SimilarityAuditError("overall closest categories do not reconcile")
    if overall.get("held_out_query_status_categories") != dict(
        held_out.status_categories
    ):
        raise SimilarityAuditError("overall query statuses do not reconcile")
