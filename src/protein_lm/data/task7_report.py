"""Render aggregate-only evidence for the Task 7 diagnostic audit."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping

from protein_lm.data.similarity_audit import (
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


def render_task7_report(report: Mapping[str, object]) -> RenderedTask7Report:
    """Validate the authority boundary and render the final aggregate report."""

    required_guards = {
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
    drift = [
        f"{key}: found {report.get(key)!r}, expected {expected!r}"
        for key, expected in required_guards.items()
        if report.get(key) != expected
    ]
    if drift:
        raise SimilarityAuditError("Task 7 authority guard failed: " + "; ".join(drift))

    strategies = report.get("strategies")
    if not isinstance(strategies, dict) or set(strategies) != {"random", "group_aware"}:
        raise SimilarityAuditError("Task 7 report must contain both strategies")
    _validate_aggregate_report(strategies)

    report_dict = dict(report)
    json_text = json.dumps(report_dict, indent=2, sort_keys=True) + "\n"
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
    for strategy in ("random", "group_aware"):
        strategy_report = strategies[strategy]
        if not isinstance(strategy_report, dict):
            raise SimilarityAuditError("strategy report is malformed")
        partitions = strategy_report["partitions"]
        for partition in ("validation", "test"):
            comparison = partitions[partition]["similarity"]
            balance = partitions[partition]["balance"]
            lines.append(
                "| "
                f"{strategy} | {partition} | "
                f"{balance['records']} ({balance['record_share_percent']}%) | "
                f"{balance['residues']} ({balance['residue_share_percent']}%) | "
                f"{comparison['held_out_queries_with_prohibited_match']} | "
                f"{comparison['held_out_query_count']} | "
                f"{comparison['prohibited_query_rate_percent']}% | "
                f"{comparison['unique_prohibited_pairs']} |"
            )
        validation_balance = partitions["validation"]["balance"]
        test_balance = partitions["test"]["balance"]
        overall = strategy_report["overall"]
        held_out_record_share = Decimal(
            validation_balance["record_share_percent"]
        ) + Decimal(test_balance["record_share_percent"])
        held_out_residue_share = Decimal(
            validation_balance["residue_share_percent"]
        ) + Decimal(test_balance["residue_share_percent"])
        lines.append(
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

    lines.extend(
        [
            "",
            "## Frozen membership structure",
            "",
            "| Strategy | Exact-hash crossings | UniRef50 crossings | Retained records | Retained residues | Excluded records | Largest group or unit (records) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for strategy in ("random", "group_aware"):
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

    lines.extend(
        [
            "",
            "## Held-out query status categories",
            "",
            "| Strategy | Partition | Category | Queries |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for strategy in ("random", "group_aware"):
        partitions = strategies[strategy]["partitions"]
        for partition in ("validation", "test"):
            categories = partitions[partition]["similarity"][
                "held_out_query_status_categories"
            ]
            for category in RESIDUAL_CATEGORIES:
                lines.append(
                    f"| {strategy} | {partition} | {category} | "
                    f"{categories[category]} |"
                )

    lines.extend(
        [
            "",
            "## Closest residual-match categories",
            "",
            "These categories describe the selected closest residual row itself. "
            "A query can be prohibited by a different returned pair.",
            "",
            "| Strategy | Partition | Category | Queries |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for strategy in ("random", "group_aware"):
        partitions = strategies[strategy]["partitions"]
        for partition in ("validation", "test"):
            categories = partitions[partition]["similarity"][
                "closest_residual_categories"
            ]
            for category in CLOSEST_RESIDUAL_CATEGORIES:
                lines.append(
                    f"| {strategy} | {partition} | {category} | "
                    f"{categories[category]} |"
                )

    lines.extend(
        [
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
    )
    markdown_text = "\n".join(lines)
    return RenderedTask7Report(
        json_text=json_text,
        markdown_text=markdown_text,
        json_sha256=hashlib.sha256(json_text.encode("utf-8")).hexdigest(),
    )


def _validate_aggregate_report(strategies: Mapping[str, object]) -> None:
    for strategy_name, raw_strategy in strategies.items():
        if not isinstance(raw_strategy, dict):
            raise SimilarityAuditError(f"{strategy_name} report is malformed")
        structural = raw_strategy.get("structural_membership")
        structural_count_fields = (
            "exact_sequence_hash_crossings",
            "uniref50_group_crossings",
            "retained_records",
            "retained_residues",
            "excluded_records",
            "excluded_residues",
            "largest_uniref50_group_records",
            "largest_uniref50_group_residues",
        )
        if not isinstance(structural, dict) or any(
            isinstance(structural.get(key), bool)
            or not isinstance(structural.get(key), int)
            or structural[key] < 0
            for key in structural_count_fields
        ):
            raise SimilarityAuditError(
                f"{strategy_name} structural evidence is malformed"
            )
        partitions = raw_strategy.get("partitions")
        if not isinstance(partitions, dict) or set(partitions) != {
            "training",
            "validation",
            "test",
        }:
            raise SimilarityAuditError(f"{strategy_name} partitions are malformed")
        total_numerator = 0
        total_denominator = 0
        summed_counts = Counter()
        summed_attribution = Counter()
        summed_closest = Counter(
            {category: 0 for category in CLOSEST_RESIDUAL_CATEGORIES}
        )
        summed_statuses = Counter({category: 0 for category in RESIDUAL_CATEGORIES})
        for partition in ("validation", "test"):
            raw_partition = partitions[partition]
            if not isinstance(raw_partition, dict):
                raise SimilarityAuditError("held-out partition report is malformed")
            similarity = raw_partition.get("similarity")
            if not isinstance(similarity, dict):
                raise SimilarityAuditError("partition similarity report is malformed")
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
            categories = similarity.get("closest_residual_categories")
            if not isinstance(categories, dict) or set(categories) != set(
                CLOSEST_RESIDUAL_CATEGORIES
            ):
                raise SimilarityAuditError("closest-hit categories are malformed")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in categories.values()
            ) or sum(categories.values()) != denominator:
                raise SimilarityAuditError("closest-hit categories do not reconcile")
            statuses = similarity.get("held_out_query_status_categories")
            if not isinstance(statuses, dict) or set(statuses) != set(
                RESIDUAL_CATEGORIES
            ):
                raise SimilarityAuditError("held-out query statuses are malformed")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in statuses.values()
            ) or sum(statuses.values()) != denominator:
                raise SimilarityAuditError("held-out query statuses do not reconcile")
            if statuses["prohibited"] != numerator:
                raise SimilarityAuditError(
                    "prohibited query status does not match the primary numerator"
                )
            if categories["closest_match_prohibited"] > numerator:
                raise SimilarityAuditError(
                    "closest prohibited count exceeds the primary numerator"
                )
            attribution = similarity.get("prohibited_pair_attribution")
            if not isinstance(attribution, dict) or set(attribution) != {
                "exact_sequence_duplicate",
                "same_uniref50_group",
                "cross_uniref50_group",
            }:
                raise SimilarityAuditError("prohibited attribution is malformed")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in attribution.values()
            ):
                raise SimilarityAuditError("prohibited attribution is invalid")
            unique_pairs = similarity.get("unique_prohibited_pairs")
            if (
                isinstance(unique_pairs, bool)
                or not isinstance(unique_pairs, int)
                or unique_pairs < 0
                or sum(attribution.values()) != unique_pairs
            ):
                raise SimilarityAuditError("prohibited attribution does not reconcile")
            total_numerator += numerator
            total_denominator += denominator
            for key in (
                "unique_prohibited_pairs",
                "enforcement_returned_pairs",
                "residual_returned_pairs",
                "unique_returned_pair_union",
            ):
                value = similarity.get(key)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise SimilarityAuditError(f"{key} is invalid")
                summed_counts[key] += value
            summed_attribution.update(attribution)
            summed_closest.update(categories)
            summed_statuses.update(statuses)

        overall = raw_strategy.get("overall")
        if not isinstance(overall, dict):
            raise SimilarityAuditError("overall strategy report is malformed")
        if (
            overall.get("held_out_queries_with_prohibited_match") != total_numerator
            or overall.get("held_out_query_count") != total_denominator
        ):
            raise SimilarityAuditError("overall strategy totals do not reconcile")
        expected_overall_rate = (
            f"{(Decimal(total_numerator) * 100 / total_denominator):.6f}"
        )
        if overall.get("prohibited_query_rate_percent") != expected_overall_rate:
            raise SimilarityAuditError("overall strategy rate does not reconcile")
        if any(overall.get(key) != value for key, value in summed_counts.items()):
            raise SimilarityAuditError("overall pair counts do not reconcile")
        if overall.get("prohibited_pair_attribution") != dict(summed_attribution):
            raise SimilarityAuditError("overall attribution does not reconcile")
        if overall.get("closest_residual_categories") != dict(summed_closest):
            raise SimilarityAuditError("overall closest categories do not reconcile")
        if overall.get("held_out_query_status_categories") != dict(summed_statuses):
            raise SimilarityAuditError("overall query statuses do not reconcile")
