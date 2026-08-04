"""Historical diagnostic report assembly, rendering, and publication."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from protein_lm.data.fixed_budget_audit.errors import AuditValidationError
from protein_lm.data.random_split import sha256_sidecar
from protein_lm.data.similarity_alignment import (
    CLOSEST_RESIDUAL_CATEGORIES,
    RESIDUAL_CATEGORIES,
)
from protein_lm.data.similarity_audit_policy import (
    APPROVED_SIMILARITY_AUDIT_CONFIG_SHA256,
    SimilarityAuditPolicy,
)
from protein_lm.data.similarity_fastas import MaterializedInputs
from protein_lm.data.similarity_manifests import (
    PARTITIONS as _ALL_PARTITIONS,
    STRATEGIES as _ALL_STRATEGIES,
    StrategyManifest,
)
from protein_lm.data.task5_report import (
    CompletedPublicArtifact,
    render_completion_index,
)

__all__ = [
    "DIAGNOSTIC_OUTPUT_STEM",
    "DIAGNOSTIC_REPORT_FILENAMES",
    "DIAGNOSTIC_COMPLETION_FILENAME",
    "DIAGNOSTIC_COMPLETION_SCOPE",
    "RenderedDiagnosticReport",
    "diagnostic_public_pass_evidence",
    "diagnostic_structural_membership_evidence",
    "diagnostic_overall_similarity",
    "build_diagnostic_report",
    "render_diagnostic_report",
    "publish_diagnostic_report",
]


# Historical diagnostic reporting


# Diagnostic records and constants


@dataclass(frozen=True)
class RenderedDiagnosticReport:
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


DIAGNOSTIC_OUTPUT_STEM = "task_07_diagnostic_similarity_audit"
DIAGNOSTIC_REPORT_FILENAMES = (
    f"{DIAGNOSTIC_OUTPUT_STEM}.json",
    f"{DIAGNOSTIC_OUTPUT_STEM}.md",
    f"{DIAGNOSTIC_OUTPUT_STEM}.sha256",
)
DIAGNOSTIC_COMPLETION_FILENAME = f"{DIAGNOSTIC_OUTPUT_STEM}.complete.json"
DIAGNOSTIC_COMPLETION_SCOPE = "week_01_task_07_public_outputs"
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


# Diagnostic assembly


def diagnostic_public_pass_evidence(
    marker: Mapping[str, object],
) -> dict[str, object]:
    """Remove private query identifiers from one pass's public evidence."""

    convergence = dict(marker["convergence"])
    convergence.pop("escalated_query_ids", None)
    accepted = dict(marker["accepted"])
    return {
        "query_count": marker["query_count"],
        "stages": marker["stages"],
        "convergence": convergence,
        "accepted": accepted,
    }


def diagnostic_structural_membership_evidence(
    strategy: str,
    manifest: StrategyManifest,
    frozen_report: Mapping[str, object],
) -> dict[str, object]:
    """Build structural membership evidence for one split strategy."""

    evidence = asdict(manifest.structural_audit)
    evidence["record_retention_percent"] = "100.000000"
    evidence["residue_retention_percent"] = "100.000000"
    evidence["exclusion_reasons"] = {}
    if strategy == "group_aware":
        assignment_units = frozen_report.get("assignment_units")
        if not isinstance(assignment_units, dict):
            raise AuditValidationError("Task 6 assignment-unit evidence is malformed")
        evidence["largest_assignment_unit_records"] = assignment_units.get(
            "largest_unit_records"
        )
        evidence["largest_assignment_unit_residues"] = assignment_units.get(
            "largest_unit_residues"
        )
        evidence["largest_assignment_unit_original_groups"] = assignment_units.get(
            "largest_unit_original_groups"
        )
    return evidence


def diagnostic_overall_similarity(
    partitions: Mapping[str, object],
) -> dict[str, object]:
    """Combine validation and test aggregates without exposing row-level data."""

    held_out = [partitions[name]["similarity"] for name in ("validation", "test")]
    numerator = sum(item["held_out_queries_with_prohibited_match"] for item in held_out)
    denominator = sum(item["held_out_query_count"] for item in held_out)
    attribution = Counter()
    closest_categories = Counter(
        {category: 0 for category in CLOSEST_RESIDUAL_CATEGORIES}
    )
    status_categories = Counter({category: 0 for category in RESIDUAL_CATEGORIES})
    for item in held_out:
        attribution.update(item["prohibited_pair_attribution"])
        closest_categories.update(item["closest_residual_categories"])
        status_categories.update(item["held_out_query_status_categories"])
    return {
        "held_out_queries_with_prohibited_match": numerator,
        "held_out_query_count": denominator,
        "prohibited_query_rate_percent": (
            f"{(Decimal(numerator) * 100 / Decimal(denominator)):.6f}"
        ),
        "unique_prohibited_pairs": sum(
            item["unique_prohibited_pairs"] for item in held_out
        ),
        "prohibited_pair_attribution": dict(attribution),
        "enforcement_returned_pairs": sum(
            item["enforcement_returned_pairs"] for item in held_out
        ),
        "residual_returned_pairs": sum(
            item["residual_returned_pairs"] for item in held_out
        ),
        "unique_returned_pair_union": sum(
            item["unique_returned_pair_union"] for item in held_out
        ),
        "closest_residual_categories": dict(closest_categories),
        "held_out_query_status_categories": dict(status_categories),
    }


def build_diagnostic_report(
    *,
    policy: SimilarityAuditPolicy,
    code_revision: str,
    fingerprint: str,
    mmseqs_version: str,
    started_at: datetime,
    completed_at: datetime,
    runtime_seconds: float,
    inputs: MaterializedInputs,
    manifests: Mapping[str, StrategyManifest],
    frozen_reports: Mapping[str, Mapping[str, object]],
    database_reports: Mapping[str, object],
    strategy_reports: Mapping[str, object],
) -> dict[str, object]:
    """Assemble the complete aggregate report before rendering validation."""

    command_count, command_runtime = _completed_command_runtime(
        database_reports,
        strategy_reports,
    )
    return {
        "schema_version": policy.schema_version,
        "scope": policy.scope,
        "adjustment_id": policy.adjustment_id,
        "diagnostic_only": True,
        "diagnostic_audit_authorized": policy.diagnostic_audit_authorized,
        "diagnostic_audit_completed": True,
        "candidate_status": policy.candidate_status,
        "repair_authorized": policy.repair_authorized,
        "repair_performed": policy.repair_performed,
        "selected_split_authorized": policy.selected_split_authorized,
        "task8_membership_use_authorized": policy.task8_membership_use_authorized,
        "model_use": policy.model_use,
        "post_audit_review_required": policy.post_audit_review_required,
        "network_requests_made": False,
        "code_revision": code_revision,
        "config_sha256": APPROVED_SIMILARITY_AUDIT_CONFIG_SHA256,
        "run_fingerprint": fingerprint,
        "inputs": {
            "task4_catalog": asdict(inputs.catalog),
            "task5_public_manifest": asdict(manifests["random"].public_manifest),
            "task5_local_assignment": asdict(manifests["random"].local_assignment),
            "task5_report_sha256": policy.task5_report_sha256,
            "task6_public_manifest": asdict(manifests["group_aware"].public_manifest),
            "task6_local_assignment": asdict(manifests["group_aware"].local_assignment),
            "task6_report_sha256": policy.task6_report_sha256,
            "task6_repair_state_sha256": policy.task6_repair_state_sha256,
            "source_checksums": frozen_reports["random"]["sources"],
            "materialized_fastas": {
                strategy: {
                    partition: asdict(inputs.fastas[strategy][partition])
                    for partition in _ALL_PARTITIONS
                }
                for strategy in _ALL_STRATEGIES
            },
        },
        "procedure": {
            "mmseqs_executable": policy.mmseqs_executable,
            "mmseqs_version": mmseqs_version,
            "threads": policy.threads,
            "format_output": policy.format_output,
            "inclusive_prohibited_boundary": {
                "minimum_identity": policy.prohibited_min_sequence_identity,
                "minimum_query_coverage": policy.prohibited_min_query_coverage,
                "minimum_target_coverage": policy.prohibited_min_target_coverage,
            },
            "prohibited_pair_evidence": policy.prohibited_pair_evidence,
            "staged_caps": [
                policy.initial_cap,
                policy.comparison_cap,
                policy.escalation_cap,
            ],
            "boundary_fixtures_passed": True,
            "target_databases": database_reports,
        },
        "runtime": {
            "final_invocation_started_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "final_invocation_wall_clock_seconds": f"{runtime_seconds:.3f}",
            "completed_mmseqs_command_count": command_count,
            "completed_mmseqs_command_runtime_seconds": command_runtime,
            "hardware": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "logical_cpu_count": os.cpu_count(),
            },
            "workspace_byte_ceiling": policy.workspace_byte_ceiling,
            "free_space_reserve": policy.free_space_reserve,
        },
        "strategies": dict(strategy_reports),
        "limitations": [
            "Only validation-to-training and test-to-training were searched.",
            "Validation-to-test similarity was not measured.",
            "The heuristic search cannot prove the absence of homology.",
            "The failed-balance candidate has shorter held-out proteins on average.",
            "The diagnostic cannot repair, select, or authorize model use of a split.",
        ],
    }


def _completed_command_runtime(
    database_reports: Mapping[str, object],
    strategy_reports: Mapping[str, object],
) -> tuple[int, str]:
    durations = []
    for raw_database in database_reports.values():
        if not isinstance(raw_database, dict):
            raise AuditValidationError("database runtime evidence is malformed")
        durations.append(Decimal(str(raw_database.get("runtime_seconds"))))
    for raw_strategy in strategy_reports.values():
        if not isinstance(raw_strategy, dict):
            raise AuditValidationError("strategy runtime evidence is malformed")
        partitions = raw_strategy["partitions"]
        for partition in ("validation", "test"):
            for raw_pass in partitions[partition]["passes"].values():
                for raw_stage in raw_pass["stages"].values():
                    durations.append(Decimal(str(raw_stage["runtime_seconds"])))
    if any(not duration.is_finite() or duration < 0 for duration in durations):
        raise AuditValidationError("command runtime evidence is invalid")
    return len(durations), f"{sum(durations, Decimal('0')):.3f}"


# Diagnostic validation and rendering


def render_diagnostic_report(
    report: Mapping[str, object],
) -> RenderedDiagnosticReport:
    """Validate the authority boundary and render the final aggregate report."""

    _validate_authority_guards(report)
    strategies = _validated_strategies(report)
    _validate_aggregate_report(strategies)

    report_dict = dict(report)
    json_text = json.dumps(report_dict, indent=2, sort_keys=True) + "\n"
    markdown_text = _render_markdown(strategies)
    return RenderedDiagnosticReport(
        json_text=json_text,
        markdown_text=markdown_text,
        json_sha256=hashlib.sha256(json_text.encode("utf-8")).hexdigest(),
    )


def _validate_authority_guards(
    report: Mapping[str, object],
) -> None:
    drift = [
        f"{key}: found {report.get(key)!r}, expected {expected!r}"
        for key, expected in _REQUIRED_AUTHORITY_GUARDS.items()
        if report.get(key) != expected
    ]
    if drift:
        raise AuditValidationError("Task 7 authority guard failed: " + "; ".join(drift))


def _validated_strategies(
    report: Mapping[str, object],
) -> dict[str, object]:
    strategies = report.get("strategies")
    if not isinstance(strategies, dict) or set(strategies) != {
        "random",
        "group_aware",
    }:
        raise AuditValidationError("Task 7 report must contain both strategies")
    return strategies


def _render_markdown(strategies: Mapping[str, object]) -> str:
    lines = []
    lines.extend(_comparison_section(strategies))
    lines.extend(_structural_membership_section(strategies))
    lines.extend(_query_status_section(strategies))
    lines.extend(_closest_residual_section(strategies))
    lines.extend(_interpretation_section())
    return "\n".join(lines)


def _comparison_section(
    strategies: Mapping[str, object],
) -> list[str]:
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
            raise AuditValidationError("strategy report is malformed")
        partitions = strategy_report["partitions"]
        for partition in _HELD_OUT_PARTITIONS:
            lines.append(
                _partition_comparison_row(
                    strategy,
                    partition,
                    partitions,
                )
            )
        lines.append(
            _overall_comparison_row(
                strategy,
                strategy_report,
                partitions,
            )
        )
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


def _structural_membership_section(
    strategies: Mapping[str, object],
) -> list[str]:
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


def _query_status_section(
    strategies: Mapping[str, object],
) -> list[str]:
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


def _closest_residual_section(
    strategies: Mapping[str, object],
) -> list[str]:
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


def _validate_aggregate_report(
    strategies: Mapping[str, object],
) -> None:
    for strategy_name, raw_strategy in strategies.items():
        _validate_strategy_report(strategy_name, raw_strategy)


def _validate_strategy_report(
    strategy_name: str,
    raw_strategy: object,
) -> None:
    if not isinstance(raw_strategy, dict):
        raise AuditValidationError(f"{strategy_name} report is malformed")

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
        raise AuditValidationError(f"{strategy_name} partitions are malformed")

    held_out = _validate_held_out_partitions(partitions)
    _validate_overall_report(raw_strategy.get("overall"), held_out)


def _validate_structural_evidence(
    strategy_name: str,
    structural: object,
) -> None:
    if not isinstance(structural, dict) or any(
        isinstance(structural.get(key), bool)
        or not isinstance(structural.get(key), int)
        or structural[key] < 0
        for key in _STRUCTURAL_COUNT_FIELDS
    ):
        raise AuditValidationError(f"{strategy_name} structural evidence is malformed")


def _validate_held_out_partitions(
    partitions: Mapping[str, object],
) -> _SimilarityEvidence:
    total_numerator = 0
    total_denominator = 0
    summed_counts = Counter()
    summed_attribution = Counter()
    summed_closest = Counter({category: 0 for category in CLOSEST_RESIDUAL_CATEGORIES})
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


def _validate_partition_report(
    raw_partition: object,
) -> _SimilarityEvidence:
    if not isinstance(raw_partition, dict):
        raise AuditValidationError("held-out partition report is malformed")
    similarity = raw_partition.get("similarity")
    if not isinstance(similarity, dict):
        raise AuditValidationError("partition similarity report is malformed")

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


def _validate_violation_counts(
    similarity: Mapping[str, object],
) -> tuple[int, int]:
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
        raise AuditValidationError("partition violation counts are invalid")
    expected_rate = f"{(Decimal(numerator) * 100 / denominator):.6f}"
    if similarity.get("prohibited_query_rate_percent") != expected_rate:
        raise AuditValidationError("partition violation rate does not reconcile")
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
        raise AuditValidationError("closest-hit categories are malformed")
    if _contains_invalid_count(closest) or sum(closest.values()) != denominator:
        raise AuditValidationError("closest-hit categories do not reconcile")

    statuses = similarity.get("held_out_query_status_categories")
    if not isinstance(statuses, dict) or set(statuses) != set(RESIDUAL_CATEGORIES):
        raise AuditValidationError("held-out query statuses are malformed")
    if _contains_invalid_count(statuses) or sum(statuses.values()) != denominator:
        raise AuditValidationError("held-out query statuses do not reconcile")
    if statuses["prohibited"] != numerator:
        raise AuditValidationError(
            "prohibited query status does not match the primary numerator"
        )
    if closest["closest_match_prohibited"] > numerator:
        raise AuditValidationError(
            "closest prohibited count exceeds the primary numerator"
        )
    return closest, statuses


def _validate_attribution(
    similarity: Mapping[str, object],
) -> dict[str, int]:
    attribution = similarity.get("prohibited_pair_attribution")
    if not isinstance(attribution, dict) or set(attribution) != _ATTRIBUTION_CATEGORIES:
        raise AuditValidationError("prohibited attribution is malformed")
    if _contains_invalid_count(attribution):
        raise AuditValidationError("prohibited attribution is invalid")

    unique_pairs = similarity.get("unique_prohibited_pairs")
    if (
        isinstance(unique_pairs, bool)
        or not isinstance(unique_pairs, int)
        or unique_pairs < 0
        or sum(attribution.values()) != unique_pairs
    ):
        raise AuditValidationError("prohibited attribution does not reconcile")
    return attribution


def _validate_pair_counts(
    similarity: Mapping[str, object],
) -> dict[str, int]:
    counts = {}
    for key in _PAIR_COUNT_FIELDS:
        value = similarity.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AuditValidationError(f"{key} is invalid")
        counts[key] = value
    return counts


def _contains_invalid_count(counts: Mapping[str, object]) -> bool:
    return any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counts.values()
    )


def _validate_overall_report(
    overall: object,
    held_out: _SimilarityEvidence,
) -> None:
    if not isinstance(overall, dict):
        raise AuditValidationError("overall strategy report is malformed")
    if (
        overall.get("held_out_queries_with_prohibited_match")
        != held_out.prohibited_queries
        or overall.get("held_out_query_count") != held_out.query_count
    ):
        raise AuditValidationError("overall strategy totals do not reconcile")
    expected_rate = (
        f"{(Decimal(held_out.prohibited_queries) * 100 / held_out.query_count):.6f}"
    )
    if overall.get("prohibited_query_rate_percent") != expected_rate:
        raise AuditValidationError("overall strategy rate does not reconcile")
    if any(overall.get(key) != value for key, value in held_out.pair_counts.items()):
        raise AuditValidationError("overall pair counts do not reconcile")
    if overall.get("prohibited_pair_attribution") != dict(held_out.attribution):
        raise AuditValidationError("overall attribution does not reconcile")
    if overall.get("closest_residual_categories") != dict(held_out.closest_categories):
        raise AuditValidationError("overall closest categories do not reconcile")
    if overall.get("held_out_query_status_categories") != dict(
        held_out.status_categories
    ):
        raise AuditValidationError("overall query statuses do not reconcile")


# Diagnostic public publication


def publish_diagnostic_report(
    rendered: RenderedDiagnosticReport,
    workspace: Path,
    report_directory: Path,
) -> None:
    """Publish validated outputs with the completion marker promoted last."""

    staging = workspace / "public_report_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    outputs = {
        f"{DIAGNOSTIC_OUTPUT_STEM}.json": rendered.json_text.encode("utf-8"),
        f"{DIAGNOSTIC_OUTPUT_STEM}.md": rendered.markdown_text.encode("utf-8"),
        f"{DIAGNOSTIC_OUTPUT_STEM}.sha256": sha256_sidecar(
            f"{DIAGNOSTIC_OUTPUT_STEM}.json",
            rendered.json_sha256,
        ).encode("ascii"),
    }
    completed = []
    for filename, content in outputs.items():
        path = staging / filename
        path.write_bytes(content)
        completed.append(
            CompletedPublicArtifact(
                relative_path=f"reports/week_01/{filename}",
                byte_size=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            )
        )
    completion_text = render_completion_index(
        tuple(completed),
        scope=DIAGNOSTIC_COMPLETION_SCOPE,
    )
    (staging / DIAGNOSTIC_COMPLETION_FILENAME).write_text(
        completion_text,
        encoding="utf-8",
    )

    report_directory.mkdir(parents=True, exist_ok=True)
    completion_path = report_directory / DIAGNOSTIC_COMPLETION_FILENAME
    completion_path.unlink(missing_ok=True)
    for filename in DIAGNOSTIC_REPORT_FILENAMES:
        (staging / filename).replace(report_directory / filename)
    (staging / DIAGNOSTIC_COMPLETION_FILENAME).replace(completion_path)
    staging.rmdir()
