"""Build and validate the decision-relevant A-004 report payload."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from decimal import Decimal

from protein_lm.data.similarity_audit_policy import SimilarityAuditError
from protein_lm.data.task7_a004_aggregation import PairUnionBundle, TrackEvidence
from protein_lm.data.task7_a004_policy import A004Policy
from protein_lm.data.task7_a004_runtime import validate_hardware_provenance
from protein_lm.data.task7_pair_union import PairUnionEvidence

COMMON_RESULT = "common_all_query_10000"
STAGED_RESULT = "staged_union_with_changed_query_100000"
_STRATEGIES = ("group_aware", "random")
_PARTITIONS = ("test", "validation")
_PASSES = ("enforcement", "residual")


def build_report_payload(
    *,
    fingerprint: str,
    policy: A004Policy,
    hardware: Mapping[str, object],
    assignment_balances: Mapping[str, object],
    assignments_unchanged: bool,
    tracks: Mapping[tuple[str, str, str], TrackEvidence],
    unions: Mapping[tuple[str, str], PairUnionBundle],
) -> dict[str, object]:
    """Build one deterministic report object from verified aggregate evidence."""

    track_records = [_track_record(track) for _, track in sorted(tracks.items())]
    partition_records = [
        _partition_record(key, bundle, tracks) for key, bundle in sorted(unions.items())
    ]
    return {
        "schema_version": 1,
        "stage": "a004_report",
        "fingerprint": fingerprint,
        "scope": policy.scope,
        "adjustment_id": policy.adjustment_id,
        "read_only": policy.read_only,
        "model_use": policy.model_use,
        "task8_membership_use_authorized": policy.task8_membership_use_authorized,
        "diagnostic_assignments_unchanged": assignments_unchanged,
        "hardware": dict(hardware),
        "result_semantics": {
            "common_result_name": COMMON_RESULT,
            "staged_result_name": STAGED_RESULT,
            "staged_cap_applies_to_all_queries": False,
            "negative_query_meaning": (
                "no prohibited pair detected through the query's highest executed cap"
            ),
        },
        "assignment_balance": assignment_balances,
        "tracks": track_records,
        "partition_results": partition_records,
        "limitations": [
            "Every prohibited-match numerator is a lower bound under the fixed search budget.",
            "The staged result adds 100000-cap evidence only for changed queries.",
            "Detected overlap is not an exhaustive biological relationship inventory.",
            "Length-distribution differences remain descriptive limitations.",
        ],
    }


def validate_report_payload(payload: Mapping[str, object], *, fingerprint: str) -> None:
    """Validate report schema, inventories, rates, sources, and semantics."""

    expected_fields = {
        "schema_version",
        "stage",
        "fingerprint",
        "scope",
        "adjustment_id",
        "read_only",
        "model_use",
        "task8_membership_use_authorized",
        "diagnostic_assignments_unchanged",
        "hardware",
        "result_semantics",
        "assignment_balance",
        "tracks",
        "partition_results",
        "limitations",
    }
    expected_authority = {
        "schema_version": 1,
        "stage": "a004_report",
        "fingerprint": fingerprint,
        "adjustment_id": "A-004",
        "read_only": True,
        "model_use": "prohibited",
        "task8_membership_use_authorized": False,
        "diagnostic_assignments_unchanged": True,
        "scope": "week_01_task_07_read_only_fixed_budget_audit",
    }
    if set(payload) != expected_fields or any(
        payload.get(name) != value for name, value in expected_authority.items()
    ):
        raise SimilarityAuditError("A-004 report authority or schema drifted")
    hardware = payload.get("hardware")
    if not isinstance(hardware, dict):
        raise SimilarityAuditError("A-004 report hardware is malformed")
    validate_hardware_provenance(hardware)
    if payload.get("result_semantics") != {
        "common_result_name": COMMON_RESULT,
        "staged_result_name": STAGED_RESULT,
        "staged_cap_applies_to_all_queries": False,
        "negative_query_meaning": (
            "no prohibited pair detected through the query's highest executed cap"
        ),
    }:
        raise SimilarityAuditError("A-004 result semantics drifted")
    tracks = payload.get("tracks")
    partitions = payload.get("partition_results")
    if not isinstance(tracks, list) or len(tracks) != 8:
        raise SimilarityAuditError("A-004 report track inventory drifted")
    if not isinstance(partitions, list) or len(partitions) != 4:
        raise SimilarityAuditError("A-004 report partition inventory drifted")
    expected_tracks = {
        (strategy, partition, pass_name)
        for strategy in _STRATEGIES
        for partition in _PARTITIONS
        for pass_name in _PASSES
    }
    observed_tracks = {_validate_track_record(track) for track in tracks}
    if observed_tracks != expected_tracks:
        raise SimilarityAuditError("A-004 report track identities drifted")
    observed_partitions = {_validate_partition_record(item) for item in partitions}
    expected_partitions = {
        (strategy, partition)
        for strategy in _STRATEGIES
        for partition in _PARTITIONS
    }
    if observed_partitions != expected_partitions:
        raise SimilarityAuditError("A-004 report partition identities drifted")
    balance = payload.get("assignment_balance")
    if not isinstance(balance, dict) or set(balance) != {"random", "group_aware"}:
        raise SimilarityAuditError("A-004 assignment-balance evidence is malformed")
    limitations = payload.get("limitations")
    if not isinstance(limitations, list) or len(limitations) < 4:
        raise SimilarityAuditError("A-004 report limitations are incomplete")
    if "all_query_100000" in json.dumps(payload, sort_keys=True):
        raise SimilarityAuditError("A-004 report makes a forbidden all-query 100k claim")


def _track_record(track: TrackEvidence) -> dict[str, object]:
    caps = {}
    by_cap = {stage.cap: stage for stage in track.stages}
    for cap, summary in sorted(track.summaries.items()):
        evidence = summary.evidence
        caps[str(cap)] = {
            "source_label": summary.source_label,
            "query_scope": (
                "changed_queries_1000_to_10000" if cap == 100_000 else "all_queries"
            ),
            "query_count": evidence.query_count,
            "returned_rows": evidence.returned_rows,
            "prohibited_pairs": evidence.prohibited_pairs,
            "prohibited_queries": evidence.prohibited_queries,
            "prohibited_query_rate": _rate(
                evidence.prohibited_queries, evidence.query_count
            ),
            "closest_categories": dict(evidence.closest_categories),
            "runtime_seconds": by_cap[cap].runtime_seconds,
        }
    return {
        "strategy": track.plan.strategy,
        "partition": track.plan.partition,
        "pass_name": track.plan.pass_name,
        "source_label": track.plan.origin,
        "all_query_denominator": len(track.all_query_ids),
        "changed_query_count_1000_to_10000": len(track.changed_query_ids),
        "caps": caps,
        "cap_sensitivity": [_comparison_record(value) for value in track.comparisons],
    }


def _partition_record(
    key: tuple[str, str],
    bundle: PairUnionBundle,
    tracks: Mapping[tuple[str, str, str], TrackEvidence],
) -> dict[str, object]:
    strategy, partition = key
    passes = [tracks[(strategy, partition, name)] for name in _PASSES]
    denominator = len(passes[0].all_query_ids)
    if any(item.all_query_ids != passes[0].all_query_ids for item in passes[1:]):
        raise SimilarityAuditError("A-004 pass query universes differ")
    common = bundle.common_all_query_10000.evidence
    staged = bundle.staged_union_with_changed_query_100000.evidence
    return {
        "strategy": strategy,
        "partition": partition,
        COMMON_RESULT: _union_record(common, denominator),
        STAGED_RESULT: _union_record(staged, denominator),
        "staged_additions": asdict(bundle.comparison),
    }


def _union_record(evidence: PairUnionEvidence, denominator: int) -> dict[str, object]:
    return {
        "prohibited_pairs": evidence.unique_pairs,
        "prohibited_queries": evidence.unique_queries,
        "denominator": denominator,
        "rate": _rate(evidence.unique_queries, denominator),
        "source_labels": list(evidence.source_labels),
    }


def _comparison_record(value: object) -> dict[str, object]:
    raw = asdict(value)  # type: ignore[arg-type]
    raw["complete_row_change_query_ids"] = list(raw["complete_row_change_query_ids"])
    return raw


def _rate(numerator: int, denominator: int) -> dict[str, object]:
    if denominator < 1 or numerator < 0 or numerator > denominator:
        raise SimilarityAuditError("A-004 prohibited-query rate cannot be computed")
    fraction = Decimal(numerator) / Decimal(denominator)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "fraction": format(fraction, ".8f"),
        "percent": format(fraction * 100, ".6f"),
    }


def _validate_track_record(raw: object) -> tuple[str, str, str]:
    if not isinstance(raw, dict):
        raise SimilarityAuditError("A-004 report track is malformed")
    key = (raw.get("strategy"), raw.get("partition"), raw.get("pass_name"))
    if not all(isinstance(item, str) for item in key):
        raise SimilarityAuditError("A-004 report track identity is malformed")
    expected_source = (
        "imported_a003" if key == ("random", "validation", "residual") else "executed_a004"
    )
    if raw.get("source_label") != expected_source:
        raise SimilarityAuditError("A-004 report source label is malformed")
    denominator = raw.get("all_query_denominator")
    changed = raw.get("changed_query_count_1000_to_10000")
    caps = raw.get("caps")
    if (
        isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator < 1
        or isinstance(changed, bool)
        or not isinstance(changed, int)
        or not 0 <= changed <= denominator
    ):
        raise SimilarityAuditError("A-004 report track denominator is malformed")
    expected_caps = {"1000", "10000"} | ({"100000"} if changed else set())
    if not isinstance(caps, dict) or set(caps) != expected_caps:
        raise SimilarityAuditError("A-004 report cap inventory is malformed")
    for cap, evidence in caps.items():
        _validate_cap_record(
            evidence,
            source=expected_source,
            cap=cap,
            expected_query_count=changed if cap == "100000" else denominator,
        )
    sensitivity = raw.get("cap_sensitivity")
    if not isinstance(sensitivity, list) or len(sensitivity) != (2 if changed else 1):
        raise SimilarityAuditError("A-004 cap sensitivity is malformed")
    _validate_sensitivity(
        sensitivity[0],
        baseline=1_000,
        comparison=10_000,
        compared_queries=denominator,
        expected_row_changes=changed,
    )
    if changed:
        _validate_sensitivity(
            sensitivity[1],
            baseline=10_000,
            comparison=100_000,
            compared_queries=changed,
        )
    return key  # type: ignore[return-value]


def _validate_cap_record(
    raw: object, *, source: str, cap: str, expected_query_count: int
) -> None:
    if not isinstance(raw, dict) or raw.get("source_label") != source:
        raise SimilarityAuditError("A-004 report cap source label drifted")
    query_count = raw.get("query_count")
    expected_scope = "changed_queries_1000_to_10000" if cap == "100000" else "all_queries"
    if raw.get("query_scope") != expected_scope or query_count != expected_query_count:
        raise SimilarityAuditError("A-004 report cap query scope drifted")
    rate = raw.get("prohibited_query_rate")
    _validate_rate(rate)
    if not isinstance(rate, dict) or rate.get("denominator") != query_count:
        raise SimilarityAuditError("A-004 report cap denominator drifted")
    if rate.get("numerator") != raw.get("prohibited_queries"):
        raise SimilarityAuditError("A-004 report cap numerator drifted")
    returned = raw.get("returned_rows")
    pairs = raw.get("prohibited_pairs")
    queries = raw.get("prohibited_queries")
    closest = raw.get("closest_categories")
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (returned, pairs, queries)
        )
        or pairs > returned
        or queries > query_count
        or not isinstance(closest, dict)
        or any(
            not isinstance(name, str)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            for name, count in closest.items()
        )
        or sum(closest.values()) != query_count
        or not isinstance(raw.get("runtime_seconds"), str)
    ):
        raise SimilarityAuditError("A-004 report cap counts do not reconcile")


def _validate_partition_record(raw: object) -> tuple[str, str]:
    if not isinstance(raw, dict):
        raise SimilarityAuditError("A-004 partition result is malformed")
    key = (raw.get("strategy"), raw.get("partition"))
    if not all(isinstance(item, str) for item in key):
        raise SimilarityAuditError("A-004 partition identity is malformed")
    for name in (COMMON_RESULT, STAGED_RESULT):
        result = raw.get(name)
        if not isinstance(result, dict):
            raise SimilarityAuditError("A-004 named partition result is missing")
        rate = result.get("rate")
        _validate_rate(rate)
        if not isinstance(rate, dict) or result.get("denominator") != rate.get("denominator"):
            raise SimilarityAuditError("A-004 partition denominator drifted")
        if result.get("prohibited_queries") != rate.get("numerator"):
            raise SimilarityAuditError("A-004 partition numerator drifted")
        if (
            isinstance(result.get("prohibited_pairs"), bool)
            or not isinstance(result.get("prohibited_pairs"), int)
            or result["prohibited_pairs"] < 0
            or not isinstance(result.get("source_labels"), list)
            or not result["source_labels"]
            or any(not isinstance(label, str) or not label for label in result["source_labels"])
        ):
            raise SimilarityAuditError("A-004 partition evidence is malformed")
    _validate_staged_additions(raw)
    return key  # type: ignore[return-value]


def _validate_staged_additions(raw: Mapping[str, object]) -> None:
    common = raw[COMMON_RESULT]
    staged = raw[STAGED_RESULT]
    additions = raw.get("staged_additions")
    if not isinstance(common, dict) or not isinstance(staged, dict) or not isinstance(
        additions, dict
    ):
        raise SimilarityAuditError("A-004 staged-addition evidence is malformed")
    expected = {
        "common_pairs": common["prohibited_pairs"],
        "staged_pairs": staged["prohibited_pairs"],
        "additional_pairs": staged["prohibited_pairs"] - common["prohibited_pairs"],
        "common_queries": common["prohibited_queries"],
        "staged_queries": staged["prohibited_queries"],
        "newly_prohibited_queries": (
            staged["prohibited_queries"] - common["prohibited_queries"]
        ),
    }
    if additions != expected or any(value < 0 for value in expected.values()):
        raise SimilarityAuditError("A-004 staged-addition counts do not reconcile")


def _validate_sensitivity(
    raw: object,
    *,
    baseline: int,
    comparison: int,
    compared_queries: int,
    expected_row_changes: int | None = None,
) -> None:
    if not isinstance(raw, dict) or (
        raw.get("baseline_cap"),
        raw.get("comparison_cap"),
        raw.get("compared_queries"),
    ) != (baseline, comparison, compared_queries):
        raise SimilarityAuditError("A-004 cap-sensitivity transition drifted")
    row_changes = raw.get("complete_row_changes")
    changed_ids = raw.get("complete_row_change_query_ids")
    counts = (
        row_changes,
        raw.get("newly_prohibited_queries"),
        raw.get("no_longer_prohibited_queries"),
        raw.get("closest_category_changes"),
    )
    if (
        any(isinstance(value, bool) or not isinstance(value, int) for value in counts)
        or any(not 0 <= value <= compared_queries for value in counts)
        or not isinstance(changed_ids, list)
        or len(changed_ids) != row_changes
        or len(set(changed_ids)) != len(changed_ids)
        or any(not isinstance(value, str) or not value for value in changed_ids)
        or (expected_row_changes is not None and row_changes != expected_row_changes)
    ):
        raise SimilarityAuditError("A-004 cap-sensitivity counts do not reconcile")


def _validate_rate(raw: object) -> None:
    if not isinstance(raw, dict):
        raise SimilarityAuditError("A-004 report rate is malformed")
    numerator = raw.get("numerator")
    denominator = raw.get("denominator")
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator < 1
        or not 0 <= numerator <= denominator
        or raw != _rate(numerator, denominator)
    ):
        raise SimilarityAuditError("A-004 report rate does not reconcile")
