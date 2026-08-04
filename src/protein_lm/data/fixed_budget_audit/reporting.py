"""A-004 report, receipt, and completion publication contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from protein_lm.data.artifacts import (
    file_identity,
    read_json,
    require_marker_identity,
    verify_file,
    write_json_atomic,
)
from protein_lm.data.fixed_budget_audit.config import (
    A004Policy,
    AuditPass,
    CandidateCap,
    DatasetPartition,
    PairUnionKind,
    QueryScope,
    SplitStrategy,
    TrackOrigin,
)
from protein_lm.data.fixed_budget_audit.errors import (
    AuditPublicationError,
    AuditValidationError,
)
from protein_lm.data.fixed_budget_audit.provenance import (
    validate_hardware_provenance,
)

if TYPE_CHECKING:
    from protein_lm.data.fixed_budget_audit.evidence import PairUnionEvidence
    from protein_lm.data.fixed_budget_audit.search import A004Database, FixedBudgetStage
    from protein_lm.data.fixed_budget_audit.source import A003Import, ImportedStage
    from protein_lm.data.fixed_budget_audit.tracks import (
        PairUnionBundle,
        TrackEvidence,
    )

__all__ = [
    "COMMON_RESULT",
    "STAGED_RESULT",
    "ReportPublication",
    "ReceiptPublication",
    "CompletionAuthorization",
    "build_report_payload",
    "validate_report_payload",
    "render_markdown_report",
    "publish_a004_report",
    "verify_report_publication",
    "publish_receipt",
    "verify_receipt_publication",
    "publish_completion_marker",
]


# A004 fixed-budget reporting


# Records and constants


@dataclass(frozen=True)
class ReportPublication:
    """Expected identities and payload for one published A-004 report."""

    directory: Path
    json_path: Path
    markdown_path: Path
    marker_path: Path
    payload: Mapping[str, object]
    json_identity: Mapping[str, object]
    markdown_identity: Mapping[str, object]
    marker_identity: Mapping[str, object]


@dataclass(frozen=True)
class ReceiptPublication:
    """Expected bytes and payload for the deterministic A-004 receipt."""

    path: Path
    payload: Mapping[str, object]
    identity: Mapping[str, object]


@dataclass(frozen=True)
class CompletionAuthorization:
    """Disk-backed final-validation result required for completion."""

    fingerprint: str
    receipt_identity: Mapping[str, object]
    report_identities: Mapping[str, Mapping[str, object]]


COMMON_RESULT = PairUnionKind.COMMON_ALL_QUERY_10000.value
STAGED_RESULT = PairUnionKind.STAGED_UNION_WITH_CHANGED_QUERY_100000.value
_STRATEGIES = tuple(item.value for item in SplitStrategy)
_PARTITIONS = tuple(item.value for item in DatasetPartition)
_PASSES = tuple(item.value for item in AuditPass)
_SOURCES = frozenset(item.value for item in TrackOrigin)
_RESULT_NAMES = tuple(item.value for item in PairUnionKind)
_QUERY_SCOPES = frozenset(item.value for item in QueryScope)


# Payload building


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

    track_records = [_report_track_record(track) for _, track in sorted(tracks.items())]
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
            "common_result_name": PairUnionKind.COMMON_ALL_QUERY_10000.value,
            "staged_result_name": (
                PairUnionKind.STAGED_UNION_WITH_CHANGED_QUERY_100000.value
            ),
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


# Payload validation


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
        raise AuditValidationError("A-004 report authority or schema drifted")
    hardware = payload.get("hardware")
    if not isinstance(hardware, dict):
        raise AuditValidationError("A-004 report hardware is malformed")
    validate_hardware_provenance(hardware)
    if payload.get("result_semantics") != {
        "common_result_name": PairUnionKind.COMMON_ALL_QUERY_10000.value,
        "staged_result_name": (
            PairUnionKind.STAGED_UNION_WITH_CHANGED_QUERY_100000.value
        ),
        "staged_cap_applies_to_all_queries": False,
        "negative_query_meaning": (
            "no prohibited pair detected through the query's highest executed cap"
        ),
    }:
        raise AuditValidationError("A-004 result semantics drifted")
    tracks = payload.get("tracks")
    partitions = payload.get("partition_results")
    if not isinstance(tracks, list) or len(tracks) != 8:
        raise AuditValidationError("A-004 report track inventory drifted")
    if not isinstance(partitions, list) or len(partitions) != 4:
        raise AuditValidationError("A-004 report partition inventory drifted")
    expected_tracks = {
        (strategy, partition, pass_name)
        for strategy in _STRATEGIES
        for partition in _PARTITIONS
        for pass_name in _PASSES
    }
    observed_tracks = {_validate_track_record(track) for track in tracks}
    if observed_tracks != expected_tracks:
        raise AuditValidationError("A-004 report track identities drifted")
    observed_partitions = {_validate_partition_record(item) for item in partitions}
    expected_partitions = {
        (strategy, partition) for strategy in _STRATEGIES for partition in _PARTITIONS
    }
    if observed_partitions != expected_partitions:
        raise AuditValidationError("A-004 report partition identities drifted")
    balance = payload.get("assignment_balance")
    if not isinstance(balance, dict) or set(balance) != set(_STRATEGIES):
        raise AuditValidationError("A-004 assignment-balance evidence is malformed")
    limitations = payload.get("limitations")
    if not isinstance(limitations, list) or len(limitations) < 4:
        raise AuditValidationError("A-004 report limitations are incomplete")
    if "all_query_100000" in json.dumps(payload, sort_keys=True):
        raise AuditValidationError(
            "A-004 report makes a forbidden all-query 100k claim"
        )


def _validate_track_record(raw: object) -> tuple[str, str, str]:
    if not isinstance(raw, dict):
        raise AuditValidationError("A-004 report track is malformed")
    key = (raw.get("strategy"), raw.get("partition"), raw.get("pass_name"))
    if not all(isinstance(item, str) for item in key):
        raise AuditValidationError("A-004 report track identity is malformed")
    expected_source = (
        TrackOrigin.IMPORTED_A003.value
        if key
        == (
            SplitStrategy.RANDOM.value,
            DatasetPartition.VALIDATION.value,
            AuditPass.RESIDUAL.value,
        )
        else TrackOrigin.EXECUTED_A004.value
    )
    if (
        raw.get("source_label") not in _SOURCES
        or raw.get("source_label") != expected_source
    ):
        raise AuditValidationError("A-004 report source label is malformed")
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
        raise AuditValidationError("A-004 report track denominator is malformed")
    initial_cap = str(CandidateCap.INITIAL.value)
    comparison_cap = str(CandidateCap.COMPARISON.value)
    escalation_cap = str(CandidateCap.ESCALATION.value)
    expected_caps = {initial_cap, comparison_cap} | (
        {escalation_cap} if changed else set()
    )
    if not isinstance(caps, dict) or set(caps) != expected_caps:
        raise AuditValidationError("A-004 report cap inventory is malformed")
    for cap, evidence in caps.items():
        _validate_cap_record(
            evidence,
            source=expected_source,
            cap=cap,
            expected_query_count=changed if cap == escalation_cap else denominator,
        )
    sensitivity = raw.get("cap_sensitivity")
    if not isinstance(sensitivity, list) or len(sensitivity) != (2 if changed else 1):
        raise AuditValidationError("A-004 cap sensitivity is malformed")
    _validate_sensitivity(
        sensitivity[0],
        baseline=CandidateCap.INITIAL.value,
        comparison=CandidateCap.COMPARISON.value,
        compared_queries=denominator,
        expected_row_changes=changed,
    )
    if changed:
        _validate_sensitivity(
            sensitivity[1],
            baseline=CandidateCap.COMPARISON.value,
            comparison=CandidateCap.ESCALATION.value,
            compared_queries=changed,
        )
    return key  # type: ignore[return-value]


def _validate_cap_record(
    raw: object, *, source: str, cap: str, expected_query_count: int
) -> None:
    if not isinstance(raw, dict) or raw.get("source_label") != source:
        raise AuditValidationError("A-004 report cap source label drifted")
    query_count = raw.get("query_count")
    expected_scope = (
        QueryScope.CHANGED_QUERIES_1000_TO_10000.value
        if cap == str(CandidateCap.ESCALATION.value)
        else QueryScope.ALL_QUERIES.value
    )
    if (
        raw.get("query_scope") not in _QUERY_SCOPES
        or raw.get("query_scope") != expected_scope
        or query_count != expected_query_count
    ):
        raise AuditValidationError("A-004 report cap query scope drifted")
    rate = raw.get("prohibited_query_rate")
    _validate_rate(rate)
    if not isinstance(rate, dict) or rate.get("denominator") != query_count:
        raise AuditValidationError("A-004 report cap denominator drifted")
    if rate.get("numerator") != raw.get("prohibited_queries"):
        raise AuditValidationError("A-004 report cap numerator drifted")
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
        raise AuditValidationError("A-004 report cap counts do not reconcile")


def _validate_partition_record(raw: object) -> tuple[str, str]:
    if not isinstance(raw, dict):
        raise AuditValidationError("A-004 partition result is malformed")
    key = (raw.get("strategy"), raw.get("partition"))
    if not all(isinstance(item, str) for item in key):
        raise AuditValidationError("A-004 partition identity is malformed")
    for name in _RESULT_NAMES:
        result = raw.get(name)
        if not isinstance(result, dict):
            raise AuditValidationError("A-004 named partition result is missing")
        rate = result.get("rate")
        _validate_rate(rate)
        if not isinstance(rate, dict) or result.get("denominator") != rate.get(
            "denominator"
        ):
            raise AuditValidationError("A-004 partition denominator drifted")
        if result.get("prohibited_queries") != rate.get("numerator"):
            raise AuditValidationError("A-004 partition numerator drifted")
        if (
            isinstance(result.get("prohibited_pairs"), bool)
            or not isinstance(result.get("prohibited_pairs"), int)
            or result["prohibited_pairs"] < 0
            or not isinstance(result.get("source_labels"), list)
            or not result["source_labels"]
            or any(
                not isinstance(label, str) or not label
                for label in result["source_labels"]
            )
        ):
            raise AuditValidationError("A-004 partition evidence is malformed")
    _validate_staged_additions(raw)
    return key  # type: ignore[return-value]


def _validate_staged_additions(raw: Mapping[str, object]) -> None:
    common = raw[PairUnionKind.COMMON_ALL_QUERY_10000.value]
    staged = raw[PairUnionKind.STAGED_UNION_WITH_CHANGED_QUERY_100000.value]
    additions = raw.get("staged_additions")
    if (
        not isinstance(common, dict)
        or not isinstance(staged, dict)
        or not isinstance(additions, dict)
    ):
        raise AuditValidationError("A-004 staged-addition evidence is malformed")
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
        raise AuditValidationError("A-004 staged-addition counts do not reconcile")


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
        raise AuditValidationError("A-004 cap-sensitivity transition drifted")
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
        raise AuditValidationError("A-004 cap-sensitivity counts do not reconcile")


def _validate_rate(raw: object) -> None:
    if not isinstance(raw, dict):
        raise AuditValidationError("A-004 report rate is malformed")
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
        raise AuditValidationError("A-004 report rate does not reconcile")


# Markdown rendering


def render_markdown_report(payload: Mapping[str, object]) -> str:
    """Render held-out, per-cap, sensitivity, and limitation tables."""

    lines = [
        "# A-004 read-only fixed-budget audit",
        "",
        "The common result covers every query through cap 10000. The staged result adds cap 100000 only for queries whose complete rows changed between caps 1000 and 10000.",
        "",
        "## Held-out results",
        "",
        "| Strategy | Partition | Result | Prohibited queries | Denominator | Rate | Prohibited pairs |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for partition in payload["partition_results"]:  # type: ignore[index]
        for name in _RESULT_NAMES:
            result = partition[name]
            lines.append(
                "| {strategy} | {partition} | `{name}` | {queries} | {denominator} | {percent}% | {pairs} |".format(
                    strategy=partition["strategy"],
                    partition=partition["partition"],
                    name=name,
                    queries=result["prohibited_queries"],
                    denominator=result["denominator"],
                    percent=result["rate"]["percent"],
                    pairs=result["prohibited_pairs"],
                )
            )
    _append_cap_table(lines, payload)
    _append_sensitivity_table(lines, payload)
    _append_staged_additions(lines, payload)
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in payload["limitations"])  # type: ignore[index]
    lines.append("")
    return "\n".join(lines)


def _append_cap_table(lines: list[str], payload: Mapping[str, object]) -> None:
    lines.extend(
        [
            "",
            "## Per-cap evidence",
            "",
            "| Strategy | Partition | Pass | Source | Cap | Query scope | Prohibited queries | Denominator | Rate | Prohibited pairs |",
            "|---|---|---|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for track in payload["tracks"]:  # type: ignore[index]
        for cap, evidence in track["caps"].items():
            rate = evidence["prohibited_query_rate"]
            lines.append(
                "| {strategy} | {partition} | {pass_name} | `{source}` | {cap} | {scope} | {queries} | {denominator} | {percent}% | {pairs} |".format(
                    strategy=track["strategy"],
                    partition=track["partition"],
                    pass_name=track["pass_name"],
                    source=track["source_label"],
                    cap=cap,
                    scope=evidence["query_scope"],
                    queries=evidence["prohibited_queries"],
                    denominator=rate["denominator"],
                    percent=rate["percent"],
                    pairs=evidence["prohibited_pairs"],
                )
            )


def _append_sensitivity_table(lines: list[str], payload: Mapping[str, object]) -> None:
    lines.extend(
        [
            "",
            "## Cap sensitivity",
            "",
            "| Strategy | Partition | Pass | Transition | Compared queries | Complete row changes | Newly prohibited | No longer prohibited | Closest-category changes |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for track in payload["tracks"]:  # type: ignore[index]
        for change in track["cap_sensitivity"]:
            lines.append(
                "| {strategy} | {partition} | {pass_name} | {baseline} to {comparison} | {queries} | {rows} | {new} | {lost} | {closest} |".format(
                    strategy=track["strategy"],
                    partition=track["partition"],
                    pass_name=track["pass_name"],
                    baseline=change["baseline_cap"],
                    comparison=change["comparison_cap"],
                    queries=change["compared_queries"],
                    rows=change["complete_row_changes"],
                    new=change["newly_prohibited_queries"],
                    lost=change["no_longer_prohibited_queries"],
                    closest=change["closest_category_changes"],
                )
            )


def _append_staged_additions(lines: list[str], payload: Mapping[str, object]) -> None:
    lines.extend(
        [
            "",
            "## Staged additions",
            "",
            "| Strategy | Partition | Additional pairs | Newly prohibited queries |",
            "|---|---|---:|---:|",
        ]
    )
    for partition in payload["partition_results"]:  # type: ignore[index]
        additions = partition["staged_additions"]
        lines.append(
            "| {strategy} | {partition} | {pairs} | {queries} |".format(
                strategy=partition["strategy"],
                partition=partition["partition"],
                pairs=additions["additional_pairs"],
                queries=additions["newly_prohibited_queries"],
            )
        )


# Report publication


def publish_a004_report(
    *,
    workspace: Path,
    fingerprint: str,
    policy: A004Policy,
    hardware: Mapping[str, object],
    assignment_balances: Mapping[str, object],
    assignments_unchanged: bool,
    tracks: Mapping[tuple[str, str, str], TrackEvidence],
    unions: Mapping[tuple[str, str], PairUnionBundle],
) -> ReportPublication:
    """Write or verify the complete A-004 JSON and Markdown report pair."""

    payload = build_report_payload(
        fingerprint=fingerprint,
        policy=policy,
        hardware=hardware,
        assignment_balances=assignment_balances,
        assignments_unchanged=assignments_unchanged,
        tracks=tracks,
        unions=unions,
    )
    validate_report_payload(payload, fingerprint=fingerprint)
    markdown = render_markdown_report(payload)
    directory = workspace / "evidence" / "report"
    marker_path = directory / "complete.json"
    json_path = directory / "a004_report.json"
    markdown_path = directory / "a004_report.md"
    if marker_path.exists():
        publication = _publication_from_disk(directory, payload, markdown, fingerprint)
        verify_report_publication(publication, fingerprint=fingerprint)
        return publication
    if directory.exists() and any(directory.iterdir()):
        raise AuditPublicationError("A-004 report output lacks its completion marker")
    staging = directory.with_name(".report.incomplete")
    if staging.exists():
        raise AuditPublicationError("A-004 report has an unmarked staging directory")
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    write_json_atomic(staging / json_path.name, payload)
    (staging / markdown_path.name).write_text(markdown, encoding="utf-8")
    marker = {
        "schema_version": 1,
        "stage": "a004_report_artifacts",
        "fingerprint": fingerprint,
        "json": file_identity(staging / json_path.name),
        "markdown": file_identity(staging / markdown_path.name),
    }
    write_json_atomic(staging / marker_path.name, marker)
    if directory.exists():
        directory.rmdir()
    staging.replace(directory)
    publication = _publication_from_disk(directory, payload, markdown, fingerprint)
    verify_report_publication(publication, fingerprint=fingerprint)
    return publication


def verify_report_publication(
    publication: ReportPublication, *, fingerprint: str
) -> None:
    """Re-read and verify report bytes, marker, schema, and expected payload."""

    marker = read_json(publication.marker_path)
    require_marker_identity(marker, fingerprint, "a004_report_artifacts")
    expected_marker = {
        "schema_version": 1,
        "stage": "a004_report_artifacts",
        "fingerprint": fingerprint,
        "json": dict(publication.json_identity),
        "markdown": dict(publication.markdown_identity),
    }
    if marker != expected_marker:
        raise AuditValidationError("A-004 report marker identity drifted")
    _verify_identity(publication.marker_path, publication.marker_identity)
    _verify_identity(publication.json_path, publication.json_identity)
    _verify_identity(publication.markdown_path, publication.markdown_identity)
    current = read_json(publication.json_path)
    validate_report_payload(current, fingerprint=fingerprint)
    if current != publication.payload:
        raise AuditValidationError("A-004 JSON report payload drifted")


# Receipt publication


def publish_receipt(
    *,
    workspace: Path,
    fingerprint: str,
    policy: A004Policy,
    source_policy_path: Path,
    code_revision: str,
    mmseqs_version: str,
    hardware: Mapping[str, object],
    assignments_before: Mapping[str, Mapping[str, object]],
    assignments_after: Mapping[str, Mapping[str, object]],
    imported: A003Import,
    databases: Mapping[str, A004Database],
    tracks: Mapping[tuple[str, str, str], TrackEvidence],
    unions: Mapping[tuple[str, str], PairUnionBundle],
    report: ReportPublication,
) -> ReceiptPublication:
    """Write or verify the deterministic receipt without completing A-004."""

    receipt_path = workspace / "a004_import_receipt.json"
    validate_hardware_provenance(hardware)
    if assignments_before != assignments_after:
        raise AuditValidationError("A-004 receipt cannot claim changed assignments")
    receipt = _normalize_json(
        _receipt_payload(
            fingerprint=fingerprint,
            policy=policy,
            source_policy_path=source_policy_path,
            code_revision=code_revision,
            mmseqs_version=mmseqs_version,
            hardware=hardware,
            assignments_before=assignments_before,
            assignments_after=assignments_after,
            imported=imported,
            databases=databases,
            tracks=tracks,
            unions=unions,
            report=report,
        )
    )
    _write_or_verify(receipt_path, receipt, "a004_import_receipt", fingerprint)
    publication = ReceiptPublication(
        path=receipt_path,
        payload=receipt,
        identity=file_identity(receipt_path),
    )
    verify_receipt_publication(publication, fingerprint=fingerprint)
    return publication


def verify_receipt_publication(
    publication: ReceiptPublication, *, fingerprint: str
) -> None:
    """Re-read and compare receipt bytes against the expected payload."""

    size = publication.identity.get("byte_size")
    digest = publication.identity.get("sha256")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or not isinstance(digest, str)
    ):
        raise AuditValidationError("A-004 receipt identity is malformed")
    verify_file(publication.path, size, digest)
    current = read_json(publication.path)
    require_marker_identity(current, fingerprint, "a004_import_receipt")
    if current != publication.payload:
        raise AuditValidationError("A-004 receipt payload drifted")


# Completion publication


def publish_completion_marker(
    *,
    workspace: Path,
    fingerprint: str,
    receipt: ReceiptPublication,
    report: ReportPublication,
    authorization: CompletionAuthorization,
) -> Path:
    """Publish only after the caller supplies the final disk-validation result."""

    expected_report_identities = {
        "json": dict(report.json_identity),
        "markdown": dict(report.markdown_identity),
        "marker": dict(report.marker_identity),
    }
    if (
        authorization.fingerprint != fingerprint
        or dict(authorization.receipt_identity) != dict(receipt.identity)
        or authorization.report_identities != expected_report_identities
    ):
        raise AuditPublicationError("A-004 completion authorization is invalid")
    completion_path = workspace / "a004_complete.json"
    completion = {
        "schema_version": 1,
        "stage": "a004_workflow_complete",
        "fingerprint": fingerprint,
        "receipt": dict(receipt.identity),
        "report": {
            "json": dict(report.json_identity),
            "markdown": dict(report.markdown_identity),
            "marker": dict(report.marker_identity),
        },
        "model_use": "prohibited",
        "task8_membership_use_authorized": False,
        "diagnostic_assignments_unchanged": True,
    }
    _write_or_verify(
        completion_path,
        completion,
        "a004_workflow_complete",
        fingerprint,
    )
    return completion_path


# Private serialization and identity helpers


def _report_track_record(track: TrackEvidence) -> dict[str, object]:
    caps = {}
    by_cap = {stage.cap: stage for stage in track.stages}
    for cap, summary in sorted(track.summaries.items()):
        evidence = summary.evidence
        caps[str(cap)] = {
            "source_label": summary.source_label,
            "query_scope": (
                QueryScope.CHANGED_QUERIES_1000_TO_10000.value
                if cap == CandidateCap.ESCALATION.value
                else QueryScope.ALL_QUERIES.value
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
        "strategy": track.plan.strategy.value,
        "partition": track.plan.partition.value,
        "pass_name": track.plan.pass_name.value,
        "source_label": track.plan.origin.value,
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
        raise AuditValidationError("A-004 pass query universes differ")
    common = bundle.common_all_query_10000.evidence
    staged = bundle.staged_union_with_changed_query_100000.evidence
    return {
        "strategy": strategy,
        "partition": partition,
        PairUnionKind.COMMON_ALL_QUERY_10000.value: _union_record(common, denominator),
        PairUnionKind.STAGED_UNION_WITH_CHANGED_QUERY_100000.value: _union_record(
            staged, denominator
        ),
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
        raise AuditValidationError("A-004 prohibited-query rate cannot be computed")
    fraction = Decimal(numerator) / Decimal(denominator)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "fraction": format(fraction, ".8f"),
        "percent": format(fraction * 100, ".6f"),
    }


def _publication_from_disk(
    directory: Path,
    payload: Mapping[str, object],
    markdown: str,
    fingerprint: str,
) -> ReportPublication:
    json_path = directory / "a004_report.json"
    markdown_path = directory / "a004_report.md"
    marker_path = directory / "complete.json"
    expected_markdown = markdown.encode("utf-8")
    if not markdown_path.is_file() or markdown_path.read_bytes() != expected_markdown:
        raise AuditValidationError("A-004 Markdown report payload drifted")
    marker = read_json(marker_path)
    require_marker_identity(marker, fingerprint, "a004_report_artifacts")
    json_identity = marker.get("json")
    markdown_identity = marker.get("markdown")
    if not isinstance(json_identity, dict) or not isinstance(markdown_identity, dict):
        raise AuditValidationError("A-004 report marker evidence is malformed")
    return ReportPublication(
        directory=directory,
        json_path=json_path,
        markdown_path=markdown_path,
        marker_path=marker_path,
        payload=payload,
        json_identity=json_identity,
        markdown_identity=markdown_identity,
        marker_identity=file_identity(marker_path),
    )


def _receipt_payload(
    *,
    fingerprint: str,
    policy: A004Policy,
    source_policy_path: Path,
    code_revision: str,
    mmseqs_version: str,
    hardware: Mapping[str, object],
    assignments_before: Mapping[str, Mapping[str, object]],
    assignments_after: Mapping[str, Mapping[str, object]],
    imported: A003Import,
    databases: Mapping[str, A004Database],
    tracks: Mapping[tuple[str, str, str], TrackEvidence],
    unions: Mapping[tuple[str, str], PairUnionBundle],
    report: ReportPublication,
) -> dict[str, object]:
    track_records = [
        _receipt_track_record(track) for _, track in sorted(tracks.items())
    ]
    return {
        "schema_version": 1,
        "stage": "a004_import_receipt",
        "fingerprint": fingerprint,
        "authority": {
            "adjustment_id": policy.adjustment_id,
            "source_adjustment_id": policy.source_adjustment_id,
            "read_only": policy.read_only,
            "repair_authorized": policy.repair_authorized,
            "selected_split_authorized": policy.selected_split_authorized,
            "model_use": policy.model_use,
            "task8_membership_use_authorized": (policy.task8_membership_use_authorized),
        },
        "code_revision": code_revision,
        "mmseqs_version": mmseqs_version,
        "hardware": dict(hardware),
        "source_policy": file_identity(source_policy_path),
        "diagnostic_assignments": {
            "unchanged": True,
            "before": assignments_before,
            "after": assignments_after,
        },
        TrackOrigin.IMPORTED_A003.value: {
            "fingerprint": imported.fingerprint,
            "fastas": {
                strategy: {
                    partition: asdict(evidence)
                    for partition, evidence in partitions.items()
                }
                for strategy, partitions in imported.fastas.items()
            },
            "database": asdict(imported.database),
            "escalated_query_ids": list(imported.escalated_query_ids),
            "stages": [_stage_record(stage) for stage in imported.stages],
        },
        "fresh_a004_databases": {
            strategy: {
                "command": list(database.command),
                "runtime_seconds": database.runtime_seconds,
                "identity": database.identity,
            }
            for strategy, database in databases.items()
        },
        "imported_tracks": [
            track
            for track in track_records
            if track["origin"] == TrackOrigin.IMPORTED_A003.value
        ],
        "executed_tracks": [
            track
            for track in track_records
            if track["origin"] == TrackOrigin.EXECUTED_A004.value
        ],
        "pair_unions": {
            f"{strategy}_{partition}": {
                PairUnionKind.COMMON_ALL_QUERY_10000.value: {
                    "marker": bundle.common_all_query_10000.marker_identity,
                    "evidence": asdict(bundle.common_all_query_10000.evidence),
                },
                PairUnionKind.STAGED_UNION_WITH_CHANGED_QUERY_100000.value: {
                    "marker": (
                        bundle.staged_union_with_changed_query_100000.marker_identity
                    ),
                    "evidence": asdict(
                        bundle.staged_union_with_changed_query_100000.evidence
                    ),
                },
                "comparison": asdict(bundle.comparison),
            }
            for (strategy, partition), bundle in sorted(unions.items())
        },
        "report": {
            "directory": str(report.directory),
            "json": dict(report.json_identity),
            "markdown": dict(report.markdown_identity),
            "marker": dict(report.marker_identity),
        },
    }


def _receipt_track_record(track: TrackEvidence) -> dict[str, object]:
    return {
        "strategy": track.plan.strategy.value,
        "partition": track.plan.partition.value,
        "pass_name": track.plan.pass_name.value,
        "origin": track.plan.origin.value,
        "all_query_count": len(track.all_query_ids),
        "changed_query_ids_between_1000_and_10000": list(track.changed_query_ids),
        "stages": [_stage_record(stage) for stage in track.stages],
        "cap_sensitivity": [asdict(comparison) for comparison in track.comparisons],
        "cap_summaries": {
            str(cap): {
                "source_label": summary.source_label,
                "marker": dict(summary.marker_identity),
                "evidence": asdict(summary.evidence),
            }
            for cap, summary in track.summaries.items()
        },
        "pass_marker": track.pass_marker_identity,
    }


def _stage_record(stage: FixedBudgetStage | ImportedStage) -> dict[str, object]:
    from protein_lm.data.fixed_budget_audit.source import (
        ImportedStage as RuntimeImportedStage,
    )

    is_imported = isinstance(stage, RuntimeImportedStage)
    marker = (
        asdict(stage.marker) if is_imported else file_identity(stage.marker_path)  # type: ignore[union-attr]
    )
    return {
        "cap": stage.cap,
        "origin": (
            TrackOrigin.IMPORTED_A003.value
            if is_imported
            else TrackOrigin.EXECUTED_A004.value
        ),
        "marker": marker,
        "query_fasta": asdict(stage.query_fasta),
        "canonical": asdict(stage.canonical),
        "command": list(stage.command),
        "runtime_seconds": stage.runtime_seconds,
    }


def _verify_identity(path: Path, identity: Mapping[str, object]) -> None:
    size = identity.get("byte_size")
    digest = identity.get("sha256")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or not isinstance(digest, str)
    ):
        raise AuditValidationError("A-004 report file identity is malformed")
    verify_file(path, size, digest)


def _write_or_verify(
    path: Path, payload: Mapping[str, object], stage: str, fingerprint: str
) -> None:
    if path.exists():
        existing = read_json(path)
        require_marker_identity(existing, fingerprint, stage)
        if existing != payload:
            raise AuditPublicationError(f"A-004 {stage} identity drifted")
        return
    write_json_atomic(path, payload)


def _normalize_json(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _normalize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    return value
