"""Build deterministic aggregate-only output for the Week 1 Task 2 audit."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from protein_lm.data.corpus_audit import SwissProtAudit, audit_swiss_prot
from protein_lm.data.proteingym import (
    ProteinGymAssay,
    ProteinGymMetadataAudit,
    ProteinGymScan,
    scan_proteingym_metadata,
)
from protein_lm.data.uniprot import SwissProtRecord, parse_swiss_prot
from protein_lm.data.uniref import (
    UniRef50Audit,
    UniRef50Scan,
    scan_uniref50_membership,
)

TASK2_SCHEMA_VERSION = 1
TASK2_SCOPE = "week_01_task_2_aggregate_only"
_REQUIRED_SOURCE_ROLES = frozenset(
    {"swiss_prot_records", "uniref50_membership", "proteingym_metadata"}
)
_REFERENCE_STATUSES = (
    "exact_swiss_prot",
    "mismatched_swiss_prot",
    "outside_swiss_prot",
)
_UNIREF_STATUSES = (
    "mapped_in_swiss_prot",
    "mapped_outside_swiss_prot",
    "missing",
    "blank_group",
    "conflicting",
    "inconsistent_accession",
)
_MARKDOWN_JSON_ONLY_PATHS = frozenset(
    {
        ("swiss_prot", "length_histogram"),
        ("swiss_prot", "complete_ec_label_counts"),
        ("uniref50", "group_size_histogram"),
        ("proteingym_metadata", "assay_reference_length_histogram"),
    }
)


class Task2AuditError(ValueError):
    """Raised when Task 2 inputs or cross-source facts are inconsistent."""


@dataclass(frozen=True)
class SourceEvidence:
    """Aggregate-safe identity for one verified source artifact."""

    release: str
    filename: str
    byte_size: int
    sha256: str
    upstream_checksum_algorithm: str
    upstream_checksum: str
    license_spdx: str
    retrieval_date: str
    retrieval_method: str


@dataclass(frozen=True)
class ProteinGymSupportAudit:
    """Aggregate cross-source support for future ProteinGym panel selection."""

    target_location_counts: dict[str, int]
    assay_reference_status_counts: dict[str, int]
    target_uniref50_status_counts: dict[str, int]
    assay_uniref50_status_counts: dict[str, int]
    duplicate_mapping_target_count: int
    reservable_target_count: int
    reservable_assay_count: int
    unique_reservable_family_count: int


@dataclass(frozen=True)
class Task2AuditReport:
    """The complete deterministic and aggregate-only Task 2 result."""

    schema_version: int
    scope: str
    code_revision: str
    sources: dict[str, SourceEvidence]
    swiss_prot: SwissProtAudit
    uniref50: UniRef50Audit
    proteingym_metadata: ProteinGymMetadataAudit
    proteingym_support: ProteinGymSupportAudit


@dataclass(frozen=True)
class RenderedTask2Audit:
    """Deterministic report bytes and the JSON checksum."""

    json_text: str
    markdown_text: str
    json_sha256: str


def build_task2_audit(
    *,
    swiss_prot_path: Path,
    uniref50_path: Path,
    proteingym_path: Path,
    sources: Mapping[str, SourceEvidence],
    code_revision: str,
) -> Task2AuditReport:
    """Parse all three sources and return only approved aggregate facts."""

    if set(sources) != _REQUIRED_SOURCE_ROLES:
        raise Task2AuditError(
            "source evidence must contain exactly: "
            + ", ".join(sorted(_REQUIRED_SOURCE_ROLES))
        )
    if not code_revision:
        raise Task2AuditError("code revision must not be empty")

    proteingym_scan = scan_proteingym_metadata(proteingym_path)
    target_entry_names = frozenset(proteingym_scan.target_entry_names)
    accessions: list[str] = []
    matched_swiss_prot_records: dict[str, SwissProtRecord] = {}

    observed_records = _observe_swiss_prot_records(
        parse_swiss_prot(swiss_prot_path),
        target_entry_names=target_entry_names,
        accessions=accessions,
        matched_records=matched_swiss_prot_records,
    )
    swiss_prot_audit = audit_swiss_prot(observed_records)

    uniref50_scan = scan_uniref50_membership(
        uniref50_path,
        accessions,
        target_entry_names=proteingym_scan.target_entry_names,
    )
    proteingym_support = audit_proteingym_support(
        proteingym_scan,
        matched_swiss_prot_records=matched_swiss_prot_records,
        uniref50_scan=uniref50_scan,
    )

    return Task2AuditReport(
        schema_version=TASK2_SCHEMA_VERSION,
        scope=TASK2_SCOPE,
        code_revision=code_revision,
        sources=dict(sorted(sources.items())),
        swiss_prot=swiss_prot_audit,
        uniref50=uniref50_scan.audit,
        proteingym_metadata=proteingym_scan.audit,
        proteingym_support=proteingym_support,
    )


def audit_proteingym_support(
    proteingym_scan: ProteinGymScan,
    *,
    matched_swiss_prot_records: Mapping[str, SwissProtRecord],
    uniref50_scan: UniRef50Scan,
) -> ProteinGymSupportAudit:
    """Cross-check ProteinGym targets against Swiss-Prot and UniRef50."""

    target_names = frozenset(proteingym_scan.target_entry_names)
    unexpected_swiss_names = set(matched_swiss_prot_records) - target_names
    if unexpected_swiss_names:
        raise Task2AuditError("Swiss-Prot matches contain unexpected entry names")

    target_location_counts: Counter[str] = Counter()
    target_statuses: dict[str, str] = {}
    reservable_families: set[str] = set()

    for entry_name in proteingym_scan.target_entry_names:
        swiss_record = matched_swiss_prot_records.get(entry_name)
        target_location_counts[
            "in_swiss_prot" if swiss_record is not None else "outside_swiss_prot"
        ] += 1

        status, family = _classify_uniref50_support(
            entry_name=entry_name,
            swiss_record=swiss_record,
            uniref50_scan=uniref50_scan,
        )
        target_statuses[entry_name] = status
        if family is not None:
            reservable_families.add(family)

    assay_reference_statuses: Counter[str] = Counter()
    assay_uniref50_statuses: Counter[str] = Counter()
    for assay in proteingym_scan.assays:
        swiss_record = matched_swiss_prot_records.get(assay.entry_name)
        assay_reference_statuses[_classify_reference_sequence(assay, swiss_record)] += 1
        assay_uniref50_statuses[target_statuses[assay.entry_name]] += 1

    target_uniref50_statuses = Counter(target_statuses.values())
    mapped_statuses = {"mapped_in_swiss_prot", "mapped_outside_swiss_prot"}
    audit = ProteinGymSupportAudit(
        target_location_counts={
            status: target_location_counts[status]
            for status in ("in_swiss_prot", "outside_swiss_prot")
        },
        assay_reference_status_counts={
            status: assay_reference_statuses[status] for status in _REFERENCE_STATUSES
        },
        target_uniref50_status_counts={
            status: target_uniref50_statuses[status] for status in _UNIREF_STATUSES
        },
        assay_uniref50_status_counts={
            status: assay_uniref50_statuses[status] for status in _UNIREF_STATUSES
        },
        duplicate_mapping_target_count=len(
            target_names.intersection(uniref50_scan.duplicate_entry_names)
        ),
        reservable_target_count=sum(
            target_uniref50_statuses[status] for status in mapped_statuses
        ),
        reservable_assay_count=sum(
            assay_uniref50_statuses[status] for status in mapped_statuses
        ),
        unique_reservable_family_count=len(reservable_families),
    )
    _validate_support_reconciliations(audit, proteingym_scan.audit)
    return audit


def render_task2_audit(report: Task2AuditReport) -> RenderedTask2Audit:
    """Render stable JSON and Markdown from the same aggregate result."""

    json_text = json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"
    markdown_text = _render_markdown(report)
    json_sha256 = hashlib.sha256(json_text.encode()).hexdigest()
    return RenderedTask2Audit(
        json_text=json_text,
        markdown_text=markdown_text,
        json_sha256=json_sha256,
    )


def _observe_swiss_prot_records(
    records: Iterable[SwissProtRecord],
    *,
    target_entry_names: frozenset[str],
    accessions: list[str],
    matched_records: dict[str, SwissProtRecord],
) -> Iterator[SwissProtRecord]:
    for record in records:
        accessions.append(record.primary_accession)
        if record.entry_name in target_entry_names:
            if record.entry_name in matched_records:
                raise Task2AuditError(
                    f"duplicate Swiss-Prot entry name {record.entry_name!r}"
                )
            matched_records[record.entry_name] = record
        yield record


def _classify_reference_sequence(
    assay: ProteinGymAssay,
    swiss_record: SwissProtRecord | None,
) -> str:
    if swiss_record is None:
        return "outside_swiss_prot"
    if assay.target_sequence == swiss_record.sequence:
        return "exact_swiss_prot"
    return "mismatched_swiss_prot"


def _classify_uniref50_support(
    *,
    entry_name: str,
    swiss_record: SwissProtRecord | None,
    uniref50_scan: UniRef50Scan,
) -> tuple[str, str | None]:
    if swiss_record is not None:
        return _classify_swiss_prot_accession(
            entry_name=entry_name,
            swiss_record=swiss_record,
            uniref50_scan=uniref50_scan,
        )

    match = uniref50_scan.entry_name_matches.get(entry_name)
    if match is not None:
        if match.accession_in_target_population:
            return "inconsistent_accession", None
        return "mapped_outside_swiss_prot", match.group
    if entry_name in uniref50_scan.conflicting_entry_names:
        return "conflicting", None
    if entry_name in uniref50_scan.blank_group_entry_names:
        return "blank_group", None
    if entry_name in uniref50_scan.missing_entry_names:
        return "missing", None
    raise Task2AuditError(
        f"ProteinGym entry name {entry_name!r} has no UniRef50 status"
    )


def _classify_swiss_prot_accession(
    *,
    entry_name: str,
    swiss_record: SwissProtRecord,
    uniref50_scan: UniRef50Scan,
) -> tuple[str, str | None]:
    accession = swiss_record.primary_accession
    if accession in uniref50_scan.conflicting_accessions:
        return "conflicting", None
    if accession in uniref50_scan.blank_group_accessions:
        return "blank_group", None
    if accession in uniref50_scan.missing_accessions:
        return "missing", None

    accession_group = uniref50_scan.accession_to_group.get(accession)
    if accession_group is None:
        raise Task2AuditError(
            f"Swiss-Prot accession {accession!r} has no UniRef50 status"
        )

    optional_entry_match = uniref50_scan.entry_name_matches.get(entry_name)
    if optional_entry_match is not None and (
        not optional_entry_match.accession_in_target_population
        or optional_entry_match.accession != accession
        or optional_entry_match.group != accession_group
    ):
        return "inconsistent_accession", None
    return "mapped_in_swiss_prot", accession_group


def _validate_support_reconciliations(
    audit: ProteinGymSupportAudit,
    metadata_audit: ProteinGymMetadataAudit,
) -> None:
    if sum(audit.target_location_counts.values()) != (
        metadata_audit.target_entry_name_count
    ):
        raise RuntimeError("ProteinGym target-location counts do not reconcile")
    if sum(audit.assay_reference_status_counts.values()) != metadata_audit.assay_count:
        raise RuntimeError("ProteinGym reference-status counts do not reconcile")
    if sum(audit.target_uniref50_status_counts.values()) != (
        metadata_audit.target_entry_name_count
    ):
        raise RuntimeError("ProteinGym target UniRef50 counts do not reconcile")
    if sum(audit.assay_uniref50_status_counts.values()) != metadata_audit.assay_count:
        raise RuntimeError("ProteinGym assay UniRef50 counts do not reconcile")


def _render_markdown(report: Task2AuditReport) -> str:
    rows = tuple(_flatten_aggregate(asdict(report)))
    lines = [
        "# Week 1 Task 2 Aggregate Corpus Audit",
        "",
        "This report contains aggregate source facts only. It does not apply "
        "eligibility filters, construct a split, calculate leakage, or run a model.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {_markdown_cell(metric)} | {_markdown_cell(value)} |"
        for metric, value in rows
    )
    lines.append("")
    return "\n".join(lines)


def _flatten_aggregate(
    value: object,
    prefix: tuple[str, ...] = (),
) -> Iterator[tuple[str, object]]:
    if prefix in _MARKDOWN_JSON_ONLY_PATHS:
        yield ".".join(prefix), "See full distribution in JSON"
        return
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            yield from _flatten_aggregate(value[key], (*prefix, str(key)))
        return
    yield ".".join(prefix), value


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")
