"""Render and validate aggregate-only Task 4 evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass

from protein_lm.data.eligibility_policy import Task4PreparationError
from protein_lm.data.task2_audit import SourceEvidence
from protein_lm.data.uniref import UniRef50Audit

_MARKDOWN_JSON_ONLY_PATHS = frozenset({("mapping", "group_size_histogram")})


@dataclass(frozen=True)
class RecordResidueCount:
    records: int
    residues: int


@dataclass(frozen=True)
class PopulationAudit:
    source: RecordResidueCount
    eligible: RecordResidueCount
    excluded: RecordResidueCount
    matched_flags: dict[str, RecordResidueCount]
    primary_exclusions: dict[str, RecordResidueCount]


@dataclass(frozen=True)
class DuplicateAudit:
    unique_sequence_hash_count: int
    duplicate_sequence_group_count: int
    records_in_duplicate_groups: int
    redundant_record_count: int
    maximum_duplicate_multiplicity: int


@dataclass(frozen=True)
class GroupAudit:
    source_unique_group_count: int
    eligible_unique_group_count: int
    maximum_source_group_size: int
    maximum_eligible_group_size: int
    eligible_duplicate_hashes_across_groups: int
    eligible_records_in_cross_group_duplicate_hashes: int


@dataclass(frozen=True)
class ProteinGymReservationAudit:
    resolvable_target_count: int
    resolvable_assay_count: int
    reserved_family_count: int
    reserved_family_set_sha256: str
    source_represented_family_count: int
    source_records: int
    source_residues: int
    eligible_represented_family_count: int
    eligible_records: int
    eligible_residues: int


@dataclass(frozen=True)
class DerivedArtifact:
    relative_path: str
    row_count: int
    byte_size: int
    sha256: str


@dataclass(frozen=True)
class Task4Report:
    """Aggregate-only evidence from one complete Task 4 preparation."""

    schema_version: int
    scope: str
    code_revision: str
    policy_sha256: str
    approved_task2_report_sha256: str
    sources: dict[str, SourceEvidence]
    population: PopulationAudit
    mapping: UniRef50Audit
    source_duplicates: DuplicateAudit
    eligible_duplicates: DuplicateAudit
    groups: GroupAudit
    proteingym_reservation: ProteinGymReservationAudit
    catalog: DerivedArtifact
    reserved_families: DerivedArtifact


@dataclass(frozen=True)
class RenderedTask4Report:
    json_text: str
    markdown_text: str
    json_sha256: str


def render_task4_report(report: Task4Report) -> RenderedTask4Report:
    """Render byte-stable aggregate JSON and Markdown."""

    json_text = json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"
    lines = [
        "# Week 1 Task 4 Eligible-Record Preparation",
        "",
        "This report contains aggregate preparation evidence only. Raw sequences, "
        "record identifiers, family identifiers, labels, splits, and model results "
        "are excluded.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {_markdown_cell(metric)} | {_markdown_cell(value)} |"
        for metric, value in _flatten_aggregate(asdict(report))
    )
    lines.append("")
    markdown_text = "\n".join(lines)
    return RenderedTask4Report(
        json_text=json_text,
        markdown_text=markdown_text,
        json_sha256=hashlib.sha256(json_text.encode()).hexdigest(),
    )


def validate_task2_anchors(report: Task4Report, task2_report: Mapping[str, object]) -> None:
    """Prove that Task 4 started from the approved Task 2 population."""

    swiss = _mapping(task2_report, "swiss_prot")
    uniref = _mapping(task2_report, "uniref50")
    support = _mapping(task2_report, "proteingym_support")
    expected = {
        "source records": (report.population.source.records, swiss.get("record_count")),
        "source residues": (
            report.population.source.residues,
            swiss.get("residue_count"),
        ),
        "source unique sequences": (
            report.source_duplicates.unique_sequence_hash_count,
            swiss.get("unique_sequence_count"),
        ),
        "source duplicate groups": (
            report.source_duplicates.duplicate_sequence_group_count,
            swiss.get("duplicate_sequence_group_count"),
        ),
        "source records in duplicate groups": (
            report.source_duplicates.records_in_duplicate_groups,
            swiss.get("records_in_duplicate_groups"),
        ),
        "source redundant records": (
            report.source_duplicates.redundant_record_count,
            swiss.get("redundant_record_count"),
        ),
        "source maximum duplicate multiplicity": (
            report.source_duplicates.maximum_duplicate_multiplicity,
            swiss.get("maximum_duplicate_multiplicity"),
        ),
        "mapping source rows": (
            report.mapping.source_row_count,
            uniref.get("source_row_count"),
        ),
        "mapping target accessions": (
            report.mapping.target_accession_count,
            uniref.get("target_accession_count"),
        ),
        "mapped accessions": (
            report.mapping.mapped_accession_count,
            uniref.get("mapped_accession_count"),
        ),
        "missing mappings": (
            report.mapping.missing_source_row_count,
            uniref.get("missing_source_row_count"),
        ),
        "blank mappings": (
            report.mapping.blank_group_accession_count,
            uniref.get("blank_group_accession_count"),
        ),
        "duplicate mappings": (
            report.mapping.duplicate_source_accession_count,
            uniref.get("duplicate_source_accession_count"),
        ),
        "conflicting mappings": (
            report.mapping.conflicting_mapping_accession_count,
            uniref.get("conflicting_mapping_accession_count"),
        ),
        "source UniRef50 groups": (
            report.mapping.unique_group_count,
            uniref.get("unique_group_count"),
        ),
        "maximum source UniRef50 group size": (
            report.mapping.maximum_group_size,
            uniref.get("maximum_group_size"),
        ),
        "resolvable ProteinGym targets": (
            report.proteingym_reservation.resolvable_target_count,
            support.get("reservable_target_count"),
        ),
        "resolvable ProteinGym assays": (
            report.proteingym_reservation.resolvable_assay_count,
            support.get("reservable_assay_count"),
        ),
        "reserved families": (
            report.proteingym_reservation.reserved_family_count,
            support.get("unique_reservable_family_count"),
        ),
    }
    task2_histogram = uniref.get("group_size_histogram")
    task4_histogram = {
        str(size): count
        for size, count in report.mapping.group_size_histogram.items()
    }
    if task4_histogram != task2_histogram:
        expected["source UniRef50 group-size histogram"] = (
            task4_histogram,
            task2_histogram,
        )
    differences = [
        f"{name}: Task 4={actual}, Task 2={anchor}"
        for name, (actual, anchor) in expected.items()
        if actual != anchor
    ]
    if differences:
        raise Task4PreparationError(
            "Task 4 does not match the approved Task 2 population: "
            + "; ".join(differences)
        )


def _flatten_aggregate(
    value: object, prefix: tuple[str, ...] = ()
) -> Iterator[tuple[str, object]]:
    if prefix in _MARKDOWN_JSON_ONLY_PATHS:
        yield ".".join(prefix), "See full distribution in JSON"
        return
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            yield from _flatten_aggregate(value[key], (*prefix, str(key)))
        return
    yield ".".join(prefix), value


def _mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise Task4PreparationError(f"Task 2 report field {key!r} is malformed")
    return value


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")
