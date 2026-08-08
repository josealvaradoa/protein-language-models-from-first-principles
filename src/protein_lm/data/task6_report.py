"""Render aggregate evidence for the Task 6 group-aware candidate."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass

from protein_lm.data.task5_report import DerivedArtifact, SplitPopulation


@dataclass(frozen=True)
class CandidatePartitionAudit:
    """Target, realized size, and balance result for one partition."""

    target_numerator: int
    target_denominator: int
    target_share_percent: str
    records: int
    residues: int
    unique_groups: int
    assignment_units: int
    record_share_percent: str
    residue_share_percent: str
    record_deviation_percentage_points: str
    residue_deviation_percentage_points: str
    record_balance_passed: bool
    residue_balance_passed: bool


@dataclass(frozen=True)
class AssignmentUnitAudit:
    """Aggregate evidence about grouping, unions, and reservations."""

    initial_group_count: int
    exact_hashes_spanning_groups: int
    assignment_unit_count: int
    merged_unit_count: int
    groups_in_merged_units: int
    largest_unit_records: int
    largest_unit_residues: int
    largest_unit_original_groups: int
    reserved_family_universe: int
    represented_reserved_groups: int
    unrepresented_reserved_families: int
    reserved_assignment_units: int
    reservation_expanded_groups: int
    source_reserved_records: int
    source_reserved_residues: int
    reserved_assignment_records: int
    reserved_assignment_residues: int
    reservation_expanded_records: int
    reservation_expanded_residues: int
    largest_reserved_unit_records: int
    largest_reserved_unit_residues: int


@dataclass(frozen=True)
class CandidateIntegrityAudit:
    """Crossing, reservation, retention, and balance acceptance evidence."""

    assigned_records: int
    accession_crossings: int
    exact_sequence_hash_crossings: int
    uniref50_group_crossings: int
    reserved_groups_outside_test: int
    reserved_records_outside_test: int
    record_retention_percent: str
    residue_retention_percent: str
    structural_invariants_passed: bool
    record_balance_passed: bool
    residue_balance_passed: bool
    all_balance_axes_passed: bool


@dataclass(frozen=True)
class RepairStateAudit:
    """Identity of the exact pre-repair state-zero serialization."""

    repair_cycle: int
    row_count: int
    byte_size: int
    sha256: str


@dataclass(frozen=True)
class GroupSplitBuild:
    """Evidence returned by one complete Task 6 manifest build."""

    population: SplitPopulation
    partitions: dict[str, CandidatePartitionAudit]
    assignment_units: AssignmentUnitAudit
    integrity: CandidateIntegrityAudit
    repair_state: RepairStateAudit
    candidate_status: str
    task6_gates_passed: bool
    failure_reasons: tuple[str, ...]
    local_assignments: DerivedArtifact
    public_manifest: DerivedArtifact


@dataclass(frozen=True)
class Task6Report:
    """Public metadata and aggregate evidence for the pre-repair candidate."""

    schema_version: int
    scope: str
    strategy: str
    stage: str
    repair_cycle: int
    candidate_status: str
    task6_gates_passed: bool
    failure_reasons: tuple[str, ...]
    similarity_audit_completed: bool
    task7_authorized: bool
    model_use: str
    selected_for_training: bool
    repeat_verified: bool
    verified_passes: int
    seed: int
    order_namespace: str
    hash_algorithm: str
    license_spdx: str
    code_revision: str
    config_sha256: str
    task4_report_sha256: str
    task4_policy_sha256: str
    sources: dict[str, dict[str, object]]
    input_catalog: DerivedArtifact
    population: SplitPopulation
    partitions: dict[str, CandidatePartitionAudit]
    assignment_units: AssignmentUnitAudit
    integrity: CandidateIntegrityAudit
    repair_state: RepairStateAudit
    local_assignments: DerivedArtifact
    public_manifest: DerivedArtifact


@dataclass(frozen=True)
class RenderedTask6Report:
    """Byte-stable JSON, Markdown, and canonical JSON digest."""

    json_text: str
    markdown_text: str
    json_sha256: str


def render_task6_report(report: Task6Report) -> RenderedTask6Report:
    """Render byte-stable aggregate JSON and Markdown."""

    report_dict = asdict(report)
    json_text = json.dumps(report_dict, indent=2, sort_keys=True) + "\n"
    if report.task6_gates_passed:
        status = (
            "The pre-repair candidate passed every Task 6 gate. "
            "It is authorized only for the Task 7 similarity audit and "
            "remains prohibited for model use."
        )
    else:
        status = (
            "The pre-repair candidate failed at least one Task 6 gate. "
            "It is not authorized for Task 7 or model use."
        )
    lines = [
        "# Week 1 Task 6 Group-Aware Pre-Repair Candidate",
        "",
        status,
        "",
        "This report contains aggregate evidence and provenance. The separate "
        "public manifest contains approved identifiers and membership fields, "
        "but no sequences, labels, scores, or model results.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {_markdown_cell(metric)} | {_markdown_cell(value)} |"
        for metric, value in _flatten(report_dict)
    )
    lines.append("")
    markdown_text = "\n".join(lines)
    return RenderedTask6Report(
        json_text=json_text,
        markdown_text=markdown_text,
        json_sha256=hashlib.sha256(json_text.encode("utf-8")).hexdigest(),
    )


def _flatten(
    value: object,
    prefix: tuple[str, ...] = (),
) -> Iterator[tuple[str, object]]:
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            yield from _flatten(value[key], (*prefix, str(key)))
        return
    if isinstance(value, (list, tuple)):
        yield ".".join(prefix), ", ".join(str(item) for item in value)
        return
    yield ".".join(prefix), value


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")
