"""Build the frozen Week 1 Task 6 group-aware pre-repair candidate."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, localcontext
from pathlib import Path

from protein_lm.data.group_split_policy import (
    APPROVED_GROUP_SPLIT_CONFIG_SHA256,
    APPROVED_GROUP_SPLIT_POLICY,
    GroupSplitError,
    GroupSplitPolicy,
)
from protein_lm.data.random_split import (
    LOCAL_ASSIGNMENT_COLUMNS,
    PUBLIC_MANIFEST_COLUMNS,
    SplitInputRecord,
    read_eligible_records,
    validate_task4_report,
)
from protein_lm.data.task5_report import DerivedArtifact, SplitPopulation
from protein_lm.data.task6_report import (
    AssignmentUnitAudit,
    CandidateIntegrityAudit,
    CandidatePartitionAudit,
    GroupSplitBuild,
    RepairStateAudit,
)

PARTITIONS = ("training", "validation", "test")
_PERCENT_QUANTUM = Decimal("0.000001")


@dataclass(frozen=True)
class AssignmentUnit:
    """One indivisible group or exact-duplicate-connected supergroup."""

    stable_unit_id: str
    original_groups: tuple[str, ...]
    record_count: int
    residue_count: int
    reserved_for_test: bool
    seeded_order_hash: bytes


class _DisjointSet:
    """Track transitive exact-duplicate connections between groups."""

    def __init__(self, values: set[str]) -> None:
        self._parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self._parent[value]
        while parent != self._parent[parent]:
            parent = self._parent[parent]
        while value != parent:
            next_value = self._parent[value]
            self._parent[value] = parent
            value = next_value
        return parent

    def union(self, first: str, second: str) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        lower, higher = sorted((first_root, second_root))
        self._parent[higher] = lower


def stable_unit_identifier(original_groups: tuple[str, ...]) -> str:
    """Return the frozen stable identifier for one assignment unit."""

    if not original_groups:
        raise GroupSplitError("an assignment unit must contain at least one group")
    groups = tuple(sorted(original_groups))
    if len(set(groups)) != len(groups):
        raise GroupSplitError("an assignment unit contains duplicate group identifiers")
    for group in groups:
        _require_visible_ascii(group, "UniRef50 group")
    if len(groups) == 1:
        return groups[0]
    return hashlib.sha256("\n".join(groups).encode("ascii")).hexdigest()


def seeded_unit_order_hash(
    stable_unit_id: str,
    policy: GroupSplitPolicy = APPROVED_GROUP_SPLIT_POLICY,
) -> bytes:
    """Hash one stable unit identifier for the frozen seeded tie-break."""

    _require_visible_ascii(stable_unit_id, "stable unit identifier")
    _require_visible_ascii(policy.order_namespace, "order namespace")
    payload = (
        policy.order_namespace.encode("ascii")
        + b"\x00"
        + str(policy.seed).encode("ascii")
        + b"\x00"
        + stable_unit_id.encode("ascii")
    )
    return hashlib.sha256(payload).digest()


def build_assignment_units(
    records: list[SplitInputRecord],
    policy: GroupSplitPolicy,
) -> tuple[
    tuple[AssignmentUnit, ...],
    dict[str, str],
    AssignmentUnitAudit,
]:
    """Create exact-duplicate-connected assignment units."""

    group_record_counts: dict[str, int] = defaultdict(int)
    group_residue_counts: dict[str, int] = defaultdict(int)
    group_reserved: dict[str, bool] = {}
    first_group_by_sequence_hash: dict[str, str] = {}
    spanning_sequence_hashes: set[str] = set()

    for record in records:
        group = record.uniref50_group
        group_record_counts[group] += 1
        group_residue_counts[group] += record.biological_length
        previous_reserved = group_reserved.setdefault(
            group,
            record.proteingym_candidate_test_reserved,
        )
        if previous_reserved != record.proteingym_candidate_test_reserved:
            raise GroupSplitError(
                f"UniRef50 group {group!r} has inconsistent reservation flags"
            )

    groups = set(group_record_counts)
    disjoint_set = _DisjointSet(groups)
    for record in records:
        group = record.uniref50_group
        first_group = first_group_by_sequence_hash.setdefault(
            record.sequence_sha256,
            group,
        )
        if first_group != group:
            spanning_sequence_hashes.add(record.sequence_sha256)
            disjoint_set.union(first_group, group)

    component_groups: dict[str, list[str]] = defaultdict(list)
    for group in groups:
        component_groups[disjoint_set.find(group)].append(group)

    units = []
    group_to_unit = {}
    seen_unit_ids = set()
    for original_group_list in component_groups.values():
        original_groups = tuple(sorted(original_group_list))
        stable_id = stable_unit_identifier(original_groups)
        if stable_id in seen_unit_ids:
            raise GroupSplitError(f"duplicate stable unit identifier: {stable_id}")
        seen_unit_ids.add(stable_id)
        for group in original_groups:
            group_to_unit[group] = stable_id
        units.append(
            AssignmentUnit(
                stable_unit_id=stable_id,
                original_groups=original_groups,
                record_count=sum(
                    group_record_counts[group] for group in original_groups
                ),
                residue_count=sum(
                    group_residue_counts[group] for group in original_groups
                ),
                reserved_for_test=any(
                    group_reserved[group] for group in original_groups
                ),
                seeded_order_hash=seeded_unit_order_hash(stable_id, policy),
            )
        )

    units.sort(key=lambda unit: unit.stable_unit_id)
    represented_reserved_groups = sum(group_reserved.values())
    source_reserved_records = sum(
        record.proteingym_candidate_test_reserved for record in records
    )
    source_reserved_residues = sum(
        record.biological_length
        for record in records
        if record.proteingym_candidate_test_reserved
    )
    reserved_units = tuple(unit for unit in units if unit.reserved_for_test)
    merged_units = tuple(unit for unit in units if len(unit.original_groups) > 1)
    reserved_assignment_records = sum(unit.record_count for unit in reserved_units)
    reserved_assignment_residues = sum(unit.residue_count for unit in reserved_units)
    audit = AssignmentUnitAudit(
        initial_group_count=len(groups),
        exact_hashes_spanning_groups=len(spanning_sequence_hashes),
        assignment_unit_count=len(units),
        merged_unit_count=len(merged_units),
        groups_in_merged_units=sum(len(unit.original_groups) for unit in merged_units),
        largest_unit_records=max(unit.record_count for unit in units),
        largest_unit_residues=max(unit.residue_count for unit in units),
        largest_unit_original_groups=max(len(unit.original_groups) for unit in units),
        reserved_family_universe=policy.expected_reserved_family_universe,
        represented_reserved_groups=represented_reserved_groups,
        unrepresented_reserved_families=(
            policy.expected_reserved_family_universe - represented_reserved_groups
        ),
        reserved_assignment_units=len(reserved_units),
        reservation_expanded_groups=(
            sum(len(unit.original_groups) for unit in reserved_units)
            - represented_reserved_groups
        ),
        source_reserved_records=source_reserved_records,
        source_reserved_residues=source_reserved_residues,
        reserved_assignment_records=reserved_assignment_records,
        reserved_assignment_residues=reserved_assignment_residues,
        reservation_expanded_records=(
            reserved_assignment_records - source_reserved_records
        ),
        reservation_expanded_residues=(
            reserved_assignment_residues - source_reserved_residues
        ),
        largest_reserved_unit_records=max(unit.record_count for unit in reserved_units),
        largest_reserved_unit_residues=max(
            unit.residue_count for unit in reserved_units
        ),
    )
    _validate_assignment_unit_totals(audit, policy)
    return tuple(units), group_to_unit, audit


def exact_allocation_score(
    partition_counts: Mapping[str, tuple[int, int]],
    *,
    total_records: int,
    total_residues: int,
    policy: GroupSplitPolicy,
) -> int:
    """Return an integer exactly proportional to the frozen rational score."""

    targets = _partition_targets(policy)
    if set(partition_counts) != set(PARTITIONS):
        raise GroupSplitError("allocation score received unexpected partitions")
    record_squared_error = 0
    residue_squared_error = 0
    for partition in PARTITIONS:
        record_count, residue_count = partition_counts[partition]
        target_numerator = targets[partition]
        record_error = (
            policy.target_denominator * record_count - target_numerator * total_records
        )
        residue_error = (
            policy.target_denominator * residue_count
            - target_numerator * total_residues
        )
        record_squared_error += record_error * record_error
        residue_squared_error += residue_error * residue_error
    return (
        total_residues * total_residues * record_squared_error
        + total_records * total_records * residue_squared_error
    )


def allocate_assignment_units(
    units: tuple[AssignmentUnit, ...],
    *,
    total_records: int,
    total_residues: int,
    policy: GroupSplitPolicy,
) -> dict[str, str]:
    """Allocate reserved units and then greedily place every remaining unit."""

    counts = {partition: [0, 0] for partition in PARTITIONS}
    assignments = {}
    for unit in units:
        if not unit.reserved_for_test:
            continue
        assignments[unit.stable_unit_id] = "test"
        counts["test"][0] += unit.record_count
        counts["test"][1] += unit.residue_count

    remaining = sorted(
        (unit for unit in units if not unit.reserved_for_test),
        key=lambda unit: (
            -unit.residue_count,
            -unit.record_count,
            unit.seeded_order_hash,
            unit.stable_unit_id,
        ),
    )
    tie_order = tuple(policy.partition_tie_order.split(","))
    if tie_order != PARTITIONS:
        raise GroupSplitError("partition tie order differs from the frozen order")

    for unit in remaining:
        best_partition = None
        best_score = None
        for partition in tie_order:
            counts[partition][0] += unit.record_count
            counts[partition][1] += unit.residue_count
            score = exact_allocation_score(
                {name: (values[0], values[1]) for name, values in counts.items()},
                total_records=total_records,
                total_residues=total_residues,
                policy=policy,
            )
            counts[partition][0] -= unit.record_count
            counts[partition][1] -= unit.residue_count
            if best_score is None or score < best_score:
                best_partition = partition
                best_score = score

        if best_partition is None:
            raise RuntimeError("allocator did not select a partition")
        assignments[unit.stable_unit_id] = best_partition
        counts[best_partition][0] += unit.record_count
        counts[best_partition][1] += unit.residue_count

    if len(assignments) != len(units):
        raise GroupSplitError("not every assignment unit received a partition")
    if sum(values[0] for values in counts.values()) != total_records:
        raise GroupSplitError("allocated record counts do not reconcile")
    if sum(values[1] for values in counts.values()) != total_residues:
        raise GroupSplitError("allocated residue counts do not reconcile")
    return assignments


def build_group_aware_candidate(
    *,
    catalog_path: Path,
    local_assignment_output_path: Path,
    public_manifest_output_path: Path,
    policy: GroupSplitPolicy,
    policy_sha256: str,
) -> GroupSplitBuild:
    """Build staged local and public Task 6 manifests."""

    _validate_build_policy(policy, policy_sha256)
    records = read_eligible_records(catalog_path, policy)
    records.sort(key=lambda record: record.primary_accession)
    population = SplitPopulation(
        records=len(records),
        residues=sum(record.biological_length for record in records),
        unique_groups=len({record.uniref50_group for record in records}),
    )
    units, group_to_unit, unit_audit = build_assignment_units(records, policy)
    assignments = allocate_assignment_units(
        units,
        total_records=population.records,
        total_residues=population.residues,
        policy=policy,
    )
    return _write_and_audit_manifests(
        records,
        units=units,
        group_to_unit=group_to_unit,
        assignments=assignments,
        population=population,
        assignment_unit_audit=unit_audit,
        local_assignment_output_path=local_assignment_output_path,
        public_manifest_output_path=public_manifest_output_path,
        policy=policy,
    )


def validate_task4_group_report(
    task4_report: Mapping[str, object],
    policy: GroupSplitPolicy,
) -> dict[str, dict[str, object]]:
    """Validate Task 4 population and reservation anchors for Task 6."""

    sources = validate_task4_report(task4_report, policy)
    reservation = task4_report.get("proteingym_reservation")
    if not isinstance(reservation, dict):
        raise GroupSplitError("Task 4 ProteinGym reservation evidence is malformed")
    expected = {
        "reserved family universe": (
            reservation.get("reserved_family_count"),
            policy.expected_reserved_family_universe,
        ),
        "eligible represented reserved groups": (
            reservation.get("eligible_represented_family_count"),
            policy.expected_eligible_reserved_groups,
        ),
        "eligible reserved records": (
            reservation.get("eligible_records"),
            policy.expected_eligible_reserved_records,
        ),
        "eligible reserved residues": (
            reservation.get("eligible_residues"),
            policy.expected_eligible_reserved_residues,
        ),
    }
    differences = [
        f"{name}: found {found!r}, expected {approved!r}"
        for name, (found, approved) in expected.items()
        if found != approved
    ]
    if differences:
        raise GroupSplitError("Task 4 reservation drift: " + "; ".join(differences))
    return sources


def _write_and_audit_manifests(
    records: list[SplitInputRecord],
    *,
    units: tuple[AssignmentUnit, ...],
    group_to_unit: Mapping[str, str],
    assignments: Mapping[str, str],
    population: SplitPopulation,
    assignment_unit_audit: AssignmentUnitAudit,
    local_assignment_output_path: Path,
    public_manifest_output_path: Path,
    policy: GroupSplitPolicy,
) -> GroupSplitBuild:
    local_assignment_output_path.parent.mkdir(parents=True, exist_ok=True)
    public_manifest_output_path.parent.mkdir(parents=True, exist_ok=True)
    local_hasher = hashlib.sha256()
    public_hasher = hashlib.sha256()
    local_byte_size = 0
    public_byte_size = 0
    partition_records = {partition: 0 for partition in PARTITIONS}
    partition_residues = {partition: 0 for partition in PARTITIONS}
    partition_groups = {partition: set() for partition in PARTITIONS}
    partition_units = {partition: set() for partition in PARTITIONS}
    accession_partitions: dict[str, set[str]] = defaultdict(set)
    hash_partitions: dict[str, set[str]] = defaultdict(set)
    group_partitions: dict[str, set[str]] = defaultdict(set)
    reserved_groups_outside_test = set()
    reserved_records_outside_test = 0

    with (
        local_assignment_output_path.open("wb") as local_output,
        public_manifest_output_path.open("wb") as public_output,
    ):
        local_header = ("\t".join(LOCAL_ASSIGNMENT_COLUMNS) + "\n").encode()
        public_header = ("\t".join(PUBLIC_MANIFEST_COLUMNS) + "\n").encode()
        local_output.write(local_header)
        public_output.write(public_header)
        local_hasher.update(local_header)
        public_hasher.update(public_header)
        local_byte_size += len(local_header)
        public_byte_size += len(public_header)

        for record in records:
            stable_unit_id = group_to_unit[record.uniref50_group]
            partition = assignments[stable_unit_id]
            partition_records[partition] += 1
            partition_residues[partition] += record.biological_length
            partition_groups[partition].add(record.uniref50_group)
            partition_units[partition].add(stable_unit_id)
            accession_partitions[record.primary_accession].add(partition)
            hash_partitions[record.sequence_sha256].add(partition)
            group_partitions[record.uniref50_group].add(partition)
            if record.proteingym_candidate_test_reserved and partition != "test":
                reserved_groups_outside_test.add(record.uniref50_group)
                reserved_records_outside_test += 1

            local_row = _tsv_row(
                (
                    policy.strategy,
                    policy.stage,
                    str(policy.repair_cycle),
                    stable_unit_id,
                    partition,
                    record.primary_accession,
                )
            )
            public_row = _tsv_row(
                (
                    record.primary_accession,
                    partition,
                    record.sequence_sha256,
                    str(record.biological_length),
                    record.uniref50_group,
                )
            )
            local_bytes = (local_row + "\n").encode("utf-8")
            public_bytes = (public_row + "\n").encode("utf-8")
            local_output.write(local_bytes)
            public_output.write(public_bytes)
            local_hasher.update(local_bytes)
            public_hasher.update(public_bytes)
            local_byte_size += len(local_bytes)
            public_byte_size += len(public_bytes)

    if local_assignment_output_path.stat().st_size != local_byte_size:
        raise GroupSplitError("local assignment byte count changed after writing")
    if public_manifest_output_path.stat().st_size != public_byte_size:
        raise GroupSplitError("public manifest byte count changed after writing")

    partitions = _partition_audits(
        partition_records=partition_records,
        partition_residues=partition_residues,
        partition_groups=partition_groups,
        partition_units=partition_units,
        population=population,
        policy=policy,
    )
    accession_crossings = sum(
        len(partitions_for_accession) > 1
        for partitions_for_accession in accession_partitions.values()
    )
    hash_crossings = sum(
        len(partitions_for_hash) > 1 for partitions_for_hash in hash_partitions.values()
    )
    group_crossings = sum(
        len(partitions_for_group) > 1
        for partitions_for_group in group_partitions.values()
    )
    structural_invariants_passed = (
        len(accession_partitions) == population.records
        and accession_crossings == 0
        and hash_crossings == 0
        and group_crossings == 0
        and not reserved_groups_outside_test
        and reserved_records_outside_test == 0
    )
    if not structural_invariants_passed:
        raise GroupSplitError("Task 6 structural assignment invariants failed")

    record_balance_passed = all(
        audit.record_balance_passed for audit in partitions.values()
    )
    residue_balance_passed = all(
        audit.residue_balance_passed for audit in partitions.values()
    )
    integrity = CandidateIntegrityAudit(
        assigned_records=len(accession_partitions),
        accession_crossings=accession_crossings,
        exact_sequence_hash_crossings=hash_crossings,
        uniref50_group_crossings=group_crossings,
        reserved_groups_outside_test=len(reserved_groups_outside_test),
        reserved_records_outside_test=reserved_records_outside_test,
        record_retention_percent=_format_percent(
            len(accession_partitions),
            population.records,
        ),
        residue_retention_percent=_format_percent(
            sum(partition_residues.values()),
            population.residues,
        ),
        structural_invariants_passed=True,
        record_balance_passed=record_balance_passed,
        residue_balance_passed=residue_balance_passed,
        all_balance_axes_passed=record_balance_passed and residue_balance_passed,
    )
    failure_reasons = tuple(
        f"{partition}_{axis}_balance"
        for partition in PARTITIONS
        for axis, passed in (
            ("record", partitions[partition].record_balance_passed),
            ("residue", partitions[partition].residue_balance_passed),
        )
        if not passed
    )
    task6_gates_passed = not failure_reasons
    repair_state = repair_state_audit(
        units=units,
        assignments=assignments,
        repair_cycle=policy.repair_cycle,
    )
    return GroupSplitBuild(
        population=population,
        partitions=partitions,
        assignment_units=assignment_unit_audit,
        integrity=integrity,
        repair_state=repair_state,
        candidate_status=("passed_task6" if task6_gates_passed else "failed_balance"),
        task6_gates_passed=task6_gates_passed,
        failure_reasons=failure_reasons,
        local_assignments=DerivedArtifact(
            relative_path=policy.local_assignment_relative_path,
            row_count=population.records,
            byte_size=local_byte_size,
            sha256=local_hasher.hexdigest(),
        ),
        public_manifest=DerivedArtifact(
            relative_path=policy.public_manifest_relative_path,
            row_count=population.records,
            byte_size=public_byte_size,
            sha256=public_hasher.hexdigest(),
        ),
    )


def _partition_audits(
    *,
    partition_records: Mapping[str, int],
    partition_residues: Mapping[str, int],
    partition_groups: Mapping[str, set[str]],
    partition_units: Mapping[str, set[str]],
    population: SplitPopulation,
    policy: GroupSplitPolicy,
) -> dict[str, CandidatePartitionAudit]:
    targets = _partition_targets(policy)
    audits = {}
    for partition in PARTITIONS:
        target_numerator = targets[partition]
        record_count = partition_records[partition]
        residue_count = partition_residues[partition]
        audits[partition] = CandidatePartitionAudit(
            target_numerator=target_numerator,
            target_denominator=policy.target_denominator,
            target_share_percent=_format_percent(
                target_numerator,
                policy.target_denominator,
            ),
            records=record_count,
            residues=residue_count,
            unique_groups=len(partition_groups[partition]),
            assignment_units=len(partition_units[partition]),
            record_share_percent=_format_percent(
                record_count,
                population.records,
            ),
            residue_share_percent=_format_percent(
                residue_count,
                population.residues,
            ),
            record_deviation_percentage_points=_format_deviation(
                record_count,
                population.records,
                target_numerator,
                policy.target_denominator,
            ),
            residue_deviation_percentage_points=_format_deviation(
                residue_count,
                population.residues,
                target_numerator,
                policy.target_denominator,
            ),
            record_balance_passed=within_balance_tolerance(
                record_count,
                population.records,
                target_numerator,
                policy,
            ),
            residue_balance_passed=within_balance_tolerance(
                residue_count,
                population.residues,
                target_numerator,
                policy,
            ),
        )
    if sum(audit.records for audit in audits.values()) != population.records:
        raise GroupSplitError("partition record counts do not reconcile")
    if sum(audit.residues for audit in audits.values()) != population.residues:
        raise GroupSplitError("partition residue counts do not reconcile")
    return audits


def repair_state_audit(
    *,
    units: tuple[AssignmentUnit, ...],
    assignments: Mapping[str, str],
    repair_cycle: int,
) -> RepairStateAudit:
    rows = []
    for unit in units:
        partition = assignments[unit.stable_unit_id]
        for original_group in unit.original_groups:
            rows.append(
                (
                    original_group,
                    f"{original_group}\tretained\t{unit.stable_unit_id}\t{partition}",
                )
            )
    state_bytes = "\n".join(row for _, row in sorted(rows)).encode("utf-8")
    return RepairStateAudit(
        repair_cycle=repair_cycle,
        row_count=len(rows),
        byte_size=len(state_bytes),
        sha256=hashlib.sha256(state_bytes).hexdigest(),
    )


def _validate_assignment_unit_totals(
    audit: AssignmentUnitAudit,
    policy: GroupSplitPolicy,
) -> None:
    expected = {
        "eligible group count": (
            audit.initial_group_count,
            policy.expected_eligible_groups,
        ),
        "reserved family universe": (
            audit.reserved_family_universe,
            policy.expected_reserved_family_universe,
        ),
        "represented reserved groups": (
            audit.represented_reserved_groups,
            policy.expected_eligible_reserved_groups,
        ),
        "reserved records": (
            audit.source_reserved_records,
            policy.expected_eligible_reserved_records,
        ),
        "reserved residues": (
            audit.source_reserved_residues,
            policy.expected_eligible_reserved_residues,
        ),
    }
    differences = [
        f"{name}: found {found}, expected {approved}"
        for name, (found, approved) in expected.items()
        if found != approved
    ]
    if differences:
        raise GroupSplitError("Task 6 assignment-unit drift: " + "; ".join(differences))


def _validate_build_policy(
    policy: GroupSplitPolicy,
    policy_sha256: str,
) -> None:
    if policy != APPROVED_GROUP_SPLIT_POLICY:
        raise GroupSplitError("group split policy is not the approved policy")
    if policy_sha256 != APPROVED_GROUP_SPLIT_CONFIG_SHA256:
        raise GroupSplitError(
            "group split policy bytes do not match the approved checksum"
        )
    if (
        policy.training_target_numerator
        + policy.validation_target_numerator
        + policy.test_target_numerator
        != policy.target_denominator
    ):
        raise GroupSplitError("group split target fractions do not sum to one")
    if policy.balance_tolerance_numerator <= 0:
        raise GroupSplitError("balance tolerance must be positive")


def _partition_targets(policy: GroupSplitPolicy) -> dict[str, int]:
    return {
        "training": policy.training_target_numerator,
        "validation": policy.validation_target_numerator,
        "test": policy.test_target_numerator,
    }


def within_balance_tolerance(
    realized_numerator: int,
    realized_denominator: int,
    target_numerator: int,
    policy: GroupSplitPolicy,
) -> bool:
    difference_numerator = abs(
        policy.target_denominator * realized_numerator
        - target_numerator * realized_denominator
    )
    return (
        difference_numerator * policy.balance_tolerance_denominator
        <= policy.balance_tolerance_numerator
        * policy.target_denominator
        * realized_denominator
    )


def _format_percent(numerator: int, denominator: int) -> str:
    with localcontext() as context:
        context.prec = 50
        percent = Decimal(numerator) * 100 / Decimal(denominator)
        return format(percent.quantize(_PERCENT_QUANTUM, ROUND_HALF_UP), "f")


def _format_deviation(
    realized_numerator: int,
    realized_denominator: int,
    target_numerator: int,
    target_denominator: int,
) -> str:
    with localcontext() as context:
        context.prec = 50
        realized = Decimal(realized_numerator) / Decimal(realized_denominator)
        target = Decimal(target_numerator) / Decimal(target_denominator)
        difference = (realized - target) * 100
        return format(difference.quantize(_PERCENT_QUANTUM, ROUND_HALF_UP), "f")


def _tsv_row(values: tuple[str, ...]) -> str:
    if any("\t" in value or "\n" in value or "\r" in value for value in values):
        raise GroupSplitError("manifest value contains a tab or newline")
    return "\t".join(values)


def _require_visible_ascii(value: str, field: str) -> None:
    if (
        not value
        or not value.isascii()
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in value)
    ):
        raise GroupSplitError(f"{field} must contain only visible ASCII")
