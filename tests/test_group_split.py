import hashlib
import json
from dataclasses import asdict, replace
from fractions import Fraction
from pathlib import Path

import pytest

from protein_lm.data import group_split
from protein_lm.data.eligibility import CATALOG_COLUMNS
from protein_lm.data.group_split import (
    AssignmentUnit,
    allocate_assignment_units,
    build_assignment_units,
    build_group_aware_candidate,
    exact_allocation_score,
    repair_state_audit,
    seeded_unit_order_hash,
    stable_unit_identifier,
    validate_task4_group_report,
    within_balance_tolerance,
)
from protein_lm.data.group_split_policy import (
    APPROVED_GROUP_SPLIT_CONFIG_SHA256,
    APPROVED_GROUP_SPLIT_POLICY,
    GroupSplitError,
    load_group_split_policy,
)
from protein_lm.data.random_split import (
    DiagnosticSplitUseError,
    SplitInputRecord,
    require_selected_training_manifest,
)
from protein_lm.data.task5_report import DerivedArtifact
from protein_lm.data.task6_report import Task6Report, render_task6_report

PROJECT_ROOT = Path(__file__).parents[1]
POLICY_PATH = PROJECT_ROOT / "experiments" / "week_01" / "group_aware_split.toml"
FIXTURE_CONFIG_SHA256 = "d" * 64
TASK4_REPORT_PATH = (
    PROJECT_ROOT / "reports" / "week_01" / "task_04_eligible_records.json"
)


def _record(
    accession: str,
    sequence: str,
    group: str,
    *,
    reserved: bool = False,
) -> SplitInputRecord:
    return SplitInputRecord(
        primary_accession=accession,
        sequence_sha256=hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        biological_length=len(sequence),
        uniref50_group=group,
        proteingym_candidate_test_reserved=reserved,
    )


def _catalog_row(
    accession: str,
    sequence: str,
    group: str,
    *,
    reserved: bool = False,
) -> str:
    values = (
        accession,
        sequence,
        hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        str(len(sequence)),
        *(("false",) * 5),
        "true",
        "",
        group,
        "true" if reserved else "false",
    )
    return "\t".join(values)


def _fixture_specs() -> list[tuple[str, str, str, bool]]:
    return [
        ("P00001", "A" * 32, "UniRef50_A", False),
        ("P00002", "C" * 33, "UniRef50_A", False),
        ("P00003", "A" * 32, "UniRef50_B", True),
        ("P00004", "D" * 34, "UniRef50_C", False),
        ("P00005", "E" * 35, "UniRef50_D", False),
        ("P00006", "F" * 36, "UniRef50_E", False),
    ]


def _write_catalog(
    tmp_path: Path,
    specs: list[tuple[str, str, str, bool]],
):
    rows = [
        _catalog_row(accession, sequence, group, reserved=reserved)
        for accession, sequence, group, reserved in specs
    ]
    content = "\t".join(CATALOG_COLUMNS) + "\n" + "\n".join(rows) + "\n"
    path = tmp_path / "catalog.tsv"
    path.write_bytes(content.encode("utf-8"))
    groups = {group for _, _, group, _ in specs}
    reserved_groups = {group for _, _, group, reserved in specs if reserved}
    reserved_records = [sequence for _, sequence, _, reserved in specs if reserved]
    policy = replace(
        APPROVED_GROUP_SPLIT_POLICY,
        task4_catalog_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        task4_catalog_byte_size=path.stat().st_size,
        task4_catalog_row_count=len(specs),
        expected_eligible_records=len(specs),
        expected_eligible_residues=sum(len(sequence) for _, sequence, _, _ in specs),
        expected_eligible_groups=len(groups),
        expected_reserved_family_universe=len(reserved_groups),
        expected_eligible_reserved_groups=len(reserved_groups),
        expected_eligible_reserved_records=len(reserved_records),
        expected_eligible_reserved_residues=sum(
            len(sequence) for sequence in reserved_records
        ),
    )
    return path, policy


def _allow_fixture_policy(
    monkeypatch: pytest.MonkeyPatch,
    policy,
) -> None:
    monkeypatch.setattr(
        group_split,
        "APPROVED_GROUP_SPLIT_POLICY",
        policy,
    )
    monkeypatch.setattr(
        group_split,
        "APPROVED_GROUP_SPLIT_CONFIG_SHA256",
        FIXTURE_CONFIG_SHA256,
    )


def _build_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    catalog_path, policy = _write_catalog(tmp_path, _fixture_specs())
    _allow_fixture_policy(monkeypatch, policy)
    first = build_group_aware_candidate(
        catalog_path=catalog_path,
        local_assignment_output_path=tmp_path / "first_local.tsv",
        public_manifest_output_path=tmp_path / "first_public.tsv",
        policy=policy,
        policy_sha256=FIXTURE_CONFIG_SHA256,
    )
    second = build_group_aware_candidate(
        catalog_path=catalog_path,
        local_assignment_output_path=tmp_path / "second_local.tsv",
        public_manifest_output_path=tmp_path / "second_public.tsv",
        policy=policy,
        policy_sha256=FIXTURE_CONFIG_SHA256,
    )
    return first, second, policy


def test_group_split_policy_is_exact_and_rejects_byte_drift(tmp_path: Path) -> None:
    policy = load_group_split_policy(POLICY_PATH)
    assert policy == APPROVED_GROUP_SPLIT_POLICY
    assert hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest() == (
        APPROVED_GROUP_SPLIT_CONFIG_SHA256
    )

    drifted_path = tmp_path / "group_aware_split.toml"
    drifted_path.write_bytes(POLICY_PATH.read_bytes() + b"\n")
    with pytest.raises(GroupSplitError, match="approved checksum"):
        load_group_split_policy(drifted_path)


def test_task4_group_and_reservation_anchors_are_required() -> None:
    report = json.loads(TASK4_REPORT_PATH.read_text())
    sources = validate_task4_group_report(
        report,
        APPROVED_GROUP_SPLIT_POLICY,
    )
    assert "swiss_prot_records" in sources

    report["proteingym_reservation"]["eligible_records"] += 1
    with pytest.raises(GroupSplitError, match="reservation drift"):
        validate_task4_group_report(
            report,
            APPROVED_GROUP_SPLIT_POLICY,
        )


def test_stable_unit_identifier_uses_sorted_groups_and_lf_bytes() -> None:
    assert stable_unit_identifier(("UniRef50_A",)) == "UniRef50_A"
    expected = hashlib.sha256(b"UniRef50_A\nUniRef50_B").hexdigest()
    assert stable_unit_identifier(("UniRef50_B", "UniRef50_A")) == expected

    with pytest.raises(GroupSplitError, match="duplicate group"):
        stable_unit_identifier(("UniRef50_A", "UniRef50_A"))
    with pytest.raises(GroupSplitError, match="visible ASCII"):
        stable_unit_identifier(("UniRef50_A\nB",))


def test_seeded_unit_order_hash_uses_the_frozen_payload() -> None:
    stable_id = "UniRef50_A"
    expected = hashlib.sha256(
        b"week1-group-order-v1\x0020260727\x00UniRef50_A"
    ).digest()
    assert seeded_unit_order_hash(stable_id) == expected


def test_exact_integer_score_matches_the_rational_definition() -> None:
    counts = {
        "training": (81, 8_300),
        "validation": (9, 800),
        "test": (10, 900),
    }
    total_records = 100
    total_residues = 10_000
    policy = APPROVED_GROUP_SPLIT_POLICY
    targets = {"training": 18, "validation": 1, "test": 1}
    rational_score = sum(
        (
            Fraction(counts[partition][0], total_records)
            - Fraction(targets[partition], policy.target_denominator)
        )
        ** 2
        + (
            Fraction(counts[partition][1], total_residues)
            - Fraction(targets[partition], policy.target_denominator)
        )
        ** 2
        for partition in ("training", "validation", "test")
    )
    exact_scale = total_records**2 * total_residues**2 * policy.target_denominator**2
    expected = rational_score * exact_scale
    assert expected.denominator == 1
    assert (
        exact_allocation_score(
            counts,
            total_records=total_records,
            total_residues=total_residues,
            policy=policy,
        )
        == expected.numerator
    )


def test_allocator_matches_the_walkthrough_and_protects_reserved_test() -> None:
    units = (
        AssignmentUnit("A", ("UniRef50_A",), 60, 6_000, False, b"a" * 32),
        AssignmentUnit("B", ("UniRef50_B",), 20, 2_000, False, b"b" * 32),
        AssignmentUnit("CD", ("UniRef50_C", "UniRef50_D"), 10, 1_000, False, b"c" * 32),
        AssignmentUnit("E", ("UniRef50_E",), 5, 500, False, b"d" * 32),
        AssignmentUnit("PG", ("UniRef50_PG",), 5, 500, True, b"e" * 32),
    )

    assignments = allocate_assignment_units(
        units,
        total_records=100,
        total_residues=10_000,
        policy=APPROVED_GROUP_SPLIT_POLICY,
    )

    assert assignments == {
        "A": "training",
        "B": "training",
        "CD": "training",
        "E": "validation",
        "PG": "test",
    }


def test_duplicate_union_is_transitive_order_independent_and_expands_reservation() -> (
    None
):
    records = [
        _record("P1", "A" * 32, "UniRef50_A"),
        _record("P2", "C" * 32, "UniRef50_A"),
        _record("P3", "A" * 32, "UniRef50_B", reserved=True),
        _record("P4", "G" * 32, "UniRef50_B", reserved=True),
        _record("P5", "G" * 32, "UniRef50_C"),
    ]
    policy = replace(
        APPROVED_GROUP_SPLIT_POLICY,
        expected_eligible_groups=3,
        expected_reserved_family_universe=1,
        expected_eligible_reserved_groups=1,
        expected_eligible_reserved_records=2,
        expected_eligible_reserved_residues=64,
    )

    first_units, first_mapping, first_audit = build_assignment_units(records, policy)
    second_units, second_mapping, second_audit = build_assignment_units(
        list(reversed(records)),
        policy,
    )

    assert first_units == second_units
    assert first_mapping == second_mapping
    assert first_audit == second_audit
    assert len(first_units) == 1
    assert first_units[0].original_groups == (
        "UniRef50_A",
        "UniRef50_B",
        "UniRef50_C",
    )
    assert first_units[0].reserved_for_test is True
    assert first_audit.exact_hashes_spanning_groups == 2
    assert first_audit.reservation_expanded_groups == 2
    assert first_audit.reservation_expanded_records == 3
    assert first_audit.reservation_expanded_residues == 96


def test_inconsistent_reservation_within_one_group_fails() -> None:
    records = [
        _record("P1", "A" * 32, "UniRef50_A", reserved=True),
        _record("P2", "C" * 32, "UniRef50_A", reserved=False),
    ]
    policy = replace(
        APPROVED_GROUP_SPLIT_POLICY,
        expected_eligible_groups=1,
        expected_reserved_family_universe=1,
        expected_eligible_reserved_groups=1,
        expected_eligible_reserved_records=1,
        expected_eligible_reserved_residues=32,
    )
    with pytest.raises(GroupSplitError, match="inconsistent reservation"):
        build_assignment_units(records, policy)


def test_balance_tolerance_uses_inclusive_exact_boundaries() -> None:
    policy = APPROVED_GROUP_SPLIT_POLICY
    assert within_balance_tolerance(179, 200, 18, policy)
    assert within_balance_tolerance(181, 200, 18, policy)
    assert not within_balance_tolerance(178, 200, 18, policy)
    assert not within_balance_tolerance(182, 200, 18, policy)


def test_fixture_build_is_repeatable_label_free_and_keeps_units_intact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second, _ = _build_fixture(tmp_path, monkeypatch)

    assert first == second
    assert (tmp_path / "first_local.tsv").read_bytes() == (
        tmp_path / "second_local.tsv"
    ).read_bytes()
    assert (tmp_path / "first_public.tsv").read_bytes() == (
        tmp_path / "second_public.tsv"
    ).read_bytes()
    assert first.integrity.structural_invariants_passed is True
    assert first.integrity.accession_crossings == 0
    assert first.integrity.exact_sequence_hash_crossings == 0
    assert first.integrity.uniref50_group_crossings == 0
    assert first.integrity.reserved_records_outside_test == 0
    assert first.assignment_units.initial_group_count == 5
    assert first.assignment_units.assignment_unit_count == 4
    assert first.assignment_units.reservation_expanded_groups == 1
    assert first.assignment_units.reservation_expanded_records == 2

    public_lines = (tmp_path / "first_public.tsv").read_text().splitlines()
    assert public_lines[0].split("\t") == [
        "primary_accession",
        "partition",
        "sequence_sha256",
        "biological_length",
        "uniref50_group",
    ]
    assert [line.split("\t")[0] for line in public_lines[1:]] == sorted(
        line.split("\t")[0] for line in public_lines[1:]
    )
    assert all(len(line.split("\t")) == 5 for line in public_lines[1:])
    assert "A" * 32 not in (tmp_path / "first_public.tsv").read_text()

    local_rows = [
        line.split("\t")
        for line in (tmp_path / "first_local.tsv").read_text().splitlines()[1:]
    ]
    merged_rows = [row for row in local_rows if row[-1] in {"P00001", "P00003"}]
    assert len({row[3] for row in merged_rows}) == 1
    assert {row[4] for row in merged_rows} == {"test"}


def test_repair_state_has_no_header_or_trailing_lf() -> None:
    units = (
        AssignmentUnit("A", ("UniRef50_A",), 1, 10, False, b"a" * 32),
        AssignmentUnit(
            "BC",
            ("UniRef50_B", "UniRef50_C"),
            2,
            20,
            False,
            b"b" * 32,
        ),
    )
    assignments = {"A": "training", "BC": "test"}
    expected_bytes = (
        b"UniRef50_A\tretained\tA\ttraining\n"
        b"UniRef50_B\tretained\tBC\ttest\n"
        b"UniRef50_C\tretained\tBC\ttest"
    )

    audit = repair_state_audit(
        units=units,
        assignments=assignments,
        repair_cycle=0,
    )

    assert audit.row_count == 3
    assert audit.byte_size == len(expected_bytes)
    assert audit.sha256 == hashlib.sha256(expected_bytes).hexdigest()


def test_pre_repair_candidate_is_rejected_by_training_guard() -> None:
    with pytest.raises(DiagnosticSplitUseError, match="final selected stage"):
        require_selected_training_manifest(
            {
                "strategy": "group_aware",
                "stage": "pre_repair",
                "diagnostic_only": False,
                "selected_for_training": False,
                "model_use": "prohibited",
            }
        )


def test_report_is_aggregate_and_preserves_failed_candidate_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build, _, policy = _build_fixture(tmp_path, monkeypatch)
    report = Task6Report(
        schema_version=policy.schema_version,
        scope=policy.scope,
        strategy=policy.strategy,
        stage=policy.stage,
        repair_cycle=policy.repair_cycle,
        candidate_status=build.candidate_status,
        task6_gates_passed=build.task6_gates_passed,
        failure_reasons=build.failure_reasons,
        similarity_audit_completed=False,
        task7_authorized=build.task6_gates_passed,
        model_use=policy.model_use,
        selected_for_training=policy.selected_for_training,
        repeat_verified=True,
        verified_passes=2,
        seed=policy.seed,
        order_namespace=policy.order_namespace,
        hash_algorithm=policy.hash_algorithm,
        license_spdx=policy.license_spdx,
        code_revision="fixture-revision",
        config_sha256=FIXTURE_CONFIG_SHA256,
        task4_report_sha256=policy.task4_report_sha256,
        task4_policy_sha256=policy.task4_policy_sha256,
        sources={"swiss_prot_records": {"license_spdx": "CC-BY-4.0"}},
        input_catalog=DerivedArtifact(
            relative_path="data/processed/week_01/task_04_record_catalog.tsv",
            row_count=policy.task4_catalog_row_count,
            byte_size=policy.task4_catalog_byte_size,
            sha256=policy.task4_catalog_sha256,
        ),
        population=build.population,
        partitions=build.partitions,
        assignment_units=build.assignment_units,
        integrity=build.integrity,
        repair_state=build.repair_state,
        local_assignments=build.local_assignments,
        public_manifest=build.public_manifest,
    )

    rendered = render_task6_report(report)

    rendered_json = json.loads(rendered.json_text)
    assert rendered_json["candidate_status"] == build.candidate_status
    assert rendered_json["task6_gates_passed"] is build.task6_gates_passed
    assert "candidate_usable" not in rendered_json
    assert asdict(report)["selected_for_training"] is False
    for private_value in ("P00001", "UniRef50_A", "A" * 32):
        assert private_value not in rendered.json_text
        assert private_value not in rendered.markdown_text
