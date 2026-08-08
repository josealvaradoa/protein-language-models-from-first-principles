import hashlib
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from protein_lm.data import eligibility
from protein_lm.data.eligibility import (
    EligibleRecord,
    EligibilityFlags,
    build_task4_catalog,
    classify_record,
    normalize_sequence,
    sequence_sha256,
)
from protein_lm.data.eligibility_policy import (
    APPROVED_ELIGIBILITY_POLICY,
    APPROVED_ELIGIBILITY_POLICY_SHA256,
    Task4PreparationError,
    load_eligibility_policy,
)
from protein_lm.data.task2_audit import SourceEvidence
from protein_lm.data.task4_report import (
    render_task4_report,
    validate_task2_anchors,
)
from protein_lm.data.uniprot import SwissProtRecord
from protein_lm.data.uniref import UniRef50Audit

PROJECT_ROOT = Path(__file__).parents[1]
FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures" / "week_01" / "corpus_audit"
POLICY_PATH = PROJECT_ROOT / "experiments" / "week_01" / "eligibility.toml"
APPROVED_TASK2_SHA256 = (
    "ab83d9a3341694dab9b4097334f43b2036e5b4fb0417c8b3a028e54f679cdd0f"
)


def _record(
    sequence: str,
    *,
    accession: str = "P00001",
    fragment: bool = False,
) -> SwissProtRecord:
    return SwissProtRecord(
        entry_name=f"{accession}_SYNTH",
        primary_accession=accession,
        declared_length=len(sequence),
        sequence=sequence,
        is_fragment=fragment,
        ec_numbers=(),
    )


def _source_evidence() -> dict[str, SourceEvidence]:
    return {
        "swiss_prot_records": SourceEvidence(
            release="fixture",
            filename="uniprot_sprot.dat",
            byte_size=1,
            sha256="a" * 64,
            upstream_checksum_algorithm="md5",
            upstream_checksum="a" * 32,
            license_spdx="CC-BY-4.0",
            retrieval_date="2026-07-28",
            retrieval_method="fixture",
        ),
        "uniref50_membership": SourceEvidence(
            release="fixture",
            filename="idmapping_selected.tab",
            byte_size=1,
            sha256="b" * 64,
            upstream_checksum_algorithm="md5",
            upstream_checksum="b" * 32,
            license_spdx="CC-BY-4.0",
            retrieval_date="2026-07-28",
            retrieval_method="fixture",
        ),
        "proteingym_metadata": SourceEvidence(
            release="fixture",
            filename="DMS_substitutions.csv",
            byte_size=1,
            sha256="c" * 64,
            upstream_checksum_algorithm="git_blob_sha1",
            upstream_checksum="c" * 40,
            license_spdx="MIT",
            retrieval_date="2026-07-28",
            retrieval_method="fixture",
        ),
    }


def _task4_policy():
    return replace(
        APPROVED_ELIGIBILITY_POLICY,
        minimum_length=1,
        maximum_length=6,
        expected_resolvable_proteingym_targets=3,
        expected_resolvable_proteingym_assays=5,
        expected_reserved_proteingym_families=3,
    )


def _allow_fixture_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        eligibility,
        "APPROVED_ELIGIBILITY_POLICY",
        _task4_policy(),
    )
    monkeypatch.setattr(
        eligibility,
        "APPROVED_ELIGIBILITY_POLICY_SHA256",
        "d" * 64,
    )


def _complete_mapping(tmp_path: Path) -> Path:
    source = (FIXTURE_DIRECTORY / "idmapping_selected.tab").read_text(
        encoding="utf-8"
    )
    path = tmp_path / "complete_mapping.tab"
    path.write_text(
        source
        + "P00006\tEPSILON_SYNTH\t3\t4\t5\t6\t7\t"
        "UniRef100_GROUP_D\tUniRef90_GROUP_D\tUniRef50_GROUP_D\n",
        encoding="utf-8",
    )
    return path


def _build_fixture(
    tmp_path: Path,
    *,
    mapping_path: Path | None = None,
    suffix: str = "first",
):
    return build_task4_catalog(
        swiss_prot_path=FIXTURE_DIRECTORY / "uniprot_sprot.dat",
        uniref50_path=mapping_path or _complete_mapping(tmp_path),
        proteingym_path=FIXTURE_DIRECTORY / "DMS_substitutions.csv",
        catalog_output_path=tmp_path / f"{suffix}_catalog.tsv",
        reserved_families_output_path=tmp_path / f"{suffix}_families.txt",
        catalog_relative_path="data/processed/week_01/task_04_record_catalog.tsv",
        reserved_families_relative_path=(
            "data/processed/week_01/"
            "task_04_candidate_test_reserved_families.txt"
        ),
        policy=_task4_policy(),
        policy_sha256="d" * 64,
        task2_report_sha256=APPROVED_TASK2_SHA256,
        sources=_source_evidence(),
        code_revision="fixture-revision",
    )


def test_policy_and_sequence_hash_are_frozen() -> None:
    policy = load_eligibility_policy(POLICY_PATH)

    assert policy == APPROVED_ELIGIBILITY_POLICY
    assert hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest() == (
        APPROVED_ELIGIBILITY_POLICY_SHA256
    )
    assert normalize_sequence("acdef") == "ACDEF"
    assert sequence_sha256("acdef") == hashlib.sha256(b"ACDEF").hexdigest()
    with pytest.raises(Task4PreparationError, match="ASCII letters"):
        normalize_sequence("ACD EF")


@pytest.mark.parametrize(
    ("length", "expected_reason"),
    [
        (31, "below_min_length"),
        (32, None),
        (2046, None),
        (2047, "above_max_length"),
    ],
)
def test_length_boundaries_are_inclusive(
    length: int, expected_reason: str | None
) -> None:
    prepared = classify_record(
        _record("A" * length),
        uniref50_group="UniRef50_BOUNDARY",
        reserved_families=frozenset(),
    )

    assert prepared.primary_exclusion_reason == expected_reason
    assert prepared.eligible is (expected_reason is None)


def test_all_flags_are_retained_before_primary_precedence() -> None:
    prepared = classify_record(
        _record("B" * 10, fragment=True),
        uniref50_group="",
        reserved_families=frozenset(),
    )

    assert prepared.flags == EligibilityFlags(
        noncanonical_residue=True,
        fragment=True,
        below_min_length=True,
        above_max_length=False,
        blank_uniref50_mapping=True,
    )
    assert prepared.primary_exclusion_reason == "noncanonical_residue"
    assert prepared.eligible is False


@pytest.mark.parametrize(
    ("fragment", "length", "group", "expected_reason"),
    [
        (True, 10, "UniRef50_ONE", "fragment"),
        (False, 10, "", "below_min_length"),
        (False, 2047, "", "above_max_length"),
    ],
)
def test_primary_exclusion_precedence_transitions(
    fragment: bool,
    length: int,
    group: str,
    expected_reason: str,
) -> None:
    prepared = classify_record(
        _record("A" * length, fragment=fragment),
        uniref50_group=group,
        reserved_families=frozenset(),
    )

    assert prepared.flags.blank_uniref50_mapping is (not group)
    assert prepared.primary_exclusion_reason == expected_reason


@pytest.mark.parametrize("residue", tuple("BJXZUO"))
def test_each_noncanonical_residue_is_excluded(residue: str) -> None:
    prepared = classify_record(
        _record(residue * 32),
        uniref50_group="UniRef50_CANONICAL_TEST",
        reserved_families=frozenset(),
    )

    assert prepared.flags.noncanonical_residue is True
    assert prepared.primary_exclusion_reason == "noncanonical_residue"


def test_reservation_is_applied_to_the_complete_family() -> None:
    groups = (
        "UniRef50_RESERVED",
        "UniRef50_RESERVED",
        "UniRef50_OTHER",
    )
    prepared = [
        classify_record(
            _record("A" * 32, accession=f"P0000{index}"),
            uniref50_group=group,
            reserved_families=frozenset({"UniRef50_RESERVED"}),
        )
        for index, group in enumerate(groups, start=1)
    ]

    assert [
        record.proteingym_candidate_test_reserved for record in prepared
    ] == [True, True, False]


def test_exact_duplicates_crossing_groups_are_reported() -> None:
    counters = eligibility._CatalogCounters()
    for accession, group in (
        ("P00001", "UniRef50_ONE"),
        ("P00002", "UniRef50_TWO"),
    ):
        counters.observe(
            classify_record(
                _record("A" * 32, accession=accession),
                uniref50_group=group,
                reserved_families=frozenset(),
            )
        )
    mapping = UniRef50Audit(
        source_row_count=2,
        target_accession_count=2,
        mapped_accession_count=2,
        missing_source_row_count=0,
        blank_group_accession_count=0,
        duplicate_source_accession_count=0,
        conflicting_mapping_accession_count=0,
        unique_group_count=2,
        maximum_group_size=1,
        group_size_histogram={1: 2},
    )

    groups = counters.group_audit(mapping)

    assert groups.eligible_duplicate_hashes_across_groups == 1
    assert groups.eligible_records_in_cross_group_duplicate_hashes == 2


def test_digest_collision_guard_rejects_different_sequences() -> None:
    counters = eligibility._CatalogCounters()
    flags = EligibilityFlags(False, False, False, False, False)
    first = EligibleRecord(
        "P00001",
        "AAAA",
        "f" * 64,
        4,
        flags,
        True,
        None,
        "UniRef50_ONE",
        False,
    )
    second = replace(first, primary_accession="P00002", sequence="CCCC")

    counters.observe(first)
    with pytest.raises(Task4PreparationError, match="different sequences"):
        counters.observe(second)


def test_fixture_catalog_reconciles_and_is_repeatable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_fixture_policy(monkeypatch)
    mapping_path = _complete_mapping(tmp_path)
    first = _build_fixture(tmp_path, mapping_path=mapping_path, suffix="first")
    second = _build_fixture(tmp_path, mapping_path=mapping_path, suffix="second")

    assert first == second
    assert (tmp_path / "first_catalog.tsv").read_bytes() == (
        tmp_path / "second_catalog.tsv"
    ).read_bytes()
    assert first.population.source.records == 6
    assert first.population.source.residues == 24
    assert first.population.eligible.records == 3
    assert first.population.eligible.residues == 11
    assert first.population.primary_exclusions[
        "noncanonical_residue"
    ].records == 2
    assert first.population.primary_exclusions[
        "blank_uniref50_mapping"
    ].records == 1
    assert first.source_duplicates.duplicate_sequence_group_count == 1
    assert first.eligible_duplicates.records_in_duplicate_groups == 2
    assert first.groups.eligible_unique_group_count == 2
    assert first.proteingym_reservation.reserved_family_count == 3
    assert first.proteingym_reservation.eligible_represented_family_count == 2
    assert first.proteingym_reservation.eligible_records == 3

    catalog = (tmp_path / "first_catalog.tsv").read_text(encoding="utf-8")
    assert catalog.count("\n") == 7
    assert "P00001\tACDEF" in catalog
    assert "UniRef50_GROUP_A\ttrue" in catalog
    assert (tmp_path / "first_families.txt").read_text(encoding="ascii") == (
        "UniRef50_GROUP_A\nUniRef50_GROUP_D\nUniRef50_OTHER\n"
    )

    rendered = render_task4_report(first)
    for private_value in (
        "P00001",
        "ACDEF",
        "UniRef50_GROUP_A",
        "PG_ALPHA_1",
    ):
        assert private_value not in rendered.json_text
        assert private_value not in rendered.markdown_text


def test_task2_anchor_drift_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_fixture_policy(monkeypatch)
    report = _build_fixture(tmp_path)
    uniref = asdict(report.mapping)
    uniref["group_size_histogram"] = {
        str(size): count
        for size, count in report.mapping.group_size_histogram.items()
    }
    task2 = {
        "swiss_prot": {
            "record_count": report.population.source.records,
            "residue_count": report.population.source.residues,
            "unique_sequence_count": (
                report.source_duplicates.unique_sequence_hash_count
            ),
            "duplicate_sequence_group_count": (
                report.source_duplicates.duplicate_sequence_group_count
            ),
            "records_in_duplicate_groups": (
                report.source_duplicates.records_in_duplicate_groups
            ),
            "redundant_record_count": (
                report.source_duplicates.redundant_record_count
            ),
            "maximum_duplicate_multiplicity": (
                report.source_duplicates.maximum_duplicate_multiplicity
            ),
        },
        "uniref50": uniref,
        "proteingym_support": {
            "reservable_target_count": (
                report.proteingym_reservation.resolvable_target_count
            ),
            "reservable_assay_count": (
                report.proteingym_reservation.resolvable_assay_count
            ),
            "unique_reservable_family_count": (
                report.proteingym_reservation.reserved_family_count
            ),
        },
    }
    validate_task2_anchors(report, task2)
    task2["swiss_prot"]["record_count"] += 1

    with pytest.raises(Task4PreparationError, match="source records"):
        validate_task2_anchors(report, task2)


def test_official_builder_boundary_rejects_policy_drift() -> None:
    common = {
        "task2_report_sha256": APPROVED_TASK2_SHA256,
        "sources": _source_evidence(),
        "code_revision": "fixture-revision",
    }
    with pytest.raises(Task4PreparationError, match="not the approved policy"):
        eligibility._validate_build_inputs(
            policy=replace(APPROVED_ELIGIBILITY_POLICY, minimum_length=31),
            policy_sha256=APPROVED_ELIGIBILITY_POLICY_SHA256,
            **common,
        )
    with pytest.raises(Task4PreparationError, match="approved checksum"):
        eligibility._validate_build_inputs(
            policy=APPROVED_ELIGIBILITY_POLICY,
            policy_sha256="f" * 64,
            **common,
        )


@pytest.mark.parametrize(
    ("mapping_transform", "message"),
    [
        (lambda source: source, "missing=1"),
        (
            lambda source: source
            + "P00001\tALPHA_SYNTH\t3\t4\t5\t6\t7\t"
            "UniRef100_GROUP_A\tUniRef90_GROUP_A\tUniRef50_GROUP_A\n",
            "duplicate=1",
        ),
        (
            lambda source: source
            + "P00001\tALPHA_SYNTH\t3\t4\t5\t6\t7\t"
            "UniRef100_GROUP_X\tUniRef90_GROUP_X\tUniRef50_GROUP_X\n",
            "conflicting=1",
        ),
    ],
)
def test_fatal_accession_mapping_evidence_stops_task4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mapping_transform,
    message: str,
) -> None:
    _allow_fixture_policy(monkeypatch)
    source = (FIXTURE_DIRECTORY / "idmapping_selected.tab").read_text(
        encoding="utf-8"
    )
    mapping_path = tmp_path / "bad_mapping.tab"
    mapping_path.write_text(mapping_transform(source), encoding="utf-8")

    with pytest.raises(Task4PreparationError, match=message):
        _build_fixture(tmp_path, mapping_path=mapping_path)
