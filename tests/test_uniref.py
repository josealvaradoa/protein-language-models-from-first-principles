import gzip
from pathlib import Path

import pytest

from protein_lm.data.uniprot import parse_swiss_prot
from protein_lm.data.uniref import (
    UniRef50Match,
    UniRef50ParseError,
    audit_uniref50_membership,
    scan_uniref50_membership,
)

FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures" / "week_01" / "corpus_audit"
MAPPING_PATH = FIXTURE_DIRECTORY / "idmapping_selected.tab"
SWISS_PROT_PATH = FIXTURE_DIRECTORY / "uniprot_sprot.dat"


def _target_accessions() -> tuple[str, ...]:
    return tuple(
        record.primary_accession for record in parse_swiss_prot(SWISS_PROT_PATH)
    )


def _mapping_row(
    accession: str,
    group: str,
    entry_name: str = "SYNTHETIC_ID",
) -> str:
    return "\t".join(
        (
            accession,
            entry_name,
            "3",
            "4",
            "5",
            "6",
            "7",
            "UniRef100_SYNTHETIC",
            "UniRef90_SYNTHETIC",
            group,
        )
    )


def test_uniref50_audit_matches_hand_calculated_fixture() -> None:
    audit = audit_uniref50_membership(MAPPING_PATH, _target_accessions())

    assert audit.source_row_count == 6
    assert audit.target_accession_count == 6
    assert audit.mapped_accession_count == 4
    assert audit.missing_source_row_count == 1
    assert audit.blank_group_accession_count == 1
    assert audit.duplicate_source_accession_count == 0
    assert audit.conflicting_mapping_accession_count == 0
    assert audit.unique_group_count == 2
    assert audit.maximum_group_size == 2
    assert audit.group_size_histogram == {2: 2}


def test_uniref50_scan_returns_only_unambiguous_requested_mappings() -> None:
    scan = scan_uniref50_membership(
        MAPPING_PATH,
        _target_accessions(),
        target_entry_names=(
            "ALPHA_SYNTH",
            "DELTA_SYNTH",
            "EPSILON_SYNTH",
            "UNRELATED_SYNTH",
        ),
    )

    assert scan.accession_to_group == {
        "P00001": "UniRef50_GROUP_A",
        "P00002": "UniRef50_GROUP_A",
        "P00003": "UniRef50_GROUP_B",
        "P00004": "UniRef50_GROUP_B",
    }
    assert scan.missing_accessions == {"P00006"}
    assert scan.blank_group_accessions == {"P00005"}
    assert scan.duplicate_accessions == set()
    assert scan.conflicting_accessions == set()

    assert scan.entry_name_matches == {
        "ALPHA_SYNTH": UniRef50Match(
            accession="P00001",
            group="UniRef50_GROUP_A",
            accession_in_target_population=True,
        ),
        "UNRELATED_SYNTH": UniRef50Match(
            accession="Q99999",
            group="UniRef50_OTHER",
            accession_in_target_population=False,
        ),
    }
    assert scan.missing_entry_names == {"EPSILON_SYNTH"}
    assert scan.blank_group_entry_names == {"DELTA_SYNTH"}
    assert scan.duplicate_entry_names == set()
    assert scan.conflicting_entry_names == set()
    assert scan.audit.mapped_accession_count == len(scan.accession_to_group)


def test_uniref50_audit_is_equal_for_plain_gzip_and_repeated_runs(
    tmp_path: Path,
) -> None:
    compressed_path = tmp_path / "idmapping_selected.tab.gz"
    with gzip.open(compressed_path, mode="wt", encoding="utf-8") as compressed:
        compressed.write(MAPPING_PATH.read_text(encoding="utf-8"))

    first = audit_uniref50_membership(MAPPING_PATH, _target_accessions())
    second = audit_uniref50_membership(MAPPING_PATH, _target_accessions())
    compressed = audit_uniref50_membership(
        compressed_path,
        _target_accessions(),
    )

    assert first == second == compressed


def test_duplicate_and_conflicting_rows_do_not_inflate_group_sizes(
    tmp_path: Path,
) -> None:
    duplicate_path = tmp_path / "duplicate_rows.tab"
    source = MAPPING_PATH.read_text(encoding="utf-8")
    source += _mapping_row("P00001", "UniRef50_GROUP_A") + "\n"
    source += _mapping_row("P00004", "UniRef50_GROUP_C") + "\n"
    duplicate_path.write_text(source, encoding="utf-8")

    audit = audit_uniref50_membership(duplicate_path, _target_accessions())

    assert audit.source_row_count == 8
    assert audit.mapped_accession_count == 3
    assert audit.duplicate_source_accession_count == 2
    assert audit.conflicting_mapping_accession_count == 1
    assert audit.unique_group_count == 2
    assert audit.maximum_group_size == 2
    assert audit.group_size_histogram == {1: 1, 2: 1}


def test_entry_name_duplicates_and_conflicts_do_not_change_accession_mapping(
    tmp_path: Path,
) -> None:
    duplicate_path = tmp_path / "duplicate_entry_names.tab"
    source = MAPPING_PATH.read_text(encoding="utf-8")
    source += (
        _mapping_row(
            "P00001",
            "UniRef50_GROUP_A",
            entry_name="ALPHA_SYNTH",
        )
        + "\n"
    )
    source += (
        _mapping_row(
            "Q11111",
            "UniRef50_GROUP_A",
            entry_name="ALPHA_SYNTH",
        )
        + "\n"
    )
    source += (
        _mapping_row(
            "Q99999",
            "UniRef50_OTHER",
            entry_name="UNRELATED_SYNTH",
        )
        + "\n"
    )
    duplicate_path.write_text(source, encoding="utf-8")

    scan = scan_uniref50_membership(
        duplicate_path,
        ("P00001",),
        target_entry_names=("ALPHA_SYNTH", "UNRELATED_SYNTH"),
    )

    assert scan.accession_to_group == {"P00001": "UniRef50_GROUP_A"}
    assert scan.duplicate_accessions == {"P00001"}
    assert scan.conflicting_accessions == set()
    assert scan.entry_name_matches == {
        "UNRELATED_SYNTH": UniRef50Match(
            accession="Q99999",
            group="UniRef50_OTHER",
            accession_in_target_population=False,
        )
    }
    assert scan.duplicate_entry_names == {"ALPHA_SYNTH", "UNRELATED_SYNTH"}
    assert scan.conflicting_entry_names == {"ALPHA_SYNTH"}


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("P00001\tTOO_SHORT\n", "fewer than 10 columns"),
        (
            _mapping_row("P00001", "UniRef90_WRONG_LEVEL") + "\n",
            "invalid UniRef50 identifier",
        ),
    ],
)
def test_uniref50_audit_rejects_malformed_matched_rows(
    tmp_path: Path,
    row: str,
    message: str,
) -> None:
    malformed_path = tmp_path / "malformed.tab"
    malformed_path.write_text(row, encoding="utf-8")

    with pytest.raises(UniRef50ParseError, match=message):
        audit_uniref50_membership(malformed_path, ("P00001",))


def test_scan_rejects_malformed_row_matched_only_by_entry_name(
    tmp_path: Path,
) -> None:
    malformed_path = tmp_path / "malformed_entry_name_row.tab"
    malformed_path.write_text("Q99999\tPROTEINGYM_TARGET\n", encoding="utf-8")

    with pytest.raises(UniRef50ParseError, match="fewer than 10 columns"):
        scan_uniref50_membership(
            malformed_path,
            ("P00001",),
            target_entry_names=("PROTEINGYM_TARGET",),
        )


def test_uniref50_audit_rejects_invalid_target_input() -> None:
    with pytest.raises(ValueError, match="without target accessions"):
        audit_uniref50_membership(MAPPING_PATH, ())
    with pytest.raises(ValueError, match="duplicate target accession"):
        audit_uniref50_membership(MAPPING_PATH, ("P00001", "P00001"))
    with pytest.raises(ValueError, match="target entry names must not be empty"):
        scan_uniref50_membership(
            MAPPING_PATH,
            ("P00001",),
            target_entry_names=("",),
        )
    with pytest.raises(ValueError, match="duplicate target entry name"):
        scan_uniref50_membership(
            MAPPING_PATH,
            ("P00001",),
            target_entry_names=("ALPHA_SYNTH", "ALPHA_SYNTH"),
        )
