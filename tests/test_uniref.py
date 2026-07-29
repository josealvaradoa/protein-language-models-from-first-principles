import gzip
from pathlib import Path

import pytest

from protein_lm.data.uniprot import parse_swiss_prot
from protein_lm.data.uniref import (
    UniRef50ParseError,
    audit_uniref50_membership,
)

FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures" / "week_01" / "corpus_audit"
MAPPING_PATH = FIXTURE_DIRECTORY / "idmapping_selected.tab"
SWISS_PROT_PATH = FIXTURE_DIRECTORY / "uniprot_sprot.dat"


def _target_accessions() -> tuple[str, ...]:
    return tuple(
        record.primary_accession for record in parse_swiss_prot(SWISS_PROT_PATH)
    )


def _mapping_row(accession: str, group: str) -> str:
    return "\t".join(
        (
            accession,
            "SYNTHETIC_ID",
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


def test_uniref50_audit_rejects_invalid_target_input() -> None:
    with pytest.raises(ValueError, match="without target accessions"):
        audit_uniref50_membership(MAPPING_PATH, ())
    with pytest.raises(ValueError, match="duplicate target accession"):
        audit_uniref50_membership(MAPPING_PATH, ("P00001", "P00001"))
