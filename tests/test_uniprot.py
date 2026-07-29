import gzip
from pathlib import Path

import pytest

from protein_lm.data.uniprot import (
    SwissProtParseError,
    SwissProtRecord,
    parse_swiss_prot,
)

FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures" / "week_01" / "corpus_audit"


def test_parser_yields_complete_records() -> None:
    records = tuple(parse_swiss_prot(FIXTURE_DIRECTORY / "uniprot_sprot.dat"))

    assert len(records) == 6
    assert records[0] == SwissProtRecord(
        entry_name="ALPHA_SYNTH",
        primary_accession="P00001",
        declared_length=5,
        sequence="ACDEF",
        is_fragment=False,
        ec_numbers=(),
    )
    assert records[1].sequence == records[0].sequence
    assert records[1].ec_numbers == ("1.1.1.1",)
    assert records[2].is_fragment is True
    assert records[2].sequence == "ABJXZU"
    assert records[2].ec_numbers == ("2.7.-.-",)
    assert records[3].ec_numbers == ("3.1.1.1", "3.1.1.2")
    assert records[4].ec_numbers == ("4.2.1.1", "4.2.-.-")
    assert records[5].sequence == "W"


def test_parser_reads_gzip_without_extracting_it(tmp_path: Path) -> None:
    plain_path = FIXTURE_DIRECTORY / "uniprot_sprot.dat"
    compressed_path = tmp_path / "uniprot_sprot.dat.gz"
    with gzip.open(compressed_path, mode="wt", encoding="utf-8") as compressed:
        compressed.write(plain_path.read_text(encoding="utf-8"))

    assert tuple(parse_swiss_prot(compressed_path)) == tuple(
        parse_swiss_prot(plain_path)
    )


def test_parser_rejects_length_mismatch() -> None:
    malformed_path = FIXTURE_DIRECTORY / "uniprot_sprot_length_mismatch.dat"

    with pytest.raises(
        SwissProtParseError,
        match=r"P99999.*ID=4, SQ=4, parsed=3",
    ):
        tuple(parse_swiss_prot(malformed_path))


def test_parser_rejects_duplicate_primary_accession(tmp_path: Path) -> None:
    source = (FIXTURE_DIRECTORY / "uniprot_sprot.dat").read_text(encoding="utf-8")
    duplicate_source = source.replace("AC   P00002;", "AC   P00001;")
    duplicate_path = tmp_path / "duplicate_accession.dat"
    duplicate_path.write_text(duplicate_source, encoding="utf-8")

    with pytest.raises(SwissProtParseError, match="duplicate accession P00001"):
        tuple(parse_swiss_prot(duplicate_path))
