import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from protein_lm.data.proteingym import (
    PROTEINGYM_V1_3_PIN,
    ProteinGymSourcePin,
    ProteinGymValidationError,
    scan_proteingym_metadata,
    verify_proteingym_source,
)

FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "week_01"
    / "corpus_audit"
    / "DMS_substitutions.csv"
)
REQUIRED_HEADER = (
    "DMS_id,UniProt_ID,target_seq,seq_len,DMS_number_single_mutants,"
    "ProteinGym_version,coarse_selection_type\n"
)


def _pin_for(path: Path) -> ProteinGymSourcePin:
    content = path.read_bytes()
    git_blob = hashlib.sha1(usedforsecurity=False)
    git_blob.update(f"blob {len(content)}\0".encode())
    git_blob.update(content)
    return replace(
        PROTEINGYM_V1_3_PIN,
        expected_bytes=len(content),
        expected_sha256=hashlib.sha256(content).hexdigest(),
        expected_git_blob_sha1=git_blob.hexdigest(),
    )


def test_proteingym_scan_matches_hand_calculated_fixture() -> None:
    scan = scan_proteingym_metadata(FIXTURE_PATH)

    assert len(scan.assays) == 6
    assert scan.target_entry_names == (
        "ALPHA_SYNTH",
        "DELTA_SYNTH",
        "EPSILON_SYNTH",
        "UNRELATED_SYNTH",
    )
    assert scan.assays[1].single_mutant_count == 0
    assert scan.assays[2].target_sequence == "ACDEG"

    audit = scan.audit
    assert audit.assay_count == 6
    assert audit.target_entry_name_count == 4
    assert audit.target_reference_pair_count == 5
    assert audit.entry_names_with_multiple_assays_count == 1
    assert audit.entry_names_with_multiple_target_sequences_count == 1
    assert audit.assays_per_target_histogram == {1: 3, 3: 1}
    assert audit.assay_reference_length_histogram == {1: 1, 2: 1, 3: 1, 5: 3}
    assert audit.canonical_reference_assay_count == 5
    assert audit.noncanonical_reference_assay_count == 1
    assert audit.noncanonical_symbol_counts == {"Z": 1}
    assert audit.assays_with_single_mutants == 5
    assert audit.assays_without_single_mutants == 1
    assert audit.single_mutant_variant_count == 27
    assert audit.cohort_version_counts == {"0.1": 3, "1": 3}
    assert audit.coarse_selection_type_counts == {
        "Activity": 2,
        "Binding": 1,
        "Expression": 1,
        "OrganismalFitness": 1,
        "Stability": 1,
    }


def test_proteingym_source_verification_checks_both_hash_identities(
    tmp_path: Path,
) -> None:
    pin = _pin_for(FIXTURE_PATH)
    verification = verify_proteingym_source(FIXTURE_PATH, pin, chunk_bytes=7)

    assert verification.byte_size == pin.expected_bytes
    assert verification.sha256 == pin.expected_sha256
    assert verification.git_blob_sha1 == pin.expected_git_blob_sha1

    corrupted_path = tmp_path / "DMS_substitutions.csv"
    corrupted = bytearray(FIXTURE_PATH.read_bytes())
    corrupted[-2] = ord("B")
    corrupted_path.write_bytes(corrupted)

    with pytest.raises(ProteinGymValidationError, match="SHA-256 mismatch"):
        verify_proteingym_source(corrupted_path, pin)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            "PG_ONE,P53,ACD,4,3,1,Activity\n",
            "target sequence length mismatch",
        ),
        (
            "PG_ONE,P53,AcD,3,3,1,Activity\n",
            "malformed target sequence",
        ),
        (
            "PG_ONE,P53,ACD,3,-1,1,Activity\n",
            "must be a nonnegative integer",
        ),
        (
            "PG_ONE,P53,ACD,3,3,1,Activity\nPG_ONE,P54,ACD,3,3,1,Binding\n",
            "duplicate DMS_id",
        ),
        (
            "PG_ONE,P53,ACD,3,3,1,Activity,EXTRA\n",
            "more values than header columns",
        ),
    ],
)
def test_proteingym_parser_rejects_malformed_rows(
    tmp_path: Path,
    body: str,
    message: str,
) -> None:
    malformed_path = tmp_path / "malformed.csv"
    malformed_path.write_text(REQUIRED_HEADER + body, encoding="utf-8")

    with pytest.raises(ProteinGymValidationError, match=message):
        scan_proteingym_metadata(malformed_path)


def test_proteingym_parser_rejects_missing_columns_and_empty_metadata(
    tmp_path: Path,
) -> None:
    missing_column_path = tmp_path / "missing_column.csv"
    missing_column_path.write_text(
        REQUIRED_HEADER.replace("DMS_id,", "") + "PG_ONE,P53,ACD,3,3,1,Activity\n",
        encoding="utf-8",
    )
    with pytest.raises(ProteinGymValidationError, match="missing required columns"):
        scan_proteingym_metadata(missing_column_path)

    empty_path = tmp_path / "empty.csv"
    empty_path.write_text(REQUIRED_HEADER, encoding="utf-8")
    with pytest.raises(ValueError, match="empty ProteinGym"):
        scan_proteingym_metadata(empty_path)


def test_proteingym_v1_3_pin_matches_verified_source_identity() -> None:
    assert PROTEINGYM_V1_3_PIN.release == "v1.3"
    assert PROTEINGYM_V1_3_PIN.commit == ("1f8de974dead8ff7501eff087b725d14a965e9f9")
    assert PROTEINGYM_V1_3_PIN.expected_bytes == 208734
    assert PROTEINGYM_V1_3_PIN.expected_sha256 == (
        "a8f498011532a74aa9fe556a50555a75e928c5837d19c06a87592ae04049b308"
    )
