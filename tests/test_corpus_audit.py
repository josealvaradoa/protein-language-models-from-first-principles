from pathlib import Path

import pytest

from protein_lm.data.corpus_audit import audit_swiss_prot
from protein_lm.data.uniprot import parse_swiss_prot

FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "week_01"
    / "corpus_audit"
    / "uniprot_sprot.dat"
)


def test_swiss_prot_audit_matches_hand_calculated_fixture() -> None:
    audit = audit_swiss_prot(parse_swiss_prot(FIXTURE_PATH))

    assert audit.record_count == 6
    assert audit.residue_count == 24
    assert audit.fragment_count == 1
    assert audit.canonical_only_record_count == 4
    assert audit.records_with_noncanonical == 2
    assert audit.length_histogram == {1: 1, 3: 1, 4: 1, 5: 2, 6: 1}
    assert audit.length_percentiles == {
        "p0": 1,
        "p1": 1,
        "p5": 1,
        "p25": 3,
        "p50": 4,
        "p75": 5,
        "p90": 6,
        "p95": 6,
        "p99": 6,
        "p99.5": 6,
        "p99.9": 6,
        "p100": 6,
    }
    assert audit.noncanonical_occurrence_counts == {
        "B": 1,
        "J": 1,
        "X": 1,
        "Z": 1,
        "U": 1,
        "O": 1,
    }
    assert audit.noncanonical_record_counts == {
        "B": 1,
        "J": 1,
        "X": 1,
        "Z": 1,
        "U": 1,
        "O": 1,
    }
    assert audit.unique_sequence_count == 5
    assert audit.duplicate_sequence_group_count == 1
    assert audit.records_in_duplicate_groups == 2
    assert audit.redundant_record_count == 1
    assert audit.maximum_duplicate_multiplicity == 2
    assert audit.duplicate_multiplicity_histogram == {1: 4, 2: 1}
    assert audit.ec_state_counts == {
        "no_ec": 2,
        "partial_only": 1,
        "single_complete": 1,
        "multiple_complete": 1,
        "mixed_complete_partial": 1,
    }
    assert audit.single_complete_ec_class_counts == {"1": 1}
    assert audit.complete_ec_label_counts == {
        "1.1.1.1": 1,
        "3.1.1.1": 1,
        "3.1.1.2": 1,
        "4.2.1.1": 1,
    }


def test_swiss_prot_audit_is_repeatable() -> None:
    first = audit_swiss_prot(parse_swiss_prot(FIXTURE_PATH))
    second = audit_swiss_prot(parse_swiss_prot(FIXTURE_PATH))

    assert first == second


def test_swiss_prot_audit_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty Swiss-Prot"):
        audit_swiss_prot(())
