"""Aggregate deterministic facts from parsed Swiss-Prot records."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from protein_lm.data.uniprot import SwissProtRecord

NONCANONICAL_SYMBOLS = ("B", "J", "X", "Z", "U", "O")
_COMPLETE_EC_PATTERN = re.compile(r"^\d+\.\d+\.\d+\.\d+$")
_PERCENTILES = (
    ("p0", 0.0),
    ("p1", 1.0),
    ("p5", 5.0),
    ("p25", 25.0),
    ("p50", 50.0),
    ("p75", 75.0),
    ("p90", 90.0),
    ("p95", 95.0),
    ("p99", 99.0),
    ("p99.5", 99.5),
    ("p99.9", 99.9),
    ("p100", 100.0),
)
_EC_STATES = (
    "no_ec",
    "partial_only",
    "single_complete",
    "multiple_complete",
    "mixed_complete_partial",
)


@dataclass(frozen=True)
class SwissProtAudit:
    """The aggregate Swiss-Prot facts allowed by the Task 2 contract."""

    record_count: int
    residue_count: int
    fragment_count: int
    canonical_only_record_count: int
    records_with_noncanonical: int
    length_histogram: dict[int, int]
    length_percentiles: dict[str, int]
    noncanonical_occurrence_counts: dict[str, int]
    noncanonical_record_counts: dict[str, int]
    unique_sequence_count: int
    duplicate_sequence_group_count: int
    records_in_duplicate_groups: int
    redundant_record_count: int
    maximum_duplicate_multiplicity: int
    duplicate_multiplicity_histogram: dict[int, int]
    ec_state_counts: dict[str, int]
    single_complete_ec_class_counts: dict[str, int]
    complete_ec_label_counts: dict[str, int]


def audit_swiss_prot(records: Iterable[SwissProtRecord]) -> SwissProtAudit:
    """Consume parsed records once and return deterministic aggregate facts."""

    lengths: list[int] = []
    fragment_count = 0
    canonical_only_record_count = 0
    noncanonical_occurrences: Counter[str] = Counter()
    noncanonical_records: Counter[str] = Counter()
    sequence_counts: Counter[str] = Counter()
    ec_states: Counter[str] = Counter()
    single_complete_ec_classes: Counter[str] = Counter()
    complete_ec_labels: Counter[str] = Counter()

    for record in records:
        lengths.append(record.declared_length)
        fragment_count += int(record.is_fragment)
        sequence_counts[record.sequence] += 1

        residue_counts = Counter(record.sequence)
        has_noncanonical = False
        for symbol in NONCANONICAL_SYMBOLS:
            occurrence_count = residue_counts[symbol]
            noncanonical_occurrences[symbol] += occurrence_count
            if occurrence_count:
                noncanonical_records[symbol] += 1
                has_noncanonical = True
        if not has_noncanonical:
            canonical_only_record_count += 1

        complete_ec = tuple(
            ec_number
            for ec_number in record.ec_numbers
            if _COMPLETE_EC_PATTERN.fullmatch(ec_number)
        )
        partial_ec_count = len(record.ec_numbers) - len(complete_ec)
        complete_ec_labels.update(complete_ec)

        if not record.ec_numbers:
            ec_states["no_ec"] += 1
        elif not complete_ec:
            ec_states["partial_only"] += 1
        elif partial_ec_count:
            ec_states["mixed_complete_partial"] += 1
        elif len(complete_ec) == 1:
            ec_states["single_complete"] += 1
            top_level_class = complete_ec[0].split(".", maxsplit=1)[0]
            single_complete_ec_classes[top_level_class] += 1
        else:
            ec_states["multiple_complete"] += 1

    if not lengths:
        raise ValueError("cannot audit an empty Swiss-Prot record stream")

    sorted_lengths = sorted(lengths)
    length_histogram = dict(sorted(Counter(lengths).items()))
    length_percentiles = {
        name: _nearest_rank(sorted_lengths, percentile)
        for name, percentile in _PERCENTILES
    }

    multiplicities = tuple(sequence_counts.values())
    multiplicity_histogram = dict(sorted(Counter(multiplicities).items()))
    duplicate_multiplicities = tuple(
        multiplicity for multiplicity in multiplicities if multiplicity > 1
    )

    audit = SwissProtAudit(
        record_count=len(lengths),
        residue_count=sum(lengths),
        fragment_count=fragment_count,
        canonical_only_record_count=canonical_only_record_count,
        records_with_noncanonical=len(lengths) - canonical_only_record_count,
        length_histogram=length_histogram,
        length_percentiles=length_percentiles,
        noncanonical_occurrence_counts={
            symbol: noncanonical_occurrences[symbol] for symbol in NONCANONICAL_SYMBOLS
        },
        noncanonical_record_counts={
            symbol: noncanonical_records[symbol] for symbol in NONCANONICAL_SYMBOLS
        },
        unique_sequence_count=len(sequence_counts),
        duplicate_sequence_group_count=len(duplicate_multiplicities),
        records_in_duplicate_groups=sum(duplicate_multiplicities),
        redundant_record_count=sum(
            multiplicity - 1 for multiplicity in duplicate_multiplicities
        ),
        maximum_duplicate_multiplicity=max(multiplicities),
        duplicate_multiplicity_histogram=multiplicity_histogram,
        ec_state_counts={state: ec_states[state] for state in _EC_STATES},
        single_complete_ec_class_counts=dict(
            sorted(single_complete_ec_classes.items())
        ),
        complete_ec_label_counts=dict(sorted(complete_ec_labels.items())),
    )
    _validate_reconciliations(audit)
    return audit


def _nearest_rank(sorted_values: list[int], percentile: float) -> int:
    if percentile == 0:
        return sorted_values[0]
    rank = math.ceil((percentile / 100) * len(sorted_values))
    return sorted_values[rank - 1]


def _validate_reconciliations(audit: SwissProtAudit) -> None:
    if sum(audit.length_histogram.values()) != audit.record_count:
        raise RuntimeError("length histogram does not reconcile with record count")
    histogram_residues = sum(
        length * count for length, count in audit.length_histogram.items()
    )
    if histogram_residues != audit.residue_count:
        raise RuntimeError("length histogram does not reconcile with residue count")
    if (
        audit.canonical_only_record_count + audit.records_with_noncanonical
        != audit.record_count
    ):
        raise RuntimeError("symbol record counts do not reconcile")
    duplicate_records = sum(
        multiplicity * group_count
        for multiplicity, group_count in audit.duplicate_multiplicity_histogram.items()
    )
    if duplicate_records != audit.record_count:
        raise RuntimeError("duplicate multiplicities do not reconcile")
    if sum(audit.ec_state_counts.values()) != audit.record_count:
        raise RuntimeError("EC state counts do not reconcile")
