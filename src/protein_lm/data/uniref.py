"""Audit UniRef50 membership for selected Swiss-Prot accessions."""

from __future__ import annotations

import gzip
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_UNIREF50_PATTERN = re.compile(r"^UniRef50_\S+$")


class UniRef50ParseError(ValueError):
    """Raised when a matched identifier-mapping row is malformed."""


@dataclass(frozen=True)
class UniRef50Audit:
    """Aggregate UniRef50 coverage for a selected accession population."""

    source_row_count: int
    target_accession_count: int
    mapped_accession_count: int
    missing_source_row_count: int
    blank_group_accession_count: int
    duplicate_source_accession_count: int
    conflicting_mapping_accession_count: int
    unique_group_count: int
    maximum_group_size: int
    group_size_histogram: dict[int, int]


def audit_uniref50_membership(
    path: Path,
    target_accessions: Iterable[str],
) -> UniRef50Audit:
    """Stream identifier mappings and audit column-10 UniRef50 membership."""

    targets: set[str] = set()
    for accession in target_accessions:
        if not accession:
            raise ValueError("target accessions must not be empty")
        if accession in targets:
            raise ValueError(f"duplicate target accession: {accession}")
        targets.add(accession)
    if not targets:
        raise ValueError("cannot audit UniRef50 membership without target accessions")

    observed_groups: dict[str, str] = {}
    duplicate_accessions: set[str] = set()
    conflicting_accessions: set[str] = set()
    source_row_count = 0

    source_path = Path(path)
    if source_path.name.endswith(".gz"):
        source = gzip.open(source_path, mode="rt", encoding="utf-8")
    else:
        source = source_path.open(encoding="utf-8")

    with source:
        for line_number, raw_line in enumerate(source, start=1):
            source_row_count += 1
            accession, separator, remainder = raw_line.partition("\t")

            if accession not in targets:
                continue
            if not separator:
                raise UniRef50ParseError(
                    f"line {line_number}: matched row has fewer than 10 columns"
                )

            columns_after_accession = remainder.rstrip("\r\n").split("\t", 9)
            if len(columns_after_accession) < 9:
                raise UniRef50ParseError(
                    f"line {line_number}: matched row has fewer than 10 columns"
                )
            group = columns_after_accession[8]
            if group and _UNIREF50_PATTERN.fullmatch(group) is None:
                raise UniRef50ParseError(
                    f"line {line_number}: invalid UniRef50 identifier {group!r}"
                )

            if accession in conflicting_accessions:
                duplicate_accessions.add(accession)
                continue
            if accession not in observed_groups:
                observed_groups[accession] = group
                continue

            duplicate_accessions.add(accession)
            if observed_groups[accession] != group:
                del observed_groups[accession]
                conflicting_accessions.add(accession)

    group_sizes = Counter(group for group in observed_groups.values() if group)
    group_size_histogram = dict(sorted(Counter(group_sizes.values()).items()))
    blank_group_accession_count = sum(not group for group in observed_groups.values())

    audit = UniRef50Audit(
        source_row_count=source_row_count,
        target_accession_count=len(targets),
        mapped_accession_count=sum(group_sizes.values()),
        missing_source_row_count=(
            len(targets) - len(observed_groups) - len(conflicting_accessions)
        ),
        blank_group_accession_count=blank_group_accession_count,
        duplicate_source_accession_count=len(duplicate_accessions),
        conflicting_mapping_accession_count=len(conflicting_accessions),
        unique_group_count=len(group_sizes),
        maximum_group_size=max(group_sizes.values(), default=0),
        group_size_histogram=group_size_histogram,
    )
    _validate_reconciliations(audit)
    return audit


def _validate_reconciliations(audit: UniRef50Audit) -> None:
    categorized_targets = (
        audit.mapped_accession_count
        + audit.missing_source_row_count
        + audit.blank_group_accession_count
        + audit.conflicting_mapping_accession_count
    )
    if categorized_targets != audit.target_accession_count:
        raise RuntimeError("UniRef50 coverage counts do not reconcile")

    histogram_group_count = sum(audit.group_size_histogram.values())
    if histogram_group_count != audit.unique_group_count:
        raise RuntimeError("UniRef50 group counts do not reconcile")

    histogram_accession_count = sum(
        group_size * group_count
        for group_size, group_count in audit.group_size_histogram.items()
    )
    if histogram_accession_count != audit.mapped_accession_count:
        raise RuntimeError("UniRef50 group sizes do not reconcile")
