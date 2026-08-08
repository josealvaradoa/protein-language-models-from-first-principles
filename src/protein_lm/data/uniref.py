"""Audit UniRef50 membership for selected Swiss-Prot accessions."""

from __future__ import annotations

import gzip
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_UNIREF50_PATTERN = re.compile(r"^UniRef50_[\x21-\x7e]+$")


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


@dataclass(frozen=True)
class UniRef50Match:
    """One unambiguous entry-name match and its corpus-membership status."""

    accession: str
    group: str
    accession_in_target_population: bool


@dataclass(frozen=True)
class UniRef50Scan:
    """Aggregate audit plus the small mappings needed by later local steps."""

    audit: UniRef50Audit
    accession_to_group: dict[str, str]
    missing_accessions: frozenset[str]
    blank_group_accessions: frozenset[str]
    duplicate_accessions: frozenset[str]
    conflicting_accessions: frozenset[str]
    entry_name_matches: dict[str, UniRef50Match]
    missing_entry_names: frozenset[str]
    blank_group_entry_names: frozenset[str]
    duplicate_entry_names: frozenset[str]
    conflicting_entry_names: frozenset[str]


def audit_uniref50_membership(
    path: Path,
    target_accessions: Iterable[str],
) -> UniRef50Audit:
    """Stream identifier mappings and audit column-10 UniRef50 membership."""

    return scan_uniref50_membership(path, target_accessions).audit


def scan_uniref50_membership(
    path: Path,
    target_accessions: Iterable[str],
    *,
    target_entry_names: Iterable[str] = (),
) -> UniRef50Scan:
    """Resolve requested accessions and entry names in one streaming pass."""

    accession_targets: set[str] = set()
    for accession in target_accessions:
        if not accession:
            raise ValueError("target accessions must not be empty")
        if accession in accession_targets:
            raise ValueError(f"duplicate target accession: {accession}")
        accession_targets.add(accession)
    if not accession_targets:
        raise ValueError("cannot audit UniRef50 membership without target accessions")

    entry_name_targets: set[str] = set()
    for entry_name in target_entry_names:
        if not entry_name:
            raise ValueError("target entry names must not be empty")
        if entry_name in entry_name_targets:
            raise ValueError(f"duplicate target entry name: {entry_name}")
        entry_name_targets.add(entry_name)

    observed_accessions: dict[str, UniRef50Match] = {}
    duplicate_accessions: set[str] = set()
    conflicting_accessions: set[str] = set()
    observed_entry_names: dict[str, UniRef50Match] = {}
    duplicate_entry_names: set[str] = set()
    conflicting_entry_names: set[str] = set()
    source_row_count = 0

    source_path = Path(path)
    if source_path.name.endswith(".gz"):
        source = gzip.open(
            source_path,
            mode="rt",
            encoding="utf-8",
            newline="",
        )
    else:
        source = source_path.open(encoding="utf-8", newline="")

    with source:
        for line_number, raw_line in enumerate(source, start=1):
            source_row_count += 1
            accession, separator, remainder = raw_line.partition("\t")
            accession_is_target = accession in accession_targets

            if not separator:
                possible_accession = _remove_line_ending(raw_line)
                if possible_accession in accession_targets:
                    raise UniRef50ParseError(
                        f"line {line_number}: matched row has fewer than 10 columns"
                    )
                continue

            entry_name, next_separator, _ = remainder.partition("\t")
            if not next_separator:
                entry_name = _remove_line_ending(entry_name)
            entry_name_is_target = entry_name in entry_name_targets
            if not accession_is_target and not entry_name_is_target:
                continue

            line = _remove_line_ending(raw_line)
            accession, _, remainder = line.partition("\t")
            columns_after_accession = remainder.split("\t", 9)
            if len(columns_after_accession) < 9:
                raise UniRef50ParseError(
                    f"line {line_number}: matched row has fewer than 10 columns"
                )
            group = columns_after_accession[8]
            if group and _UNIREF50_PATTERN.fullmatch(group) is None:
                raise UniRef50ParseError(
                    f"line {line_number}: invalid UniRef50 identifier {group!r}"
                )

            match = UniRef50Match(
                accession=accession,
                group=group,
                accession_in_target_population=accession_is_target,
            )
            if accession_is_target:
                _record_match(
                    key=accession,
                    match=match,
                    observed=observed_accessions,
                    duplicates=duplicate_accessions,
                    conflicts=conflicting_accessions,
                )
            if entry_name_is_target:
                _record_match(
                    key=entry_name,
                    match=match,
                    observed=observed_entry_names,
                    duplicates=duplicate_entry_names,
                    conflicts=conflicting_entry_names,
                )

    accession_to_group = {
        accession: match.group
        for accession, match in observed_accessions.items()
        if match.group
    }
    missing_accessions = (
        accession_targets - observed_accessions.keys() - conflicting_accessions
    )
    blank_group_accessions = {
        accession for accession, match in observed_accessions.items() if not match.group
    }
    group_sizes = Counter(accession_to_group.values())
    group_size_histogram = dict(sorted(Counter(group_sizes.values()).items()))

    audit = UniRef50Audit(
        source_row_count=source_row_count,
        target_accession_count=len(accession_targets),
        mapped_accession_count=sum(group_sizes.values()),
        missing_source_row_count=len(missing_accessions),
        blank_group_accession_count=len(blank_group_accessions),
        duplicate_source_accession_count=len(duplicate_accessions),
        conflicting_mapping_accession_count=len(conflicting_accessions),
        unique_group_count=len(group_sizes),
        maximum_group_size=max(group_sizes.values(), default=0),
        group_size_histogram=group_size_histogram,
    )
    _validate_reconciliations(audit)

    entry_name_matches = {
        entry_name: match
        for entry_name, match in observed_entry_names.items()
        if match.group
    }
    missing_entry_names = (
        entry_name_targets - observed_entry_names.keys() - conflicting_entry_names
    )
    blank_group_entry_names = {
        entry_name
        for entry_name, match in observed_entry_names.items()
        if not match.group
    }

    return UniRef50Scan(
        audit=audit,
        accession_to_group=accession_to_group,
        missing_accessions=frozenset(missing_accessions),
        blank_group_accessions=frozenset(blank_group_accessions),
        duplicate_accessions=frozenset(duplicate_accessions),
        conflicting_accessions=frozenset(conflicting_accessions),
        entry_name_matches=entry_name_matches,
        missing_entry_names=frozenset(missing_entry_names),
        blank_group_entry_names=frozenset(blank_group_entry_names),
        duplicate_entry_names=frozenset(duplicate_entry_names),
        conflicting_entry_names=frozenset(conflicting_entry_names),
    )


def _remove_line_ending(raw_line: str) -> str:
    if raw_line.endswith("\r\n"):
        return raw_line[:-2]
    if raw_line.endswith("\n"):
        return raw_line[:-1]
    if raw_line.endswith("\r"):
        raise UniRef50ParseError("unsupported bare CR line ending")
    return raw_line


def _record_match(
    *,
    key: str,
    match: UniRef50Match,
    observed: dict[str, UniRef50Match],
    duplicates: set[str],
    conflicts: set[str],
) -> None:
    if key in conflicts:
        duplicates.add(key)
        return
    if key not in observed:
        observed[key] = match
        return

    duplicates.add(key)
    if observed[key] != match:
        del observed[key]
        conflicts.add(key)


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
