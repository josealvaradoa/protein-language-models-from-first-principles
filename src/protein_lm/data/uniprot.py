"""Parse the fields needed from reviewed Swiss-Prot records."""

from __future__ import annotations

import gzip
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

_ID_PATTERN = re.compile(r"^ID\s+(?P<name>\S+)\s+Reviewed;\s+(?P<length>\d+)\s+AA\.$")
_SQ_PATTERN = re.compile(r"^SQ\s+SEQUENCE\s+(?P<length>\d+)\s+AA;")
_SEQUENCE_PATTERN = re.compile(r"^[A-Z]+$")
_EC_PATTERN = re.compile(r"\bEC=([^;]+);")
_FRAGMENT_PATTERN = re.compile(r"^DE\s+Flags:.*\bFragments?;")


class SwissProtParseError(ValueError):
    """Raised when a Swiss-Prot record is incomplete or inconsistent."""


@dataclass(frozen=True)
class SwissProtRecord:
    """One reviewed Swiss-Prot entry and its canonical sequence."""

    entry_name: str
    primary_accession: str
    declared_length: int
    sequence: str
    is_fragment: bool
    ec_numbers: tuple[str, ...]


def parse_swiss_prot(path: Path) -> Iterator[SwissProtRecord]:
    """Read a flat file one line at a time and yield complete records."""

    source_path = Path(path)
    entry_name: str | None = None
    primary_accession: str | None = None
    declared_length: int | None = None
    sequence_declared_length: int | None = None
    sequence_parts: list[str] = []
    is_fragment = False
    ec_numbers: list[str] = []
    reading_sequence = False
    seen_primary_accessions: set[str] = set()

    if source_path.name.endswith(".gz"):
        source = gzip.open(source_path, mode="rt", encoding="utf-8")
    else:
        source = source_path.open(encoding="utf-8")

    with source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.rstrip()

            if line.startswith("ID"):
                if entry_name is not None:
                    raise SwissProtParseError(f"line {line_number}: new ID before //")
                match = _ID_PATTERN.fullmatch(line)
                if match is None:
                    raise SwissProtParseError(
                        f"line {line_number}: malformed or unreviewed ID"
                    )
                entry_name = match.group("name")
                declared_length = int(match.group("length"))
                continue

            if entry_name is None:
                if line:
                    raise SwissProtParseError(
                        f"line {line_number}: content outside a record"
                    )
                continue

            if line == "//":
                if primary_accession is None or sequence_declared_length is None:
                    raise SwissProtParseError(
                        f"line {line_number}: incomplete record {entry_name}"
                    )
                if primary_accession in seen_primary_accessions:
                    raise SwissProtParseError(
                        f"line {line_number}: duplicate accession {primary_accession}"
                    )
                sequence = "".join(sequence_parts)
                if (
                    declared_length != sequence_declared_length
                    or declared_length != len(sequence)
                ):
                    raise SwissProtParseError(
                        f"{primary_accession}: sequence length mismatch "
                        f"(ID={declared_length}, SQ={sequence_declared_length}, "
                        f"parsed={len(sequence)})"
                    )
                record = SwissProtRecord(
                    entry_name=entry_name,
                    primary_accession=primary_accession,
                    declared_length=declared_length,
                    sequence=sequence,
                    is_fragment=is_fragment,
                    ec_numbers=tuple(ec_numbers),
                )
                seen_primary_accessions.add(primary_accession)
                entry_name = None
                primary_accession = None
                declared_length = None
                sequence_declared_length = None
                sequence_parts = []
                is_fragment = False
                ec_numbers = []
                reading_sequence = False
                yield record
                continue

            if reading_sequence:
                sequence_chunk = "".join(line.split())
                if _SEQUENCE_PATTERN.fullmatch(sequence_chunk) is None:
                    raise SwissProtParseError(f"line {line_number}: malformed sequence")
                sequence_parts.append(sequence_chunk)
                continue

            if line.startswith("AC") and primary_accession is None:
                primary_accession = line[2:].split(";", maxsplit=1)[0].strip()
                if not primary_accession:
                    raise SwissProtParseError(
                        f"line {line_number}: empty primary accession"
                    )
                continue

            if line.startswith("DE"):
                if _FRAGMENT_PATTERN.search(line):
                    is_fragment = True
                for match in _EC_PATTERN.finditer(line):
                    ec_number = match.group(1).strip()
                    if ec_number not in ec_numbers:
                        ec_numbers.append(ec_number)
                continue

            if line.startswith("SQ"):
                match = _SQ_PATTERN.match(line)
                if match is None:
                    raise SwissProtParseError(f"line {line_number}: malformed SQ")
                sequence_declared_length = int(match.group("length"))
                reading_sequence = True

    if entry_name is not None:
        raise SwissProtParseError(f"record {entry_name} is missing //")
