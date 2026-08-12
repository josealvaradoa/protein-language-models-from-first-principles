"""Read and validate only the frozen Task 4 catalog and reservation set."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from protein_lm.data.eligibility import CATALOG_COLUMNS
from protein_lm.data.model_data.contracts import (
    CatalogRecord,
    ModelDataConfig,
    ModelDataError,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GROUP = re.compile(r"^UniRef50_[\x21-\x7e]+$")
_VISIBLE_ASCII = re.compile(r"^[\x21-\x7e]+$")


def load_reserved_families(path: Path, config: ModelDataConfig) -> frozenset[str]:
    """Load the exact pinned ProteinGym family set without accepting duplicates."""

    content = _read_pinned(
        path, config.reserved_families_sha256, None, "reserved families"
    )
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ModelDataError("reserved families are not UTF-8") from error
    if not lines or any(not _GROUP.fullmatch(line) for line in lines):
        raise ModelDataError("reserved families contain an invalid UniRef50 group")
    families = frozenset(lines)
    if len(families) != len(lines) or len(families) != config.reserved_family_count:
        raise ModelDataError("reserved families are duplicated or have the wrong count")
    return families


def load_catalog(
    path: Path, config: ModelDataConfig, reserved: frozenset[str]
) -> list[CatalogRecord]:
    """Stream the catalog once, preserving only validated eligible rows."""

    records: list[CatalogRecord] = []
    accessions: set[str] = set()
    hasher = hashlib.sha256()
    byte_size = 0
    row_count = 0
    try:
        source = path.open("rb")
    except OSError as error:
        raise ModelDataError(f"could not open Task 4 catalog: {error}") from error
    with source:
        header = source.readline()
        hasher.update(header)
        byte_size += len(header)
        if _line(header, 1).split("\t") != list(CATALOG_COLUMNS):
            raise ModelDataError("Task 4 catalog header is not approved")
        for line_number, raw in enumerate(source, start=2):
            row_count += 1
            hasher.update(raw)
            byte_size += len(raw)
            columns = _line(raw, line_number).split("\t")
            if len(columns) != len(CATALOG_COLUMNS):
                raise ModelDataError(
                    f"catalog line {line_number} has the wrong column count"
                )
            eligible = _boolean(columns[9], line_number, "eligible")
            flags = [
                _boolean(columns[index], line_number, CATALOG_COLUMNS[index])
                for index in range(4, 9)
            ]
            reserved_flag = _boolean(columns[12], line_number, "reservation flag")
            if not eligible:
                continue
            if any(flags) or columns[10]:
                raise ModelDataError(
                    f"catalog line {line_number}: eligible row violates exclusion contract"
                )
            accession, sequence, sequence_digest, raw_length, group = (
                columns[index] for index in (0, 1, 2, 3, 11)
            )
            if _VISIBLE_ASCII.fullmatch(accession) is None or accession in accessions:
                raise ModelDataError(
                    f"catalog line {line_number}: accession is invalid or duplicated"
                )
            if set(sequence) - set(config.canonical_amino_acids) or not sequence:
                raise ModelDataError(
                    f"catalog line {line_number}: sequence is not canonical"
                )
            if (
                _SHA256.fullmatch(sequence_digest) is None
                or hashlib.sha256(sequence.encode("ascii")).hexdigest()
                != sequence_digest
            ):
                raise ModelDataError(
                    f"catalog line {line_number}: sequence SHA-256 drifted"
                )
            try:
                biological_length = int(raw_length)
            except ValueError as error:
                raise ModelDataError(
                    f"catalog line {line_number}: biological length is invalid"
                ) from error
            if (
                biological_length != len(sequence)
                or not config.minimum_length
                <= biological_length
                <= config.maximum_length
            ):
                raise ModelDataError(
                    f"catalog line {line_number}: biological length violates contract"
                )
            if _GROUP.fullmatch(group) is None or reserved_flag != (group in reserved):
                raise ModelDataError(
                    f"catalog line {line_number}: ProteinGym reservation disagrees"
                )
            accessions.add(accession)
            records.append(
                CatalogRecord(
                    accession, sequence, sequence_digest, biological_length, group
                )
            )
    if (
        hasher.hexdigest() != config.task4_catalog_sha256
        or byte_size != config.task4_catalog_byte_size
        or row_count != config.task4_catalog_row_count
    ):
        raise ModelDataError("Task 4 catalog identity drifted")
    return records


def _read_pinned(
    path: Path, expected_sha256: str, expected_size: int | None, name: str
) -> bytes:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ModelDataError(f"could not read {name}: {error}") from error
    if hashlib.sha256(content).hexdigest() != expected_sha256 or (
        expected_size is not None and len(content) != expected_size
    ):
        raise ModelDataError(f"{name} identity drifted")
    return content


def _line(raw: bytes, line_number: int) -> str:
    if not raw.endswith(b"\n") or b"\r" in raw:
        raise ModelDataError(f"catalog line {line_number} must be LF-terminated")
    try:
        return raw[:-1].decode("utf-8")
    except UnicodeDecodeError as error:
        raise ModelDataError(f"catalog line {line_number} is not UTF-8") from error


def _boolean(value: str, line_number: int, name: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ModelDataError(f"catalog line {line_number}: {name} is not a boolean")
