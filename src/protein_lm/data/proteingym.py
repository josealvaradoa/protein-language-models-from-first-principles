"""Validate and audit the pinned ProteinGym v1.3 metadata."""

from __future__ import annotations

import csv
import hashlib
import hmac
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

_REQUIRED_COLUMNS = (
    "DMS_id",
    "UniProt_ID",
    "target_seq",
    "seq_len",
    "DMS_number_single_mutants",
    "ProteinGym_version",
    "coarse_selection_type",
)
_CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_IDENTIFIER_PATTERN = re.compile(r"^\S+$")
_SEQUENCE_PATTERN = re.compile(r"^[A-Z]+$")
_POSITIVE_INTEGER_PATTERN = re.compile(r"^[1-9]\d*$")
_NONNEGATIVE_INTEGER_PATTERN = re.compile(r"^\d+$")


class ProteinGymValidationError(ValueError):
    """Raised when the pinned source or one metadata row is invalid."""


@dataclass(frozen=True)
class ProteinGymSourcePin:
    """Immutable identity of the approved ProteinGym metadata source."""

    release: str
    tag: str
    commit: str
    filename: str
    url: str
    expected_bytes: int
    expected_sha256: str
    expected_git_blob_sha1: str
    license_spdx: str


PROTEINGYM_V1_3_PIN = ProteinGymSourcePin(
    release="v1.3",
    tag="PG_v1.3",
    commit="1f8de974dead8ff7501eff087b725d14a965e9f9",
    filename="DMS_substitutions.csv",
    url=(
        "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/"
        "1f8de974dead8ff7501eff087b725d14a965e9f9/"
        "reference_files/DMS_substitutions.csv"
    ),
    expected_bytes=208734,
    expected_sha256=(
        "a8f498011532a74aa9fe556a50555a75e928c5837d19c06a87592ae04049b308"
    ),
    expected_git_blob_sha1="8d1ea9a19c0404b511cd24378b25c2a5f86f10e9",
    license_spdx="MIT",
)


@dataclass(frozen=True)
class ProteinGymSourceVerification:
    """Calculated identity of one local ProteinGym metadata file."""

    byte_size: int
    sha256: str
    git_blob_sha1: str


@dataclass(frozen=True)
class ProteinGymAssay:
    """The fields needed from one ProteinGym substitution assay."""

    assay_id: str
    entry_name: str
    target_sequence: str
    declared_length: int
    single_mutant_count: int
    cohort_version: str
    coarse_selection_type: str


@dataclass(frozen=True)
class ProteinGymMetadataAudit:
    """Aggregate source facts that do not expose sequences or assay labels."""

    assay_count: int
    target_entry_name_count: int
    target_reference_pair_count: int
    entry_names_with_multiple_assays_count: int
    entry_names_with_multiple_target_sequences_count: int
    assays_per_target_histogram: dict[int, int]
    assay_reference_length_histogram: dict[int, int]
    canonical_reference_assay_count: int
    noncanonical_reference_assay_count: int
    noncanonical_symbol_counts: dict[str, int]
    assays_with_single_mutants: int
    assays_without_single_mutants: int
    single_mutant_variant_count: int
    cohort_version_counts: dict[str, int]
    coarse_selection_type_counts: dict[str, int]


@dataclass(frozen=True)
class ProteinGymScan:
    """Private assay rows plus their safe aggregate audit and lookup names."""

    audit: ProteinGymMetadataAudit
    assays: tuple[ProteinGymAssay, ...]
    target_entry_names: tuple[str, ...]


def verify_proteingym_source(
    path: Path,
    pin: ProteinGymSourcePin = PROTEINGYM_V1_3_PIN,
    *,
    chunk_bytes: int = 1024 * 1024,
) -> ProteinGymSourceVerification:
    """Verify byte size, SHA-256, and Git blob identity without network access."""

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")

    source_path = Path(path)
    if not source_path.is_file():
        raise ProteinGymValidationError(
            f"ProteinGym metadata file does not exist: {source_path}"
        )

    byte_size = source_path.stat().st_size
    if byte_size != pin.expected_bytes:
        raise ProteinGymValidationError(
            "ProteinGym byte-size mismatch: "
            f"expected {pin.expected_bytes}, found {byte_size}"
        )

    sha256_hasher = hashlib.sha256()
    git_blob_hasher = hashlib.sha1(usedforsecurity=False)
    git_blob_hasher.update(f"blob {byte_size}\0".encode())

    bytes_read = 0
    with source_path.open("rb") as source:
        while chunk := source.read(chunk_bytes):
            bytes_read += len(chunk)
            sha256_hasher.update(chunk)
            git_blob_hasher.update(chunk)

    if bytes_read != byte_size:
        raise ProteinGymValidationError(
            "ProteinGym file size changed during verification"
        )

    sha256 = sha256_hasher.hexdigest()
    if not hmac.compare_digest(sha256, pin.expected_sha256):
        raise ProteinGymValidationError(
            "ProteinGym SHA-256 mismatch: "
            f"expected {pin.expected_sha256}, found {sha256}"
        )

    git_blob_sha1 = git_blob_hasher.hexdigest()
    if not hmac.compare_digest(git_blob_sha1, pin.expected_git_blob_sha1):
        raise ProteinGymValidationError(
            "ProteinGym Git blob mismatch: "
            f"expected {pin.expected_git_blob_sha1}, found {git_blob_sha1}"
        )

    return ProteinGymSourceVerification(
        byte_size=byte_size,
        sha256=sha256,
        git_blob_sha1=git_blob_sha1,
    )


def parse_proteingym_metadata(path: Path) -> Iterator[ProteinGymAssay]:
    """Yield validated assay metadata in official source order."""

    source_path = Path(path)
    with source_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        headers = reader.fieldnames
        if headers is None:
            raise ProteinGymValidationError("ProteinGym metadata has no header")
        if len(headers) != len(set(headers)):
            raise ProteinGymValidationError(
                "ProteinGym metadata contains duplicate column names"
            )

        missing_columns = tuple(
            column for column in _REQUIRED_COLUMNS if column not in headers
        )
        if missing_columns:
            raise ProteinGymValidationError(
                "ProteinGym metadata is missing required columns: "
                + ", ".join(missing_columns)
            )

        seen_assay_ids: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ProteinGymValidationError(
                    f"row {row_number}: more values than header columns"
                )

            assay_id = _required_cell(row, "DMS_id", row_number)
            entry_name = _required_cell(row, "UniProt_ID", row_number)
            target_sequence = _required_cell(row, "target_seq", row_number)
            length_text = _required_cell(row, "seq_len", row_number)
            single_mutant_count_text = _required_cell(
                row,
                "DMS_number_single_mutants",
                row_number,
            )
            cohort_version = _required_cell(
                row,
                "ProteinGym_version",
                row_number,
            )
            coarse_selection_type = _required_cell(
                row,
                "coarse_selection_type",
                row_number,
            )

            if _IDENTIFIER_PATTERN.fullmatch(assay_id) is None:
                raise ProteinGymValidationError(
                    f"row {row_number}: malformed DMS_id {assay_id!r}"
                )
            if assay_id in seen_assay_ids:
                raise ProteinGymValidationError(
                    f"row {row_number}: duplicate DMS_id {assay_id!r}"
                )
            seen_assay_ids.add(assay_id)

            if _IDENTIFIER_PATTERN.fullmatch(entry_name) is None:
                raise ProteinGymValidationError(
                    f"row {row_number}: malformed UniProt_ID {entry_name!r}"
                )
            if _SEQUENCE_PATTERN.fullmatch(target_sequence) is None:
                raise ProteinGymValidationError(
                    f"row {row_number}: malformed target sequence"
                )
            if _POSITIVE_INTEGER_PATTERN.fullmatch(length_text) is None:
                raise ProteinGymValidationError(
                    f"row {row_number}: seq_len must be a positive integer"
                )
            if _NONNEGATIVE_INTEGER_PATTERN.fullmatch(single_mutant_count_text) is None:
                raise ProteinGymValidationError(
                    f"row {row_number}: DMS_number_single_mutants "
                    "must be a nonnegative integer"
                )

            declared_length = int(length_text)
            if declared_length != len(target_sequence):
                raise ProteinGymValidationError(
                    f"row {row_number}: target sequence length mismatch "
                    f"(declared={declared_length}, parsed={len(target_sequence)})"
                )

            yield ProteinGymAssay(
                assay_id=assay_id,
                entry_name=entry_name,
                target_sequence=target_sequence,
                declared_length=declared_length,
                single_mutant_count=int(single_mutant_count_text),
                cohort_version=cohort_version,
                coarse_selection_type=coarse_selection_type,
            )


def scan_proteingym_metadata(path: Path) -> ProteinGymScan:
    """Load the small metadata source once and expose a deduplicated lookup list."""

    assays = tuple(parse_proteingym_metadata(path))
    audit = audit_proteingym_metadata(assays)
    target_entry_names = tuple(sorted({assay.entry_name for assay in assays}))
    return ProteinGymScan(
        audit=audit,
        assays=assays,
        target_entry_names=target_entry_names,
    )


def audit_proteingym_metadata(
    assays: Iterable[ProteinGymAssay],
) -> ProteinGymMetadataAudit:
    """Return deterministic source-only aggregates for ProteinGym metadata."""

    assay_count = 0
    target_reference_pairs: set[tuple[str, str]] = set()
    assay_counts_by_entry_name: Counter[str] = Counter()
    sequences_by_entry_name: defaultdict[str, set[str]] = defaultdict(set)
    reference_lengths: Counter[int] = Counter()
    canonical_reference_assay_count = 0
    noncanonical_symbols: Counter[str] = Counter()
    assays_with_single_mutants = 0
    single_mutant_variant_count = 0
    cohort_versions: Counter[str] = Counter()
    coarse_selection_types: Counter[str] = Counter()

    for assay in assays:
        assay_count += 1
        target_reference_pairs.add((assay.entry_name, assay.target_sequence))
        assay_counts_by_entry_name[assay.entry_name] += 1
        sequences_by_entry_name[assay.entry_name].add(assay.target_sequence)
        reference_lengths[assay.declared_length] += 1

        sequence_symbols = Counter(assay.target_sequence)
        unexpected_symbols = set(sequence_symbols) - _CANONICAL_AMINO_ACIDS
        if unexpected_symbols:
            for symbol in unexpected_symbols:
                noncanonical_symbols[symbol] += sequence_symbols[symbol]
        else:
            canonical_reference_assay_count += 1

        assays_with_single_mutants += int(assay.single_mutant_count > 0)
        single_mutant_variant_count += assay.single_mutant_count
        cohort_versions[assay.cohort_version] += 1
        coarse_selection_types[assay.coarse_selection_type] += 1

    if not assay_count:
        raise ValueError("cannot audit empty ProteinGym metadata")

    audit = ProteinGymMetadataAudit(
        assay_count=assay_count,
        target_entry_name_count=len(assay_counts_by_entry_name),
        target_reference_pair_count=len(target_reference_pairs),
        entry_names_with_multiple_assays_count=sum(
            count > 1 for count in assay_counts_by_entry_name.values()
        ),
        entry_names_with_multiple_target_sequences_count=sum(
            len(sequences) > 1 for sequences in sequences_by_entry_name.values()
        ),
        assays_per_target_histogram=dict(
            sorted(Counter(assay_counts_by_entry_name.values()).items())
        ),
        assay_reference_length_histogram=dict(sorted(reference_lengths.items())),
        canonical_reference_assay_count=canonical_reference_assay_count,
        noncanonical_reference_assay_count=(
            assay_count - canonical_reference_assay_count
        ),
        noncanonical_symbol_counts=dict(sorted(noncanonical_symbols.items())),
        assays_with_single_mutants=assays_with_single_mutants,
        assays_without_single_mutants=assay_count - assays_with_single_mutants,
        single_mutant_variant_count=single_mutant_variant_count,
        cohort_version_counts=dict(sorted(cohort_versions.items())),
        coarse_selection_type_counts=dict(sorted(coarse_selection_types.items())),
    )
    _validate_audit_reconciliations(audit)
    return audit


def _required_cell(
    row: dict[str | None, str | None], column: str, row_number: int
) -> str:
    value = row.get(column)
    if value is None or not value:
        raise ProteinGymValidationError(
            f"row {row_number}: required column {column!r} is blank"
        )
    if value != value.strip():
        raise ProteinGymValidationError(
            f"row {row_number}: required column {column!r} has outer whitespace"
        )
    return value


def _validate_audit_reconciliations(audit: ProteinGymMetadataAudit) -> None:
    histogram_target_count = sum(audit.assays_per_target_histogram.values())
    if histogram_target_count != audit.target_entry_name_count:
        raise RuntimeError("ProteinGym target counts do not reconcile")
    histogram_assay_count = sum(
        assays_per_target * target_count
        for assays_per_target, target_count in audit.assays_per_target_histogram.items()
    )
    if histogram_assay_count != audit.assay_count:
        raise RuntimeError("ProteinGym assays-per-target counts do not reconcile")
    if sum(audit.assay_reference_length_histogram.values()) != audit.assay_count:
        raise RuntimeError("ProteinGym reference-length counts do not reconcile")
    if (
        audit.canonical_reference_assay_count + audit.noncanonical_reference_assay_count
        != audit.assay_count
    ):
        raise RuntimeError("ProteinGym reference-symbol counts do not reconcile")
    if (
        audit.assays_with_single_mutants + audit.assays_without_single_mutants
        != audit.assay_count
    ):
        raise RuntimeError("ProteinGym single-mutant support counts do not reconcile")
    if sum(audit.cohort_version_counts.values()) != audit.assay_count:
        raise RuntimeError("ProteinGym cohort-version counts do not reconcile")
    if sum(audit.coarse_selection_type_counts.values()) != audit.assay_count:
        raise RuntimeError("ProteinGym selection-type counts do not reconcile")
