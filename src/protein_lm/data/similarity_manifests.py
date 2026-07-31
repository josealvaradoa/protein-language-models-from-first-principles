"""Validate frozen split memberships for the Task 7 similarity audit."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from protein_lm.data.random_split import (
    LOCAL_ASSIGNMENT_COLUMNS,
    PUBLIC_MANIFEST_COLUMNS,
)
from protein_lm.data.similarity_audit_models import FileEvidence, SequenceMetadata
from protein_lm.data.similarity_audit_policy import SimilarityAuditError

PARTITIONS = ("training", "validation", "test")
STRATEGIES = ("random", "group_aware")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class PartitionPopulation:
    """Record, residue, and group totals for one frozen partition."""

    records: int
    residues: int
    unique_groups: int


@dataclass(frozen=True)
class StructuralMembershipAudit:
    """Crossing, retention, and largest-group evidence for one strategy."""

    exact_sequence_hash_crossings: int
    uniref50_group_crossings: int
    retained_records: int
    retained_residues: int
    excluded_records: int
    excluded_residues: int
    largest_uniref50_group_records: int
    largest_uniref50_group_residues: int


@dataclass(frozen=True)
class StrategyManifest:
    """Validated public metadata and local assignment agreement."""

    strategy: str
    stage: str
    records: Mapping[str, SequenceMetadata]
    partitions: Mapping[str, PartitionPopulation]
    structural_audit: StructuralMembershipAudit
    public_manifest: FileEvidence
    local_assignment: FileEvidence


@dataclass(frozen=True)
class _PairedManifestRead:
    """Typed records and file identities from one public/local manifest pair."""

    records: dict[str, SequenceMetadata]
    public_evidence: FileEvidence
    local_evidence: FileEvidence


def load_strategy_manifest(
    *,
    public_path: Path,
    local_path: Path,
    strategy: str,
    stage: str,
    expected_public_sha256: str,
    expected_local_sha256: str,
) -> StrategyManifest:
    """Require exact public and private membership agreement, row by row."""

    if strategy not in STRATEGIES:
        raise SimilarityAuditError(f"unknown strategy: {strategy}")
    try:
        public_source = public_path.open("rb")
    except OSError as error:
        raise SimilarityAuditError(f"could not open a frozen manifest: {error}") from error
    try:
        local_source = local_path.open("rb")
    except OSError as error:
        public_source.close()
        raise SimilarityAuditError(f"could not open a frozen manifest: {error}") from error

    with public_source, local_source:
        paired = _read_paired_manifest(
            public_source=public_source,
            local_source=local_source,
            public_path=public_path,
            local_path=local_path,
            strategy=strategy,
            stage=stage,
        )
    if paired.public_evidence.sha256 != expected_public_sha256:
        raise SimilarityAuditError("public split manifest checksum drifted")
    if paired.local_evidence.sha256 != expected_local_sha256:
        raise SimilarityAuditError("local assignment checksum drifted")

    partitions, structural_audit = _summarize_membership(paired.records)
    return StrategyManifest(
        strategy=strategy,
        stage=stage,
        records=paired.records,
        partitions=partitions,
        structural_audit=structural_audit,
        public_manifest=paired.public_evidence,
        local_assignment=paired.local_evidence,
    )


def metadata_by_partition(
    manifest: StrategyManifest,
    partition: str,
) -> dict[str, SequenceMetadata]:
    """Select metadata for one partition without changing membership."""

    if partition not in PARTITIONS:
        raise SimilarityAuditError(f"unknown partition: {partition}")
    return {
        accession: metadata
        for accession, metadata in manifest.records.items()
        if metadata.partition == partition
    }


def _read_paired_manifest(
    *,
    public_source: BinaryIO,
    local_source: BinaryIO,
    public_path: Path,
    local_path: Path,
    strategy: str,
    stage: str,
) -> _PairedManifestRead:
    public_hasher = hashlib.sha256()
    local_hasher = hashlib.sha256()
    public_bytes = 0
    local_bytes = 0
    row_count = 0
    records: dict[str, SequenceMetadata] = {}

    public_header = public_source.readline()
    local_header = local_source.readline()
    public_hasher.update(public_header)
    local_hasher.update(local_header)
    public_bytes += len(public_header)
    local_bytes += len(local_header)
    _validate_manifest_headers(public_header, local_header, public_path, local_path)

    previous_accession: str | None = None
    line_number = 1
    while True:
        public_line = public_source.readline()
        local_line = local_source.readline()
        if not public_line and not local_line:
            break
        line_number += 1
        if not public_line or not local_line:
            raise SimilarityAuditError("public and local assignment lengths differ")
        public_hasher.update(public_line)
        local_hasher.update(local_line)
        public_bytes += len(public_line)
        local_bytes += len(local_line)
        accession, metadata = _parse_paired_row(
            public_line=public_line,
            local_line=local_line,
            public_path=public_path,
            local_path=local_path,
            line_number=line_number,
            strategy=strategy,
            stage=stage,
            previous_accession=previous_accession,
        )
        previous_accession = accession
        records[accession] = metadata
        row_count += 1

    return _PairedManifestRead(
        records=records,
        public_evidence=FileEvidence(
            row_count=row_count,
            byte_size=public_bytes,
            sha256=public_hasher.hexdigest(),
        ),
        local_evidence=FileEvidence(
            row_count=row_count,
            byte_size=local_bytes,
            sha256=local_hasher.hexdigest(),
        ),
    )


def _validate_manifest_headers(
    public_header: bytes,
    local_header: bytes,
    public_path: Path,
    local_path: Path,
) -> None:
    if tuple(_decode_lf(public_header, f"{public_path.name} header").split("\t")) != (
        PUBLIC_MANIFEST_COLUMNS
    ):
        raise SimilarityAuditError("public split manifest header is not approved")
    if tuple(_decode_lf(local_header, f"{local_path.name} header").split("\t")) != (
        LOCAL_ASSIGNMENT_COLUMNS
    ):
        raise SimilarityAuditError("local assignment header is not approved")


def _parse_paired_row(
    *,
    public_line: bytes,
    local_line: bytes,
    public_path: Path,
    local_path: Path,
    line_number: int,
    strategy: str,
    stage: str,
    previous_accession: str | None,
) -> tuple[str, SequenceMetadata]:
    public_cells = _decode_lf(
        public_line,
        f"{public_path.name} line {line_number}",
    ).split("\t")
    local_cells = _decode_lf(
        local_line,
        f"{local_path.name} line {line_number}",
    ).split("\t")
    if len(public_cells) != len(PUBLIC_MANIFEST_COLUMNS):
        raise SimilarityAuditError("public split manifest row is malformed")
    if len(local_cells) != len(LOCAL_ASSIGNMENT_COLUMNS):
        raise SimilarityAuditError("local assignment row is malformed")

    accession, partition, digest, length_token, group = public_cells
    if partition not in PARTITIONS:
        raise SimilarityAuditError(f"unknown partition: {partition}")
    _require_visible_ascii(accession, "manifest accession")
    _require_visible_ascii(group, "manifest UniRef50 group")
    if not _SHA256_PATTERN.fullmatch(digest):
        raise SimilarityAuditError("manifest sequence SHA-256 is malformed")
    try:
        length = int(length_token)
    except ValueError as error:
        raise SimilarityAuditError("manifest biological length is malformed") from error
    if length <= 0 or str(length) != length_token:
        raise SimilarityAuditError(
            "manifest biological length must be a canonical positive integer"
        )
    if previous_accession is not None and accession <= previous_accession:
        raise SimilarityAuditError("manifest accessions must be unique and sorted")

    expected_local = (strategy, stage, "0", partition, accession)
    observed_local = (
        local_cells[0],
        local_cells[1],
        local_cells[2],
        local_cells[4],
        local_cells[5],
    )
    if observed_local != expected_local:
        raise SimilarityAuditError(
            f"public and local assignments disagree for {accession}"
        )
    stable_unit = local_cells[3]
    _require_visible_ascii(stable_unit, "stable assignment unit")
    if strategy == "random" and stable_unit != accession:
        raise SimilarityAuditError(
            "random diagnostic assignment unit must equal its accession"
        )
    return accession, SequenceMetadata(
        sequence_sha256=digest,
        biological_length=length,
        uniref50_group=group,
        partition=partition,
    )


def _summarize_membership(
    records: Mapping[str, SequenceMetadata],
) -> tuple[dict[str, PartitionPopulation], StructuralMembershipAudit]:
    groups: dict[str, set[str]] = defaultdict(set)
    partition_records = defaultdict(int)
    partition_residues = defaultdict(int)
    first_partition_by_hash: dict[str, str] = {}
    first_partition_by_group: dict[str, str] = {}
    crossing_hashes: set[str] = set()
    crossing_groups: set[str] = set()
    group_records = defaultdict(int)
    group_residues = defaultdict(int)

    for metadata in records.values():
        partition = metadata.partition
        group = metadata.uniref50_group
        length = metadata.biological_length
        partition_records[partition] += 1
        partition_residues[partition] += length
        groups[partition].add(group)
        first_hash_partition = first_partition_by_hash.setdefault(
            metadata.sequence_sha256,
            partition,
        )
        if first_hash_partition != partition:
            crossing_hashes.add(metadata.sequence_sha256)
        first_group_partition = first_partition_by_group.setdefault(group, partition)
        if first_group_partition != partition:
            crossing_groups.add(group)
        group_records[group] += 1
        group_residues[group] += length

    if set(partition_records) != set(PARTITIONS):
        raise SimilarityAuditError("a frozen manifest is missing a partition")
    partitions = {
        partition: PartitionPopulation(
            records=partition_records[partition],
            residues=partition_residues[partition],
            unique_groups=len(groups[partition]),
        )
        for partition in PARTITIONS
    }
    return partitions, StructuralMembershipAudit(
        exact_sequence_hash_crossings=len(crossing_hashes),
        uniref50_group_crossings=len(crossing_groups),
        retained_records=len(records),
        retained_residues=sum(partition_residues.values()),
        excluded_records=0,
        excluded_residues=0,
        largest_uniref50_group_records=max(group_records.values()),
        largest_uniref50_group_residues=max(group_residues.values()),
    )


def _decode_lf(raw_line: bytes, context: str) -> str:
    if not raw_line.endswith(b"\n"):
        raise SimilarityAuditError(f"{context}: line must end with LF")
    if b"\r" in raw_line:
        raise SimilarityAuditError(f"{context}: CR line endings are prohibited")
    if raw_line == b"\n":
        raise SimilarityAuditError(f"{context}: blank row is prohibited")
    try:
        return raw_line[:-1].decode("utf-8")
    except UnicodeDecodeError as error:
        raise SimilarityAuditError(f"{context}: invalid UTF-8") from error


def _require_visible_ascii(value: str, context: str) -> None:
    if not value or any(ord(character) < 33 or ord(character) > 126 for character in value):
        raise SimilarityAuditError(f"{context} must contain only visible ASCII")
