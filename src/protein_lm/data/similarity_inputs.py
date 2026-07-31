"""Verify frozen Task 7 memberships and materialize ignored FASTA inputs."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from protein_lm.data.eligibility import CATALOG_COLUMNS
from protein_lm.data.random_split import (
    LOCAL_ASSIGNMENT_COLUMNS,
    PUBLIC_MANIFEST_COLUMNS,
)
from protein_lm.data.similarity_audit import FileEvidence, SequenceMetadata
from protein_lm.data.similarity_audit_policy import (
    SimilarityAuditError,
    SimilarityAuditPolicy,
)

PARTITIONS = ("training", "validation", "test")
STRATEGIES = ("random", "group_aware")
_CANONICAL_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")
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
class FastaEvidence:
    """Identity and population of one generated FASTA file."""

    record_count: int
    residue_count: int
    byte_size: int
    sha256: str


@dataclass(frozen=True)
class MaterializedInputs:
    """Evidence for all six strategy-by-partition FASTA files."""

    catalog: FileEvidence
    fastas: Mapping[str, Mapping[str, FastaEvidence]]


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
    public_hasher = hashlib.sha256()
    local_hasher = hashlib.sha256()
    public_bytes = 0
    local_bytes = 0
    row_count = 0
    records: dict[str, SequenceMetadata] = {}
    groups: dict[str, set[str]] = defaultdict(set)
    partition_records = defaultdict(int)
    partition_residues = defaultdict(int)
    first_partition_by_hash: dict[str, str] = {}
    first_partition_by_group: dict[str, str] = {}
    crossing_hashes: set[str] = set()
    crossing_groups: set[str] = set()
    group_records = defaultdict(int)
    group_residues = defaultdict(int)

    try:
        public_source = public_path.open("rb")
        local_source = local_path.open("rb")
    except OSError as error:
        raise SimilarityAuditError(f"could not open a frozen manifest: {error}") from error

    with public_source, local_source:
        public_header = public_source.readline()
        local_header = local_source.readline()
        public_hasher.update(public_header)
        local_hasher.update(local_header)
        public_bytes += len(public_header)
        local_bytes += len(local_header)
        if tuple(_decode_lf(public_header, f"{public_path.name} header").split("\t")) != (
            PUBLIC_MANIFEST_COLUMNS
        ):
            raise SimilarityAuditError("public split manifest header is not approved")
        if tuple(_decode_lf(local_header, f"{local_path.name} header").split("\t")) != (
            LOCAL_ASSIGNMENT_COLUMNS
        ):
            raise SimilarityAuditError("local assignment header is not approved")

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
                raise SimilarityAuditError(
                    "manifest biological length is malformed"
                ) from error
            if length <= 0 or str(length) != length_token:
                raise SimilarityAuditError(
                    "manifest biological length must be a canonical positive integer"
                )
            if previous_accession is not None and accession <= previous_accession:
                raise SimilarityAuditError(
                    "manifest accessions must be unique and sorted"
                )
            previous_accession = accession

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

            records[accession] = SequenceMetadata(
                sequence_sha256=digest,
                biological_length=length,
                uniref50_group=group,
                partition=partition,
            )
            partition_records[partition] += 1
            partition_residues[partition] += length
            groups[partition].add(group)
            first_hash_partition = first_partition_by_hash.setdefault(digest, partition)
            if first_hash_partition != partition:
                crossing_hashes.add(digest)
            first_group_partition = first_partition_by_group.setdefault(group, partition)
            if first_group_partition != partition:
                crossing_groups.add(group)
            group_records[group] += 1
            group_residues[group] += length
            row_count += 1

    public_evidence = FileEvidence(
        row_count=row_count,
        byte_size=public_bytes,
        sha256=public_hasher.hexdigest(),
    )
    local_evidence = FileEvidence(
        row_count=row_count,
        byte_size=local_bytes,
        sha256=local_hasher.hexdigest(),
    )
    if public_evidence.sha256 != expected_public_sha256:
        raise SimilarityAuditError("public split manifest checksum drifted")
    if local_evidence.sha256 != expected_local_sha256:
        raise SimilarityAuditError("local assignment checksum drifted")
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
    retained_residues = sum(partition_residues.values())
    structural_audit = StructuralMembershipAudit(
        exact_sequence_hash_crossings=len(crossing_hashes),
        uniref50_group_crossings=len(crossing_groups),
        retained_records=row_count,
        retained_residues=retained_residues,
        excluded_records=0,
        excluded_residues=0,
        largest_uniref50_group_records=max(group_records.values()),
        largest_uniref50_group_residues=max(group_residues.values()),
    )
    return StrategyManifest(
        strategy=strategy,
        stage=stage,
        records=records,
        partitions=partitions,
        structural_audit=structural_audit,
        public_manifest=public_evidence,
        local_assignment=local_evidence,
    )


def materialize_strategy_fastas(
    *,
    catalog_path: Path,
    manifests: Mapping[str, StrategyManifest],
    output_directory: Path,
    policy: SimilarityAuditPolicy,
) -> MaterializedInputs:
    """Join the pinned catalog to both memberships and write six FASTAs."""

    if set(manifests) != set(STRATEGIES):
        raise SimilarityAuditError("both frozen strategies are required")
    for manifest in manifests.values():
        if len(manifest.records) != policy.expected_eligible_records:
            raise SimilarityAuditError("manifest eligible population drifted")

    output_directory.mkdir(parents=True, exist_ok=True)
    writers = {
        strategy: {
            partition: _AtomicFastaWriter(
                output_directory / f"{strategy}_{partition}.fasta"
            )
            for partition in PARTITIONS
        }
        for strategy in STRATEGIES
    }
    catalog_hasher = hashlib.sha256()
    catalog_bytes = 0
    source_rows = 0
    eligible_rows = 0
    eligible_residues = 0
    seen_accessions: set[str] = set()
    try:
        with catalog_path.open("rb") as source:
            raw_header = source.readline()
            catalog_hasher.update(raw_header)
            catalog_bytes += len(raw_header)
            if tuple(_decode_lf(raw_header, "Task 4 catalog header").split("\t")) != (
                CATALOG_COLUMNS
            ):
                raise SimilarityAuditError("Task 4 catalog header is not approved")

            for line_number, raw_line in enumerate(source, start=2):
                source_rows += 1
                catalog_hasher.update(raw_line)
                catalog_bytes += len(raw_line)
                cells = _decode_lf(
                    raw_line,
                    f"Task 4 catalog line {line_number}",
                ).split("\t")
                if len(cells) != len(CATALOG_COLUMNS):
                    raise SimilarityAuditError("Task 4 catalog row is malformed")
                if cells[9] == "false":
                    continue
                if cells[9] != "true":
                    raise SimilarityAuditError("Task 4 eligible flag is malformed")

                accession, sequence, digest = cells[0], cells[1], cells[2]
                try:
                    length = int(cells[3])
                except ValueError as error:
                    raise SimilarityAuditError(
                        "Task 4 biological length is malformed"
                    ) from error
                group = cells[11]
                _validate_catalog_sequence(
                    accession=accession,
                    sequence=sequence,
                    digest=digest,
                    length=length,
                    line_number=line_number,
                )
                if accession in seen_accessions:
                    raise SimilarityAuditError("duplicate eligible Task 4 accession")
                seen_accessions.add(accession)
                eligible_rows += 1
                eligible_residues += length

                for strategy, manifest in manifests.items():
                    metadata = manifest.records.get(accession)
                    if metadata is None:
                        raise SimilarityAuditError(
                            f"{strategy} manifest is missing {accession}"
                        )
                    if (
                        metadata.sequence_sha256 != digest
                        or metadata.biological_length != length
                        or metadata.uniref50_group != group
                    ):
                        raise SimilarityAuditError(
                            f"{strategy} metadata differs from Task 4 for {accession}"
                        )
                    writers[strategy][metadata.partition].write(accession, sequence)

        catalog_evidence = FileEvidence(
            row_count=source_rows,
            byte_size=catalog_bytes,
            sha256=catalog_hasher.hexdigest(),
        )
        _validate_catalog_anchors(
            catalog_evidence,
            eligible_rows=eligible_rows,
            eligible_residues=eligible_residues,
            policy=policy,
        )
        generated = {
            strategy: {
                partition: writers[strategy][partition].finish()
                for partition in PARTITIONS
            }
            for strategy in STRATEGIES
        }
    except BaseException:
        for strategy_writers in writers.values():
            for writer in strategy_writers.values():
                writer.abort()
        raise

    for strategy, partition_evidence in generated.items():
        for partition, fasta in partition_evidence.items():
            population = manifests[strategy].partitions[partition]
            if (fasta.record_count, fasta.residue_count) != (
                population.records,
                population.residues,
            ):
                raise SimilarityAuditError(
                    f"{strategy} {partition} FASTA population does not reconcile"
                )
    return MaterializedInputs(catalog=catalog_evidence, fastas=generated)


def write_fasta_subset(
    source_path: Path,
    output_path: Path,
    selected_accessions: set[str],
) -> FastaEvidence:
    """Write an escalation FASTA containing only the changed queries."""

    if not selected_accessions:
        raise SimilarityAuditError("an escalation FASTA requires at least one query")
    writer = _AtomicFastaWriter(output_path)
    found: set[str] = set()
    try:
        for accession, sequence in iter_one_line_fasta(source_path):
            if accession in selected_accessions:
                writer.write(accession, sequence)
                found.add(accession)
        evidence = writer.finish()
    except BaseException:
        writer.abort()
        raise
    missing = selected_accessions - found
    if missing:
        raise SimilarityAuditError(
            f"escalation FASTA is missing {len(missing)} selected queries"
        )
    return evidence


def iter_one_line_fasta(path: Path) -> Iterator[tuple[str, str]]:
    """Read the exact two-line FASTA representation written by this module."""

    with path.open("rb") as source:
        line_number = 0
        while True:
            header = source.readline()
            if not header:
                break
            line_number += 1
            sequence_line = source.readline()
            if not sequence_line:
                raise SimilarityAuditError(f"{path.name} ends after a FASTA header")
            line_number += 1
            header_text = _decode_lf(header, f"{path.name} line {line_number - 1}")
            sequence = _decode_lf(sequence_line, f"{path.name} line {line_number}")
            if not header_text.startswith(">") or len(header_text) == 1:
                raise SimilarityAuditError(f"{path.name} has a malformed FASTA header")
            accession = header_text[1:]
            _require_visible_ascii(accession, "FASTA accession")
            if not sequence or any(residue not in _CANONICAL_RESIDUES for residue in sequence):
                raise SimilarityAuditError(f"{path.name} has an invalid FASTA sequence")
            yield accession, sequence


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


def _validate_catalog_sequence(
    *,
    accession: str,
    sequence: str,
    digest: str,
    length: int,
    line_number: int,
) -> None:
    _require_visible_ascii(accession, f"catalog line {line_number} accession")
    if not sequence or any(residue not in _CANONICAL_RESIDUES for residue in sequence):
        raise SimilarityAuditError(
            f"catalog line {line_number}: eligible sequence is not canonical"
        )
    if length != len(sequence) or str(length) != str(len(sequence)):
        raise SimilarityAuditError(f"catalog line {line_number}: sequence length differs")
    calculated = hashlib.sha256(sequence.encode("ascii")).hexdigest()
    if digest != calculated:
        raise SimilarityAuditError(f"catalog line {line_number}: sequence hash differs")


def _validate_catalog_anchors(
    evidence: FileEvidence,
    *,
    eligible_rows: int,
    eligible_residues: int,
    policy: SimilarityAuditPolicy,
) -> None:
    observed = {
        "catalog SHA-256": (evidence.sha256, policy.task4_catalog_sha256),
        "catalog byte size": (evidence.byte_size, policy.task4_catalog_byte_size),
        "catalog rows": (evidence.row_count, policy.task4_catalog_row_count),
        "eligible records": (eligible_rows, policy.expected_eligible_records),
        "eligible residues": (eligible_residues, policy.expected_eligible_residues),
    }
    drift = [
        f"{name}: found {found}, expected {expected}"
        for name, (found, expected) in observed.items()
        if found != expected
    ]
    if drift:
        raise SimilarityAuditError("Task 4 catalog drift: " + "; ".join(drift))


class _AtomicFastaWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.temporary_path = path.with_name(f".{path.name}.incomplete")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary_path.unlink(missing_ok=True)
        self._output = self.temporary_path.open("wb")
        self._hasher = hashlib.sha256()
        self._records = 0
        self._residues = 0
        self._bytes = 0

    def write(self, accession: str, sequence: str) -> None:
        content = f">{accession}\n{sequence}\n".encode("ascii")
        self._output.write(content)
        self._hasher.update(content)
        self._records += 1
        self._residues += len(sequence)
        self._bytes += len(content)

    def finish(self) -> FastaEvidence:
        if self._output.closed:
            raise SimilarityAuditError("FASTA writer was already closed")
        self._output.close()
        self.temporary_path.replace(self.path)
        return FastaEvidence(
            record_count=self._records,
            residue_count=self._residues,
            byte_size=self._bytes,
            sha256=self._hasher.hexdigest(),
        )

    def abort(self) -> None:
        if not self._output.closed:
            self._output.close()
        self.temporary_path.unlink(missing_ok=True)


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
