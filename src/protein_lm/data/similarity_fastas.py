"""Materialize and verify FASTA inputs for the Task 7 similarity audit."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from protein_lm.data.eligibility import CATALOG_COLUMNS
from protein_lm.data.similarity_audit_models import FileEvidence
from protein_lm.data.similarity_audit_policy import (
    SimilarityAuditError,
    SimilarityAuditPolicy,
)
from protein_lm.data.similarity_manifests import (
    PARTITIONS,
    STRATEGIES,
    StrategyManifest,
)

_CANONICAL_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")


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
            if not sequence or any(
                residue not in _CANONICAL_RESIDUES for residue in sequence
            ):
                raise SimilarityAuditError(f"{path.name} has an invalid FASTA sequence")
            yield accession, sequence


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
