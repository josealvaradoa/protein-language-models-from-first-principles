"""Deterministic artifact writing, identity, decoding, and verification."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from protein_lm.data.similarity_audit_models import (
    CanonicalAlignmentEvidence,
    FileEvidence,
)
from protein_lm.data.similarity_audit_policy import SimilarityAuditError

if TYPE_CHECKING:
    from protein_lm.data.similarity_fastas import FastaEvidence


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Write one JSON checkpoint without exposing a partial final file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary.write_bytes(content)
    temporary.replace(path)


def read_json(path: Path) -> dict[str, object]:
    """Read one completion marker and require an object at its root."""

    try:
        parsed = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SimilarityAuditError(f"completion marker is malformed: {path}") from error
    if not isinstance(parsed, dict):
        raise SimilarityAuditError(f"completion marker root is not an object: {path}")
    return parsed


def require_marker_identity(
    marker: Mapping[str, object],
    fingerprint: str,
    stage: str,
) -> None:
    """Require a marker to belong to this exact frozen run and stage."""

    if marker.get("schema_version") != 1 or marker.get("stage") != stage:
        raise SimilarityAuditError(f"{stage} completion marker is malformed")
    if marker.get("fingerprint") != fingerprint:
        raise SimilarityAuditError(
            f"{stage} completion marker belongs to a different frozen run"
        )


class EvidenceWriter:
    """Write one evidence file through a sibling temporary file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.temporary_path = path.with_name(f".{path.name}.incomplete")
        self.temporary_path.unlink(missing_ok=True)
        self.output = self.temporary_path.open("wb")
        self.hasher = hashlib.sha256()
        self.byte_size = 0
        self.row_count = 0

    def write(self, content: bytes) -> None:
        self.output.write(content)
        self.hasher.update(content)
        self.byte_size += len(content)
        self.row_count += 1

    def finish(self) -> FileEvidence:
        self.output.close()
        self.temporary_path.replace(self.path)
        return FileEvidence(self.row_count, self.byte_size, self.hasher.hexdigest())

    def abort(self) -> None:
        if not self.output.closed:
            self.output.close()
        self.temporary_path.unlink(missing_ok=True)
        self.path.unlink(missing_ok=True)


def file_identity(path: Path) -> dict[str, object]:
    """Return the byte size and SHA-256 identity of a file."""

    hasher = hashlib.sha256()
    byte_size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            hasher.update(chunk)
            byte_size += len(chunk)
    return {"byte_size": byte_size, "sha256": hasher.hexdigest()}


def verify_file(path: Path, expected_size: int, expected_sha256: str) -> None:
    """Require a file to match its frozen byte size and digest."""

    if not path.is_file():
        raise SimilarityAuditError(f"completed artifact is missing: {path}")
    identity = file_identity(path)
    if identity != {"byte_size": expected_size, "sha256": expected_sha256}:
        raise SimilarityAuditError(f"completed artifact checksum drifted: {path}")


def verify_database_artifacts(directory: Path, raw_index: object) -> None:
    """Require the current database inventory to equal its frozen index."""

    if not isinstance(raw_index, dict) or not raw_index:
        raise SimilarityAuditError("database artifact index is malformed")
    if any(
        not isinstance(filename, str) or Path(filename).name != filename
        for filename in raw_index
    ):
        raise SimilarityAuditError("database artifact filename is unsafe")
    indexed_names = set(raw_index)
    if "target" not in indexed_names:
        raise SimilarityAuditError("database artifact index lacks its target prefix")
    actual_names = {
        path.name for path in directory.iterdir() if path.name != "complete.json"
    }
    if actual_names != indexed_names:
        raise SimilarityAuditError("database artifact inventory drifted")
    for filename, evidence in raw_index.items():
        if not (directory / filename).is_file():
            raise SimilarityAuditError(
                "database artifact inventory contains a non-file"
            )
        if not isinstance(evidence, dict):
            raise SimilarityAuditError("database artifact evidence is malformed")
        verify_file(
            directory / filename,
            _strict_int(evidence.get("byte_size"), "database byte size"),
            _strict_string(evidence.get("sha256"), "database SHA-256"),
        )


def verify_compact_file(path: Path, raw_evidence: object) -> None:
    """Verify one compact pass artifact against serialized evidence."""

    if not isinstance(raw_evidence, dict):
        raise SimilarityAuditError("compact artifact evidence is malformed")
    evidence = file_evidence_from(raw_evidence)
    verify_file(path, evidence.byte_size, evidence.sha256)


def file_evidence_from(raw: object) -> FileEvidence:
    """Decode serialized generic file evidence."""

    if not isinstance(raw, dict):
        raise SimilarityAuditError("file evidence is malformed")
    return FileEvidence(
        row_count=_strict_int(raw.get("row_count"), "row count"),
        byte_size=_strict_int(raw.get("byte_size"), "byte size"),
        sha256=_strict_string(raw.get("sha256"), "SHA-256"),
    )


def fasta_evidence_from(raw: object) -> FastaEvidence:
    """Decode serialized FASTA evidence."""

    from protein_lm.data.similarity_fastas import FastaEvidence

    if not isinstance(raw, dict):
        raise SimilarityAuditError("FASTA evidence is malformed")
    return FastaEvidence(
        record_count=_strict_int(raw.get("record_count"), "FASTA record count"),
        residue_count=_strict_int(raw.get("residue_count"), "FASTA residue count"),
        byte_size=_strict_int(raw.get("byte_size"), "FASTA byte size"),
        sha256=_strict_string(raw.get("sha256"), "FASTA SHA-256"),
    )


def canonical_evidence_from(raw: object) -> CanonicalAlignmentEvidence:
    """Decode serialized raw and canonical alignment evidence."""

    if not isinstance(raw, dict):
        raise SimilarityAuditError("canonical alignment evidence is malformed")
    return CanonicalAlignmentEvidence(
        raw=file_evidence_from(raw.get("raw")),
        canonical=file_evidence_from(raw.get("canonical")),
    )


def _strict_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SimilarityAuditError(f"{name} must be a nonnegative integer")
    return value


def _strict_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SimilarityAuditError(f"{name} must be a nonempty string")
    return value
