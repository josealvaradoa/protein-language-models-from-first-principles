"""Focused contracts for deterministic artifact primitives."""

import hashlib
from pathlib import Path

import pytest

from protein_lm.data.artifacts import (
    EvidenceWriter,
    canonical_evidence_from,
    fasta_evidence_from,
    file_evidence_from,
    file_identity,
    read_json,
    verify_database_artifacts,
    verify_file,
    write_json_atomic,
)
from protein_lm.data.similarity_audit_models import (
    CanonicalAlignmentEvidence,
    FileEvidence,
)
from protein_lm.data.similarity_audit_policy import SimilarityAuditError
from protein_lm.data.similarity_fastas import FastaEvidence


def test_atomic_json_bytes_and_file_identity_are_stable(tmp_path: Path) -> None:
    path = tmp_path / "nested/complete.json"
    expected = b'{\n  "a": 1,\n  "z": 2\n}\n'

    write_json_atomic(path, {"z": 2, "a": 1})

    assert path.read_bytes() == expected
    assert not path.with_name(".complete.json.incomplete").exists()
    assert read_json(path) == {"a": 1, "z": 2}
    assert file_identity(path) == {
        "byte_size": len(expected),
        "sha256": hashlib.sha256(expected).hexdigest(),
    }
    verify_file(path, len(expected), hashlib.sha256(expected).hexdigest())


def test_evidence_writer_preserves_rows_bytes_and_atomic_finish(tmp_path: Path) -> None:
    path = tmp_path / "pairs.tsv"
    writer = EvidenceWriter(path)

    writer.write(b"A\tB\n")
    writer.write(b"C\tD\n")
    evidence = writer.finish()

    expected = b"A\tB\nC\tD\n"
    assert path.read_bytes() == expected
    assert not path.with_name(".pairs.tsv.incomplete").exists()
    assert evidence == FileEvidence(
        row_count=2,
        byte_size=len(expected),
        sha256=hashlib.sha256(expected).hexdigest(),
    )


def test_serialized_evidence_decoders_retain_types_and_strict_errors() -> None:
    raw_file = {"row_count": 2, "byte_size": 8, "sha256": "a" * 64}
    canonical_file = {"row_count": 1, "byte_size": 4, "sha256": "b" * 64}

    assert file_evidence_from(raw_file) == FileEvidence(2, 8, "a" * 64)
    assert fasta_evidence_from(
        {"record_count": 1, "residue_count": 4, "byte_size": 8, "sha256": "c" * 64}
    ) == FastaEvidence(1, 4, 8, "c" * 64)
    assert canonical_evidence_from(
        {"raw": raw_file, "canonical": canonical_file}
    ) == CanonicalAlignmentEvidence(
        raw=FileEvidence(2, 8, "a" * 64),
        canonical=FileEvidence(1, 4, "b" * 64),
    )

    with pytest.raises(
        SimilarityAuditError, match="row count must be a nonnegative integer"
    ):
        file_evidence_from({**raw_file, "row_count": True})


def test_evidence_writer_abort_removes_partial_and_final_paths(tmp_path: Path) -> None:
    path = tmp_path / "partial.tsv"
    path.write_bytes(b"stale final\n")
    writer = EvidenceWriter(path)
    writer.write(b"partial\n")

    writer.abort()

    assert not path.exists()
    assert not path.with_name(".partial.tsv.incomplete").exists()


def test_database_artifact_index_must_name_target(tmp_path: Path) -> None:
    database = tmp_path / "database"
    database.mkdir()
    (database / "target").write_bytes(b"target")
    dbtype = database / "target.dbtype"
    dbtype.write_bytes(b"dbtype")

    with pytest.raises(SimilarityAuditError, match="lacks its target prefix"):
        verify_database_artifacts(database, {"target.dbtype": file_identity(dbtype)})
