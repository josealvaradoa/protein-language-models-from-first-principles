"""Completion-marker and artifact verification for the Task 7 audit."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from protein_lm.data.similarity_audit_models import (
    CanonicalAlignmentEvidence,
    FileEvidence,
)
from protein_lm.data.similarity_audit_policy import (
    SimilarityAuditError,
    SimilarityAuditPolicy,
)
from protein_lm.data.similarity_fastas import FastaEvidence


def validate_completed_pass(
    marker: Mapping[str, object],
    *,
    marker_path: Path,
    fingerprint: str,
    strategy: str,
    partition: str,
    pass_name: str,
    expected_query_ids: frozenset[str],
    policy: SimilarityAuditPolicy,
) -> None:
    """Prove that a resumed search pass is complete and internally consistent."""

    require_marker_identity(marker, fingerprint, "completed_search_pass")
    expected = {
        "strategy": strategy,
        "partition": partition,
        "pass_name": pass_name,
        "query_count": len(expected_query_ids),
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        raise SimilarityAuditError("completed search-pass identity drifted")
    convergence = marker.get("convergence")
    if not isinstance(convergence, dict) or convergence.get("final_differing_queries") != 0:
        raise SimilarityAuditError("completed search pass is not converged")
    escalated_ids = convergence.get("escalated_query_ids")
    if (
        not isinstance(escalated_ids, list)
        or any(not isinstance(value, str) for value in escalated_ids)
        or escalated_ids != sorted(set(escalated_ids))
        or not set(escalated_ids) <= expected_query_ids
    ):
        raise SimilarityAuditError("completed escalation query set is malformed")
    escalated_count = len(escalated_ids)
    expected_convergence = {
        "expected_queries": len(expected_query_ids),
        "converged_at_comparison_cap": len(expected_query_ids) - escalated_count,
        "escalated_queries": escalated_count,
        "converged_at_escalation_cap": escalated_count,
        "final_differing_queries": 0,
    }
    if any(convergence.get(key) != value for key, value in expected_convergence.items()):
        raise SimilarityAuditError("completed convergence counts do not reconcile")
    expected_stage_caps = {str(policy.initial_cap), str(policy.comparison_cap)}
    if escalated_count:
        expected_stage_caps.add(str(policy.escalation_cap))
    stages = marker.get("stages")
    if not isinstance(stages, dict) or set(stages) != expected_stage_caps:
        raise SimilarityAuditError("completed search stages do not reconcile")
    accepted = marker.get("accepted")
    if not isinstance(accepted, dict):
        raise SimilarityAuditError("accepted pass evidence is malformed")
    expected_accepted = {
        "pass_name": pass_name,
        "accepted_at_comparison_cap": len(expected_query_ids) - escalated_count,
        "accepted_at_escalation_cap": escalated_count,
    }
    if any(accepted.get(key) != value for key, value in expected_accepted.items()):
        raise SimilarityAuditError("accepted cap distribution does not reconcile")
    accepted_rows = _strict_int(accepted.get("accepted_rows"), "accepted row count")
    compact = marker_path.parent / "compact"
    verify_compact_file(compact / "returned_pairs.tsv", accepted.get("returned_pairs"))
    verify_compact_file(
        compact / "prohibited_pairs.tsv",
        accepted.get("prohibited_pairs"),
    )
    returned = file_evidence_from(accepted["returned_pairs"])
    prohibited = file_evidence_from(accepted["prohibited_pairs"])
    if returned.row_count != accepted_rows or prohibited.row_count > accepted_rows:
        raise SimilarityAuditError("accepted pair counts do not reconcile")
    summaries = accepted.get("residual_summaries")
    if pass_name == "residual":
        verify_compact_file(compact / "residual_summaries.tsv", summaries)
        if file_evidence_from(summaries).row_count != len(expected_query_ids):
            raise SimilarityAuditError("residual summary query count drifted")
    elif summaries is not None:
        raise SimilarityAuditError("enforcement pass has residual summaries")


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


def verify_artifact_index(directory: Path, raw_index: object) -> None:
    """Verify every file recorded for a completed MMseqs2 database."""

    if not isinstance(raw_index, dict) or not raw_index:
        raise SimilarityAuditError("database artifact index is malformed")
    for filename, evidence in raw_index.items():
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise SimilarityAuditError("database artifact filename is unsafe")
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
