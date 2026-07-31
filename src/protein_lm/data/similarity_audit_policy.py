"""Load the frozen Week 1 Task 7 diagnostic similarity policy."""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path, PurePosixPath

APPROVED_SIMILARITY_AUDIT_CONFIG_SHA256 = (
    "ce767f0ce843e4f40edbcd2f9da6ca4642996046cb4042a2410c27c39cbae742"
)


class SimilarityAuditError(ValueError):
    """Raised when Task 7 cannot produce trustworthy diagnostic evidence."""


@dataclass(frozen=True)
class SimilarityAuditPolicy:
    """Every frozen input, boundary, command option, and safety limit."""

    schema_version: int
    scope: str
    adjustment_id: str
    diagnostic_audit_authorized: bool
    repair_authorized: bool
    selected_split_authorized: bool
    task8_membership_use_authorized: bool
    model_use: str
    post_audit_review_required: bool
    candidate_status: str
    repair_performed: bool
    mmseqs_executable: str
    mmseqs_version: str
    createdb_mode: int
    createdb_shuffle: int
    search_type: int
    alignment_mode: int
    sequence_identity_mode: int
    sensitivity: str
    evalue_cutoff: str
    mask: int
    composition_bias_correction: int
    threads: int
    format_output: str
    enforcement_min_sequence_identity: str
    enforcement_coverage: str
    enforcement_coverage_mode: int
    residual_min_sequence_identity: str
    residual_coverage: str
    residual_coverage_mode: int
    prohibited_min_sequence_identity: str
    prohibited_min_query_coverage: str
    prohibited_min_target_coverage: str
    prohibited_pair_evidence: str
    initial_cap: int
    comparison_cap: int
    escalation_cap: int
    parser_chunk_rows: int
    workspace_byte_ceiling: int
    free_space_reserve: int
    disk_check_interval_seconds: int
    task4_catalog_relative_path: str
    task4_catalog_sha256: str
    task4_catalog_byte_size: int
    task4_catalog_row_count: int
    expected_eligible_records: int
    expected_eligible_residues: int
    task5_public_manifest_relative_path: str
    task5_public_manifest_sha256: str
    task5_local_assignment_relative_path: str
    task5_local_assignment_sha256: str
    task5_report_relative_path: str
    task5_report_sha256: str
    task6_public_manifest_relative_path: str
    task6_public_manifest_sha256: str
    task6_local_assignment_relative_path: str
    task6_local_assignment_sha256: str
    task6_report_relative_path: str
    task6_report_sha256: str
    task6_repair_state_sha256: str
    workspace_relative_path: str


def load_similarity_audit_policy(path: Path) -> SimilarityAuditPolicy:
    """Load the exact committed policy and reject any byte drift."""

    try:
        content = Path(path).read_bytes()
        raw = tomllib.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise SimilarityAuditError(
            f"could not load similarity audit policy: {error}"
        ) from error

    if hashlib.sha256(content).hexdigest() != APPROVED_SIMILARITY_AUDIT_CONFIG_SHA256:
        raise SimilarityAuditError(
            "similarity audit policy bytes do not match the approved checksum"
        )
    expected_fields = {field.name for field in fields(SimilarityAuditPolicy)}
    if set(raw) != expected_fields:
        raise SimilarityAuditError(
            "similarity audit policy fields differ from the approved schema"
        )

    try:
        policy = SimilarityAuditPolicy(
            **{
                field.name: _typed_value(raw[field.name], field.name, field.type)
                for field in fields(SimilarityAuditPolicy)
            }
        )
    except (TypeError, ValueError) as error:
        raise SimilarityAuditError(f"invalid similarity audit policy: {error}") from error
    _validate_policy(policy)
    return policy


def _typed_value(value: object, name: str, annotation: object) -> object:
    expected_type = {"int": int, "str": str, "bool": bool}.get(str(annotation))
    if expected_type is None:
        raise TypeError(f"unsupported policy field type for {name}")
    if expected_type is int and isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if not isinstance(value, expected_type):
        raise TypeError(f"{name} must be {expected_type.__name__}")
    if expected_type is str and not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _validate_policy(policy: SimilarityAuditPolicy) -> None:
    required = {
        "schema_version": (policy.schema_version, 1),
        "scope": (policy.scope, "week_01_task_07_diagnostic_similarity_audit"),
        "adjustment_id": (policy.adjustment_id, "A-003"),
        "diagnostic_audit_authorized": (
            policy.diagnostic_audit_authorized,
            True,
        ),
        "repair_authorized": (policy.repair_authorized, False),
        "selected_split_authorized": (policy.selected_split_authorized, False),
        "task8_membership_use_authorized": (
            policy.task8_membership_use_authorized,
            False,
        ),
        "model_use": (policy.model_use, "prohibited"),
        "post_audit_review_required": (policy.post_audit_review_required, True),
        "candidate_status": (policy.candidate_status, "failed_balance"),
        "repair_performed": (policy.repair_performed, False),
        "createdb order": (
            (policy.createdb_mode, policy.createdb_shuffle),
            (0, 0),
        ),
        "format_output": (
            policy.format_output,
            "query,target,fident,qcov,tcov,alnlen,qlen,tlen,qstart,qend,"
            "tstart,tend,evalue,bits",
        ),
        "prohibited pair evidence": (
            policy.prohibited_pair_evidence,
            "union_of_accepted_enforcement_and_residual_passes",
        ),
        "staged caps": (
            (policy.initial_cap, policy.comparison_cap, policy.escalation_cap),
            (1_000, 10_000, 100_000),
        ),
    }
    drift = [
        f"{name}: found {found!r}, expected {expected!r}"
        for name, (found, expected) in required.items()
        if found != expected
    ]
    if drift:
        raise SimilarityAuditError("policy authority drift: " + "; ".join(drift))

    positive_fields = (
        "threads",
        "initial_cap",
        "comparison_cap",
        "escalation_cap",
        "parser_chunk_rows",
        "workspace_byte_ceiling",
        "free_space_reserve",
        "disk_check_interval_seconds",
        "task4_catalog_byte_size",
        "task4_catalog_row_count",
        "expected_eligible_records",
        "expected_eligible_residues",
    )
    for name in positive_fields:
        if getattr(policy, name) <= 0:
            raise SimilarityAuditError(f"{name} must be positive")

    checksum_fields = tuple(
        field.name
        for field in fields(policy)
        if field.name.endswith("_sha256")
    )
    for name in checksum_fields:
        value = getattr(policy, name)
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise SimilarityAuditError(f"{name} must be a lowercase SHA-256")

    path_fields = tuple(
        field.name
        for field in fields(policy)
        if field.name.endswith("_relative_path")
    )
    for name in path_fields:
        value = getattr(policy, name)
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
            raise SimilarityAuditError(f"{name} must stay inside the repository")
