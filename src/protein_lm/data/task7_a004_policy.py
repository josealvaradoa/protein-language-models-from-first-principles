"""Load the frozen A-004 read-only Task 7 continuation policy."""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path, PurePosixPath

from protein_lm.data.similarity_audit_policy import (
    APPROVED_SIMILARITY_AUDIT_CONFIG_SHA256,
    SimilarityAuditError,
)

APPROVED_A004_CONFIG_SHA256 = (
    "3a21edeaf45057a8e50b5643abc14d3b633edb69b66aefd184e5f59963931a04"
)
FIXED_CAPS = (1_000, 10_000, 100_000)


@dataclass(frozen=True)
class A004Policy:
    """Authority, paths, reporting caps, and imported A-003 marker identities."""

    schema_version: int
    scope: str
    adjustment_id: str
    source_adjustment_id: str
    read_only: bool
    repair_authorized: bool
    selected_split_authorized: bool
    task8_membership_use_authorized: bool
    model_use: str
    source_policy_relative_path: str
    source_policy_sha256: str
    source_workspace_relative_path: str
    workspace_relative_path: str
    source_code_revision: str
    source_run_fingerprint: str
    source_mmseqs_version: str
    all_query_cap: int
    staged_escalation_cap: int
    import_strategy: str
    import_partition: str
    import_pass: str
    source_fastas_marker_sha256: str
    source_database_marker_sha256: str
    source_stage_marker_sha256: tuple[tuple[int, str], ...]

    def stage_marker_sha256(self, cap: int) -> str:
        """Return the frozen marker hash for one fixed search cap."""

        try:
            return dict(self.source_stage_marker_sha256)[cap]
        except KeyError as error:
            raise SimilarityAuditError(f"cap is not frozen for A-004: {cap}") from error


def load_a004_policy(path: Path) -> A004Policy:
    """Load the exact approved A-004 config and reject byte or schema drift."""

    try:
        content = path.read_bytes()
        raw = tomllib.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise SimilarityAuditError(f"could not load A-004 policy: {error}") from error

    if hashlib.sha256(content).hexdigest() != APPROVED_A004_CONFIG_SHA256:
        raise SimilarityAuditError("A-004 policy bytes do not match the approved checksum")
    if set(raw) != {field.name for field in fields(A004Policy)}:
        raise SimilarityAuditError("A-004 policy fields differ from the approved schema")

    stages = _stage_marker_pins(raw["source_stage_marker_sha256"])
    try:
        policy = A004Policy(
            **{
                field.name: _typed_value(raw[field.name], field.name, field.type)
                for field in fields(A004Policy)
                if field.name != "source_stage_marker_sha256"
            },
            source_stage_marker_sha256=stages,
        )
    except (TypeError, ValueError) as error:
        raise SimilarityAuditError(f"invalid A-004 policy: {error}") from error
    _validate_policy(policy)
    return policy


def resolve_a004_paths(policy: A004Policy, project_root: Path) -> dict[str, Path]:
    """Resolve configured paths while keeping both workspaces inside the repo."""

    configured = {
        "source_policy": policy.source_policy_relative_path,
        "source_workspace": policy.source_workspace_relative_path,
        "workspace": policy.workspace_relative_path,
    }
    root = project_root.resolve()
    resolved: dict[str, Path] = {}
    for name, relative_path in configured.items():
        path = (project_root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise SimilarityAuditError(f"configured {name} path leaves the repository") from error
        resolved[name] = path
    source_workspace = resolved["source_workspace"]
    workspace = resolved["workspace"]
    if workspace.is_relative_to(source_workspace) or source_workspace.is_relative_to(
        workspace
    ):
        raise SimilarityAuditError("A-004 must use a non-overlapping sibling workspace")
    return resolved


def _stage_marker_pins(raw: object) -> tuple[tuple[int, str], ...]:
    if not isinstance(raw, dict) or set(raw) != {str(cap) for cap in FIXED_CAPS}:
        raise SimilarityAuditError("A-004 stage-marker caps differ from the fixed caps")
    pins = tuple((cap, raw[str(cap)]) for cap in FIXED_CAPS)
    for cap, checksum in pins:
        _require_sha256(checksum, f"cap {cap} marker SHA-256")
    return pins


def _typed_value(value: object, name: str, annotation: object) -> object:
    expected_type = {"int": int, "str": str, "bool": bool}.get(str(annotation))
    if expected_type is None:
        raise TypeError(f"unsupported A-004 policy field type for {name}")
    if expected_type is int and isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if not isinstance(value, expected_type):
        raise TypeError(f"{name} must be {expected_type.__name__}")
    if expected_type is str and not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _validate_policy(policy: A004Policy) -> None:
    required = {
        "schema_version": (policy.schema_version, 1),
        "scope": (policy.scope, "week_01_task_07_read_only_fixed_budget_audit"),
        "adjustment_id": (policy.adjustment_id, "A-004"),
        "source_adjustment_id": (policy.source_adjustment_id, "A-003"),
        "read_only": (policy.read_only, True),
        "repair_authorized": (policy.repair_authorized, False),
        "selected_split_authorized": (policy.selected_split_authorized, False),
        "task8_membership_use_authorized": (
            policy.task8_membership_use_authorized,
            False,
        ),
        "model_use": (policy.model_use, "prohibited"),
        "source policy SHA-256": (
            policy.source_policy_sha256,
            APPROVED_SIMILARITY_AUDIT_CONFIG_SHA256,
        ),
        "reporting caps": (
            (policy.all_query_cap, policy.staged_escalation_cap),
            (10_000, 100_000),
        ),
        "import identity": (
            (policy.import_strategy, policy.import_partition, policy.import_pass),
            ("random", "validation", "residual"),
        ),
    }
    drift = [
        f"{name}: found {found!r}, expected {expected!r}"
        for name, (found, expected) in required.items()
        if found != expected
    ]
    if drift:
        raise SimilarityAuditError("A-004 authority drift: " + "; ".join(drift))

    for name in (
        "source_policy_relative_path",
        "source_workspace_relative_path",
        "workspace_relative_path",
    ):
        value = getattr(policy, name)
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
            raise SimilarityAuditError(f"{name} must stay inside the repository")
    if policy.source_workspace_relative_path == policy.workspace_relative_path:
        raise SimilarityAuditError("A-004 must use a sibling workspace")

    for name in (
        "source_policy_sha256",
        "source_run_fingerprint",
        "source_fastas_marker_sha256",
        "source_database_marker_sha256",
    ):
        _require_sha256(getattr(policy, name), name)
    if len(policy.source_code_revision) != 40:
        raise SimilarityAuditError("source_code_revision must be a full Git commit hash")
    _require_hex(policy.source_code_revision, "source_code_revision")


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise SimilarityAuditError(f"{name} must be a lowercase SHA-256")
    _require_hex(value, name)


def _require_hex(value: str, name: str) -> None:
    if any(character not in "0123456789abcdef" for character in value):
        raise SimilarityAuditError(f"{name} must use lowercase hexadecimal")
