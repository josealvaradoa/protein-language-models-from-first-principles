"""Pure fingerprint, hardware, and frozen-input provenance contracts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from protein_lm.data.artifacts import file_identity
from protein_lm.data.fixed_budget_audit.config import (
    APPROVED_A004_CONFIG_SHA256,
    A004Policy,
    DatasetPartition,
    SplitStrategy,
)
from protein_lm.data.fixed_budget_audit.errors import (
    AuditConfigurationError,
    SourceEvidenceError,
)
from protein_lm.data.similarity_audit_policy import SimilarityAuditPolicy

if TYPE_CHECKING:
    from protein_lm.data.fixed_budget_audit.source import A003Import
    from protein_lm.data.similarity_fastas import MaterializedInputs

HARDWARE_FIELDS = ("platform", "machine", "processor", "logical_cpu_count")
_ASSIGNMENT_IDENTITY_NAMES = (
    "task5_public",
    "task5_local",
    "task5_report",
    "task6_public",
    "task6_local",
    "task6_report",
)
_FASTA_PARTITIONS = (
    "training",
    DatasetPartition.VALIDATION.value,
    DatasetPartition.TEST.value,
)


def hardware_provenance() -> dict[str, object]:
    """Capture the same host fields used by the A-003 audit report."""

    hardware = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
    }
    validate_hardware_provenance(hardware)
    return hardware


def validate_hardware_provenance(hardware: Mapping[str, object]) -> None:
    """Require one complete, JSON-safe hardware provenance object."""

    if not isinstance(hardware, Mapping):
        raise AuditConfigurationError("A-004 hardware provenance is malformed")
    if set(hardware) != set(HARDWARE_FIELDS):
        raise AuditConfigurationError("A-004 hardware provenance fields drifted")
    if any(not isinstance(hardware[name], str) for name in HARDWARE_FIELDS[:3]):
        raise AuditConfigurationError("A-004 hardware provenance is malformed")
    cpu_count = hardware["logical_cpu_count"]
    if isinstance(cpu_count, bool) or not isinstance(cpu_count, int) or cpu_count < 1:
        raise AuditConfigurationError("A-004 logical CPU count is malformed")


def a004_fingerprint(
    *,
    policy: A004Policy,
    source_policy: SimilarityAuditPolicy,
    code_revision: str,
    mmseqs_version: str,
) -> str:
    """Bind A-004 output to its code, two policies, tool, and frozen inputs."""

    if (
        not isinstance(policy, A004Policy)
        or not isinstance(source_policy, SimilarityAuditPolicy)
        or not isinstance(code_revision, str)
        or not code_revision
        or not isinstance(mmseqs_version, str)
        or not mmseqs_version
    ):
        raise AuditConfigurationError("A-004 fingerprint inputs are malformed")
    payload = {
        "a004_policy_sha256": APPROVED_A004_CONFIG_SHA256,
        "a003_policy_sha256": policy.source_policy_sha256,
        "a003_run_fingerprint": policy.source_run_fingerprint,
        "a003_code_revision": policy.source_code_revision,
        "a004_code_revision": code_revision,
        "mmseqs_version": mmseqs_version,
        "task4_catalog_sha256": source_policy.task4_catalog_sha256,
        "task5_local_assignment_sha256": source_policy.task5_local_assignment_sha256,
        "task6_local_assignment_sha256": source_policy.task6_local_assignment_sha256,
    }
    if any(not isinstance(value, str) or not value for value in payload.values()):
        raise AuditConfigurationError("A-004 fingerprint inputs are malformed")
    content = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(content).hexdigest()


def frozen_assignment_identities(
    paths: Mapping[str, Path],
) -> dict[str, dict[str, object]]:
    """Capture only immutable Task 5/6 evidence identities, never memberships."""

    if not isinstance(paths, Mapping) or not set(_ASSIGNMENT_IDENTITY_NAMES) <= set(
        paths
    ):
        raise SourceEvidenceError("A-004 frozen assignment paths are incomplete")
    return {name: file_identity(paths[name]) for name in _ASSIGNMENT_IDENTITY_NAMES}


def require_same_six_fastas(
    inputs: MaterializedInputs,
    imported: A003Import,
) -> None:
    """Require newly materialized A-004 inputs to equal all preserved A-003 FASTAs."""

    from protein_lm.data.similarity_fastas import MaterializedInputs

    expected_strategies = {strategy.value for strategy in SplitStrategy}
    expected_partitions = set(_FASTA_PARTITIONS)
    if (
        not isinstance(inputs, MaterializedInputs)
        or set(inputs.fastas) != expected_strategies
        or any(
            set(inputs.fastas[strategy]) != expected_partitions
            for strategy in expected_strategies
        )
    ):
        raise SourceEvidenceError("A-004 FASTA differs from preserved A-003 input")
    for strategy_member in SplitStrategy:
        strategy = strategy_member.value
        for partition in _FASTA_PARTITIONS:
            if inputs.fastas[strategy][partition] != imported.fasta(
                strategy, partition
            ):
                raise SourceEvidenceError(
                    "A-004 FASTA differs from preserved A-003 input"
                )
