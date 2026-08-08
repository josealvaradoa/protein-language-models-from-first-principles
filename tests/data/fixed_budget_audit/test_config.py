"""Frozen configuration and enum contracts for the fixed-budget audit."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from protein_lm.data.fixed_budget_audit.config import (
    APPROVED_A004_CONFIG_SHA256,
    FIXED_CAPS,
    AuditPass,
    CandidateCap,
    DatasetPartition,
    PairUnionKind,
    QueryScope,
    SplitStrategy,
    TrackOrigin,
    load_a004_policy,
    resolve_a004_paths,
)
from protein_lm.data.fixed_budget_audit.errors import AuditConfigurationError
from protein_lm.data.similarity_audit_policy import SimilarityAuditError

PROJECT_ROOT = Path(__file__).parents[3]
POLICY_PATH = (
    PROJECT_ROOT / "experiments" / "week_01" / "read_only_similarity_audit_a004.toml"
)


def test_core_enum_wire_values_and_fixed_caps_are_exact() -> None:
    assert tuple((item.name, item.value) for item in SplitStrategy) == (
        ("RANDOM", "random"),
        ("GROUP_AWARE", "group_aware"),
    )
    assert tuple((item.name, item.value) for item in DatasetPartition) == (
        ("VALIDATION", "validation"),
        ("TEST", "test"),
    )
    assert tuple((item.name, item.value) for item in AuditPass) == (
        ("ENFORCEMENT", "enforcement"),
        ("RESIDUAL", "residual"),
    )
    assert tuple((item.name, item.value) for item in TrackOrigin) == (
        ("IMPORTED_A003", "imported_a003"),
        ("EXECUTED_A004", "executed_a004"),
    )
    assert tuple((item.name, item.value) for item in PairUnionKind) == (
        ("COMMON_ALL_QUERY_10000", "common_all_query_10000"),
        (
            "STAGED_UNION_WITH_CHANGED_QUERY_100000",
            "staged_union_with_changed_query_100000",
        ),
    )
    assert tuple((item.name, item.value) for item in QueryScope) == (
        ("ALL_QUERIES", "all_queries"),
        (
            "CHANGED_QUERIES_1000_TO_10000",
            "changed_queries_1000_to_10000",
        ),
    )
    assert tuple((item.name, item.value) for item in CandidateCap) == (
        ("INITIAL", 1_000),
        ("COMPARISON", 10_000),
        ("ESCALATION", 100_000),
    )
    assert FIXED_CAPS == tuple(cap.value for cap in CandidateCap)


def test_policy_freezes_read_only_import_and_separate_workspace() -> None:
    policy = load_a004_policy(POLICY_PATH)
    paths = resolve_a004_paths(policy, PROJECT_ROOT)

    assert hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest() == (
        APPROVED_A004_CONFIG_SHA256
    )
    assert policy.read_only is True
    assert policy.repair_authorized is False
    assert policy.task8_membership_use_authorized is False
    assert policy.model_use == "prohibited"
    assert tuple(cap for cap, _ in policy.source_stage_marker_sha256) == FIXED_CAPS
    assert policy.all_query_cap is CandidateCap.COMPARISON
    assert policy.staged_escalation_cap is CandidateCap.ESCALATION
    assert policy.import_strategy is SplitStrategy.RANDOM
    assert policy.import_partition is DatasetPartition.VALIDATION
    assert policy.import_pass is AuditPass.RESIDUAL
    assert paths["source_workspace"] != paths["workspace"]


def test_explicit_json_boundary_values_remain_plain_strings_and_integers() -> None:
    policy = load_a004_policy(POLICY_PATH)
    wire = {
        "strategy": policy.import_strategy.value,
        "partition": policy.import_partition.value,
        "pass_name": policy.import_pass.value,
        "cap": CandidateCap.INITIAL.value,
    }

    assert type(wire["strategy"]) is str
    assert type(wire["partition"]) is str
    assert type(wire["pass_name"]) is str
    assert type(wire["cap"]) is int
    assert json.dumps(wire, sort_keys=True, separators=(",", ":")) == (
        '{"cap":1000,"partition":"validation","pass_name":"residual",'
        '"strategy":"random"}'
    )


def test_policy_rejects_byte_drift_and_unknown_cap_with_typed_error(
    tmp_path: Path,
) -> None:
    policy = load_a004_policy(POLICY_PATH)
    drifted = tmp_path / "a004.toml"
    drifted.write_bytes(POLICY_PATH.read_bytes() + b"\n")

    with pytest.raises(AuditConfigurationError, match="approved checksum") as raised:
        load_a004_policy(drifted)
    assert isinstance(raised.value, SimilarityAuditError)
    with pytest.raises(AuditConfigurationError, match="not frozen"):
        policy.stage_marker_sha256(42)


def test_paths_reject_escape_and_shared_workspace() -> None:
    policy = load_a004_policy(POLICY_PATH)

    with pytest.raises(AuditConfigurationError, match="leaves the repository"):
        resolve_a004_paths(
            replace(policy, workspace_relative_path="../outside"),
            PROJECT_ROOT,
        )
    with pytest.raises(AuditConfigurationError, match="non-overlapping sibling"):
        resolve_a004_paths(
            replace(
                policy,
                workspace_relative_path=policy.source_workspace_relative_path,
            ),
            PROJECT_ROOT,
        )
    with pytest.raises(AuditConfigurationError, match="non-overlapping sibling"):
        resolve_a004_paths(
            replace(
                policy,
                workspace_relative_path=(
                    policy.source_workspace_relative_path + "/a004"
                ),
            ),
            PROJECT_ROOT,
        )
    with pytest.raises(AuditConfigurationError, match="non-overlapping sibling"):
        resolve_a004_paths(
            replace(policy, workspace_relative_path="data/processed/week_01"),
            PROJECT_ROOT,
        )
