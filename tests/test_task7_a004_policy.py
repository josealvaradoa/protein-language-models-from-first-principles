import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from protein_lm.data.similarity_audit_policy import SimilarityAuditError
from protein_lm.data.task7_a004_policy import (
    APPROVED_A004_CONFIG_SHA256,
    FIXED_CAPS,
    load_a004_policy,
    resolve_a004_paths,
)

PROJECT_ROOT = Path(__file__).parents[1]
POLICY_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "week_01"
    / "read_only_similarity_audit_a004.toml"
)


def test_a004_policy_freezes_read_only_import_and_separate_workspace() -> None:
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
    assert policy.all_query_cap == 10_000
    assert policy.staged_escalation_cap == 100_000
    assert paths["source_workspace"] != paths["workspace"]


def test_a004_policy_rejects_byte_drift_and_unknown_cap(tmp_path: Path) -> None:
    policy = load_a004_policy(POLICY_PATH)
    drifted = tmp_path / "a004.toml"
    drifted.write_bytes(POLICY_PATH.read_bytes() + b"\n")

    with pytest.raises(SimilarityAuditError, match="approved checksum"):
        load_a004_policy(drifted)
    with pytest.raises(SimilarityAuditError, match="not frozen"):
        policy.stage_marker_sha256(42)


def test_a004_paths_reject_escape_and_shared_workspace() -> None:
    policy = load_a004_policy(POLICY_PATH)

    with pytest.raises(SimilarityAuditError, match="leaves the repository"):
        resolve_a004_paths(
            replace(policy, workspace_relative_path="../outside"),
            PROJECT_ROOT,
        )
    with pytest.raises(SimilarityAuditError, match="non-overlapping sibling"):
        resolve_a004_paths(
            replace(
                policy,
                workspace_relative_path=policy.source_workspace_relative_path,
            ),
            PROJECT_ROOT,
        )
    with pytest.raises(SimilarityAuditError, match="non-overlapping sibling"):
        resolve_a004_paths(
            replace(
                policy,
                workspace_relative_path=(
                    policy.source_workspace_relative_path + "/a004"
                ),
            ),
            PROJECT_ROOT,
        )
    with pytest.raises(SimilarityAuditError, match="non-overlapping sibling"):
        resolve_a004_paths(
            replace(policy, workspace_relative_path="data/processed/week_01"),
            PROJECT_ROOT,
        )
