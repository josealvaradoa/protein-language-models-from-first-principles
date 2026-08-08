import hashlib
from pathlib import Path

import pytest

from protein_lm.data.similarity_audit_policy import (
    APPROVED_SIMILARITY_AUDIT_CONFIG_SHA256,
    SimilarityAuditError,
    load_similarity_audit_policy,
)

PROJECT_ROOT = Path(__file__).parents[1]
POLICY_PATH = (
    PROJECT_ROOT / "experiments" / "week_01" / "diagnostic_similarity_audit.toml"
)


def test_policy_is_byte_pinned_and_rejects_drift(tmp_path: Path) -> None:
    policy = load_similarity_audit_policy(POLICY_PATH)
    assert policy.adjustment_id == "A-003"
    assert policy.repair_authorized is False
    assert policy.model_use == "prohibited"
    assert hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest() == (
        APPROVED_SIMILARITY_AUDIT_CONFIG_SHA256
    )

    drifted = tmp_path / "policy.toml"
    drifted.write_bytes(POLICY_PATH.read_bytes() + b"\n")
    with pytest.raises(SimilarityAuditError, match="approved checksum"):
        load_similarity_audit_policy(drifted)
