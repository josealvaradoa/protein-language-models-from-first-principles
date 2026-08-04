import hashlib
from pathlib import Path

import protein_lm.data.fixed_budget_audit.source as source_module
import pytest

from protein_lm.data.fixed_budget_audit.errors import SourceEvidenceError
from protein_lm.data.fixed_budget_audit.source import (
    ensure_materialized_inputs,
    reverify_frozen_run_state,
)
from protein_lm.data.similarity_audit_policy import (
    load_similarity_audit_policy,
)

PROJECT_ROOT = Path(__file__).parents[3]
POLICY_PATH = (
    PROJECT_ROOT / "experiments" / "week_01" / "diagnostic_similarity_audit.toml"
)


def test_materialized_inputs_preserve_unmarked_nonempty_fasta_directory(
    tmp_path: Path,
) -> None:
    partial = tmp_path / "workspace/fastas/random_training.fasta"
    partial.parent.mkdir(parents=True)
    partial.write_text(">T1\nAAAA\n")

    with pytest.raises(SourceEvidenceError, match="lacks its completion marker"):
        ensure_materialized_inputs(
            workspace=tmp_path / "workspace",
            catalog_path=tmp_path / "missing-catalog.tsv",
            manifests={},
            policy=load_similarity_audit_policy(POLICY_PATH),
            fingerprint="synthetic-fingerprint",
        )

    assert partial.read_text() == ">T1\nAAAA\n"


def test_frozen_run_reverification_classifies_source_policy_byte_drift(
    monkeypatch,
    tmp_path: Path,
) -> None:
    policy = load_similarity_audit_policy(POLICY_PATH)
    config_path = tmp_path / "diagnostic_similarity_audit.toml"
    config_path.write_bytes(POLICY_PATH.read_bytes() + b"\n# valid TOML comment\n")
    expected = {
        "catalog": policy.task4_catalog_sha256,
        "task5_public": policy.task5_public_manifest_sha256,
        "task5_local": policy.task5_local_assignment_sha256,
        "task5_report": policy.task5_report_sha256,
        "task6_public": policy.task6_public_manifest_sha256,
        "task6_local": policy.task6_local_assignment_sha256,
        "task6_report": policy.task6_report_sha256,
    }
    paths = {name: tmp_path / name for name in expected}

    def synthetic_identity(path: Path) -> dict[str, object]:
        if path == config_path:
            content = path.read_bytes()
            return {
                "byte_size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        return {"byte_size": 1, "sha256": expected[path.name]}

    monkeypatch.setattr(source_module, "file_identity", synthetic_identity)

    with pytest.raises(SourceEvidenceError, match="source policy changed"):
        reverify_frozen_run_state(
            paths=paths,
            policy=policy,
            code_revision="b" * 40,
            mmseqs_version=policy.mmseqs_version,
            config_path=config_path,
            project_root=tmp_path,
        )
