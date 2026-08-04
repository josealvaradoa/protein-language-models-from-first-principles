import hashlib
import json
from pathlib import Path

import pytest

from protein_lm.data.fixed_budget_audit.errors import (
    AuditExecutionError,
    AuditValidationError,
)
from protein_lm.data.fixed_budget_audit.search import ensure_a004_target_database
from protein_lm.data.similarity_audit_policy import (
    SimilarityAuditError,
    load_similarity_audit_policy,
)
from protein_lm.data.similarity_fastas import FastaEvidence

PROJECT_ROOT = Path(__file__).parents[3]
POLICY_PATH = (
    PROJECT_ROOT / "experiments" / "week_01" / "diagnostic_similarity_audit.toml"
)


def test_a004_database_is_separate_and_resumes_only_after_identity_checks(
    tmp_path: Path,
) -> None:
    policy = load_similarity_audit_policy(POLICY_PATH)
    training, evidence = _write_fasta(tmp_path / "training.fasta")
    workspace = tmp_path / "a004_workspace"
    calls: list[tuple[str, ...]] = []

    def runner(command, project_root, workspace, log_path, policy):
        calls.append(tuple(command))
        Path(command[3]).write_bytes(b"synthetic target database")
        return "0.01"

    database = ensure_a004_target_database(
        strategy="random",
        training_fasta=training,
        training_fasta_evidence=evidence,
        project_root=tmp_path,
        workspace=workspace,
        policy=policy,
        fingerprint="synthetic-fingerprint",
        command_runner=runner,
    )

    assert database.prefix == workspace / "databases/random/target"
    assert database.prefix.is_file()
    assert "/.random.incomplete/target" in calls[0][3]
    assert not (workspace / "databases/.random.incomplete").exists()

    resumed = ensure_a004_target_database(
        strategy="random",
        training_fasta=training,
        training_fasta_evidence=evidence,
        project_root=tmp_path,
        workspace=workspace,
        policy=policy,
        fingerprint="synthetic-fingerprint",
        command_runner=lambda *args: (_ for _ in ()).throw(AssertionError("rerun")),
    )
    assert resumed.identity == database.identity

    training.write_text(">T1\nCCCC\n")
    with pytest.raises(SimilarityAuditError, match="checksum drifted"):
        ensure_a004_target_database(
            strategy="random",
            training_fasta=training,
            training_fasta_evidence=evidence,
            project_root=tmp_path,
            workspace=workspace,
            policy=policy,
            fingerprint="synthetic-fingerprint",
        )


def test_a004_database_resume_rejects_unindexed_artifact(tmp_path: Path) -> None:
    policy = load_similarity_audit_policy(POLICY_PATH)
    training, evidence = _write_fasta(tmp_path / "training.fasta")
    workspace = tmp_path / "a004_workspace"

    def runner(command, *args):
        Path(command[3]).write_bytes(b"synthetic target database")
        return "0.01"

    ensure_a004_target_database(
        strategy="random",
        training_fasta=training,
        training_fasta_evidence=evidence,
        project_root=tmp_path,
        workspace=workspace,
        policy=policy,
        fingerprint="synthetic-fingerprint",
        command_runner=runner,
    )
    (workspace / "databases/random/unexpected.dbtype").write_bytes(b"stale")

    with pytest.raises(SimilarityAuditError, match="artifact inventory drifted"):
        ensure_a004_target_database(
            strategy="random",
            training_fasta=training,
            training_fasta_evidence=evidence,
            project_root=tmp_path,
            workspace=workspace,
            policy=policy,
            fingerprint="synthetic-fingerprint",
            command_runner=lambda *args: (_ for _ in ()).throw(
                AssertionError("completed database must not rerun")
            ),
        )


def test_a004_database_requires_target_artifact(tmp_path: Path) -> None:
    policy = load_similarity_audit_policy(POLICY_PATH)
    training, evidence = _write_fasta(tmp_path / "training.fasta")

    def runner(command, *args):
        Path(f"{command[3]}.dbtype").write_bytes(b"incomplete database")
        return "0.01"

    with pytest.raises(AuditExecutionError, match="no target database"):
        ensure_a004_target_database(
            strategy="random",
            training_fasta=training,
            training_fasta_evidence=evidence,
            project_root=tmp_path,
            workspace=tmp_path / "a004_workspace",
            policy=policy,
            fingerprint="synthetic-fingerprint",
            command_runner=runner,
        )


def test_a004_database_preserves_unmarked_final_directory(tmp_path: Path) -> None:
    policy = load_similarity_audit_policy(POLICY_PATH)
    training, evidence = _write_fasta(tmp_path / "training.fasta")
    preserved = tmp_path / "a004_workspace/databases/random/evidence.bin"
    preserved.parent.mkdir(parents=True)
    preserved.write_bytes(b"preserve me")

    with pytest.raises(
        AuditExecutionError,
        match="database directory lacks its completion marker",
    ):
        ensure_a004_target_database(
            strategy="random",
            training_fasta=training,
            training_fasta_evidence=evidence,
            project_root=tmp_path,
            workspace=tmp_path / "a004_workspace",
            policy=policy,
            fingerprint="synthetic-fingerprint",
        )

    assert preserved.read_bytes() == b"preserve me"


def test_a004_database_maps_completed_identity_drift_to_validation(
    tmp_path: Path,
) -> None:
    policy = load_similarity_audit_policy(POLICY_PATH)
    training, evidence = _write_fasta(tmp_path / "training.fasta")
    workspace = tmp_path / "a004_workspace"

    def runner(command, *args):
        Path(command[3]).write_bytes(b"synthetic target database")
        return "0.01"

    database = ensure_a004_target_database(
        strategy="random",
        training_fasta=training,
        training_fasta_evidence=evidence,
        project_root=tmp_path,
        workspace=workspace,
        policy=policy,
        fingerprint="synthetic-fingerprint",
        command_runner=runner,
    )
    marker = json.loads(database.marker_path.read_text())
    marker["strategy"] = "group_aware"
    database.marker_path.write_text(json.dumps(marker))

    with pytest.raises(AuditValidationError, match="identity drifted"):
        ensure_a004_target_database(
            strategy="random",
            training_fasta=training,
            training_fasta_evidence=evidence,
            project_root=tmp_path,
            workspace=workspace,
            policy=policy,
            fingerprint="synthetic-fingerprint",
            command_runner=lambda *args: (_ for _ in ()).throw(
                AssertionError("completed database must not rerun")
            ),
        )


def _write_fasta(path: Path) -> tuple[Path, FastaEvidence]:
    content = ">T1\nAAAA\n"
    path.write_text(content)
    return path, FastaEvidence(
        record_count=1,
        residue_count=4,
        byte_size=len(content.encode()),
        sha256=hashlib.sha256(content.encode()).hexdigest(),
    )
