import hashlib
import json
from pathlib import Path

import pytest

from similarity_evidence_test_support import alignment_tsv_row, metadata
from protein_lm.data.fixed_budget_audit.errors import (
    AuditConfigurationError,
    AuditExecutionError,
    AuditValidationError,
)
from protein_lm.data.fixed_budget_audit.config import (
    AuditPass,
    DatasetPartition,
    SplitStrategy,
)
from protein_lm.data.fixed_budget_audit.tracks import ensure_fixed_budget_pass
from protein_lm.data.similarity_audit_policy import load_similarity_audit_policy
from protein_lm.data.similarity_fastas import FastaEvidence
from protein_lm.data.similarity_results import compare_canonical_results

PROJECT_ROOT = Path(__file__).parents[3]
POLICY_PATH = (
    PROJECT_ROOT / "experiments" / "week_01" / "diagnostic_similarity_audit.toml"
)


def test_fixed_budget_executor_retains_cap_sensitive_stages_and_resumes(
    tmp_path: Path,
) -> None:
    policy = load_similarity_audit_policy(POLICY_PATH)
    query_fasta, query_evidence = _write_fasta(
        tmp_path / "queries.fasta", (("Q1", "AAAA"), ("Q2", "CCCC"))
    )
    queries = {"Q1": metadata("q1"), "Q2": metadata("q2")}
    targets = {
        "T1": metadata("t1", partition="training"),
        "T2": metadata("t2", partition="training"),
    }
    calls: list[tuple[int, tuple[str, ...]]] = []

    def runner(command, project_root, workspace, log_path, policy):
        cap = int(command[command.index("--max-seqs") + 1])
        query_ids = tuple(
            line[1:]
            for line in Path(command[2]).read_text().splitlines()
            if line.startswith(">")
        )
        calls.append((cap, query_ids))
        rows = {
            1_000: (alignment_tsv_row("Q1", "T1"),),
            10_000: (
                alignment_tsv_row("Q1", "T1"),
                alignment_tsv_row("Q2", "T2", fident="0.45"),
            ),
            100_000: (alignment_tsv_row("Q2", "T1", fident="0.60"),),
        }[cap]
        Path(command[4]).write_text("\n".join(rows) + "\n")
        return "0.01"

    result = ensure_fixed_budget_pass(
        strategy=SplitStrategy.RANDOM,
        partition=DatasetPartition.VALIDATION,
        pass_name=AuditPass.ENFORCEMENT,
        query_fasta=query_fasta,
        query_fasta_evidence=query_evidence,
        query_metadata=queries,
        target_database=tmp_path / "target_database",
        target_database_identity={"marker": "synthetic-target"},
        target_metadata=targets,
        project_root=tmp_path,
        workspace=tmp_path / "a004_workspace",
        policy=policy,
        fingerprint="synthetic-fingerprint",
        command_runner=runner,
    )

    assert calls == [(1_000, ("Q1", "Q2")), (10_000, ("Q1", "Q2")), (100_000, ("Q2",))]
    assert result.changed_query_ids == ("Q2",)
    assert tuple(stage.cap for stage in result.stages) == (1_000, 10_000, 100_000)
    for stage in result.stages:
        assert stage.canonical_path.is_file()
        assert not (stage.canonical_path.parent / "raw.tsv").exists()
    assert compare_canonical_results(
        result.stage(10_000).canonical_path,
        result.stage(100_000).canonical_path,
        expected_query_ids=("Q2",),
        left_may_contain_other_queries=True,
    ) == ("Q2",)
    with pytest.raises(AuditConfigurationError, match="cap is unavailable"):
        result.stage(999)

    def fail_runner(*args):
        raise AssertionError("a completed stage must not rerun")

    resumed = ensure_fixed_budget_pass(
        strategy=SplitStrategy.RANDOM,
        partition=DatasetPartition.VALIDATION,
        pass_name=AuditPass.ENFORCEMENT,
        query_fasta=query_fasta,
        query_fasta_evidence=query_evidence,
        query_metadata=queries,
        target_database=tmp_path / "target_database",
        target_database_identity={"marker": "synthetic-target"},
        target_metadata=targets,
        project_root=tmp_path,
        workspace=tmp_path / "a004_workspace",
        policy=policy,
        fingerprint="synthetic-fingerprint",
        command_runner=fail_runner,
    )
    assert resumed.changed_query_ids == ("Q2",)

    with pytest.raises(AuditValidationError, match="search-stage identity drifted"):
        ensure_fixed_budget_pass(
            strategy=SplitStrategy.RANDOM,
            partition=DatasetPartition.VALIDATION,
            pass_name=AuditPass.ENFORCEMENT,
            query_fasta=query_fasta,
            query_fasta_evidence=query_evidence,
            query_metadata=queries,
            target_database=tmp_path / "target_database",
            target_database_identity={"marker": "different-target"},
            target_metadata=targets,
            project_root=tmp_path,
            workspace=tmp_path / "a004_workspace",
            policy=policy,
            fingerprint="synthetic-fingerprint",
            command_runner=fail_runner,
        )

    marker = json.loads(result.marker_path.read_text())
    marker["changed_query_ids"] = []
    result.marker_path.write_text(json.dumps(marker))
    with pytest.raises(AuditValidationError, match="pass identity drifted"):
        ensure_fixed_budget_pass(
            strategy=SplitStrategy.RANDOM,
            partition=DatasetPartition.VALIDATION,
            pass_name=AuditPass.ENFORCEMENT,
            query_fasta=query_fasta,
            query_fasta_evidence=query_evidence,
            query_metadata=queries,
            target_database=tmp_path / "target_database",
            target_database_identity={"marker": "synthetic-target"},
            target_metadata=targets,
            project_root=tmp_path,
            workspace=tmp_path / "a004_workspace",
            policy=policy,
            fingerprint="synthetic-fingerprint",
            command_runner=fail_runner,
        )


def test_fixed_budget_executor_preserves_unmarked_stage_for_inspection(
    tmp_path: Path,
) -> None:
    policy = load_similarity_audit_policy(POLICY_PATH)
    query_fasta, query_evidence = _write_fasta(
        tmp_path / "queries.fasta", (("Q1", "AAAA"),)
    )
    stage = tmp_path / "workspace/tracks/random/validation/enforcement/cap_1000"
    stage.mkdir(parents=True)
    (stage / "raw.tsv").write_text("incomplete evidence\n")

    with pytest.raises(AuditExecutionError, match="unmarked output"):
        ensure_fixed_budget_pass(
            strategy=SplitStrategy.RANDOM,
            partition=DatasetPartition.VALIDATION,
            pass_name=AuditPass.ENFORCEMENT,
            query_fasta=query_fasta,
            query_fasta_evidence=query_evidence,
            query_metadata={"Q1": metadata("q1")},
            target_database=tmp_path / "target_database",
            target_database_identity={"marker": "synthetic-target"},
            target_metadata={"T1": metadata("t1", partition="training")},
            project_root=tmp_path,
            workspace=tmp_path / "workspace",
            policy=policy,
            fingerprint="synthetic-fingerprint",
            command_runner=lambda *args: "0.01",
        )
    assert (stage / "raw.tsv").read_text() == "incomplete evidence\n"


def test_fixed_budget_executor_rejects_stale_cap_when_no_query_changed(
    tmp_path: Path,
) -> None:
    policy = load_similarity_audit_policy(POLICY_PATH)
    query_fasta, query_evidence = _write_fasta(
        tmp_path / "queries.fasta", (("Q1", "AAAA"),)
    )
    pass_directory = tmp_path / "workspace/tracks/random/test/residual"
    stale = pass_directory / "cap_100000/stale.tsv"
    stale.parent.mkdir(parents=True)
    stale.write_text("preserve me\n")

    def runner(command, *args):
        Path(command[4]).write_text(
            alignment_tsv_row(
                "Q1",
                "T1",
                qlen=4,
                tlen=4,
                alnlen=4,
                qend=4,
                tend=4,
            )
            + "\n"
        )
        return "0.01"

    with pytest.raises(AuditExecutionError, match="cap inventory drifted"):
        ensure_fixed_budget_pass(
            strategy=SplitStrategy.RANDOM,
            partition=DatasetPartition.TEST,
            pass_name=AuditPass.RESIDUAL,
            query_fasta=query_fasta,
            query_fasta_evidence=query_evidence,
            query_metadata={"Q1": metadata("q1", length=4, partition="test")},
            target_database=tmp_path / "target_database",
            target_database_identity={"marker": "synthetic-target"},
            target_metadata={"T1": metadata("t1", length=4, partition="training")},
            project_root=tmp_path,
            workspace=tmp_path / "workspace",
            policy=policy,
            fingerprint="synthetic-fingerprint",
            command_runner=runner,
        )
    assert stale.read_text() == "preserve me\n"
    assert not (pass_directory / "complete.json").exists()


def test_fixed_budget_executor_rejects_the_imported_track_identity(
    tmp_path: Path,
) -> None:
    policy = load_similarity_audit_policy(POLICY_PATH)

    with pytest.raises(
        AuditConfigurationError,
        match="outside the fixed execution topology",
    ):
        ensure_fixed_budget_pass(
            strategy=SplitStrategy.RANDOM,
            partition=DatasetPartition.VALIDATION,
            pass_name=AuditPass.RESIDUAL,
            query_fasta=tmp_path / "missing.fasta",
            query_fasta_evidence=FastaEvidence(0, 0, 0, ""),
            query_metadata={},
            target_database=tmp_path / "target_database",
            target_database_identity={},
            target_metadata={},
            project_root=tmp_path,
            workspace=tmp_path / "workspace",
            policy=policy,
            fingerprint="synthetic-fingerprint",
        )


def _write_fasta(
    path: Path, records: tuple[tuple[str, str], ...]
) -> tuple[Path, FastaEvidence]:
    content = "".join(f">{accession}\n{sequence}\n" for accession, sequence in records)
    path.write_text(content)
    encoded = content.encode()
    return path, FastaEvidence(
        record_count=len(records),
        residue_count=sum(len(sequence) for _, sequence in records),
        byte_size=len(encoded),
        sha256=hashlib.sha256(encoded).hexdigest(),
    )
