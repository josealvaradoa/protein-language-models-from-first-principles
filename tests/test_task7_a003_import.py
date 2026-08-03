import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from protein_lm.data.similarity_audit_policy import (
    SimilarityAuditError,
    load_similarity_audit_policy,
)
from protein_lm.data.similarity_fastas import FastaEvidence
from protein_lm.data.task7_a003_import import verify_a003_residual_import
from protein_lm.data.task7_a004_policy import A004Policy, FIXED_CAPS, load_a004_policy

PROJECT_ROOT = Path(__file__).parents[1]
SOURCE_POLICY = (
    PROJECT_ROOT / "experiments" / "week_01" / "diagnostic_similarity_audit.toml"
)
A004_POLICY = (
    PROJECT_ROOT
    / "experiments"
    / "week_01"
    / "read_only_similarity_audit_a004.toml"
)


def test_import_verifies_fixture_without_writing(tmp_path: Path) -> None:
    root, policy, _ = _build_fixture(tmp_path)
    before = _snapshot(policy, root)

    imported = verify_a003_residual_import(project_root=root, policy=policy)

    assert imported.fingerprint == policy.source_run_fingerprint
    assert imported.database.artifact_count == 1
    assert tuple(stage.cap for stage in imported.stages) == FIXED_CAPS
    assert imported.escalated_query_ids == ("Q2",)
    assert set(imported.fastas) == {"random", "group_aware"}
    assert set(imported.fastas["random"]) == {"training", "validation", "test"}
    assert set(imported.fastas["group_aware"]) == {"training", "validation", "test"}
    assert _snapshot(policy, root) == before


def test_import_rejects_changed_canonical_output(tmp_path: Path) -> None:
    root, policy, canonical_path = _build_fixture(tmp_path)
    canonical_path.write_bytes(canonical_path.read_bytes() + b"corrupt\n")

    with pytest.raises(SimilarityAuditError, match="checksum drifted"):
        verify_a003_residual_import(project_root=root, policy=policy)


def test_import_rejects_wrong_escalation_membership(tmp_path: Path) -> None:
    root, policy, _ = _build_fixture(tmp_path, escalated_accession="Q1")

    with pytest.raises(SimilarityAuditError, match="differs from cap changes"):
        verify_a003_residual_import(project_root=root, policy=policy)


def test_import_verifies_all_six_preserved_fastas(tmp_path: Path) -> None:
    root, policy, _ = _build_fixture(tmp_path)
    group_test = (
        root
        / policy.source_workspace_relative_path
        / "fastas"
        / "group_aware_test.fasta"
    )
    group_test.write_text(">GX1\nAAAA\n")

    with pytest.raises(SimilarityAuditError, match="checksum drifted"):
        verify_a003_residual_import(project_root=root, policy=policy)


def test_import_rejects_unindexed_database_artifact(tmp_path: Path) -> None:
    root, policy, _ = _build_fixture(tmp_path)
    database = root / policy.source_workspace_relative_path / "databases/random"
    (database / "unexpected.index").write_bytes(b"unexpected")

    with pytest.raises(SimilarityAuditError, match="artifact inventory drifted"):
        verify_a003_residual_import(project_root=root, policy=policy)


def test_import_rejects_missing_indexed_database_artifact(tmp_path: Path) -> None:
    root, policy, _ = _build_fixture(tmp_path)
    database = root / policy.source_workspace_relative_path / "databases/random"
    (database / "target").unlink()

    with pytest.raises(SimilarityAuditError, match="artifact inventory drifted"):
        verify_a003_residual_import(project_root=root, policy=policy)


def _build_fixture(
    tmp_path: Path,
    *,
    escalated_accession: str = "Q2",
) -> tuple[Path, A004Policy, Path]:
    root = tmp_path / "repo"
    source_config = root / "experiments/week_01/diagnostic_similarity_audit.toml"
    source_config.parent.mkdir(parents=True)
    source_config.write_bytes(SOURCE_POLICY.read_bytes())
    source_policy = load_similarity_audit_policy(source_config)
    policy = load_a004_policy(A004_POLICY)
    workspace = root / policy.source_workspace_relative_path
    fingerprint = _source_fingerprint(
        policy=source_policy,
        code_revision=policy.source_code_revision,
        mmseqs_version=policy.source_mmseqs_version,
    )

    fasta_records = {
        "random": {
            "training": (("T1", "AAAA"),),
            "validation": (("Q1", "AAAA"), ("Q2", "CCCC")),
            "test": (("RT1", "DDDD"),),
        },
        "group_aware": {
            "training": (("GT1", "EEEE"),),
            "validation": (("GV1", "FFFF"),),
            "test": (("GX1", "GGGG"),),
        },
    }
    fastas = {
        strategy: {
            partition: _write_fasta(
                workspace / "fastas" / f"{strategy}_{partition}.fasta", records
            )
            for partition, records in partitions.items()
        }
        for strategy, partitions in fasta_records.items()
    }
    training_path = workspace / "fastas/random_training.fasta"
    validation_path = workspace / "fastas/random_validation.fasta"
    training = fastas["random"]["training"]
    validation = fastas["random"]["validation"]
    fastas_marker = workspace / "fastas/complete.json"
    _write_json(
        fastas_marker,
        {
            "schema_version": 1,
            "stage": "materialized_inputs",
            "fingerprint": fingerprint,
            "fastas": {
                strategy: {
                    partition: asdict(evidence)
                    for partition, evidence in partitions.items()
                }
                for strategy, partitions in fastas.items()
            },
        },
    )

    database_directory = workspace / "databases/random"
    database_artifact = database_directory / "target"
    database_artifact.parent.mkdir(parents=True)
    database_artifact.write_bytes(b"fixture database")
    database_marker = database_directory / "complete.json"
    _write_json(
        database_marker,
        {
            "schema_version": 1,
            "stage": "target_database",
            "fingerprint": fingerprint,
            "strategy": "random",
            "training_fasta": asdict(training),
            "command": list(
                _literal_createdb_command(
                    training_path,
                    workspace / "databases/.random.incomplete/target",
                )
            ),
            "runtime_seconds": "0.1",
            "artifacts": {"target": _identity(database_artifact)},
        },
    )

    escalation_path = workspace / "tracks/random/validation/residual/escalated_queries.fasta"
    escalation = _write_fasta(escalation_path, ((escalated_accession, "CCCC"),))
    rows_by_cap = {
        1_000: (_row("Q1"),),
        10_000: (_row("Q1"), _row("Q2")),
        100_000: (_row(escalated_accession),),
    }
    stage_hashes = []
    first_canonical = None
    for cap in FIXED_CAPS:
        stage_directory = workspace / f"tracks/random/validation/residual/cap_{cap}"
        canonical_path = stage_directory / "canonical.tsv"
        canonical_path.parent.mkdir(parents=True)
        canonical_path.write_text("\n".join(rows_by_cap[cap]) + "\n")
        first_canonical = first_canonical or canonical_path
        query_path = escalation_path if cap == 100_000 else validation_path
        query = escalation if cap == 100_000 else validation
        canonical = _file_evidence(canonical_path, len(rows_by_cap[cap]))
        marker_path = stage_directory / "complete.json"
        _write_json(
            marker_path,
            {
                "schema_version": 1,
                "stage": "search_stage",
                "fingerprint": fingerprint,
                "cap": cap,
                "query_count": query.record_count,
                "query_fasta": asdict(query),
                "command": list(
                    _literal_search_command(
                        cap=cap,
                        query_fasta=query_path,
                        target_database=workspace / "databases/random/target",
                        raw_output=stage_directory / "raw.tsv",
                        temp_directory=stage_directory / "mmseqs_tmp",
                    )
                ),
                "runtime_seconds": "0.2",
                "raw_retained": False,
                "alignment_evidence": {
                    "raw": canonical,
                    "canonical": canonical,
                },
            },
        )
        stage_hashes.append((cap, _identity(marker_path)["sha256"]))

    policy = replace(
        policy,
        source_fastas_marker_sha256=str(_identity(fastas_marker)["sha256"]),
        source_database_marker_sha256=str(_identity(database_marker)["sha256"]),
        source_stage_marker_sha256=tuple(stage_hashes),
    )
    assert first_canonical is not None
    return root, policy, first_canonical


def _write_fasta(path: Path, records: tuple[tuple[str, str], ...]) -> FastaEvidence:
    content = "".join(f">{accession}\n{sequence}\n" for accession, sequence in records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    encoded = content.encode()
    return FastaEvidence(
        record_count=len(records),
        residue_count=sum(len(sequence) for _, sequence in records),
        byte_size=len(encoded),
        sha256=hashlib.sha256(encoded).hexdigest(),
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _file_evidence(path: Path, row_count: int) -> dict[str, object]:
    identity = _identity(path)
    return {"row_count": row_count, **identity}


def _row(query: str) -> str:
    return f"{query}\tT1\t5e-1\t1e0\t1e0\t4\t4\t4\t1\t4\t1\t4\t1e-5\t1e1"


def _snapshot(policy: A004Policy, root: Path) -> dict[str, bytes]:
    workspace = root / policy.source_workspace_relative_path
    return {
        path.relative_to(workspace).as_posix(): path.read_bytes()
        for path in workspace.rglob("*")
        if path.is_file()
    }


def _identity(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {"byte_size": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def _source_fingerprint(*, policy, code_revision: str, mmseqs_version: str) -> str:
    payload = {
        "code_revision": code_revision,
        "config_sha256": (
            "ce767f0ce843e4f40edbcd2f9da6ca4642996046cb4042a2410c27c39cbae742"
        ),
        "mmseqs_version": mmseqs_version,
        "task4_catalog_sha256": policy.task4_catalog_sha256,
        "task5_public_manifest_sha256": policy.task5_public_manifest_sha256,
        "task5_local_assignment_sha256": policy.task5_local_assignment_sha256,
        "task5_report_sha256": policy.task5_report_sha256,
        "task6_public_manifest_sha256": policy.task6_public_manifest_sha256,
        "task6_local_assignment_sha256": policy.task6_local_assignment_sha256,
        "task6_report_sha256": policy.task6_report_sha256,
        "task6_repair_state_sha256": policy.task6_repair_state_sha256,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def _literal_createdb_command(
    training_fasta: Path, database_prefix: Path
) -> tuple[str, ...]:
    return (
        "/opt/homebrew/bin/mmseqs",
        "createdb",
        str(training_fasta),
        str(database_prefix),
        "--dbtype",
        "1",
        "--shuffle",
        "0",
        "--createdb-mode",
        "0",
        "--threads",
        "10",
    )


def _literal_search_command(
    *,
    cap: int,
    query_fasta: Path,
    target_database: Path,
    raw_output: Path,
    temp_directory: Path,
) -> tuple[str, ...]:
    return (
        "/opt/homebrew/bin/mmseqs",
        "easy-search",
        str(query_fasta),
        str(target_database),
        str(raw_output),
        str(temp_directory),
        "--search-type",
        "1",
        "--alignment-mode",
        "3",
        "--seq-id-mode",
        "0",
        "-s",
        "7.5",
        "-e",
        "10",
        "--mask",
        "0",
        "--comp-bias-corr",
        "0",
        "--max-seqs",
        str(cap),
        "--threads",
        "10",
        "--format-output",
        (
            "query,target,fident,qcov,tcov,alnlen,qlen,tlen,qstart,qend,"
            "tstart,tend,evalue,bits"
        ),
        "--min-seq-id",
        "0.3",
        "-c",
        "0.0",
        "--cov-mode",
        "0",
    )
