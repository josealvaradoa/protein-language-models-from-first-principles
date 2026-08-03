import hashlib
import json
from pathlib import Path

from protein_lm.data.similarity_fastas import FastaEvidence
from protein_lm.data.task7_a004_evidence import ensure_cap_summary, ensure_pair_union
from task7_test_support import alignment_tsv_row, canonicalize_rows, metadata


def test_a004_evidence_marks_imported_and_executed_sources_separately(
    tmp_path: Path,
) -> None:
    query_fasta, query_evidence = _write_fasta(tmp_path / "queries.fasta")
    queries = {"Q1": metadata("q1")}
    targets = {"T1": metadata("t1", partition="training")}
    canonical = canonicalize_rows(
        tmp_path,
        "canonical",
        (alignment_tsv_row("Q1", "T1"),),
        queries,
        targets,
    )
    identity = _evidence(canonical, 1)
    imported = ensure_cap_summary(
        source_label="imported_a003",
        cap=1_000,
        canonical_path=canonical,
        canonical_evidence=identity,
        query_fasta=query_fasta,
        query_fasta_evidence=query_evidence,
        expected_query_ids=queries,
        output_directory=tmp_path / "imported",
        fingerprint="synthetic-fingerprint",
    )
    executed = ensure_cap_summary(
        source_label="executed_a004",
        cap=1_000,
        canonical_path=canonical,
        canonical_evidence=identity,
        query_fasta=query_fasta,
        query_fasta_evidence=query_evidence,
        expected_query_ids=queries,
        output_directory=tmp_path / "executed",
        fingerprint="synthetic-fingerprint",
    )

    assert _marker(imported.directory)["source_label"] == "imported_a003"
    assert _marker(executed.directory)["source_label"] == "executed_a004"
    common = ensure_pair_union(
        label="common_all_query_10000_random_validation",
        source_paths={
            "enforcement_executed_a004_cap_1000": executed.directory
            / "prohibited_pairs.tsv",
            "residual_imported_a003_cap_1000": imported.directory
            / "prohibited_pairs.tsv",
        },
        output_directory=tmp_path / "common",
        fingerprint="synthetic-fingerprint",
    )
    staged = ensure_pair_union(
        label="staged_union_with_changed_query_100000_random_validation",
        source_paths={"common_all_query_10000": common.directory / "prohibited_pairs.tsv"},
        output_directory=tmp_path / "staged",
        fingerprint="synthetic-fingerprint",
    )

    assert _marker(common.directory)["label"].startswith("common_all_query_10000")
    assert _marker(staged.directory)["label"].startswith(
        "staged_union_with_changed_query_100000"
    )
    assert "all_query_100000" not in _marker(staged.directory)["label"]


def _write_fasta(path: Path) -> tuple[Path, FastaEvidence]:
    content = ">Q1\nAAAA\n"
    path.write_text(content)
    return path, FastaEvidence(
        record_count=1,
        residue_count=4,
        byte_size=len(content.encode()),
        sha256=hashlib.sha256(content.encode()).hexdigest(),
    )


def _evidence(path: Path, rows: int):
    content = path.read_bytes()
    from protein_lm.data.similarity_audit_models import FileEvidence

    return FileEvidence(rows, len(content), hashlib.sha256(content).hexdigest())


def _marker(directory: Path) -> dict[str, object]:
    return json.loads((directory / "complete.json").read_text())
