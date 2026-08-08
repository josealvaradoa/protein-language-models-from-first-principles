import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from protein_lm.data import similarity_inputs
from protein_lm.data.eligibility import CATALOG_COLUMNS
from protein_lm.data.similarity_audit_policy import (
    SimilarityAuditError,
    load_similarity_audit_policy,
)
from protein_lm.data.similarity_fastas import (
    iter_one_line_fasta,
    materialize_strategy_fastas,
)
from protein_lm.data.similarity_manifests import load_strategy_manifest

PROJECT_ROOT = Path(__file__).parents[1]
POLICY_PATH = (
    PROJECT_ROOT / "experiments" / "week_01" / "diagnostic_similarity_audit.toml"
)


def test_input_facade_preserves_original_public_imports() -> None:
    assert similarity_inputs.load_strategy_manifest is load_strategy_manifest
    assert (
        similarity_inputs.materialize_strategy_fastas
        is materialize_strategy_fastas
    )


def _catalog_row(accession: str, sequence: str, group: str) -> str:
    return "\t".join(
        (
            accession,
            sequence,
            hashlib.sha256(sequence.encode()).hexdigest(),
            str(len(sequence)),
            "false",
            "false",
            "false",
            "false",
            "false",
            "true",
            "",
            group,
            "false",
        )
    )


def _write_manifest_pair(
    tmp_path: Path,
    name: str,
    strategy: str,
    stage: str,
    rows: list[tuple[str, str, str, str]],
) -> tuple[Path, Path]:
    public = tmp_path / f"{name}_public.tsv"
    local = tmp_path / f"{name}_local.tsv"
    public_lines = [
        "primary_accession\tpartition\tsequence_sha256\tbiological_length\t"
        "uniref50_group"
    ]
    local_lines = [
        "strategy\tstage\trepair_cycle\tstable_assignment_unit\t"
        "partition_or_exclusion_status\taccession"
    ]
    for accession, sequence, group, partition in rows:
        digest = hashlib.sha256(sequence.encode()).hexdigest()
        public_lines.append(
            f"{accession}\t{partition}\t{digest}\t{len(sequence)}\t{group}"
        )
        unit = accession if strategy == "random" else group
        local_lines.append(
            f"{strategy}\t{stage}\t0\t{unit}\t{partition}\t{accession}"
        )
    public.write_text("\n".join(public_lines) + "\n", encoding="utf-8")
    local.write_text("\n".join(local_lines) + "\n", encoding="utf-8")
    return public, local


def test_manifest_join_materializes_exact_six_fastas(tmp_path: Path) -> None:
    specs = [
        ("A1", "A" * 32, "UniRef50_A", "training"),
        ("A2", "C" * 33, "UniRef50_B", "validation"),
        ("A3", "D" * 34, "UniRef50_C", "test"),
    ]
    random_public, random_local = _write_manifest_pair(
        tmp_path,
        "random",
        "random",
        "diagnostic",
        specs,
    )
    candidate_public, candidate_local = _write_manifest_pair(
        tmp_path,
        "candidate",
        "group_aware",
        "pre_repair",
        specs,
    )
    random_manifest = load_strategy_manifest(
        public_path=random_public,
        local_path=random_local,
        strategy="random",
        stage="diagnostic",
        expected_public_sha256=hashlib.sha256(random_public.read_bytes()).hexdigest(),
        expected_local_sha256=hashlib.sha256(random_local.read_bytes()).hexdigest(),
    )
    candidate_manifest = load_strategy_manifest(
        public_path=candidate_public,
        local_path=candidate_local,
        strategy="group_aware",
        stage="pre_repair",
        expected_public_sha256=hashlib.sha256(candidate_public.read_bytes()).hexdigest(),
        expected_local_sha256=hashlib.sha256(candidate_local.read_bytes()).hexdigest(),
    )
    catalog = tmp_path / "catalog.tsv"
    catalog.write_text(
        "\t".join(CATALOG_COLUMNS)
        + "\n"
        + "\n".join(
            _catalog_row(accession, sequence, group)
            for accession, sequence, group, _ in specs
        )
        + "\n",
        encoding="utf-8",
    )
    policy = replace(
        load_similarity_audit_policy(POLICY_PATH),
        task4_catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
        task4_catalog_byte_size=catalog.stat().st_size,
        task4_catalog_row_count=len(specs),
        expected_eligible_records=len(specs),
        expected_eligible_residues=sum(
            len(sequence) for _, sequence, _, _ in specs
        ),
    )
    output = tmp_path / "fastas"
    materialized = materialize_strategy_fastas(
        catalog_path=catalog,
        manifests={"random": random_manifest, "group_aware": candidate_manifest},
        output_directory=output,
        policy=policy,
    )
    assert set(materialized.fastas) == {"random", "group_aware"}
    assert len(list(output.glob("*.fasta"))) == 6
    assert list(iter_one_line_fasta(output / "random_validation.fasta")) == [
        ("A2", "C" * 33)
    ]


def test_manifest_reports_unique_hash_and_group_crossings(tmp_path: Path) -> None:
    specs = [
        ("A1", "A" * 32, "UniRef50_SHARED", "training"),
        ("A2", "A" * 32, "UniRef50_OTHER", "validation"),
        ("A3", "C" * 33, "UniRef50_SHARED", "test"),
    ]
    public, local = _write_manifest_pair(
        tmp_path,
        "crossings",
        "random",
        "diagnostic",
        specs,
    )
    manifest = load_strategy_manifest(
        public_path=public,
        local_path=local,
        strategy="random",
        stage="diagnostic",
        expected_public_sha256=hashlib.sha256(public.read_bytes()).hexdigest(),
        expected_local_sha256=hashlib.sha256(local.read_bytes()).hexdigest(),
    )
    assert manifest.structural_audit.exact_sequence_hash_crossings == 1
    assert manifest.structural_audit.uniref50_group_crossings == 1
    assert manifest.structural_audit.largest_uniref50_group_records == 2


def test_manifest_order_error_precedes_local_assignment_error(tmp_path: Path) -> None:
    specs = [
        ("A2", "A" * 32, "UniRef50_A", "training"),
        ("A1", "C" * 33, "UniRef50_B", "validation"),
    ]
    public, local = _write_manifest_pair(
        tmp_path,
        "error_order",
        "random",
        "diagnostic",
        specs,
    )
    local_lines = local.read_text(encoding="utf-8").splitlines()
    local_lines[2] = local_lines[2].replace("\tA1", "\tBROKEN")
    local.write_text("\n".join(local_lines) + "\n", encoding="utf-8")

    with pytest.raises(SimilarityAuditError, match="unique and sorted"):
        load_strategy_manifest(
            public_path=public,
            local_path=local,
            strategy="random",
            stage="diagnostic",
            expected_public_sha256=hashlib.sha256(public.read_bytes()).hexdigest(),
            expected_local_sha256=hashlib.sha256(local.read_bytes()).hexdigest(),
        )
