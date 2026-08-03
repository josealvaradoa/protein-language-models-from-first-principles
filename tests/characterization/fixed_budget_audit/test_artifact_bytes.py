"""Hand-reviewed byte goldens for the synthetic A-004 evidence graph."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import protein_lm.data.task7_a004_workflow as workflow_module
from a004_workflow_test_support import SOURCE_CONFIG, install_synthetic_workflow
from protein_lm.data.similarity_audit_models import FileEvidence, SequenceMetadata
from protein_lm.data.similarity_audit_policy import load_similarity_audit_policy
from protein_lm.data.similarity_manifests import (
    PartitionPopulation,
    StrategyManifest,
    StructuralMembershipAudit,
)
from protein_lm.data.task7_inputs import ensure_materialized_inputs

from .golden_support import GoldenAudit, fasta_evidence, identity, json_bytes

GOLDEN_REPORT = Path(__file__).parent / "goldens/a004_report.md"
FINGERPRINT = "synthetic-materialization-fingerprint"


def test_full_synthetic_workflow_matches_independent_byte_goldens(
    monkeypatch, tmp_path: Path
) -> None:
    synthetic = install_synthetic_workflow(monkeypatch, tmp_path, changed_search=True)

    workflow_module.run_a004_fixed_budget_audit(
        project_root=synthetic.project_root,
        config_path=synthetic.config_path,
        search_runner=synthetic.search_runner,
        database_runner=synthetic.database_runner,
        hardware=synthetic.hardware,
    )

    golden = GoldenAudit(
        synthetic.project_root,
        synthetic.source_policy_path.read_bytes(),
    )
    golden.add_publication(GOLDEN_REPORT.read_bytes())
    for relative_path, expected in golden.artifacts.items():
        actual_path = synthetic.project_root / relative_path
        assert actual_path.read_bytes() == expected, relative_path


def test_materialized_input_marker_matches_independent_byte_golden(
    tmp_path: Path,
) -> None:
    catalog = _catalog_bytes()
    catalog_path = tmp_path / "catalog.tsv"
    catalog_path.write_bytes(catalog)
    workspace = tmp_path / "workspace"
    policy = replace(
        load_similarity_audit_policy(SOURCE_CONFIG),
        task4_catalog_sha256=hashlib.sha256(catalog).hexdigest(),
        task4_catalog_byte_size=len(catalog),
        task4_catalog_row_count=3,
        expected_eligible_records=3,
        expected_eligible_residues=12,
    )

    ensure_materialized_inputs(
        workspace=workspace,
        catalog_path=catalog_path,
        manifests=_manifests(),
        policy=policy,
        fingerprint=FINGERPRINT,
    )

    expected_fastas = {
        "random": {
            "training": b">A\nAAAA\n",
            "validation": b">B\nCCCC\n",
            "test": b">C\nDDDD\n",
        },
        "group_aware": {
            "training": b">B\nCCCC\n",
            "validation": b">C\nDDDD\n",
            "test": b">A\nAAAA\n",
        },
    }
    expected_marker = json_bytes(
        {
            "schema_version": 1,
            "stage": "materialized_inputs",
            "fingerprint": FINGERPRINT,
            "catalog": {"row_count": 3, **identity(catalog)},
            "fastas": {
                strategy: {
                    partition: fasta_evidence(content)
                    for partition, content in partitions.items()
                }
                for strategy, partitions in expected_fastas.items()
            },
        }
    )
    for strategy, partitions in expected_fastas.items():
        for partition, expected in partitions.items():
            path = workspace / "fastas" / f"{strategy}_{partition}.fasta"
            assert path.read_bytes() == expected
    assert (workspace / "fastas/complete.json").read_bytes() == expected_marker


def _catalog_bytes() -> bytes:
    header = (
        "primary_accession\tsequence\tsequence_sha256\tbiological_length\t"
        "noncanonical_residue\tfragment\tbelow_min_length\tabove_max_length\t"
        "blank_uniref50_mapping\teligible\tprimary_exclusion_reason\t"
        "uniref50_group\tproteingym_candidate_test_reserved\n"
    )
    rows = []
    for accession, sequence, group in (
        ("A", "AAAA", "UniRef50_A"),
        ("B", "CCCC", "UniRef50_B"),
        ("C", "DDDD", "UniRef50_C"),
    ):
        rows.append(
            "\t".join(
                (
                    accession,
                    sequence,
                    hashlib.sha256(sequence.encode()).hexdigest(),
                    "4",
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
            + "\n"
        )
    return (header + "".join(rows)).encode()


def _manifests() -> dict[str, StrategyManifest]:
    sequences = {"A": "AAAA", "B": "CCCC", "C": "DDDD"}
    groups = {accession: f"UniRef50_{accession}" for accession in sequences}
    assignments = {
        "random": {"A": "training", "B": "validation", "C": "test"},
        "group_aware": {"B": "training", "C": "validation", "A": "test"},
    }
    manifests = {}
    for strategy, partitions in assignments.items():
        records = {
            accession: SequenceMetadata(
                sequence_sha256=hashlib.sha256(sequences[accession].encode()).hexdigest(),
                biological_length=4,
                uniref50_group=groups[accession],
                partition=partition,
            )
            for accession, partition in partitions.items()
        }
        manifests[strategy] = StrategyManifest(
            strategy=strategy,
            stage="synthetic",
            records=records,
            partitions={
                partition: PartitionPopulation(1, 4, 1)
                for partition in ("training", "validation", "test")
            },
            structural_audit=StructuralMembershipAudit(0, 0, 3, 12, 0, 0, 1, 4),
            public_manifest=FileEvidence(3, 1, "1" * 64),
            local_assignment=FileEvidence(3, 1, "2" * 64),
        )
    return manifests
