"""Synthetic filesystem builders for Week 2 promotion and loader tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from protein_lm.data.eligibility import CATALOG_COLUMNS
from protein_lm.data.model_data.contracts import LengthBucket, ModelDataConfig
from protein_lm.data.model_data.manifests import MEMBERSHIP_HEADER
from protein_lm.data.model_data.promotion import (
    PromotionContract,
    PromotedArtifact,
    promote,
    registry_payload,
)


def build_synthetic_environment(
    tmp_path: Path,
) -> tuple[Path, ModelDataConfig, PromotionContract]:
    """Create one complete synthetic candidate, readiness report, and Task 4 catalog."""

    root = tmp_path / "project"
    catalog_path = root / "data/processed/week_01/task_04_record_catalog.tsv"
    reserved_path = (
        root / "data/processed/week_01/task_04_candidate_test_reserved_families.txt"
    )
    report_path = root / "reports/week_01/task_04_eligible_records.json"
    policy_path = root / "experiments/week_01/eligibility.toml"
    config_path = root / "experiments/week_02/model_data_readiness.toml"
    for path in (catalog_path, reserved_path, report_path, policy_path, config_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    sequences = {
        "P00001": ("A" * 32, "UniRef50_G1"),
        "P00002": ("C" * 33, "UniRef50_G2"),
        "P00003": ("D" * 34, "UniRef50_G3"),
    }
    catalog_path.write_text(
        "\t".join(CATALOG_COLUMNS)
        + "\n"
        + "\n".join(
            catalog_row(accession, sequence, group)
            for accession, (sequence, group) in sequences.items()
        )
        + "\n",
        encoding="utf-8",
    )
    reserved_path.write_text("UniRef50_RESERVED\n", encoding="utf-8")
    report_path.write_text('{"synthetic": true}\n', encoding="utf-8")
    policy_path.write_text("synthetic = true\n", encoding="utf-8")
    config_path.write_text("synthetic config\n", encoding="utf-8")
    config = synthetic_config(root, catalog_path, reserved_path, report_path, policy_path)
    artifacts = {
        "shared_validation.tsv": membership_bytes(
            [("P00002", "shared_validation", "C" * 33, "UniRef50_G2")]
        ),
        "random_arm.tsv": membership_bytes(
            [
                ("P00001", "training", "A" * 32, "UniRef50_G1"),
                ("P00003", "native_validation", "D" * 34, "UniRef50_G3"),
            ]
        ),
        "family_aware_arm.tsv": membership_bytes(
            [
                ("P00001", "training", "A" * 32, "UniRef50_G1"),
                ("P00003", "native_validation", "D" * 34, "UniRef50_G3"),
            ]
        ),
    }
    contract = synthetic_contract(artifacts)
    write_readiness(root, config, contract)
    write_candidate(root, config, contract, artifacts, config_path)
    return root, config, contract


def promote_synthetic(
    root: Path,
    config: ModelDataConfig,
    contract: PromotionContract,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Promote a synthetic candidate without requiring a synthetic Git repository."""

    monkeypatch.setattr(
        "protein_lm.data.model_data.promotion._require_clean_committed_revision",
        lambda _: None,
    )
    return promote(root, config, contract=contract)


def replace_public_artifact(
    root: Path,
    config: ModelDataConfig,
    contract: PromotionContract,
    filename: str,
    content: bytes,
) -> PromotionContract:
    """Replace one public artifact and its injected synthetic identity together."""

    replacement = artifact_evidence(filename, content)
    changed = replace(
        contract,
        artifacts=tuple(
            replacement if item.filename == filename else item
            for item in contract.artifacts
        ),
    )
    public = root / "manifests/week_02"
    (public / filename).write_bytes(content)
    (public / "model_data_v1.json").write_text(
        json.dumps(registry_payload(config, changed), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return changed


def membership_bytes(rows: list[tuple[str, str, str, str]]) -> bytes:
    """Build a deterministic six-column synthetic membership manifest."""

    content = ["\t".join(MEMBERSHIP_HEADER)]
    for accession, partition, sequence, group in rows:
        content.append(
            "\t".join(
                (
                    accession,
                    partition,
                    hashlib.sha256(sequence.encode("ascii")).hexdigest(),
                    str(len(sequence)),
                    "32-2046",
                    group,
                )
            )
        )
    return ("\n".join(content) + "\n").encode("utf-8")


def artifact_evidence(filename: str, content: bytes) -> PromotedArtifact:
    """Return the exact identity for one synthetic public artifact."""

    return PromotedArtifact(
        filename,
        len(content.splitlines()) - 1,
        len(content),
        hashlib.sha256(content).hexdigest(),
    )


def catalog_row(accession: str, sequence: str, group: str) -> str:
    """Build one eligible Task 4 catalog row."""

    return "\t".join(
        (
            accession,
            sequence,
            hashlib.sha256(sequence.encode("ascii")).hexdigest(),
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


def synthetic_config(
    root: Path,
    catalog_path: Path,
    reserved_path: Path,
    report_path: Path,
    policy_path: Path,
) -> ModelDataConfig:
    """Return a pinned config for the synthetic Task 4 files."""

    return ModelDataConfig(
        schema_version=1,
        scope="week_02_model_data_candidate",
        candidate_identifier="v1",
        source_release="2026_02",
        proteingym_release="v1.3",
        license_spdx="CC-BY-4.0",
        task4_catalog_relative_path=str(catalog_path.relative_to(root)),
        task4_catalog_sha256=hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        task4_catalog_byte_size=catalog_path.stat().st_size,
        task4_catalog_row_count=3,
        reserved_families_relative_path=str(reserved_path.relative_to(root)),
        reserved_families_sha256=hashlib.sha256(reserved_path.read_bytes()).hexdigest(),
        reserved_family_count=1,
        task4_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        task4_report_relative_path=str(report_path.relative_to(root)),
        task4_eligibility_policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        task4_eligibility_policy_relative_path=str(policy_path.relative_to(root)),
        canonical_amino_acids="ACDEFGHIKLMNPQRSTVWY",
        minimum_length=32,
        maximum_length=2046,
        sequence_hash="sha256",
        prediction_token_target=1,
        minimum_evaluation_predictions=1,
        maximum_evaluation_predictions=1,
        minimum_bucket_predictions=1,
        base_seed=20260812,
        hash_algorithm="sha256",
        allocation_namespaces=(),
        length_buckets=(LengthBucket("32-2046", 32, 2046),),
        candidate_directory_relative_path="data/processed/week_02/model_data_candidates/v1",
        readiness_json_relative_path="reports/week_02/model_data_readiness_v1.json",
        readiness_markdown_relative_path="reports/week_02/model_data_readiness_v1.md",
        readiness_sha256_relative_path="reports/week_02/model_data_readiness_v1.sha256",
        mmseqs2_status="diagnostic_only",
        model_use="candidate_pending_readiness",
    )


def synthetic_contract(artifacts: dict[str, bytes]) -> PromotionContract:
    """Create the promotion contract and readiness identity for synthetic artifacts."""

    readiness_content = json.dumps(
        {
            "schema_version": 1,
            "scope": "week_02_model_data_readiness",
            "candidate_status": "passed",
            "hard_gates": {
                "artifact_and_provenance_checksums": True,
                "complete_per_arm_population_accounting": True,
                "deterministic_regeneration": True,
                "evaluation_bucket_bounds": True,
                "evaluation_token_bounds": True,
                "family_aware_groups_do_not_cross": True,
                "no_accession_or_hash_crossings_within_arm": True,
                "reserved_members_are_absent": True,
                "shared_and_sealed_groups_are_isolated": True,
                "source_sequence_contract": True,
            },
        },
        sort_keys=True,
    ).encode("utf-8")
    return PromotionContract(
        identifier="synthetic-promotion-v1",
        candidate_revision="synthetic-revision",
        readiness_json_sha256=hashlib.sha256(readiness_content).hexdigest(),
        artifacts=tuple(artifact_evidence(name, content) for name, content in artifacts.items()),
        sealed_rows=1,
        sealed_bytes=7,
        sealed_sha256="b" * 64,
    )


def write_readiness(
    root: Path, config: ModelDataConfig, contract: PromotionContract
) -> None:
    """Write matching synthetic aggregate-only readiness evidence."""

    content = json.dumps(
        {
            "schema_version": 1,
            "scope": "week_02_model_data_readiness",
            "candidate_status": "passed",
            "hard_gates": {
                "artifact_and_provenance_checksums": True,
                "complete_per_arm_population_accounting": True,
                "deterministic_regeneration": True,
                "evaluation_bucket_bounds": True,
                "evaluation_token_bounds": True,
                "family_aware_groups_do_not_cross": True,
                "no_accession_or_hash_crossings_within_arm": True,
                "reserved_members_are_absent": True,
                "shared_and_sealed_groups_are_isolated": True,
                "source_sequence_contract": True,
            },
        },
        sort_keys=True,
    ).encode("utf-8")
    path = root / config.readiness_json_relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    (root / config.readiness_sha256_relative_path).write_text(
        f"{contract.readiness_json_sha256}  {path.name}\n", encoding="utf-8"
    )


def write_candidate(
    root: Path,
    config: ModelDataConfig,
    contract: PromotionContract,
    artifacts: dict[str, bytes],
    config_path: Path,
) -> None:
    """Write the synthetic candidate inventory without ever using real membership."""

    candidate = root / config.candidate_directory_relative_path
    candidate.mkdir(parents=True)
    for filename, content in artifacts.items():
        (candidate / filename).write_bytes(content)
    (candidate / "shared_sealed_test.tsv").write_bytes(b"ignored\n")
    (candidate / "deduplication_aliases.tsv").write_text("header\n", encoding="utf-8")
    checksums = {
        item.filename: {
            "row_count": item.row_count,
            "byte_size": item.byte_size,
            "sha256": item.sha256,
        }
        for item in contract.artifacts
    }
    checksums["shared_sealed_test.tsv"] = {
        "row_count": contract.sealed_rows,
        "byte_size": contract.sealed_bytes,
        "sha256": contract.sealed_sha256,
    }
    checksums["deduplication_aliases.tsv"] = {
        "row_count": 0,
        "byte_size": 7,
        "sha256": hashlib.sha256(b"header\n").hexdigest(),
    }
    (candidate / "artifact_checksums.json").write_text(
        json.dumps(checksums, sort_keys=True), encoding="utf-8"
    )
    (candidate / "preparation_record.json").write_text(
        json.dumps(
            {
                "code_revision": contract.candidate_revision,
                "candidate_identifier": config.candidate_identifier,
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "artifact_evidence": checksums,
            }
        ),
        encoding="utf-8",
    )
