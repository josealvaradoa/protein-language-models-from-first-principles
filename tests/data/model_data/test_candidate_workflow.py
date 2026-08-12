"""Synthetic-only checks for the Week 2 model-data candidate contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from protein_lm.data.eligibility import CATALOG_COLUMNS
from protein_lm.data.model_data.allocation import (
    assignment_digest,
    ordered_units,
    select_closest_to_target,
)
from protein_lm.data.model_data.catalog import load_catalog, load_reserved_families
from protein_lm.data.model_data.config import _validate, load_config
from protein_lm.data.model_data.contracts import (
    CandidateRecord,
    LengthBucket,
    ModelDataConfig,
    ModelDataError,
)
from protein_lm.data.model_data.deduplication import prepare_population
from protein_lm.data.model_data.reporting import write_readiness_evidence
from protein_lm.data.model_data.validation import validate_candidate
from protein_lm.data.model_data.workflow import create_candidate, preflight


def _row(accession: str, sequence: str, group: str, *, reserved: bool = False) -> str:
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
            str(reserved).lower(),
        )
    )


def _fixture(
    tmp_path: Path, *, include_reserved_hash_crossing: bool = False
) -> tuple[Path, ModelDataConfig]:
    root = tmp_path / "project"
    catalog = root / "data/processed/week_01/task_04_record_catalog.tsv"
    reserved = (
        root / "data/processed/week_01/task_04_candidate_test_reserved_families.txt"
    )
    config_path = root / "experiments/week_02/model_data_readiness.toml"
    report = root / "reports/week_01/task_04_eligible_records.json"
    policy = root / "experiments/week_01/eligibility.toml"
    catalog.parent.mkdir(parents=True)
    reserved.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True)
    policy.parent.mkdir(parents=True)
    report.parent.mkdir(parents=True)
    rows = [
        _row("P00005", "A" * 32, "UniRef50_G1"),
        _row("P00001", "A" * 32, "UniRef50_G1"),
        _row("P00002", "C" * 33, "UniRef50_G2"),
        _row("P00003", "D" * 34, "UniRef50_G3"),
        _row("P00004", "E" * 35, "UniRef50_G4"),
        _row("P00006", "G" * 36, "UniRef50_G5"),
        _row("P00007", "H" * 37, "UniRef50_G6"),
        _row("P99999", "F" * 32, "UniRef50_RESERVED", reserved=True),
    ]
    if include_reserved_hash_crossing:
        rows.append(_row("P00008", "F" * 32, "UniRef50_G7"))
    catalog.write_text(
        "\t".join(CATALOG_COLUMNS) + "\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )
    reserved.write_text("UniRef50_RESERVED\n", encoding="utf-8")
    config_path.write_text("synthetic config identity\n", encoding="utf-8")
    report.write_text('{"synthetic": true}\n', encoding="utf-8")
    policy.write_text("synthetic = true\n", encoding="utf-8")
    config = ModelDataConfig(
        schema_version=1,
        scope="week_02_model_data_candidate",
        candidate_identifier="v1",
        source_release="2026_02",
        proteingym_release="v1.3",
        license_spdx="CC-BY-4.0",
        task4_catalog_relative_path="data/processed/week_01/task_04_record_catalog.tsv",
        task4_catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
        task4_catalog_byte_size=catalog.stat().st_size,
        task4_catalog_row_count=len(rows),
        reserved_families_relative_path="data/processed/week_01/task_04_candidate_test_reserved_families.txt",
        reserved_families_sha256=hashlib.sha256(reserved.read_bytes()).hexdigest(),
        reserved_family_count=1,
        task4_report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        task4_report_relative_path="reports/week_01/task_04_eligible_records.json",
        task4_eligibility_policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
        task4_eligibility_policy_relative_path="experiments/week_01/eligibility.toml",
        canonical_amino_acids="ACDEFGHIKLMNPQRSTVWY",
        minimum_length=32,
        maximum_length=2046,
        sequence_hash="sha256",
        prediction_token_target=67,
        minimum_evaluation_predictions=1,
        maximum_evaluation_predictions=1000,
        minimum_bucket_predictions=1,
        base_seed=20260812,
        hash_algorithm="sha256",
        allocation_namespaces=(
            "week2/shared-validation/v1",
            "week2/shared-sealed-test/v1",
            "week2/random-native-validation/v1",
            "week2/family-native-validation/v1",
            "week2/training-stream/random/v1",
            "week2/training-stream/family-aware/v1",
            "week2/sampling/random/v1",
            "week2/sampling/family-aware/v1",
        ),
        length_buckets=(LengthBucket("32-2046", 32, 2046),),
        candidate_directory_relative_path="data/processed/week_02/model_data_candidates/v1",
        readiness_json_relative_path="reports/week_02/model_data_readiness_v1.json",
        readiness_markdown_relative_path="reports/week_02/model_data_readiness_v1.md",
        readiness_sha256_relative_path="reports/week_02/model_data_readiness_v1.sha256",
        mmseqs2_status="diagnostic_only",
        model_use="candidate_pending_readiness",
    )
    return root, config


def test_frozen_config_is_strict_and_covers_pins(tmp_path: Path) -> None:
    project_root = Path(__file__).parents[3]
    path = project_root / "experiments/week_02/model_data_readiness.toml"
    assert load_config(path).candidate_identifier == "v1"
    drifted = tmp_path / "drifted.toml"
    drifted.write_bytes(path.read_bytes() + b"unknown = 1\n")
    with pytest.raises(ModelDataError, match="bytes do not match"):
        load_config(drifted)


def test_hash_ordering_is_domain_separated_and_selection_is_closest(
    tmp_path: Path,
) -> None:
    shared = assignment_digest("week2/shared-validation/v1", 20260812, "UniRef50_G1")
    sealed = assignment_digest("week2/shared-sealed-test/v1", 20260812, "UniRef50_G1")
    assert (
        shared.hex()
        == "cf235d55784e6d98d26062fe5bdeded25c75ddab95b17e71662315e5057b6df2"
    )
    assert shared != sealed
    _, config = _fixture(tmp_path)
    record = CandidateRecord("P1", "a" * 64, 32, "bucket", "UniRef50_G1")
    units = ordered_units(
        (record,),
        namespace=config.allocation_namespaces[0],
        config=config,
        grouped_by_family=True,
    )
    assert select_closest_to_target(
        units, replace(config, prediction_token_target=33)
    ) == (record,)
    assert (
        select_closest_to_target(units, replace(config, prediction_token_target=10))
        == ()
    )


def test_synthetic_candidate_is_atomic_deduplicated_and_validated(
    tmp_path: Path,
) -> None:
    root, config = _fixture(tmp_path)
    reserved = load_reserved_families(
        root / config.reserved_families_relative_path, config
    )
    catalog = load_catalog(root / config.task4_catalog_relative_path, config, reserved)
    population, aliases, excluded = prepare_population(catalog, reserved, config)
    assert excluded == 1
    assert [record.primary_accession for record in population] == [
        "P00001",
        "P00002",
        "P00003",
        "P00004",
        "P00006",
        "P00007",
    ]
    assert aliases[0].alias_accession == "P00005"
    destination = root / config.candidate_directory_relative_path
    create_candidate(
        root=root,
        config_path=root / "experiments/week_02/model_data_readiness.toml",
        config=config,
        require_clean_revision=False,
        destination=destination,
    )
    assert destination.is_dir()
    assert "sequence\t" not in (destination / "shared_validation.tsv").read_text(
        encoding="utf-8"
    )
    with pytest.raises(ModelDataError, match="already exists"):
        create_candidate(
            root=root,
            config_path=root / "experiments/week_02/model_data_readiness.toml",
            config=config,
            require_clean_revision=False,
            destination=destination,
        )
    result = validate_candidate(
        root=root,
        candidate_directory=destination,
        config=config,
        allow_synthetic_revision=True,
    )
    assert result.gates["deterministic_regeneration"]
    assert result.gates["complete_per_arm_population_accounting"]
    assert result.status == "passed", result.gates
    assert result.collection_aggregates["shared_validation"]["prediction_tokens"] > 0
    preparation = json.loads(
        (destination / "preparation_record.json").read_text(encoding="utf-8")
    )
    assert preparation["collection_aggregates"] == result.collection_aggregates
    evidence = (
        root / config.readiness_json_relative_path,
        root / config.readiness_markdown_relative_path,
        root / config.readiness_sha256_relative_path,
    )
    write_readiness_evidence(evidence, result)
    public_text = "\n".join(path.read_text(encoding="utf-8") for path in evidence[:2])
    assert "P99999" not in public_text
    assert "UniRef50_RESERVED" not in public_text
    with pytest.raises(ModelDataError, match="already exists"):
        write_readiness_evidence(evidence, result)
    random_arm = destination / "random_arm.tsv"
    random_arm.write_bytes(random_arm.read_bytes() + b"tampered\n")
    with pytest.raises(ModelDataError):
        validate_candidate(
            root=root,
            candidate_directory=destination,
            config=config,
            allow_synthetic_revision=True,
        )


def test_failed_candidate_evidence_is_preserved_after_provenance_tampering(
    tmp_path: Path,
) -> None:
    root, config = _fixture(tmp_path)
    destination = root / config.candidate_directory_relative_path
    create_candidate(
        root=root,
        config_path=root / "experiments/week_02/model_data_readiness.toml",
        config=config,
        require_clean_revision=False,
        destination=destination,
    )
    record_path = destination / "preparation_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["artifact_evidence"]["random_arm.tsv"]["row_count"] += 1
    record_path.write_text(json.dumps(record), encoding="utf-8")
    result = validate_candidate(
        root=root,
        candidate_directory=destination,
        config=config,
        allow_synthetic_revision=True,
    )
    assert result.status == "failed"
    assert not result.gates["artifact_and_provenance_checksums"]
    evidence = (
        root / config.readiness_json_relative_path,
        root / config.readiness_markdown_relative_path,
        root / config.readiness_sha256_relative_path,
    )
    write_readiness_evidence(evidence, result)
    payload = json.loads(evidence[0].read_text(encoding="utf-8"))
    assert payload["candidate_status"] == "failed"
    assert payload["collection_aggregates"] == result.collection_aggregates


@pytest.mark.parametrize(
    "path",
    (
        ("population", "collapsed_aliases"),
        ("collection_aggregates", "shared_validation", "prediction_tokens"),
    ),
)
def test_recomputed_provenance_rejects_population_and_aggregate_tampering(
    tmp_path: Path, path: tuple[str, ...]
) -> None:
    root, config = _fixture(tmp_path)
    destination = root / config.candidate_directory_relative_path
    create_candidate(
        root=root,
        config_path=root / "experiments/week_02/model_data_readiness.toml",
        config=config,
        require_clean_revision=False,
        destination=destination,
    )
    record_path = destination / "preparation_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    target = record
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] += 1
    record_path.write_text(json.dumps(record), encoding="utf-8")
    result = validate_candidate(
        root=root,
        candidate_directory=destination,
        config=config,
        allow_synthetic_revision=True,
    )
    assert not result.gates["artifact_and_provenance_checksums"]


def test_reserved_sequence_hash_crossing_fails_reserved_identity_gate(
    tmp_path: Path,
) -> None:
    root, config = _fixture(tmp_path, include_reserved_hash_crossing=True)
    destination = root / config.candidate_directory_relative_path
    create_candidate(
        root=root,
        config_path=root / "experiments/week_02/model_data_readiness.toml",
        config=config,
        require_clean_revision=False,
        destination=destination,
    )
    result = validate_candidate(
        root=root,
        candidate_directory=destination,
        config=config,
        allow_synthetic_revision=True,
    )
    assert result.status == "failed"
    assert not result.gates["reserved_members_are_absent"]


def test_preflight_never_creates_candidate_paths(tmp_path: Path) -> None:
    root, config = _fixture(tmp_path)
    assert preflight(config, root)["network_requests_made"] == 0
    assert not (root / config.candidate_directory_relative_path).exists()


@pytest.mark.parametrize(
    "relative_path",
    (
        "data/processed/week_01/task_04_record_catalog.tsv",
        "data/processed/week_01/task_04_candidate_test_reserved_families.txt",
        "reports/week_01/task_04_eligible_records.json",
        "experiments/week_01/eligibility.toml",
    ),
)
def test_preflight_detects_each_pinned_input_drift_without_output(
    tmp_path: Path, relative_path: str
) -> None:
    root, config = _fixture(tmp_path)
    target = root / relative_path
    target.write_bytes(target.read_bytes() + b"drift\n")
    with pytest.raises(ModelDataError):
        preflight(config, root)
    assert not (root / config.candidate_directory_relative_path).exists()


def test_config_rejects_a_length_bucket_gap(tmp_path: Path) -> None:
    _, config = _fixture(tmp_path)
    gapped = replace(
        config,
        length_buckets=(
            LengthBucket("32-127", 32, 127),
            LengthBucket("129-2046", 129, 2046),
        ),
    )
    with pytest.raises(ModelDataError, match="contiguous"):
        _validate(gapped)
