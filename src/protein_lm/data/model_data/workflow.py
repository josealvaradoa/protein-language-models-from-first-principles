"""Operator-gated candidate creation and non-mutating preflight planning."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from protein_lm.data.model_data.aggregates import collection_aggregate
from protein_lm.data.model_data.allocation import allocate
from protein_lm.data.model_data.catalog import load_catalog, load_reserved_families
from protein_lm.data.model_data.config import config_sha256
from protein_lm.data.model_data.contracts import (
    CatalogRecord,
    ModelDataConfig,
    ModelDataError,
)
from protein_lm.data.model_data.deduplication import prepare_population
from protein_lm.data.model_data.manifests import (
    TSV_FILENAMES,
    write_candidate_tsvs,
    write_checksums,
)


def preflight(config: ModelDataConfig, root: Path) -> dict[str, object]:
    """Verify all pinned local inputs and describe the non-mutating operation."""

    verify_frozen_inputs(root, config)

    return {
        "candidate_identifier": config.candidate_identifier,
        "destination": str(root / config.candidate_directory_relative_path),
        "verified_inputs": (
            config.task4_catalog_relative_path,
            config.reserved_families_relative_path,
            config.task4_report_relative_path,
            config.task4_eligibility_policy_relative_path,
        ),
        "selection_order": (
            "shared_validation",
            "shared_sealed_test",
            "random_native_validation",
            "family_native_validation",
        ),
        "allocation_namespaces": config.allocation_namespaces,
        "expected_artifacts": (
            *TSV_FILENAMES,
            "artifact_checksums.json",
            "preparation_record.json",
        ),
        "mmseqs2": "will not run",
        "model_code": "will not run",
        "network_requests_made": 0,
    }


def create_candidate(
    *,
    root: Path,
    config_path: Path,
    config: ModelDataConfig,
    require_clean_revision: bool = True,
    destination: Path | None = None,
) -> Path:
    """Build a local candidate atomically after every frozen input is verified."""

    if require_clean_revision:
        _require_clean_committed_revision(root)
    code_revision = (
        _git_revision(root) if require_clean_revision else "synthetic-test-only"
    )
    final_directory = destination or root / config.candidate_directory_relative_path
    _require_under_root(final_directory, root)
    if final_directory.exists():
        raise ModelDataError("candidate destination already exists")
    if destination is None:
        _require_ignored(root, final_directory)
    catalog, reserved = verify_frozen_inputs(root, config)
    population, aliases, reserved_records = prepare_population(
        catalog, reserved, config
    )
    allocation = allocate(population, config)
    final_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{config.candidate_identifier}-", dir=final_directory.parent
    ) as temporary:
        stage = Path(temporary) / final_directory.name
        stage.mkdir()
        evidence = write_candidate_tsvs(stage, allocation, aliases)
        write_checksums(stage / "artifact_checksums.json", evidence)
        record = _preparation_record(
            config=config,
            config_path=config_path,
            root=root,
            catalog_records=len(catalog),
            reserved_records=reserved_records,
            population_records=len(population),
            aliases=len(aliases),
            aggregates=_allocation_aggregates(allocation, config),
            evidence=evidence,
            code_revision=code_revision,
        )
        (stage / "preparation_record.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(stage, final_directory)
    return final_directory


def _preparation_record(**values: object) -> dict[str, object]:
    config = values["config"]
    assert isinstance(config, ModelDataConfig)
    config_path = values["config_path"]
    root = values["root"]
    assert isinstance(config_path, Path) and isinstance(root, Path)
    evidence = values["evidence"]
    return {
        "schema_version": 1,
        "contract_identifier": "2026-08-12-week-02-model-data-candidate-v1",
        "candidate_identifier": config.candidate_identifier,
        "config_sha256": config_sha256(config_path),
        "source_release": config.source_release,
        "proteingym_release": config.proteingym_release,
        "task4_catalog_sha256": config.task4_catalog_sha256,
        "reserved_families_sha256": config.reserved_families_sha256,
        "task4_report_sha256": config.task4_report_sha256,
        "task4_eligibility_policy_sha256": config.task4_eligibility_policy_sha256,
        "base_seed": config.base_seed,
        "allocation_namespaces": list(config.allocation_namespaces),
        "code_revision": values["code_revision"],
        "population": {
            "eligible_catalog_records": values["catalog_records"],
            "protein_gym_reserved_records": values["reserved_records"],
            "deduplicated_unreserved_records": values["population_records"],
            "collapsed_aliases": values["aliases"],
        },
        "collection_aggregates": values["aggregates"],
        "artifact_evidence": {
            filename: {
                "row_count": item.row_count,
                "byte_size": item.byte_size,
                "sha256": item.sha256,
            }
            for filename, item in sorted(evidence.items())
        },
        "network_requests_made": 0,
    }


def verify_frozen_inputs(
    root: Path, config: ModelDataConfig
) -> tuple[list[CatalogRecord], frozenset[str]]:
    """Verify the four pinned Week 1 inputs before any candidate allocation."""

    _verify_sha256(
        root / config.task4_report_relative_path,
        config.task4_report_sha256,
        "Task 4 report",
    )
    _verify_sha256(
        root / config.task4_eligibility_policy_relative_path,
        config.task4_eligibility_policy_sha256,
        "Task 4 eligibility policy",
    )
    reserved = load_reserved_families(
        root / config.reserved_families_relative_path, config
    )
    catalog = load_catalog(root / config.task4_catalog_relative_path, config, reserved)
    return catalog, reserved


def _verify_sha256(path: Path, expected: str, description: str) -> None:
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ModelDataError(f"could not read {description}: {error}") from error
    if actual != expected:
        raise ModelDataError(f"{description} identity drifted")


def _allocation_aggregates(
    allocation: object, config: ModelDataConfig
) -> dict[str, object]:
    from protein_lm.data.model_data.allocation import CandidateAllocation

    assert isinstance(allocation, CandidateAllocation)
    return {
        "shared_validation": collection_aggregate(allocation.shared_validation, config),
        "shared_sealed_test": collection_aggregate(
            allocation.shared_sealed_test, config
        ),
        "random_training": collection_aggregate(allocation.random_training, config),
        "random_native_validation": collection_aggregate(
            allocation.random_native_validation, config
        ),
        "family_aware_training": collection_aggregate(
            allocation.family_training, config
        ),
        "family_aware_native_validation": collection_aggregate(
            allocation.family_native_validation, config
        ),
    }


def _require_clean_committed_revision(root: Path) -> None:
    if _git_output(root, "status", "--porcelain"):
        raise ModelDataError(
            "production candidate creation requires a clean Git revision"
        )
    _git_revision(root)


def _git_revision(root: Path) -> str:
    return _git_output(root, "rev-parse", "HEAD")


def _git_output(root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ("git", *arguments), cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    except subprocess.SubprocessError as error:
        raise ModelDataError(
            "could not establish the Git execution revision"
        ) from error


def _require_ignored(root: Path, path: Path) -> None:
    try:
        result = subprocess.run(
            ("git", "check-ignore", "--quiet", "--", str(path.relative_to(root))),
            cwd=root,
            check=False,
        )
    except (OSError, ValueError) as error:
        raise ModelDataError("could not prove candidate output is ignored") from error
    if result.returncode != 0:
        raise ModelDataError("candidate output path is not ignored")


def _require_under_root(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ModelDataError(
            "candidate destination must remain under the repository root"
        ) from error
