"""Preparation-record and deterministic-regeneration evidence checks."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from protein_lm.data.model_data.config import config_sha256
from protein_lm.data.model_data.contracts import ModelDataConfig, ModelDataError
from protein_lm.data.model_data.manifests import ArtifactEvidence
from protein_lm.data.model_data.workflow import _git_revision, create_candidate

_PREPARATION_RECORD_KEYS = {
    "schema_version",
    "contract_identifier",
    "candidate_identifier",
    "config_sha256",
    "source_release",
    "proteingym_release",
    "task4_catalog_sha256",
    "reserved_families_sha256",
    "task4_report_sha256",
    "task4_eligibility_policy_sha256",
    "base_seed",
    "allocation_namespaces",
    "code_revision",
    "population",
    "collection_aggregates",
    "artifact_evidence",
    "network_requests_made",
}


def preparation_record_matches(
    *,
    root: Path,
    directory: Path,
    config: ModelDataConfig,
    evidence: dict[str, ArtifactEvidence],
    population: dict[str, int],
    collection_aggregates: dict[str, dict[str, object]],
    allow_synthetic_revision: bool,
) -> bool:
    """Require provenance to equal values recomputed from current frozen inputs."""

    try:
        record = json.loads(
            (directory / "preparation_record.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(record, dict) or set(record) != _PREPARATION_RECORD_KEYS:
        return False
    expected = {
        "schema_version": 1,
        "contract_identifier": "2026-08-12-week-02-model-data-candidate-v1",
        "candidate_identifier": config.candidate_identifier,
        "config_sha256": config_sha256(
            root / "experiments/week_02/model_data_readiness.toml"
        ),
        "source_release": config.source_release,
        "proteingym_release": config.proteingym_release,
        "task4_catalog_sha256": config.task4_catalog_sha256,
        "reserved_families_sha256": config.reserved_families_sha256,
        "task4_report_sha256": config.task4_report_sha256,
        "task4_eligibility_policy_sha256": config.task4_eligibility_policy_sha256,
        "base_seed": config.base_seed,
        "allocation_namespaces": list(config.allocation_namespaces),
        "population": population,
        "collection_aggregates": collection_aggregates,
        "artifact_evidence": {
            name: {
                "row_count": item.row_count,
                "byte_size": item.byte_size,
                "sha256": item.sha256,
            }
            for name, item in evidence.items()
        },
        "network_requests_made": 0,
    }
    if any(record.get(name) != value for name, value in expected.items()):
        return False
    if allow_synthetic_revision:
        return record.get("code_revision") == "synthetic-test-only"
    try:
        return record.get("code_revision") == _git_revision(root)
    except ModelDataError:
        return False


def regeneration_is_identical(
    root: Path, candidate_directory: Path, config: ModelDataConfig
) -> bool:
    """Rebuild to a disposable sibling directory and compare all immutable TSVs."""

    try:
        with tempfile.TemporaryDirectory(
            prefix=".week2-validate-", dir=candidate_directory.parent
        ) as temporary:
            rebuilt = create_candidate(
                root=root,
                config_path=root / "experiments/week_02/model_data_readiness.toml",
                config=config,
                require_clean_revision=False,
                destination=Path(temporary) / "rebuilt",
            )
            return all(
                (candidate_directory / filename).read_bytes()
                == (rebuilt / filename).read_bytes()
                for filename in (
                    "shared_validation.tsv",
                    "shared_sealed_test.tsv",
                    "random_arm.tsv",
                    "family_aware_arm.tsv",
                    "deduplication_aliases.tsv",
                )
            )
    except (ModelDataError, OSError):
        return False
