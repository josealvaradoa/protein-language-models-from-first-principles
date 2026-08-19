"""Aggregate payload assembly and fixed comparisons for the Week 2 public report."""

from __future__ import annotations

import math
import re
from pathlib import Path

from protein_lm.bigram.evaluation_results import hypothesis
from protein_lm.bigram.public_report_contract import PublicReportConfig, config_sha256
from protein_lm.data.model_data.contracts import ModelDataError


FORBIDDEN_MEMBERSHIP_KEYS = {
    "primary_accession",
    "sequence",
    "sequence_sha256",
    "uniref50_group",
}
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SOURCE_GATES = {
    "candidate_validation": True,
    "twelve_principal_records": True,
    "shared_validation_loaded_once": True,
    "sealed_test_never_loaded": True,
    "evaluation_only_no_retraining_or_selection": True,
    "no_network_requests": True,
}
_SOURCE_LOADS = {
    "random_native_validation": 1,
    "family_aware_native_validation": 1,
    "shared_validation": 1,
    "shared_sealed_test": 0,
}


def report_payload(
    *,
    config_path: Path,
    config: PublicReportConfig,
    source: dict[str, object],
    run: dict[str, object],
    publication_code_revision: str,
) -> dict[str, object]:
    """Copy validated aggregates and calculate the fixed comparison summaries."""

    records = source.get("records")
    provenance = source.get("provenance")
    if not isinstance(records, list) or not all(
        isinstance(item, dict) for item in records
    ):
        raise ModelDataError("source evaluation records are invalid")
    if not isinstance(provenance, dict) or source.get("hypothesis") != hypothesis(
        records
    ):
        raise ModelDataError("source evaluation provenance or hypothesis is invalid")
    _validate_passed_run(run)
    if not isinstance(publication_code_revision, str) or (
        _REVISION.fullmatch(publication_code_revision) is None
    ):
        raise ModelDataError("publication code revision is invalid")
    payload = {
        "schema_version": 1,
        "scope": "week_02_bigram_evaluation_public_report",
        "contract_identifier": config.contract_identifier,
        "status": "passed",
        "hard_gates": {
            "validated_source_evaluation": True,
            "exact_twelve_records": True,
            "aggregate_only": True,
            "no_network_requests": True,
            "sealed_test_inaccessible": True,
        },
        "publication_configuration_sha256": config_sha256(config_path),
        "publication_code_revision": publication_code_revision,
        "source": {
            "evaluation_id": config.source_evaluation_id,
            "relative_path": config.source_evaluation_relative_path,
            "evaluation_sha256": config.source_evaluation_sha256,
            "run_record_sha256": config.source_run_record_sha256,
            "registry_sha256": config.source_registry_sha256,
            "code_revision": config.source_evaluation_code_revision,
            "evaluation_configuration_sha256": config.source_evaluation_config_sha256,
        },
        "evaluation_provenance": provenance,
        "evaluation_runtime": {
            "runtime_seconds": run["runtime_seconds"],
            "collection_loads": run["collection_loads"],
        },
        "records": records,
        "hypothesis": hypothesis(records),
        "derived_comparisons": derived_comparisons(records),
        "network_requests_made": 0,
    }
    reject_forbidden_keys(payload)
    return payload


def derived_comparisons(records: list[dict[str, object]]) -> dict[str, object]:
    """Compute comparisons only from the approved twelve aggregate records."""

    def metric(arm: str, model: str, collection: str, name: str) -> float:
        record = next(
            item
            for item in records
            if item["model_arm"] == arm
            and item["model_type"] == model
            and item["collection"] == collection
        )
        metrics = record["metrics"]
        assert isinstance(metrics, dict)
        overall = metrics["overall"]
        assert isinstance(overall, dict) and type(overall[name]) is float
        return overall[name]

    pairs = (
        ("random_training", "random_native_validation"),
        ("random_training", "shared_validation"),
        ("family_aware_training", "family_aware_native_validation"),
        ("family_aware_training", "shared_validation"),
    )
    improvements = [
        {
            "model_arm": arm,
            "collection": collection,
            "unigram_minus_count_cross_entropy": metric(
                arm, "unigram", collection, "cross_entropy"
            )
            - metric(arm, "count_bigram", collection, "cross_entropy"),
            "unigram_minus_neural_cross_entropy": metric(
                arm, "unigram", collection, "cross_entropy"
            )
            - metric(arm, "neural_bigram", collection, "cross_entropy"),
        }
        for arm, collection in pairs
    ]
    family_native = "family_aware_native_validation"
    return {
        "positive_cross_entropy_improvement_means_lower_comparator_cross_entropy": True,
        "within_arm_collection": improvements,
        "shared_neural_head_to_head": {
            "random_minus_family_cross_entropy": metric(
                "random_training", "neural_bigram", "shared_validation", "cross_entropy"
            )
            - metric(
                "family_aware_training",
                "neural_bigram",
                "shared_validation",
                "cross_entropy",
            )
        },
        "week_03_baseline": {
            "model_arm": "family_aware_training",
            "model_type": "neural_bigram",
            "native_collection": family_native,
            "native_cross_entropy": metric(
                "family_aware_training", "neural_bigram", family_native, "cross_entropy"
            ),
            "native_accuracy": metric(
                "family_aware_training", "neural_bigram", family_native, "accuracy"
            ),
            "shared_collection": "shared_validation",
            "shared_cross_entropy": metric(
                "family_aware_training",
                "neural_bigram",
                "shared_validation",
                "cross_entropy",
            ),
            "shared_accuracy": metric(
                "family_aware_training",
                "neural_bigram",
                "shared_validation",
                "accuracy",
            ),
            "optimism_gap": metric(
                "family_aware_training",
                "neural_bigram",
                "shared_validation",
                "cross_entropy",
            )
            - metric(
                "family_aware_training", "neural_bigram", family_native, "cross_entropy"
            ),
        },
    }


def reject_forbidden_keys(value: object) -> None:
    """Reject private membership fields recursively before publication."""

    if isinstance(value, dict):
        if FORBIDDEN_MEMBERSHIP_KEYS & set(value):
            raise ModelDataError("public report contains forbidden membership data")
        for item in value.values():
            reject_forbidden_keys(item)
    elif isinstance(value, list):
        for item in value:
            reject_forbidden_keys(item)


def _validate_passed_run(run: dict[str, object]) -> None:
    runtime = run.get("runtime_seconds")
    if (
        run.get("status") != "passed"
        or run.get("hard_gates") != _SOURCE_GATES
        or run.get("collection_loads") != _SOURCE_LOADS
        or run.get("network_requests_made") != 0
        or run.get("failure_reason") is not None
        or type(runtime) not in (int, float)
        or not math.isfinite(runtime)
        or runtime < 0
    ):
        raise ModelDataError("source evaluation run record is not a passed evaluation")
