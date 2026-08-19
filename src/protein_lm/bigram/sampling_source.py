"""Strict small-record validation for the pinned sampling source candidate."""

from __future__ import annotations

from protein_lm.bigram.candidate_contract import CandidatePlan
from protein_lm.bigram.sampling_contract import SamplingConfig
from protein_lm.data.model_data.contracts import ModelDataError


RUN_GATES = {
    "only_promoted_training_collections",
    "one_pass_per_arm",
    "exact_audited_streams",
    "six_logical_models",
    "dual_serialization",
}


def neural_sources(config: SamplingConfig) -> tuple[tuple[str, str, str, str], ...]:
    """Return only the two approved neural artifact identities."""
    return (
        (
            config.arms[0],
            config.namespaces[0],
            config.random_neural_json_sha256,
            config.random_neural_safetensors_sha256,
        ),
        (
            config.arms[1],
            config.namespaces[1],
            config.family_aware_neural_json_sha256,
            config.family_aware_neural_safetensors_sha256,
        ),
    )


def validate_passed_source(
    record: dict[str, object],
    registry: dict[str, object],
    candidate_plan: CandidatePlan,
    config: SamplingConfig,
) -> None:
    """Validate records and neural entries without deserializing any model."""
    training = record.get("training")
    if (
        set(record)
        != {
            "schema_version",
            "scope",
            "contract_identifier",
            "candidate_id",
            "status",
            "hard_gates",
            "training",
            "source_identity",
            "code_revision",
            "arms",
            "runtime_seconds",
            "network_requests_made",
            "failure_reason",
        }
        or record.get("schema_version") != 1
        or record.get("scope") != "week_02_bigram_model_candidate"
        or record.get("contract_identifier")
        != candidate_plan.training_config.contract_identifier
        or record.get("candidate_id") != config.candidate_id
        or record.get("status") != "passed"
        or record.get("code_revision") != config.candidate_code_revision
        or record.get("network_requests_made") != 0
        or record.get("failure_reason") is not None
        or not isinstance(training, dict)
        or training.get("base_seed") != config.base_seed
        or not isinstance(record.get("hard_gates"), dict)
        or set(record["hard_gates"]) != RUN_GATES
        or not all(value is True for value in record["hard_gates"].values())
    ):
        raise ModelDataError(
            "sampling candidate run record is not the pinned passed source"
        )
    expected_names = {
        f"{arm}__{model}.{format_name}"
        for arm in config.arms
        for model in ("unigram", "count_bigram", "neural_bigram")
        for format_name in ("json", "safetensors")
    }
    artifacts = registry.get("artifacts")
    if (
        set(registry)
        != {
            "schema_version",
            "scope",
            "candidate_id",
            "logical_model_count",
            "serialization_file_count",
            "artifacts",
        }
        or registry.get("schema_version") != 1
        or registry.get("scope") != "week_02_bigram_model_candidate_registry"
        or registry.get("candidate_id") != config.candidate_id
        or registry.get("logical_model_count") != 6
        or registry.get("serialization_file_count") != 12
        or not isinstance(artifacts, dict)
        or set(artifacts) != expected_names
    ):
        raise ModelDataError("sampling candidate registry is not the pinned source")
    for arm, _namespace, json_hash, safe_hash in neural_sources(config):
        for format_name, expected_hash in (
            ("json", json_hash),
            ("safetensors", safe_hash),
        ):
            entry = artifacts[f"{arm}__neural_bigram.{format_name}"]
            if (
                not isinstance(entry, dict)
                or set(entry) != {"arm", "model_type", "format", "byte_size", "sha256"}
                or entry.get("arm") != arm
                or entry.get("model_type") != "neural_bigram"
                or entry.get("format") != format_name
                or entry.get("sha256") != expected_hash
                or type(entry.get("byte_size")) is not int
                or entry["byte_size"] <= 0
            ):
                raise ModelDataError("sampling neural artifact registry entry drifted")
