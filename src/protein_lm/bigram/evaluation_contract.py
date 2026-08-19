"""Byte-pinned, evaluation-only commitments for Week 2 bigram scoring."""

from __future__ import annotations

import hashlib
import re
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from protein_lm.data.model_data.contracts import ModelDataError


# Updated with the digest of the reviewed public TOML.  Loading rejects any edit.
APPROVED_EVALUATION_CONFIG_SHA256 = (
    "219e7a3bc06a6c227ed27b9b4b7e917083b537bd5ac5d11a7526ee8415c2d97c"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class EvaluationConfig:
    """The small public contract that makes this a fixed evaluation, not a search."""

    schema_version: int
    scope: str
    contract_identifier: str
    model_candidate_id: str
    model_candidate_relative_path: str
    model_candidate_registry_sha256: str
    model_candidate_run_record_sha256: str
    model_data_registry_relative_path: str
    model_data_registry_sha256: str
    output_root_relative_path: str
    model_arms: tuple[str, ...]
    model_types: tuple[str, ...]
    native_collections: tuple[str, ...]
    shared_collection: str
    metric_dtype: str
    count_smoothing_alpha: int
    valid_context_roles: tuple[str, ...]
    valid_target_roles: tuple[str, ...]
    tie_rule: str
    median_rule: str
    length_buckets: tuple[str, ...]
    sealed_test_collection: str
    network_requests_made: int
    selection: str
    retraining: str


def config_sha256(path: Path) -> str:
    """Return the byte identity that is carried into every evaluation record."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_evaluation_config(path: Path) -> EvaluationConfig:
    """Load only the reviewed public contract, without opening model or data files."""

    try:
        content = path.read_bytes()
        raw = tomllib.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ModelDataError(
            f"could not load bigram evaluation configuration: {error}"
        ) from error
    if hashlib.sha256(content).hexdigest() != APPROVED_EVALUATION_CONFIG_SHA256:
        raise ModelDataError(
            "bigram evaluation configuration bytes do not match approval"
        )
    if not isinstance(raw, dict) or set(raw) != {
        field.name for field in fields(EvaluationConfig)
    }:
        raise ModelDataError(
            "bigram evaluation configuration keys differ from the schema"
        )
    values = dict(raw)
    for name in (
        "model_arms",
        "model_types",
        "native_collections",
        "valid_context_roles",
        "valid_target_roles",
        "length_buckets",
    ):
        value = values[name]
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item for item in value
        ):
            raise ModelDataError(f"{name} must be a list of nonempty strings")
        values[name] = tuple(value)
    if (
        type(values["schema_version"]) is not int
        or type(values["count_smoothing_alpha"]) is not int
        or type(values["network_requests_made"]) is not int
    ):
        raise ModelDataError("bigram evaluation integer values are invalid")
    for name, value in values.items():
        if name not in {
            "schema_version",
            "count_smoothing_alpha",
            "network_requests_made",
            "model_arms",
            "model_types",
            "native_collections",
            "valid_context_roles",
            "valid_target_roles",
            "length_buckets",
        } and (not isinstance(value, str) or not value):
            raise ModelDataError(f"{name} must be a nonempty string")
    config = EvaluationConfig(**values)
    _validate(config)
    return config


def _validate(config: EvaluationConfig) -> None:
    if (
        config.schema_version != 1
        or config.scope != "week_02_bigram_evaluation"
        or config.contract_identifier != "2026-08-19-week-02-bigram-evaluation-v1"
        or config.model_candidate_id != "week2-bigram-v1-001"
        or config.model_candidate_relative_path
        != "data/processed/week_02/bigram_model_candidates/week2-bigram-v1-001"
        or config.model_candidate_registry_sha256
        != "18bca1ec67b639d0ae68ee022a18cb2cdd9e9f571c7b305e5c9960c3a8257e5f"
        or config.model_candidate_run_record_sha256
        != "e9bee5dadd91c2090304cae301b2eedec5b4e5ef3c1f3bdcadc3fbc5681391e9"
        or config.model_data_registry_relative_path
        != "manifests/week_02/model_data_v1.json"
        or config.model_data_registry_sha256
        != "13b8e1b3bb371df46f6d363b20882b91a06dde51c64d39b4e5406e0dc44efb5c"
        or config.output_root_relative_path
        != "data/processed/week_02/bigram_evaluation_candidates"
    ):
        raise ModelDataError("bigram evaluation contract identity is not approved")
    if not all(
        _SHA256.fullmatch(value)
        for value in (
            config.model_candidate_registry_sha256,
            config.model_candidate_run_record_sha256,
            config.model_data_registry_sha256,
        )
    ):
        raise ModelDataError("bigram evaluation candidate checksum is invalid")
    if (
        config.model_arms != ("random_training", "family_aware_training")
        or config.model_types != ("unigram", "count_bigram", "neural_bigram")
        or config.native_collections
        != ("random_native_validation", "family_aware_native_validation")
        or config.shared_collection != "shared_validation"
        or config.sealed_test_collection != "shared_sealed_test"
        or config.metric_dtype != "float64"
        or config.count_smoothing_alpha != 1
        or config.valid_context_roles
        != (
            "BOS",
            "A",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "V",
            "W",
            "Y",
        )
        or config.valid_target_roles
        != (
            "A",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "V",
            "W",
            "Y",
            "EOS",
        )
        or config.tie_rule != "lowest_valid_target_id"
        or config.median_rule
        != "arithmetic_mean_of_two_middle_values_for_even_population"
        or config.length_buckets
        != ("32-127", "128-255", "256-511", "512-1023", "1024-2046")
        or config.network_requests_made != 0
        or config.selection != "forbidden"
        or config.retraining != "forbidden"
    ):
        raise ModelDataError("bigram evaluation settings are not approved")
