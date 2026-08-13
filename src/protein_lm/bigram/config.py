"""Strict public configuration for the Week 2 bigram stream audit."""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.data.model_data.loaders import ModelDataCollection
from protein_lm.data.model_data.promotion import PROMOTION_CONTRACT


APPROVED_BIGRAM_STREAM_CONFIG_SHA256 = (
    "c3960675dcc810464e24c8f53617408a6ecc1915f9647382ec07ac7066d77ad1"
)
_CONTRACT_IDENTIFIER = "2026-08-13-week-02-bigram-training-stream-v1"
_STREAM_DOMAIN = "protein-lm/week2/bigram-transition-stream/v1"
_MODEL_DATA_CONFIG_SHA256 = (
    "b35ec4003b002a065c29e3c70ee72ff115edafc6645f9370603e7020b4a05f12"
)
_MODEL_DATA_REGISTRY_SHA256 = (
    "13b8e1b3bb371df46f6d363b20882b91a06dde51c64d39b4e5406e0dc44efb5c"
)
_COLLECTIONS = (
    ModelDataCollection.RANDOM_TRAINING.value,
    ModelDataCollection.FAMILY_AWARE_TRAINING.value,
)
_NAMESPACES = (
    "week2/training-stream/random/v1",
    "week2/training-stream/family-aware/v1",
)
_CONTEXT_ROLES = (
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
_TARGET_ROLES = (
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
_OUTPUTS = (
    "reports/week_02/bigram_training_streams_v1.json",
    "reports/week_02/bigram_training_streams_v1.md",
    "reports/week_02/bigram_training_streams_v1.sha256",
)


@dataclass(frozen=True)
class BigramStreamConfig:
    """Every public setting that defines the two Week 2 training streams."""

    schema_version: int
    scope: str
    contract_identifier: str
    model_data_contract_identifier: str
    model_data_config_relative_path: str
    model_data_config_sha256: str
    model_data_registry_relative_path: str
    model_data_registry_sha256: str
    base_seed: int
    hash_algorithm: str
    stream_hash_domain: str
    prediction_pair_budget: int
    batch_size: int
    full_batches: int
    final_partial_batch_pairs: int
    context_roles: tuple[str, ...]
    target_roles: tuple[str, ...]
    training_collections: tuple[str, ...]
    training_namespaces: tuple[str, ...]
    report_json_relative_path: str
    report_markdown_relative_path: str
    report_sha256_relative_path: str

    @property
    def output_paths(self) -> tuple[str, str, str]:
        """Return the report paths in their atomic-publication order."""

        return (
            self.report_json_relative_path,
            self.report_markdown_relative_path,
            self.report_sha256_relative_path,
        )


def config_sha256(path: Path) -> str:
    """Return the byte identity of an audit configuration file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path) -> BigramStreamConfig:
    """Load exactly the approved public configuration bytes and values."""

    try:
        content = path.read_bytes()
        raw = tomllib.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ModelDataError(f"could not load bigram stream configuration: {error}") from error
    if hashlib.sha256(content).hexdigest() != APPROVED_BIGRAM_STREAM_CONFIG_SHA256:
        raise ModelDataError("bigram stream configuration bytes do not match approval")
    expected = {field.name for field in fields(BigramStreamConfig)}
    if set(raw) != expected:
        raise ModelDataError("bigram stream configuration keys differ from the schema")
    values = dict(raw)
    list_fields = (
        "training_collections",
        "training_namespaces",
        "context_roles",
        "target_roles",
    )
    for name in list_fields:
        value = values[name]
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item for item in value
        ):
            raise ModelDataError(f"{name} must be a list of strings")
        values[name] = tuple(value)
    for name in (
        "schema_version",
        "base_seed",
        "prediction_pair_budget",
        "batch_size",
        "full_batches",
        "final_partial_batch_pairs",
    ):
        if type(values[name]) is not int:
            raise ModelDataError(f"{name} must be an integer")
    for name in set(values) - set(list_fields) - {
        "schema_version",
        "base_seed",
        "prediction_pair_budget",
        "batch_size",
        "full_batches",
        "final_partial_batch_pairs",
    }:
        if not isinstance(values[name], str) or not values[name]:
            raise ModelDataError(f"{name} must be a nonempty string")
    config = BigramStreamConfig(**values)
    _validate(config)
    return config


def _validate(config: BigramStreamConfig) -> None:
    if (
        config.schema_version != 1
        or config.scope != "week_02_bigram_training_stream"
        or config.contract_identifier != _CONTRACT_IDENTIFIER
        or config.model_data_contract_identifier != PROMOTION_CONTRACT.identifier
    ):
        raise ModelDataError("bigram stream contract identity is not approved")
    if (
        config.model_data_config_relative_path
        != "experiments/week_02/model_data_readiness.toml"
        or config.model_data_config_sha256 != _MODEL_DATA_CONFIG_SHA256
        or config.model_data_registry_relative_path != "manifests/week_02/model_data_v1.json"
        or config.model_data_registry_sha256 != _MODEL_DATA_REGISTRY_SHA256
        or config.base_seed != 20260812
        or config.hash_algorithm != "sha256"
        or config.stream_hash_domain != _STREAM_DOMAIN
    ):
        raise ModelDataError("bigram stream source identity is not approved")
    if (
        config.prediction_pair_budget != 100_000_000
        or config.batch_size != 65_536
        or config.full_batches != 1_525
        or config.final_partial_batch_pairs != 57_600
    ):
        raise ModelDataError("bigram stream budget or batching is not approved")
    if config.context_roles != _CONTEXT_ROLES or config.target_roles != _TARGET_ROLES:
        raise ModelDataError("bigram role-specific coordinates are not approved")
    if config.training_collections != _COLLECTIONS or config.training_namespaces != _NAMESPACES:
        raise ModelDataError("bigram stream collections or namespaces are not approved")
    if config.output_paths != _OUTPUTS:
        raise ModelDataError("bigram stream report paths are not approved")
    if config.batch_size * config.full_batches + config.final_partial_batch_pairs != config.prediction_pair_budget:
        raise ModelDataError("bigram stream batch arithmetic is inconsistent")
