"""Pinned public settings for the Week 2 bigram model fitting contract."""

from __future__ import annotations

import hashlib
import math
import re
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from protein_lm.data.model_data.contracts import ModelDataError


APPROVED_BIGRAM_TRAINING_CONFIG_SHA256 = (
    "a77851513d5c7be10ee34afe09ef46c334d028647d22c4745e17729dc35b1ffc"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
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
_TARGET_ROLES = _CONTEXT_ROLES[1:] + ("EOS",)
_ZERO_MATRIX_SHA256 = "f9a8c9b949a6f1bb9a980b9575944abeaeac8907c22d7386c95909c67e11982a"


@dataclass(frozen=True)
class BigramTrainingConfig:
    """All public values that define the approved production fitting run."""

    schema_version: int
    scope: str
    contract_identifier: str
    stream_config_relative_path: str
    stream_config_sha256: str
    stream_report_relative_path: str
    stream_report_sha256: str
    base_seed: int
    device: str
    optimizer: str
    learning_rate: float
    momentum: float
    weight_decay: float
    count_smoothing_alpha: int
    role_space_size: int
    unigram_tensor_dtype: str
    unigram_tensor_shape: tuple[int, ...]
    count_bigram_tensor_dtype: str
    count_bigram_tensor_shape: tuple[int, ...]
    neural_tensor_dtype: str
    neural_tensor_shape: tuple[int, ...]
    neural_bias: bool
    neural_loss: str
    neural_loss_reduction: str
    neural_zero_grad_set_to_none: bool
    initialization: str
    initial_weights_sha256: str
    batch_size: int
    prediction_pair_budget: int
    full_batches: int
    final_partial_batch_pairs: int
    total_optimizer_steps: int
    early_stopping: str
    context_roles: tuple[str, ...]
    target_roles: tuple[str, ...]
    model_types: tuple[str, ...]
    serialization_formats: tuple[str, ...]


def config_sha256(path: Path) -> str:
    """Return the exact byte identity of a training configuration."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_training_config(path: Path) -> BigramTrainingConfig:
    """Load only the byte-pinned production configuration."""

    try:
        content = path.read_bytes()
        raw = tomllib.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ModelDataError(
            f"could not load bigram training configuration: {error}"
        ) from error
    if hashlib.sha256(content).hexdigest() != APPROVED_BIGRAM_TRAINING_CONFIG_SHA256:
        raise ModelDataError(
            "bigram training configuration bytes do not match approval"
        )
    expected = {field.name for field in fields(BigramTrainingConfig)}
    if set(raw) != expected:
        raise ModelDataError(
            "bigram training configuration keys differ from the schema"
        )
    values = dict(raw)
    string_lists = (
        "context_roles",
        "target_roles",
        "model_types",
        "serialization_formats",
    )
    integer_lists = (
        "unigram_tensor_shape",
        "count_bigram_tensor_shape",
        "neural_tensor_shape",
    )
    for name in string_lists:
        values[name] = _string_list(values[name], name)
    for name in integer_lists:
        values[name] = _integer_list(values[name], name)
    for name in (
        "schema_version",
        "base_seed",
        "count_smoothing_alpha",
        "role_space_size",
        "batch_size",
        "prediction_pair_budget",
        "full_batches",
        "final_partial_batch_pairs",
        "total_optimizer_steps",
    ):
        if type(values[name]) is not int:
            raise ModelDataError(f"{name} must be an integer")
    for name in ("learning_rate", "momentum", "weight_decay"):
        if type(values[name]) not in (int, float) or not math.isfinite(values[name]):
            raise ModelDataError(f"{name} must be a finite number")
        values[name] = float(values[name])
    for name in ("neural_bias", "neural_zero_grad_set_to_none"):
        if type(values[name]) is not bool:
            raise ModelDataError(f"{name} must be a boolean")
    protected = (
        set(string_lists)
        | set(integer_lists)
        | {
            "schema_version",
            "base_seed",
            "count_smoothing_alpha",
            "role_space_size",
            "batch_size",
            "prediction_pair_budget",
            "full_batches",
            "final_partial_batch_pairs",
            "total_optimizer_steps",
            "learning_rate",
            "momentum",
            "weight_decay",
            "neural_bias",
            "neural_zero_grad_set_to_none",
        }
    )
    for name in set(values) - protected:
        if not isinstance(values[name], str) or not values[name]:
            raise ModelDataError(f"{name} must be a nonempty string")
    config = BigramTrainingConfig(**values)
    _validate(config)
    return config


def _validate(config: BigramTrainingConfig) -> None:
    if (
        config.schema_version != 1
        or config.scope != "week_02_bigram_training"
        or config.contract_identifier != "2026-08-18-week-02-bigram-training-v1"
        or config.stream_config_relative_path
        != "experiments/week_02/bigram_training_stream_v1.toml"
        or config.stream_config_sha256
        != "c3960675dcc810464e24c8f53617408a6ecc1915f9647382ec07ac7066d77ad1"
        or config.stream_report_relative_path
        != "reports/week_02/bigram_training_streams_v1.json"
        or config.stream_report_sha256
        != "914bcae29e989d550b2db2cc16fe2245821caa496de68e779e69b102439386b8"
        or config.base_seed != 20260812
    ):
        raise ModelDataError("bigram training contract identity is not approved")
    if (
        config.device != "cpu"
        or config.optimizer != "SGD"
        or config.learning_rate != 1.0
        or config.momentum != 0.0
        or config.weight_decay != 0.0
        or config.count_smoothing_alpha != 1
        or config.role_space_size != 21
    ):
        raise ModelDataError("bigram training numeric settings are not approved")
    if (
        config.unigram_tensor_dtype != "int64"
        or config.unigram_tensor_shape != (21,)
        or config.count_bigram_tensor_dtype != "int64"
        or config.count_bigram_tensor_shape != (21, 21)
        or config.neural_tensor_dtype != "float32"
        or config.neural_tensor_shape != (21, 21)
        or config.neural_bias
        or config.neural_loss != "cross_entropy"
        or config.neural_loss_reduction != "mean"
        or not config.neural_zero_grad_set_to_none
        or config.initialization != "all_zero"
        or config.initial_weights_sha256 != _ZERO_MATRIX_SHA256
    ):
        raise ModelDataError("bigram tensor settings are not approved")
    if (
        _SHA256.fullmatch(config.initial_weights_sha256) is None
        or _SHA256.fullmatch(config.stream_report_sha256) is None
    ):
        raise ModelDataError("training checksum is invalid")
    if (
        config.early_stopping != "none"
        or config.batch_size != 65_536
        or config.prediction_pair_budget != 100_000_000
        or config.full_batches != 1_525
        or config.final_partial_batch_pairs != 57_600
        or config.total_optimizer_steps != 1_526
    ):
        raise ModelDataError("bigram training budget or batching is not approved")
    if (
        config.batch_size * config.full_batches + config.final_partial_batch_pairs
        != config.prediction_pair_budget
    ):
        raise ModelDataError("bigram training batch arithmetic is inconsistent")
    if config.total_optimizer_steps != config.full_batches + 1:
        raise ModelDataError("bigram training step arithmetic is inconsistent")
    if config.context_roles != _CONTEXT_ROLES or config.target_roles != _TARGET_ROLES:
        raise ModelDataError("bigram training role spaces are not approved")
    if config.model_types != ("unigram", "count_bigram", "neural_bigram"):
        raise ModelDataError("bigram model types are not approved")
    if config.serialization_formats != ("json", "safetensors"):
        raise ModelDataError("bigram serialization formats are not approved")


def _string_list(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ModelDataError(f"{name} must be a list of strings")
    return tuple(value)


def _integer_list(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ModelDataError(f"{name} must be a list of integers")
    return tuple(value)
