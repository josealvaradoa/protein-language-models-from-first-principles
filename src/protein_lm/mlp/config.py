"""Strict, byte-pinned settings for the one approved Week 3 MLP run."""

from __future__ import annotations

import hashlib
import math
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from protein_lm.data.model_data.contracts import ModelDataError


APPROVED_MLP_TRAINING_CONFIG_SHA256 = (
    "3df5ed5c13e3f562fdf8cdb2166470f2642f9079305347d9d4c513f8b72d18a7"
)
_MILESTONES = (1_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000, 100_000_000)
_CHECKPOINTS = _MILESTONES[:-1] + (90_000_000, 100_000_000)
_SEEDS = (20260821, 20260822, 20260823)


@dataclass(frozen=True)
class MLPTrainingConfig:
    schema_version: int
    scope: str
    contract_identifier: str
    model_data_contract_identifier: str
    model_data_config_sha256: str
    model_data_registry_relative_path: str
    model_data_registry_sha256: str
    training_stream_report_relative_path: str
    training_stream_report_sha256: str
    training_collection: str
    native_validation_collection: str
    training_namespace: str
    stream_base_seed: int
    context_length: int
    embedding_width: int
    hidden_width: int
    context_vocab_size: int
    target_vocab_size: int
    tensor_dtype: str
    context_roles: tuple[str, ...]
    target_roles: tuple[str, ...]
    activation: str
    initialization: str
    optimizer: str
    base_learning_rate: float
    post_boundary_learning_rate: float
    learning_rate_boundary_predictions: int
    momentum: float
    weight_decay: float
    loss: str
    loss_reduction: str
    prediction_budget: int
    batch_size: int
    milestone_predictions: tuple[int, ...]
    checkpoint_predictions: tuple[int, ...]
    run_seeds: tuple[int, ...]
    early_stopping: str
    output_relative_root: str

    @property
    def parameter_count(self) -> int:
        return (
            self.context_vocab_size * self.embedding_width
            + self.context_length * self.embedding_width * self.hidden_width
            + self.hidden_width
            + self.hidden_width * self.target_vocab_size
            + self.target_vocab_size
        )

    @property
    def event_predictions(self) -> tuple[int, ...]:
        return tuple(
            sorted(set(self.milestone_predictions + self.checkpoint_predictions))
        )

    def expected_optimizer_steps(self, predictions: int) -> int:
        """Count batches after exact event splits through one approved cursor."""

        if (
            type(predictions) is not int
            or predictions < 0
            or predictions > self.prediction_budget
        ):
            raise ModelDataError("prediction position is invalid")
        total = 0
        previous = 0
        for boundary in self.event_predictions:
            end = min(boundary, predictions)
            if end > previous:
                total += math.ceil((end - previous) / self.batch_size)
            previous = boundary
            if boundary >= predictions:
                return total
        return total + math.ceil((predictions - previous) / self.batch_size)


def config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path) -> MLPTrainingConfig:
    """Load exactly the reviewed byte sequence, then validate its types and values."""

    try:
        content = path.read_bytes()
        raw = tomllib.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ModelDataError(
            f"could not load MLP training configuration: {error}"
        ) from error
    if hashlib.sha256(content).hexdigest() != APPROVED_MLP_TRAINING_CONFIG_SHA256:
        raise ModelDataError("MLP training configuration bytes do not match approval")
    if set(raw) != {field.name for field in fields(MLPTrainingConfig)}:
        raise ModelDataError("MLP training configuration keys differ from the schema")
    values = dict(raw)
    for name in ("milestone_predictions", "checkpoint_predictions", "run_seeds"):
        value = values[name]
        if not isinstance(value, list) or any(type(item) is not int for item in value):
            raise ModelDataError(f"{name} must be a list of integers")
        values[name] = tuple(value)
    for name in ("context_roles", "target_roles"):
        value = values[name]
        if not isinstance(value, list) or any(
            not isinstance(item, str) for item in value
        ):
            raise ModelDataError(f"{name} must be a list of strings")
        values[name] = tuple(value)
    integer_names = {
        "schema_version",
        "stream_base_seed",
        "context_length",
        "embedding_width",
        "hidden_width",
        "context_vocab_size",
        "target_vocab_size",
        "learning_rate_boundary_predictions",
        "prediction_budget",
        "batch_size",
    }
    for name in integer_names:
        if type(values[name]) is not int:
            raise ModelDataError(f"{name} must be an integer")
    for name in (
        "base_learning_rate",
        "post_boundary_learning_rate",
        "momentum",
        "weight_decay",
    ):
        if type(values[name]) not in (int, float) or not math.isfinite(values[name]):
            raise ModelDataError(f"{name} must be a finite number")
        values[name] = float(values[name])
    for name in (
        set(values)
        - integer_names
        - {
            "milestone_predictions",
            "checkpoint_predictions",
            "run_seeds",
            "context_roles",
            "target_roles",
            "base_learning_rate",
            "post_boundary_learning_rate",
            "momentum",
            "weight_decay",
        }
    ):
        if not isinstance(values[name], str) or not values[name]:
            raise ModelDataError(f"{name} must be a nonempty string")
    config = MLPTrainingConfig(**values)
    _validate(config)
    return config


def _validate(config: MLPTrainingConfig) -> None:
    if (
        config.schema_version != 1
        or config.scope != "week_03_scalable_mlp_training"
        or config.contract_identifier != "2026-08-23-week-03-mlp-training-v1"
        or config.model_data_contract_identifier
        != "2026-08-12-week-02-model-data-promotion-v1"
        or config.model_data_config_sha256
        != "b35ec4003b002a065c29e3c70ee72ff115edafc6645f9370603e7020b4a05f12"
        or config.model_data_registry_relative_path
        != "manifests/week_02/model_data_v1.json"
        or config.model_data_registry_sha256
        != "13b8e1b3bb371df46f6d363b20882b91a06dde51c64d39b4e5406e0dc44efb5c"
        or config.training_stream_report_relative_path
        != "reports/week_02/bigram_training_streams_v1.json"
        or config.training_stream_report_sha256
        != "914bcae29e989d550b2db2cc16fe2245821caa496de68e779e69b102439386b8"
        or config.training_collection != "family_aware_training"
        or config.native_validation_collection != "family_aware_native_validation"
        or config.training_namespace != "week2/training-stream/family-aware/v1"
        or config.stream_base_seed != 20260812
    ):
        raise ModelDataError("MLP training source identity is not approved")
    if (
        config.context_length != 10
        or config.embedding_width != 32
        or config.hidden_width != 800
        or config.context_vocab_size != 21
        or config.target_vocab_size != 21
        or config.tensor_dtype != "float32"
        or config.parameter_count != 274_293
        or config.context_roles != ("BOS",) + tuple("ACDEFGHIKLMNPQRSTVWY")
        or config.target_roles != tuple("ACDEFGHIKLMNPQRSTVWY") + ("EOS",)
        or config.activation != "tanh"
        or config.initialization
        != "pilot_scaled_normal_embedding_e_minus_half_w1_ce_minus_half_w2_h_minus_half_zero_bias_v1"
    ):
        raise ModelDataError("MLP model shape is not approved")
    if (
        config.optimizer != "SGD"
        or config.base_learning_rate != 0.1
        or config.post_boundary_learning_rate != 0.01
        or config.learning_rate_boundary_predictions != 90_000_000
        or config.momentum != 0.0
        or config.weight_decay != 0.0
        or config.loss != "cross_entropy"
        or config.loss_reduction != "mean"
        or config.early_stopping != "none"
    ):
        raise ModelDataError("MLP optimizer settings are not approved")
    if (
        config.prediction_budget != 100_000_000
        or config.batch_size != 1_024
        or config.milestone_predictions != _MILESTONES
        or config.checkpoint_predictions != _CHECKPOINTS
        or config.run_seeds != _SEEDS
        or config.output_relative_root != "data/processed/week_03/mlp_training_runs"
    ):
        raise ModelDataError("MLP run budget or events are not approved")
    if config.event_predictions[-1] != config.prediction_budget:
        raise ModelDataError("MLP event schedule does not end at the prediction budget")
