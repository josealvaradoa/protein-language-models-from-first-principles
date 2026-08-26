"""Byte-pinned contract for the Week 3 position-availability diagnostic."""

from __future__ import annotations

import hashlib
import math
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from protein_lm.data.model_data.contracts import ModelDataError


APPROVED_POSITION_AVAILABILITY_DIAGNOSTIC_CONFIG_SHA256 = (
    "392c571f1dbef0b2288179d59c0cfb3569765c0751679cc7c95ad75cee429417"
)
SEEDS = (20260821, 20260822, 20260823)
BINS = (
    "available_prior_residues_0_10",
    "available_prior_residues_11_19",
    "available_prior_residues_20_plus",
)


@dataclass(frozen=True)
class FinalRun:
    seed: int
    run_id: str
    run_status_sha256: str
    metadata_sha256: str
    tensor_sha256: str
    native_cross_entropy: float
    native_accuracy: float
    native_nll_numerator: float
    native_correct_predictions: int


@dataclass(frozen=True)
class PositionAvailabilityDiagnosticConfig:
    schema_version: int
    scope: str
    contract_identifier: str
    frozen_comparison_scope: str
    frozen_context20_mean_native_cross_entropy: float
    frozen_context20_sample_standard_deviation: float
    frozen_embedding64_mean_native_cross_entropy: float
    frozen_embedding64_sample_standard_deviation: float
    frozen_embedding64_minus_context20_mean_native_cross_entropy: float
    frozen_material_gap: float
    frozen_category: str
    context20_config_relative_path: str
    context20_config_sha256: str
    embedding64_config_relative_path: str
    embedding64_config_sha256: str
    readiness_report_relative_path: str
    readiness_report_sha256: str
    native_validation_collection: str
    native_validation_prediction_tokens: int
    native_validation_records: int
    batch_size: int
    device: str
    output_relative_root: str
    overall_metric_absolute_tolerance: float
    bins: tuple[str, ...]
    context20_runs: tuple[FinalRun, ...]
    embedding64_runs: tuple[FinalRun, ...]

    def run(self, arm: str, seed: int) -> FinalRun:
        if arm == "context20":
            runs = self.context20_runs
        elif arm == "embedding64":
            runs = self.embedding64_runs
        else:
            raise ModelDataError("diagnostic arm is not approved")
        for run in runs:
            if run.seed == seed:
                return run
        raise ModelDataError("diagnostic seed is not approved")


def config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_position_availability_diagnostic_config(
    path: Path,
) -> PositionAvailabilityDiagnosticConfig:
    """Load exact approved bytes and reject every unapproved schema variation."""

    try:
        content = path.read_bytes()
        raw = tomllib.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ModelDataError(
            f"could not load position-availability diagnostic configuration: {error}"
        ) from error
    if (
        hashlib.sha256(content).hexdigest()
        != APPROVED_POSITION_AVAILABILITY_DIAGNOSTIC_CONFIG_SHA256
    ):
        raise ModelDataError(
            "position-availability diagnostic configuration bytes do not match approval"
        )
    if set(raw) != {
        field.name for field in fields(PositionAvailabilityDiagnosticConfig)
    }:
        raise ModelDataError(
            "position-availability diagnostic configuration keys differ from schema"
        )
    values = dict(raw)
    values["bins"] = _strings(values["bins"], "bins")
    values["context20_runs"] = _runs(values["context20_runs"])
    values["embedding64_runs"] = _runs(values["embedding64_runs"])
    integers = {
        "schema_version",
        "native_validation_prediction_tokens",
        "native_validation_records",
        "batch_size",
    }
    floats = {
        "overall_metric_absolute_tolerance",
        "frozen_context20_mean_native_cross_entropy",
        "frozen_context20_sample_standard_deviation",
        "frozen_embedding64_mean_native_cross_entropy",
        "frozen_embedding64_sample_standard_deviation",
        "frozen_embedding64_minus_context20_mean_native_cross_entropy",
        "frozen_material_gap",
    }
    for name in integers:
        if type(values[name]) is not int:
            raise ModelDataError(f"{name} must be an integer")
    for name in floats:
        if type(values[name]) not in (int, float) or not math.isfinite(values[name]):
            raise ModelDataError(f"{name} must be a finite number")
        values[name] = float(values[name])
    for name in (
        set(values) - integers - floats - {"bins", "context20_runs", "embedding64_runs"}
    ):
        if not isinstance(values[name], str) or not values[name]:
            raise ModelDataError(f"{name} must be a nonempty string")
    config = PositionAvailabilityDiagnosticConfig(**values)
    _validate(config)
    return config


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ModelDataError(f"{name} must be a list of nonempty strings")
    return tuple(value)


def _runs(value: object) -> tuple[FinalRun, ...]:
    if not isinstance(value, list) or len(value) != len(SEEDS):
        raise ModelDataError("diagnostic run pins are invalid")
    result = []
    for item in value:
        if not isinstance(item, dict) or set(item) != set(
            FinalRun.__dataclass_fields__
        ):
            raise ModelDataError("diagnostic run pin schema is invalid")
        if (
            type(item["seed"]) is not int
            or type(item["native_correct_predictions"]) is not int
        ):
            raise ModelDataError("diagnostic run pin integer types are invalid")
        for name in ("native_cross_entropy", "native_accuracy", "native_nll_numerator"):
            if type(item[name]) not in (int, float) or not math.isfinite(item[name]):
                raise ModelDataError("diagnostic run pin numeric types are invalid")
        for name in ("run_id", "run_status_sha256", "metadata_sha256", "tensor_sha256"):
            if not isinstance(item[name], str) or not item[name]:
                raise ModelDataError("diagnostic run pin string types are invalid")
        result.append(
            FinalRun(
                **{
                    **item,
                    "native_cross_entropy": float(item["native_cross_entropy"]),
                    "native_accuracy": float(item["native_accuracy"]),
                    "native_nll_numerator": float(item["native_nll_numerator"]),
                }
            )
        )
    return tuple(result)


def _validate(config: PositionAvailabilityDiagnosticConfig) -> None:
    if (
        config.schema_version != 1
        or config.scope != "week_03_mlp_position_availability_diagnostic_exploratory"
        or config.contract_identifier
        != "2026-08-25-week-03-position-availability-diagnostic-v1"
        or config.frozen_comparison_scope
        != "provenance_only_diagnostic_cannot_reopen_selection"
        or config.frozen_context20_mean_native_cross_entropy != 2.863665856220289
        or config.frozen_context20_sample_standard_deviation != 0.00001985865257320209
        or config.frozen_embedding64_mean_native_cross_entropy != 2.8708249214089068
        or config.frozen_embedding64_sample_standard_deviation
        != 0.000007648534920387316
        or config.frozen_embedding64_minus_context20_mean_native_cross_entropy
        != 0.0071590651886177525
        or config.frozen_material_gap != 0.001
        or config.frozen_category != "context20_materially_better"
        or config.context20_config_relative_path
        != "experiments/week_03/mlp_context20_100m_continuation_v1.toml"
        or config.context20_config_sha256
        != "a0b1aa37c647aa5dc62c9d7a2b7c051bf66a81380448a5719d9ebe34eb6a3fb6"
        or config.embedding64_config_relative_path
        != "experiments/week_03/mlp_embedding64_100m_challenger_v1.toml"
        or config.embedding64_config_sha256
        != "5d441c6e27746f59428ee1a5ee9d3a0aa32cac28b3dea9c188095c873ed7d92a"
        or config.readiness_report_relative_path
        != "reports/week_02/model_data_readiness_v1.json"
        or config.readiness_report_sha256
        != "19d4ee82eae49b600e9e83e4bb19d468b7a9fc2cfd6b78ffebc995f77db9b881"
        or config.native_validation_collection != "family_aware_native_validation"
        or config.native_validation_prediction_tokens != 1_000_495
        or config.native_validation_records != 2_645
        or config.batch_size != 1_024
        or config.device != "cpu"
        or config.output_relative_root
        != "data/processed/week_03/mlp_position_availability_diagnostic_runs"
        or config.overall_metric_absolute_tolerance != 0.000001
        or config.bins != BINS
        or tuple(run.seed for run in config.context20_runs) != SEEDS
        or tuple(run.seed for run in config.embedding64_runs) != SEEDS
    ):
        raise ModelDataError(
            "position-availability diagnostic configuration values are not approved"
        )
    for run in config.context20_runs + config.embedding64_runs:
        if (
            run.native_correct_predictions < 0
            or run.native_correct_predictions
            > config.native_validation_prediction_tokens
            or run.native_nll_numerator <= 0
            or abs(
                run.native_cross_entropy
                - run.native_nll_numerator / config.native_validation_prediction_tokens
            )
            > config.overall_metric_absolute_tolerance
            or abs(
                run.native_accuracy
                - run.native_correct_predictions
                / config.native_validation_prediction_tokens
            )
            > config.overall_metric_absolute_tolerance
        ):
            raise ModelDataError("diagnostic expected overall metric is invalid")
    if (
        abs(
            config.frozen_embedding64_minus_context20_mean_native_cross_entropy
            - (
                config.frozen_embedding64_mean_native_cross_entropy
                - config.frozen_context20_mean_native_cross_entropy
            )
        )
        > 1e-15
        or config.frozen_category != "context20_materially_better"
        or config.frozen_embedding64_minus_context20_mean_native_cross_entropy
        < config.frozen_material_gap
    ):
        raise ModelDataError("diagnostic frozen comparison provenance is invalid")
