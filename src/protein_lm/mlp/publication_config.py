"""Strict, local-only contract for the aggregate Week 3 public report."""

from __future__ import annotations

import hashlib
import math
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from protein_lm.data.model_data.contracts import ModelDataError


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OUTPUTS = (
    "reports/week_03/mlp_evaluation_v1.json",
    "reports/week_03/mlp_evaluation_v1.md",
    "reports/week_03/mlp_evaluation_v1.sha256",
)
_SOURCE_COUNTS = {
    "config": 7, "week2_public_baseline": 1, "c10_learning_curve_status": 3,
    "capacity_status": 9, "lr_tail_status": 6, "one_epoch_status": 3,
}
_PUBLICATION_SCOPE = (
    "aggregate_only_no_sequences_no_accessions_no_family_ids_no_raw_tensors"
)
_FORBIDDEN_PUBLIC_KEYS = (
    "sequence", "sequences", "accession", "accessions", "family_id",
    "family_ids", "raw_tensor", "raw_tensors", "weights", "model.safetensors",
)
_EXPECTED_KEYS = {
    "native_validation_token_count", "final_prediction_budget",
    "context20_parameter_count", "embedding64_parameter_count",
    "context20_mean_cross_entropy", "context20_sample_standard_deviation",
    "context20_mean_accuracy", "embedding64_mean_cross_entropy",
    "embedding64_sample_standard_deviation", "embedding64_mean_accuracy",
    "material_gap", "embedding64_minus_context20_mean_cross_entropy",
    "baseline_cross_entropy", "baseline_accuracy",
    "context20_cross_entropy_gain_over_baseline",
    "context20_accuracy_gain_over_baseline",
    "capacity_context20_cross_entropy", "capacity_embedding64_cross_entropy",
    "capacity_hidden1600_cross_entropy", "position_20_plus_advantage_share",
}
_INTEGER_EXPECTED_KEYS = {
    "native_validation_token_count", "final_prediction_budget",
    "context20_parameter_count", "embedding64_parameter_count",
}


@dataclass(frozen=True)
class PinnedFile:
    kind: str
    relative_path: str
    sha256: str


@dataclass(frozen=True)
class FinalCheckpoint:
    arm: str
    seed: int
    status_relative_path: str
    status_sha256: str
    checkpoint_relative_path: str
    metadata_sha256: str
    tensor_sha256: str


@dataclass(frozen=True)
class DiagnosticStatus:
    seed: int
    relative_path: str
    sha256: str


@dataclass(frozen=True)
class PublicationConfig:
    schema_version: int
    scope: str
    contract_identifier: str
    publication_scope: str
    report_json_relative_path: str
    report_markdown_relative_path: str
    report_sha256_relative_path: str
    run_seeds: tuple[int, ...]
    forbidden_public_keys: tuple[str, ...]
    expected: dict[str, object]
    sources: tuple[PinnedFile, ...]
    final_checkpoints: tuple[FinalCheckpoint, ...]
    diagnostic_statuses: tuple[DiagnosticStatus, ...]

    @property
    def output_paths(self) -> tuple[str, str, str]:
        return (
            self.report_json_relative_path,
            self.report_markdown_relative_path,
            self.report_sha256_relative_path,
        )


def config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_publication_config(path: Path) -> PublicationConfig:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ModelDataError("could not load Week 3 publication configuration") from error
    required = {
        "schema_version", "scope", "contract_identifier", "publication_scope",
        "report_json_relative_path", "report_markdown_relative_path",
        "report_sha256_relative_path", "run_seeds", "forbidden_public_keys",
        "expected", "sources", "final_checkpoints", "diagnostic_statuses",
    }
    if set(raw) != required:
        raise ModelDataError("Week 3 publication configuration keys differ from schema")
    try:
        config = PublicationConfig(
            schema_version=raw["schema_version"], scope=raw["scope"],
            contract_identifier=raw["contract_identifier"],
            publication_scope=raw["publication_scope"],
            report_json_relative_path=raw["report_json_relative_path"],
            report_markdown_relative_path=raw["report_markdown_relative_path"],
            report_sha256_relative_path=raw["report_sha256_relative_path"],
            run_seeds=tuple(raw["run_seeds"]),
            forbidden_public_keys=tuple(raw["forbidden_public_keys"]),
            expected=raw["expected"],
            sources=tuple(PinnedFile(**item) for item in raw["sources"]),
            final_checkpoints=tuple(FinalCheckpoint(**item) for item in raw["final_checkpoints"]),
            diagnostic_statuses=tuple(DiagnosticStatus(**item) for item in raw["diagnostic_statuses"]),
        )
    except (KeyError, TypeError) as error:
        raise ModelDataError("Week 3 publication configuration is malformed") from error
    _validate(config)
    return config


def _validate(config: PublicationConfig) -> None:
    if (
        config.schema_version != 1
        or config.scope != "week_03_mlp_publication"
        or config.contract_identifier != "2026-08-26-week-03-mlp-publication-v1"
        or config.publication_scope != _PUBLICATION_SCOPE
        or config.run_seeds != (20260821, 20260822, 20260823)
        or config.output_paths != _OUTPUTS
        or not all(_safe_relative(value) for value in config.output_paths)
        or config.forbidden_public_keys != _FORBIDDEN_PUBLIC_KEYS
        or len(config.forbidden_public_keys) != len(set(config.forbidden_public_keys))
        or not isinstance(config.expected, dict)
        or set(config.expected) != _EXPECTED_KEYS
        or len(config.final_checkpoints) != 6
        or len(config.diagnostic_statuses) != 3
        or {(pin.arm, pin.seed) for pin in config.final_checkpoints}
        != {(arm, seed) for arm in ("context20", "embedding64") for seed in config.run_seeds}
        or {pin.seed for pin in config.diagnostic_statuses} != set(config.run_seeds)
        or len({pin.seed for pin in config.diagnostic_statuses}) != 3
    ):
        raise ModelDataError("Week 3 publication contract identity is invalid")
    _validate_expected(config.expected)
    paths = [pin.relative_path for pin in config.sources]
    kinds = {kind: sum(pin.kind == kind for pin in config.sources) for kind in _SOURCE_COUNTS}
    if (
        len(config.sources) != sum(_SOURCE_COUNTS.values())
        or kinds != _SOURCE_COUNTS
        or {pin.kind for pin in config.sources} != set(_SOURCE_COUNTS)
        or len(paths) != len(set(paths))
        or not all(_valid_pin(pin) for pin in config.sources)
    ):
        raise ModelDataError("Week 3 publication source pins are invalid")
    final_status_paths: list[str] = []
    checkpoint_paths: list[str] = []
    for pin in config.final_checkpoints:
        if not all(_SHA256.fullmatch(value) for value in (
            pin.status_sha256, pin.metadata_sha256, pin.tensor_sha256
        )) or not all(_safe_relative(value) for value in (
            pin.status_relative_path, pin.checkpoint_relative_path
        )):
            raise ModelDataError("Week 3 final checkpoint pins are invalid")
        final_status_paths.append(pin.status_relative_path)
        checkpoint_paths.append(pin.checkpoint_relative_path)
    diagnostic_paths = [pin.relative_path for pin in config.diagnostic_statuses]
    if not all(
        pin.seed in config.run_seeds
        and _safe_relative(pin.relative_path)
        and _SHA256.fullmatch(pin.sha256) is not None
        for pin in config.diagnostic_statuses
    ):
        raise ModelDataError("Week 3 diagnostic pins are invalid")
    if (
        len(final_status_paths) != len(set(final_status_paths))
        or len(checkpoint_paths) != len(set(checkpoint_paths))
        or len(diagnostic_paths) != len(set(diagnostic_paths))
        or set(final_status_paths) & set(checkpoint_paths)
        or set(final_status_paths) & set(diagnostic_paths)
        or set(checkpoint_paths) & set(diagnostic_paths)
        or set(paths) & (set(final_status_paths) | set(checkpoint_paths) | set(diagnostic_paths))
    ):
        raise ModelDataError("Week 3 publication paths are not unique")


def _validate_expected(expected: dict[str, object]) -> None:
    for key, value in expected.items():
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise ModelDataError("Week 3 expected values must be finite numerics")
        if key in _INTEGER_EXPECTED_KEYS:
            if type(value) is not int or value <= 0:
                raise ModelDataError("Week 3 expected integer is invalid")
        elif float(value) < 0:
            raise ModelDataError("Week 3 expected value is invalid")
    if (
        expected["native_validation_token_count"] != 1_000_495
        or expected["final_prediction_budget"] != 100_000_000
        or expected["context20_parameter_count"] != 530_293
        or expected["embedding64_parameter_count"] != 530_965
        or expected["material_gap"] <= 0
        or any(expected[key] > 1 for key in (
            "context20_mean_accuracy", "embedding64_mean_accuracy", "baseline_accuracy",
            "position_20_plus_advantage_share",
        ))
    ):
        raise ModelDataError("Week 3 expected contract values are invalid")


def _valid_pin(pin: PinnedFile) -> bool:
    return (
        isinstance(pin.kind, str) and bool(pin.kind)
        and _safe_relative(pin.relative_path)
        and _SHA256.fullmatch(pin.sha256) is not None
    )


def _safe_relative(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts
