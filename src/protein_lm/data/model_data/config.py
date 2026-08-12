"""Strict loader for the frozen Week 2 model-data configuration."""

from __future__ import annotations

import hashlib
import re
import tomllib
from dataclasses import fields
from pathlib import Path

from protein_lm.data.model_data.contracts import (
    LengthBucket,
    ModelDataConfig,
    ModelDataError,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
APPROVED_MODEL_DATA_CONFIG_SHA256 = (
    "b35ec4003b002a065c29e3c70ee72ff115edafc6645f9370603e7020b4a05f12"
)
_NAMESPACES = (
    "week2/shared-validation/v1",
    "week2/shared-sealed-test/v1",
    "week2/random-native-validation/v1",
    "week2/family-native-validation/v1",
    "week2/training-stream/random/v1",
    "week2/training-stream/family-aware/v1",
    "week2/sampling/random/v1",
    "week2/sampling/family-aware/v1",
)


def config_sha256(path: Path) -> str:
    """Return the byte identity of a configuration file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path) -> ModelDataConfig:
    """Decode the complete schema and reject unknown or unsafe settings."""

    try:
        content = path.read_bytes()
        raw = tomllib.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ModelDataError(
            f"could not load model-data configuration: {error}"
        ) from error
    if hashlib.sha256(content).hexdigest() != APPROVED_MODEL_DATA_CONFIG_SHA256:
        raise ModelDataError("model-data configuration bytes do not match approval")
    expected = {field.name for field in fields(ModelDataConfig)}
    if set(raw) != expected:
        raise ModelDataError("model-data configuration keys differ from the schema")
    buckets = _buckets(raw.get("length_buckets"))
    config = ModelDataConfig(
        **{
            field.name: _value(raw[field.name], field.name, field.type, buckets)
            for field in fields(ModelDataConfig)
        }
    )
    _validate(config)
    return config


def _value(
    value: object, name: str, annotation: object, buckets: tuple[LengthBucket, ...]
) -> object:
    annotation_name = str(annotation)
    if name == "length_buckets":
        return buckets
    if annotation_name == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ModelDataError(f"{name} must be an integer")
        return value
    if annotation_name == "str":
        if not isinstance(value, str) or not value:
            raise ModelDataError(f"{name} must be a nonempty string")
        return value
    if name == "allocation_namespaces":
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item for item in value
        ):
            raise ModelDataError("allocation_namespaces must be nonempty strings")
        return tuple(value)
    raise ModelDataError(f"unsupported configuration field: {name}")


def _buckets(raw: object) -> tuple[LengthBucket, ...]:
    if not isinstance(raw, list) or not raw:
        raise ModelDataError("length_buckets must be a nonempty array")
    buckets = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"name", "minimum", "maximum"}:
            raise ModelDataError("length bucket schema is invalid")
        name, minimum, maximum = item.values()
        if (
            not isinstance(name, str)
            or not name
            or isinstance(minimum, bool)
            or not isinstance(minimum, int)
            or isinstance(maximum, bool)
            or not isinstance(maximum, int)
        ):
            raise ModelDataError("length bucket values are invalid")
        buckets.append(LengthBucket(name, minimum, maximum))
    return tuple(buckets)


def _validate(config: ModelDataConfig) -> None:
    for name in (
        "task4_catalog_sha256",
        "reserved_families_sha256",
        "task4_report_sha256",
        "task4_eligibility_policy_sha256",
    ):
        if _SHA256.fullmatch(getattr(config, name)) is None:
            raise ModelDataError(f"{name} must be a lowercase SHA-256")
    for name in (
        "task4_catalog_relative_path",
        "reserved_families_relative_path",
        "task4_report_relative_path",
        "task4_eligibility_policy_relative_path",
        "candidate_directory_relative_path",
        "readiness_json_relative_path",
        "readiness_markdown_relative_path",
        "readiness_sha256_relative_path",
    ):
        _safe_relative_path(getattr(config, name), name)
    if config.schema_version != 1 or config.scope != "week_02_model_data_candidate":
        raise ModelDataError("model-data schema version or scope is not approved")
    if config.candidate_identifier != "v1" or config.source_release != "2026_02":
        raise ModelDataError("candidate identifier or source release is not approved")
    if config.proteingym_release != "v1.3" or config.license_spdx != "CC-BY-4.0":
        raise ModelDataError("release or license is not approved")
    if (
        config.canonical_amino_acids != "ACDEFGHIKLMNPQRSTVWY"
        or config.sequence_hash != "sha256"
    ):
        raise ModelDataError("sequence contract is not approved")
    if (config.minimum_length, config.maximum_length) != (32, 2046):
        raise ModelDataError("length contract is not approved")
    if config.hash_algorithm != "sha256" or config.base_seed != 20260812:
        raise ModelDataError("hash algorithm or seed is not approved")
    if config.allocation_namespaces != _NAMESPACES:
        raise ModelDataError("allocation namespaces are not approved")
    if (
        config.mmseqs2_status != "diagnostic_only"
        or config.model_use != "candidate_pending_readiness"
    ):
        raise ModelDataError("execution boundary is not approved")
    if not (
        0
        < config.minimum_evaluation_predictions
        <= config.prediction_token_target
        <= config.maximum_evaluation_predictions
    ):
        raise ModelDataError("evaluation token limits are contradictory")
    if config.minimum_bucket_predictions <= 0:
        raise ModelDataError("minimum bucket predictions must be positive")
    if (
        config.task4_catalog_byte_size <= 0
        or config.task4_catalog_row_count <= 0
        or config.reserved_family_count <= 0
    ):
        raise ModelDataError("frozen input sizes must be positive")
    ordered = sorted(config.length_buckets, key=lambda bucket: bucket.minimum)
    if (
        tuple(ordered) != config.length_buckets
        or ordered[0].minimum != config.minimum_length
        or ordered[-1].maximum != config.maximum_length
    ):
        raise ModelDataError("length buckets do not cover the approved range")
    if any(bucket.minimum > bucket.maximum for bucket in ordered) or any(
        left.maximum + 1 != right.minimum for left, right in zip(ordered, ordered[1:])
    ):
        raise ModelDataError("length buckets must be contiguous and non-overlapping")


def _safe_relative_path(value: str, name: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ModelDataError(f"{name} must be a safe relative path")
