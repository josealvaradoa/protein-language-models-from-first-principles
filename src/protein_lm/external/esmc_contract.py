"""Read and validate the public, local-only ESMC smoke contract."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CANONICAL_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")
REQUIRED_MODEL_FILES = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
)


class ContractValidationError(ValueError):
    """The local model or declared smoke contract violates a frozen pin."""


@dataclass(frozen=True)
class SyntheticFixture:
    """One harmless, fixed sequence and its sole masked residue coordinate."""

    identifier: str
    sequence: str
    mask_residue_index: int


@dataclass(frozen=True)
class ExpectedConfig:
    """Exact raw config fields from the pinned model artifact."""

    architectures: tuple[str, ...]
    model_type: str
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    mask_token_id: int
    pad_token_id: int
    dtype: str
    transformers_version: str

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "ExpectedConfig":
        try:
            architectures = raw["architectures"]
            if not isinstance(architectures, list) or not all(
                isinstance(value, str) for value in architectures
            ):
                raise TypeError("architectures must be a list of strings")
            return cls(
                architectures=tuple(architectures),
                model_type=str(raw["model_type"]),
                vocab_size=int(raw["vocab_size"]),
                d_model=int(raw["d_model"]),
                n_layers=int(raw["n_layers"]),
                n_heads=int(raw["n_heads"]),
                mask_token_id=int(raw["mask_token_id"]),
                pad_token_id=int(raw["pad_token_id"]),
                dtype=str(raw["dtype"]),
                transformers_version=str(raw["transformers_version"]),
            )
        except (KeyError, TypeError, ValueError) as exception:
            raise ContractValidationError("invalid expected_config declaration") from exception

    def raw_fields(self) -> dict[str, object]:
        return {
            "architectures": list(self.architectures),
            "model_type": self.model_type,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "mask_token_id": self.mask_token_id,
            "pad_token_id": self.pad_token_id,
            "dtype": self.dtype,
            "transformers_version": self.transformers_version,
        }


@dataclass(frozen=True)
class ESMCContract:
    """The complete public contract needed before optional model loading."""

    identifier: str
    model_id: str
    model_revision: str
    published_context_limit: int
    code_repository: str
    code_revision: str
    code_version: str
    transformers_repository: str
    transformers_revision: str
    weight_filename: str
    weight_sha256: str
    expected_reported_weight_size: str
    expected_config: ExpectedConfig
    expected_shapes: dict[str, int | list[int]]
    runtime_limit_seconds: int
    mps_driver_memory_limit_bytes: int
    fixtures: tuple[SyntheticFixture, ...]

    def pins(self) -> dict[str, object]:
        """Return serializable provenance without exposing a local cache path."""
        return {
            "contract_identifier": self.identifier,
            "model": {
                "id": self.model_id,
                "revision": self.model_revision,
                "published_context_limit": self.published_context_limit,
            },
            "code": {
                "repository": self.code_repository,
                "revision": self.code_revision,
                "version": self.code_version,
            },
            "dependencies": {
                "transformers_repository": self.transformers_repository,
                "transformers_revision": self.transformers_revision,
            },
            "weights": {
                "filename": self.weight_filename,
                "expected_sha256": self.weight_sha256,
                "expected_reported_size": self.expected_reported_weight_size,
            },
            "expected_config": self.expected_config.raw_fields(),
            "fixture_ids": [fixture.identifier for fixture in self.fixtures],
            "fixture_lengths": [len(fixture.sequence) for fixture in self.fixtures],
            "mask_residue_indices": [
                fixture.mask_residue_index for fixture in self.fixtures
            ],
        }


def load_esmc_contract(path: Path) -> ESMCContract:
    """Load the TOML contract and reject altered fixture or model declarations."""
    with path.open("rb") as contract_file:
        raw = tomllib.load(contract_file)

    fixtures = tuple(
        SyntheticFixture(
            identifier=str(item["identifier"]),
            sequence=str(item["sequence"]),
            mask_residue_index=int(item["mask_residue_index"]),
        )
        for item in raw["fixtures"]
    )
    contract = ESMCContract(
        identifier=str(raw["contract"]["identifier"]),
        model_id=str(raw["model"]["id"]),
        model_revision=str(raw["model"]["revision"]),
        published_context_limit=int(raw["model"]["published_context_limit"]),
        code_repository=str(raw["code"]["repository"]),
        code_revision=str(raw["code"]["revision"]),
        code_version=str(raw["code"]["version"]),
        transformers_repository=str(raw["dependencies"]["transformers_repository"]),
        transformers_revision=str(raw["dependencies"]["transformers_revision"]),
        weight_filename=str(raw["weights"]["filename"]),
        weight_sha256=str(raw["weights"]["sha256"]),
        expected_reported_weight_size=str(
            raw["weights"]["expected_reported_weight_size"]
        ),
        expected_config=ExpectedConfig.from_raw(raw["expected_config"]),
        expected_shapes={
            key: list(value) if isinstance(value, list) else int(value)
            for key, value in raw["expected_shapes"].items()
        },
        runtime_limit_seconds=int(raw["contract"]["runtime_limit_seconds"]),
        mps_driver_memory_limit_bytes=int(
            raw["contract"]["mps_driver_memory_limit_bytes"]
        ),
        fixtures=fixtures,
    )
    _validate_contract(contract)
    return contract


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file hash without reading a model weight file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        while chunk := source_file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_local_model_dir(model_dir: Path, contract: ESMCContract) -> dict[str, Any]:
    """Validate config, tokenizer files, and the local weight hash before load."""
    if not model_dir.is_dir():
        raise ContractValidationError(f"model directory does not exist: {model_dir}")
    for filename in REQUIRED_MODEL_FILES:
        file_path = model_dir / filename
        if not file_path.is_file():
            raise ContractValidationError(f"required local model file is missing: {filename}")

    config_path = model_dir / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exception:
        raise ContractValidationError("config.json is not valid JSON") from exception
    validate_expected_config(config, contract.expected_config)

    weight_path = model_dir / contract.weight_filename
    if not weight_path.is_file():
        raise ContractValidationError(
            f"required local model file is missing: {contract.weight_filename}"
        )
    actual_sha256 = sha256_file(weight_path)
    if actual_sha256 != contract.weight_sha256:
        raise ContractValidationError(
            "model.safetensors SHA-256 does not match the frozen contract"
        )
    return {"config": config, "weight_sha256": actual_sha256}


def validate_expected_config(config: dict[str, Any], expected_config: ExpectedConfig) -> None:
    """Validate the exact fields known to exist in the pinned raw config JSON."""
    for field, expected_value in expected_config.raw_fields().items():
        actual = config.get(field)
        if field == "architectures":
            if isinstance(actual, list) and set(expected_value).issubset(actual):
                continue
            raise ContractValidationError(
                f"expected architectures to include {expected_value!r}, found {actual!r}"
            )
        if actual != expected_value:
            raise ContractValidationError(
                f"expected {field}={expected_value!r}, found {actual!r}"
            )


def validate_runtime_config(
    config: Any,
    expected_config: ExpectedConfig,
) -> dict[str, object]:
    """Verify runtime identity only for fields exposed by Transformers config objects."""
    runtime_fields = (
        "architectures",
        "model_type",
        "vocab_size",
        "d_model",
        "n_layers",
        "n_heads",
        "mask_token_id",
        "pad_token_id",
    )
    observed = {field: getattr(config, field, None) for field in runtime_fields}
    for field, actual in observed.items():
        expected_value = expected_config.raw_fields()[field]
        if field == "architectures":
            if isinstance(actual, (list, tuple)) and set(expected_value).issubset(actual):
                continue
            raise ContractValidationError(
                f"runtime config architectures omit {expected_value!r}: {actual!r}"
            )
        if actual != expected_value:
            raise ContractValidationError(
                f"runtime config expected {field}={expected_value!r}, "
                f"found {actual!r}"
            )
    return observed


def validate_runtime_tokenizer(tokenizer: Any, expected_config: ExpectedConfig) -> None:
    """Reject a tokenizer whose special-token identities differ from the model pin."""
    expected_tokens = {
        "mask_token_id": expected_config.mask_token_id,
        "pad_token_id": expected_config.pad_token_id,
    }
    for field, expected_value in expected_tokens.items():
        actual = getattr(tokenizer, field, None)
        if actual != expected_value:
            raise ContractValidationError(
                f"tokenizer expected {field}={expected_value}, found {actual!r}"
            )


def _validate_contract(contract: ESMCContract) -> None:
    if len(contract.fixtures) != 2:
        raise ContractValidationError("the smoke contract must declare exactly two fixtures")
    expected_lengths = (32, 64)
    fixture_ids = [fixture.identifier for fixture in contract.fixtures]
    if len(set(fixture_ids)) != len(fixture_ids):
        raise ContractValidationError("synthetic fixture identifiers must be unique")
    for fixture, expected_length in zip(contract.fixtures, expected_lengths, strict=True):
        if len(fixture.sequence) != expected_length:
            raise ContractValidationError(
                f"{fixture.identifier} must have length {expected_length}"
            )
        if set(fixture.sequence) - CANONICAL_RESIDUES:
            raise ContractValidationError(
                f"{fixture.identifier} contains a noncanonical residue"
            )
        if not 0 <= fixture.mask_residue_index < len(fixture.sequence):
            raise ContractValidationError(
                f"{fixture.identifier} mask coordinate is outside its sequence"
            )
