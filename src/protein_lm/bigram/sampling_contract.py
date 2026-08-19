"""Byte-pinned contract for one educational Week 2 sampling diagnostic."""

from __future__ import annotations

import hashlib
import re
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from protein_lm.data.model_data.contracts import ModelDataError


APPROVED_SAMPLING_CONFIG_SHA256 = (
    "6274acfa7407670b2c98e2b05a2b8089e24bc13029fc7d2fd6154d7c166f6304"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class SamplingConfig:
    schema_version: int
    scope: str
    contract_identifier: str
    candidate_id: str
    candidate_relative_path: str
    candidate_registry_sha256: str
    candidate_run_record_sha256: str
    candidate_code_revision: str
    uv_lock_relative_path: str
    uv_lock_sha256: str
    torch_version: str
    random_neural_json_sha256: str
    random_neural_safetensors_sha256: str
    family_aware_neural_json_sha256: str
    family_aware_neural_safetensors_sha256: str
    base_seed: int
    arms: tuple[str, ...]
    namespaces: tuple[str, ...]
    samples_per_model: int
    temperature: float
    top_k: str
    top_p: str
    max_residues: int
    start_context: str
    termination: str
    network_requests_made: int
    selection: str
    biological_claims: str
    report_json_relative_path: str
    report_markdown_relative_path: str
    report_sha256_relative_path: str

    @property
    def output_paths(self) -> tuple[str, str, str]:
        return (
            self.report_json_relative_path,
            self.report_markdown_relative_path,
            self.report_sha256_relative_path,
        )


def config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sampling_config(path: Path) -> SamplingConfig:
    try:
        content = path.read_bytes()
        raw = tomllib.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ModelDataError(
            f"could not load sampling configuration: {error}"
        ) from error
    if hashlib.sha256(content).hexdigest() != APPROVED_SAMPLING_CONFIG_SHA256:
        raise ModelDataError("sampling configuration bytes do not match approval")
    if not isinstance(raw, dict) or set(raw) != {
        field.name for field in fields(SamplingConfig)
    }:
        raise ModelDataError("sampling configuration keys differ from the schema")
    values = dict(raw)
    for name in ("arms", "namespaces"):
        if not isinstance(values[name], list) or not all(
            isinstance(v, str) and v for v in values[name]
        ):
            raise ModelDataError(f"{name} must be a list of nonempty strings")
        values[name] = tuple(values[name])
    for name in (
        "schema_version",
        "base_seed",
        "samples_per_model",
        "max_residues",
        "network_requests_made",
    ):
        if type(values[name]) is not int:
            raise ModelDataError("sampling integer values are invalid")
    if type(values["temperature"]) not in (int, float):
        raise ModelDataError("sampling temperature is invalid")
    config = SamplingConfig(**values)
    _validate(config)
    return config


def _validate(config: SamplingConfig) -> None:
    expected = (
        config.schema_version == 1
        and config.scope == "week_02_bigram_sampling_diagnostic"
        and config.contract_identifier == "2026-08-19-week-02-bigram-sampling-v1"
        and config.candidate_id == "week2-bigram-v1-001"
        and config.candidate_relative_path
        == "data/processed/week_02/bigram_model_candidates/week2-bigram-v1-001"
        and config.candidate_code_revision == "c661315641049ef9f8d3b372f75ac34c86ad67e2"
        and config.uv_lock_relative_path == "uv.lock"
        and config.torch_version == "2.13.0"
        and config.arms == ("random_training", "family_aware_training")
        and config.namespaces
        == ("week2/sampling/random/v1", "week2/sampling/family-aware/v1")
        and config.base_seed == 20260812
        and config.samples_per_model == 10
        and config.temperature == 1.0
        and config.top_k == "none"
        and config.top_p == "none"
        and config.max_residues == 128
        and config.start_context == "BOS"
        and config.termination == "EOS_or_max_residues"
        and config.network_requests_made == 0
        and config.selection == "forbidden"
        and config.biological_claims == "forbidden"
        and config.output_paths
        == (
            "reports/week_02/bigram_sampling_v1.json",
            "reports/week_02/bigram_sampling_v1.md",
            "reports/week_02/bigram_sampling_v1.sha256",
        )
    )
    hashes = (
        config.candidate_registry_sha256,
        config.candidate_run_record_sha256,
        config.uv_lock_sha256,
        config.random_neural_json_sha256,
        config.random_neural_safetensors_sha256,
        config.family_aware_neural_json_sha256,
        config.family_aware_neural_safetensors_sha256,
    )
    if (
        not expected
        or not all(_SHA256.fullmatch(value) for value in hashes)
        or not _REVISION.fullmatch(config.candidate_code_revision)
    ):
        raise ModelDataError("sampling contract identity is not approved")
