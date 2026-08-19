"""Operator-gated publication from two pinned Week 2 neural bigram artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch

from protein_lm.bigram.candidate_contract import preflight as candidate_preflight
from protein_lm.bigram.sampling import sample_neural_bigram
from protein_lm.bigram.sampling_contract import (
    SamplingConfig,
    config_sha256,
    load_sampling_config,
)
from protein_lm.bigram.sampling_io import write_evidence
from protein_lm.bigram.sampling_source import neural_sources, validate_passed_source
from protein_lm.bigram.serialization import load_model_artifacts
from protein_lm.data.model_data.contracts import ModelDataError


_REVISION = re.compile(r"^[0-9a-f]{40}$")
_CONTEXT_ROLES = ["BOS", *"ACDEFGHIKLMNPQRSTVWY"]
_TARGET_ROLES = [*"ACDEFGHIKLMNPQRSTVWY", "EOS"]


@dataclass(frozen=True)
class SamplingPlan:
    root: Path
    config_path: Path
    config: SamplingConfig
    candidate: Path
    output_paths: tuple[Path, Path, Path]


def preflight(root: Path) -> SamplingPlan:
    """Validate local source records and bytes without opening non-neural models."""
    config_path = root / "experiments/week_02/bigram_sampling_v1.toml"
    config = load_sampling_config(config_path)
    candidate_plan = candidate_preflight(root, config.candidate_id)
    candidate = root / config.candidate_relative_path
    if candidate_plan.destination != candidate:
        raise ModelDataError("sampling candidate path drifted")
    _verify_bytes(
        candidate / "candidate_registry.json", config.candidate_registry_sha256
    )
    _verify_bytes(candidate / "run_record.json", config.candidate_run_record_sha256)
    validate_passed_source(
        _load_json(candidate / "run_record.json"),
        _load_json(candidate / "candidate_registry.json"),
        candidate_plan,
        config,
    )
    _verify_bytes(root / config.uv_lock_relative_path, config.uv_lock_sha256)
    if torch.__version__ != config.torch_version:
        raise ModelDataError("sampling Torch version drifted")
    for arm, _namespace, json_hash, safe_hash in neural_sources(config):
        _verify_bytes(candidate / f"{arm}__neural_bigram.json", json_hash)
        _verify_bytes(candidate / f"{arm}__neural_bigram.safetensors", safe_hash)
    return SamplingPlan(
        root,
        config_path,
        config,
        candidate,
        tuple(root / path for path in config.output_paths),
    )


def execute_publication(root: Path, plan: SamplingPlan) -> tuple[Path, Path, Path]:
    if not isinstance(plan, SamplingPlan) or plan.root != root:
        raise ModelDataError("sampling plan root is invalid")
    if preflight(root) != plan:
        raise ModelDataError("sampling plan or source candidate drifted")
    if any(path.exists() for path in plan.output_paths):
        raise ModelDataError("sampling diagnostic report already exists")
    write_evidence(plan.output_paths, _payload(plan, _clean_revision(root)))
    return plan.output_paths


def regenerate_payload(
    plan: SamplingPlan, publication_code_revision: str
) -> dict[str, object]:
    """Independently rebuild the deterministic payload from the two pinned logits."""
    if _REVISION.fullmatch(publication_code_revision) is None:
        raise ModelDataError("sampling publication code revision is invalid")
    return _payload(plan, publication_code_revision)


def _payload(plan: SamplingPlan, revision: str) -> dict[str, object]:
    samples: list[dict[str, object]] = []
    for arm, namespace, _json_hash, _safe_hash in neural_sources(plan.config):
        model_type, logits, metadata = load_model_artifacts(
            json_path=plan.candidate / f"{arm}__neural_bigram.json",
            safetensors_path=plan.candidate / f"{arm}__neural_bigram.safetensors",
        )
        _validate_neural_metadata(arm, model_type, metadata, plan.config)
        for index in range(plan.config.samples_per_model):
            samples.append(
                {
                    "model_arm": arm,
                    "namespace": namespace,
                    **sample_neural_bigram(
                        logits,
                        base_seed=plan.config.base_seed,
                        namespace=namespace,
                        sample_index=index,
                        max_residues=plan.config.max_residues,
                    ),
                }
            )
    return {
        "schema_version": 1,
        "scope": "week_02_bigram_sampling_diagnostic",
        "contract_identifier": plan.config.contract_identifier,
        "status": "passed",
        "hard_gates": {
            "validated_passed_candidate": True,
            "two_neural_artifacts_only": True,
            "twenty_samples_preserved": True,
            "synthetic_nonfunctional_educational_output": True,
            "no_selection_or_biological_claims": True,
            "no_network_requests": True,
        },
        "sampling_configuration_sha256": config_sha256(plan.config_path),
        "publication_code_revision": revision,
        "runtime": {
            "uv_lock_sha256": plan.config.uv_lock_sha256,
            "torch_version": plan.config.torch_version,
        },
        "source": {
            "candidate_id": plan.config.candidate_id,
            "relative_path": plan.config.candidate_relative_path,
            "candidate_registry_sha256": plan.config.candidate_registry_sha256,
            "candidate_run_record_sha256": plan.config.candidate_run_record_sha256,
            "candidate_code_revision": plan.config.candidate_code_revision,
            "neural_artifacts": {
                arm: {"json_sha256": json_hash, "safetensors_sha256": safe_hash}
                for arm, _namespace, json_hash, safe_hash in neural_sources(plan.config)
            },
        },
        "sampling": {
            "base_seed": plan.config.base_seed,
            "seed_derivation": "sha256_utf8_v1_nul_base_seed_nul_namespace_nul_sample_index_first_8_bytes_big_endian_mask_63_bits",
            "temperature": plan.config.temperature,
            "top_k": plan.config.top_k,
            "top_p": plan.config.top_p,
            "max_residues": plan.config.max_residues,
            "start_context": plan.config.start_context,
            "termination": plan.config.termination,
            "samples_per_model": plan.config.samples_per_model,
        },
        "samples": samples,
        "network_requests_made": 0,
    }


def _validate_neural_metadata(
    arm: str, model_type: str, metadata: dict[str, object], config: SamplingConfig
) -> None:
    if (
        model_type != "neural_bigram"
        or metadata.get("arm") != arm
        or metadata.get("model_type") != "neural_bigram"
        or metadata.get("code_revision") != config.candidate_code_revision
        or metadata.get("seed") != config.base_seed
        or metadata.get("context_roles") != _CONTEXT_ROLES
        or metadata.get("target_roles") != _TARGET_ROLES
    ):
        raise ModelDataError("sampling neural artifact metadata drifted")


def _verify_bytes(path: Path, expected: str) -> None:
    try:
        found = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ModelDataError("sampling source artifact is unavailable") from error
    if found != expected:
        raise ModelDataError("sampling source artifact hash drifted")


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelDataError("sampling source record is malformed") from error
    if not isinstance(value, dict):
        raise ModelDataError("sampling source record is malformed")
    return value


def _clean_revision(root: Path) -> str:
    try:
        if subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout:
            raise ModelDataError(
                "sampling publication requires a clean committed revision"
            )
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.SubprocessError as error:
        raise ModelDataError(
            "could not establish sampling publication code revision"
        ) from error
    if _REVISION.fullmatch(revision) is None:
        raise ModelDataError("sampling publication code revision is invalid")
    return revision
