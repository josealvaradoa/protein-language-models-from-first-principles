"""Offline, synthetic-only ESMC inference smoke orchestration."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

import torch

from protein_lm.benchmarks.metrics import (
    current_mps_memory,
    error_details,
    read_swap_state,
    synchronize,
)
from protein_lm.external.esmc_contract import (
    ContractValidationError,
    ESMCContract,
    validate_local_model_dir,
    validate_runtime_config,
    validate_runtime_tokenizer,
)
from protein_lm.external.esmc_pooling import (
    build_residue_mask,
    mask_one_residue_per_fixture,
    padding_aware_mean_pool,
    padding_poison_invariant,
    validate_masked_residue_counts,
)
from protein_lm.external.esmc_provenance import validate_installed_package_provenance
from protein_lm.external.esmc_result import (
    create_result,
    finish_result,
)


Loader = Callable[[Path], tuple[Any, Any]]


def load_local_transformers(model_dir: Path) -> tuple[Any, Any]:
    """Lazily import optional dependencies and load only from the local directory."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError as exception:
        raise RuntimeError(
            "ESMC dependencies are missing; install the optional esmc group first"
        ) from exception
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForMaskedLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
    )
    return tokenizer, model


def run_esmc_smoke(
    contract: ESMCContract,
    *,
    model_dir: Path,
    device: str,
    project_root: Path,
    loader: Loader = load_local_transformers,
    package_provenance_validator: Callable[[ESMCContract], dict[str, object]] = (
        validate_installed_package_provenance
    ),
) -> dict[str, object]:
    """Run the fixed smoke paths and return a serializable success or failure record."""
    requested_device = torch.device(device)
    swap_before = read_swap_state()
    started_at = time.perf_counter()
    result = create_result(contract, requested_device, project_root, swap_before)
    maximum_allocated: int | None = None
    maximum_driver: int | None = None

    try:
        _require_explicit_device(requested_device)
        local_provenance = validate_local_model_dir(model_dir, contract)
        result["local_weight_sha256"] = local_provenance["weight_sha256"]
        result["validated_local_config"] = local_provenance["config"]
        result["installed_packages"] = package_provenance_validator(contract)

        tokenizer, model = loader(model_dir)
        validate_runtime_tokenizer(tokenizer, contract.expected_config)
        result["validated_runtime_config"] = validate_runtime_config(
            model.config, contract.expected_config
        )
        model = model.to(requested_device, dtype=torch.float32)
        model.eval()
        result["parameter_count"] = sum(
            parameter.numel() for parameter in model.parameters()
        )
        maximum_allocated, maximum_driver = _sample_memory(
            requested_device, maximum_allocated, maximum_driver
        )
        _run_inference_paths(result, contract, tokenizer, model, requested_device)
        synchronize(requested_device)
        maximum_allocated, maximum_driver = _sample_memory(
            requested_device, maximum_allocated, maximum_driver
        )
        _validate_output_contract(result, contract)
    except Exception as exception:  # Preserve each preflight or runtime failure.
        result["status"] = "failed"
        result["error"] = error_details(exception)

    finish_result(
        result,
        contract,
        requested_device,
        started_at=started_at,
        maximum_allocated=maximum_allocated,
        maximum_driver=maximum_driver,
        swap_before=swap_before,
        swap_after=read_swap_state(),
    )
    return result


def _run_inference_paths(
    result: dict[str, object],
    contract: ESMCContract,
    tokenizer: Any,
    model: Any,
    device: torch.device,
) -> None:
    encoded = tokenizer(
        [fixture.sequence for fixture in contract.fixtures],
        padding=True,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    special_tokens_mask = encoded["special_tokens_mask"].to(device)
    residue_mask = build_residue_mask(attention_mask, special_tokens_mask)
    residue_counts = [int(value) for value in residue_mask.sum(dim=1).tolist()]
    if residue_counts != [len(fixture.sequence) for fixture in contract.fixtures]:
        raise ContractValidationError(
            "attention and special-token masks do not preserve fixture residue counts"
        )

    with torch.inference_mode():
        unmasked = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = _final_hidden_states(unmasked)
        pooled = padding_aware_mean_pool(hidden_states, residue_mask)
        masked_input_ids = mask_one_residue_per_fixture(
            input_ids, residue_mask, contract.fixtures, tokenizer.mask_token_id
        )
        masked = model(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = masked.logits

    result.update(
        {
            "unmasked_hidden_state_shape": list(hidden_states.shape),
            "pooled_embedding_shape": list(pooled.shape),
            "masked_mlm_logit_shape": list(logits.shape),
            "residue_counts": residue_counts,
            "masked_residue_counts": validate_masked_residue_counts(
                input_ids, masked_input_ids, residue_mask
            ),
            "finite_outputs": bool(
                torch.isfinite(hidden_states).all()
                and torch.isfinite(pooled).all()
                and torch.isfinite(logits).all()
            ),
            "padding_poison_invariant": padding_poison_invariant(
                hidden_states, attention_mask, residue_mask
            ),
        }
    )


def _require_explicit_device(device: torch.device) -> None:
    if device.type not in {"mps", "cpu"}:
        raise ValueError("device must be mps or cpu")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available; CPU fallback is prohibited")


def _sample_memory(
    device: torch.device,
    maximum_allocated: int | None,
    maximum_driver: int | None,
) -> tuple[int | None, int | None]:
    allocated, driver = current_mps_memory(device)
    return (
        max(maximum_allocated or 0, allocated) if allocated is not None else maximum_allocated,
        max(maximum_driver or 0, driver) if driver is not None else maximum_driver,
    )


def _final_hidden_states(outputs: Any) -> torch.Tensor:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states:
        return hidden_states[-1]
    last_hidden_state = getattr(outputs, "last_hidden_state", None)
    if last_hidden_state is None:
        raise ContractValidationError("model output does not provide final hidden states")
    return last_hidden_state


def _validate_output_contract(result: dict[str, object], contract: ESMCContract) -> None:
    expected = contract.expected_shapes
    for result_field, expected_field in (
        ("unmasked_hidden_state_shape", "hidden_states"),
        ("pooled_embedding_shape", "pooled_embeddings"),
        ("masked_mlm_logit_shape", "mlm_logits"),
    ):
        if result[result_field] != expected[expected_field]:
            raise ContractValidationError(f"{result_field} differs from the contract")
    if result["finite_outputs"] is not True:
        raise ContractValidationError("model outputs are non-finite")
    if result["padding_poison_invariant"] is not True:
        raise ContractValidationError("padding poison changed pooled outputs")
