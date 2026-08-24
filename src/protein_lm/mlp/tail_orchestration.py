"""One-pass, non-resumable Week 3 exploratory learning-rate tails."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import save_file

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.data.model_data.loaders import (
    ModelDataCollection,
    ProteinSequence,
    load_collection,
)
from protein_lm.mlp.checkpoint import load_checkpoint
from protein_lm.mlp.config import MLPTrainingConfig, config_sha256, load_config
from protein_lm.mlp.metrics import NativeMetrics, evaluate_native
from protein_lm.mlp.model import ContextMLP, resolve_device
from protein_lm.mlp.orchestration import (
    _require_ignored,
    _require_revision,
    _verify_source_pins,
)
from protein_lm.mlp.stream import (
    StreamCursor,
    iter_context_batches,
    iter_native_context_batches,
)
from protein_lm.mlp.tail_config import (
    MLPTailConfig,
    config_sha256 as tail_config_sha256,
    load_tail_config,
)
from protein_lm.mlp.tail_training import (
    schedule_provenance,
    train_tail_batch,
)
from protein_lm.mlp.training import TrainingState, new_optimizer


_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")
CollectionLoader = Callable[[Path, ModelDataCollection], Iterable[ProteinSequence]]
ProgressCallback = Callable[[str, dict[str, object]], None]


@dataclass(frozen=True)
class MLPTailPlan:
    config: MLPTailConfig
    config_path: Path
    base_config: MLPTrainingConfig
    base_config_path: Path
    run_id: str
    arm: str
    destination: Path


def preflight(root: Path, run_id: str, arm: str) -> MLPTailPlan:
    """Read approved config bytes only. This deliberately touches no run state."""

    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ModelDataError(
            "run identifier must be 3-64 lowercase letters, digits, or hyphens"
        )
    config_path = root / "experiments/week_03/mlp_lr_tail_v1.toml"
    config = load_tail_config(config_path)
    if arm not in config.approved_arms:
        raise ModelDataError("tail arm is not approved")
    base_config_path = root / config.base_config_relative_path
    base_config = load_config(base_config_path)
    if (
        config_sha256(base_config_path) != config.base_config_sha256
        or base_config.contract_identifier != config.base_contract_identifier
        or base_config.batch_size != config.batch_size
        or base_config.prediction_budget != config.final_prediction_position
    ):
        raise ModelDataError("tail base configuration identity is invalid")
    return MLPTailPlan(
        config=config,
        config_path=config_path,
        base_config=base_config,
        base_config_path=base_config_path,
        run_id=run_id,
        arm=arm,
        destination=root / config.output_relative_root / run_id,
    )


def execute_tail(
    *,
    root: Path,
    plan: MLPTailPlan,
    seed: int,
    device_name: str,
    loader: CollectionLoader = load_collection,
    code_revision: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Train one approved arm from its byte-pinned 90M checkpoint exactly once."""

    if device_name != "cpu":
        raise ModelDataError("MLP learning-rate tails require explicit CPU execution")
    if plan.arm not in plan.config.approved_arms:
        raise ModelDataError("tail arm is not approved")
    parent_pin = plan.config.parent_pin(seed)
    if plan.destination.exists():
        raise ModelDataError("tail run destination already exists; tails cannot resume")
    derived_revision = _require_revision(root, code_revision)
    _verify_source_pins(root, plan.base_config)
    _require_ignored(root, plan.destination)
    plan.destination.mkdir(parents=True)
    status_path = plan.destination / "run_status.json"
    started = time.perf_counter()
    parent_path = _parent_checkpoint_path(
        root, plan.base_config, parent_pin.run_id, plan.config
    )
    state: TrainingState | None = None
    metric: NativeMetrics | None = None
    model_sha256: str | None = None
    try:
        _write_status(
            status_path,
            _status_payload(
                plan,
                seed,
                parent_pin.run_id,
                parent_path,
                derived_revision,
                "running",
                state,
                metric,
                None,
                None,
                started,
            ),
        )
        _verify_parent_bytes(
            parent_path, parent_pin.metadata_sha256, parent_pin.tensor_sha256
        )
        device = resolve_device("cpu")
        model = ContextMLP(plan.base_config, seed, device)
        optimizer = new_optimizer(model, plan.base_config)
        state = load_checkpoint(
            parent_path,
            model=model,
            optimizer=optimizer,
            config=plan.base_config,
            config_path=plan.base_config_path,
            seed=seed,
            run_id=parent_pin.run_id,
            device_name="cpu",
            code_revision=plan.config.parent_code_revision,
        )
        _validate_parent_state(state, plan.config)
        # The inherited numerator belongs to the primary run. This field is
        # deliberately reset so every saved tail loss is clearly online-tail only.
        state.training_loss_numerator = 0.0
        training = loader(root, ModelDataCollection.FAMILY_AWARE_TRAINING)
        for batch in iter_context_batches(
            training,
            namespace=plan.base_config.training_namespace,
            base_seed=plan.base_config.stream_base_seed,
            prediction_budget=plan.config.final_prediction_position,
            batch_size=plan.config.batch_size,
            event_predictions=(plan.config.final_prediction_position,),
            cursor=state.cursor,
            context_length=plan.base_config.context_length,
        ):
            train_tail_batch(model, optimizer, batch, state, plan.config, plan.arm)
        _validate_final_state(state, plan.config)
        native_validation = loader(
            root, ModelDataCollection.FAMILY_AWARE_NATIVE_VALIDATION
        )
        metric = evaluate_native(
            model,
            iter_native_context_batches(
                native_validation,
                namespace=plan.base_config.training_namespace,
                base_seed=plan.base_config.stream_base_seed,
                batch_size=plan.config.batch_size,
                context_length=plan.base_config.context_length,
            ),
        )
        model_sha256 = _save_final_model(
            plan.destination / "final_model.safetensors", model
        )
        _verify_parent_bytes(
            parent_path, parent_pin.metadata_sha256, parent_pin.tensor_sha256
        )
        _write_status(
            status_path,
            _status_payload(
                plan,
                seed,
                parent_pin.run_id,
                parent_path,
                derived_revision,
                "passed",
                state,
                metric,
                model_sha256,
                None,
                started,
            ),
        )
        _notify(
            progress_callback,
            "completed",
            _completion_payload(plan, seed, state, metric, started),
        )
    except Exception as error:
        _write_status(
            status_path,
            _status_payload(
                plan,
                seed,
                parent_pin.run_id,
                parent_path,
                derived_revision,
                "failed",
                state,
                metric,
                model_sha256,
                str(error),
                started,
            ),
        )
        if isinstance(error, ModelDataError):
            raise
        raise ModelDataError(f"MLP learning-rate tail failed: {error}") from error
    return plan.destination


def _parent_checkpoint_path(
    root: Path,
    base_config: MLPTrainingConfig,
    parent_run_id: str,
    config: MLPTailConfig,
) -> Path:
    return (
        root
        / base_config.output_relative_root
        / parent_run_id
        / f"checkpoint-{config.parent_prediction_position}"
    )


def _verify_parent_bytes(path: Path, metadata_sha256: str, tensor_sha256: str) -> None:
    """Hash both immutable parent files before their tensors are deserialized."""

    try:
        metadata = hashlib.sha256((path / "checkpoint.json").read_bytes()).hexdigest()
        tensors = hashlib.sha256((path / "model.safetensors").read_bytes()).hexdigest()
    except OSError as error:
        raise ModelDataError("approved parent checkpoint is unavailable") from error
    if metadata != metadata_sha256 or tensors != tensor_sha256:
        raise ModelDataError("approved parent checkpoint bytes do not match pin")


def _validate_parent_state(state: TrainingState, config: MLPTailConfig) -> None:
    cursor = StreamCursor(
        config.parent_cursor_prediction_index,
        config.parent_cursor_protein_index,
        config.parent_cursor_within_protein_target_offset,
    )
    if (
        state.predictions_seen != config.parent_prediction_position
        or state.optimizer_steps != config.parent_optimizer_steps
        or state.cursor != cursor
    ):
        raise ModelDataError(
            "parent checkpoint state does not match approved tail cursor"
        )


def _validate_final_state(state: TrainingState, config: MLPTailConfig) -> None:
    if (
        state.predictions_seen != config.final_prediction_position
        or state.optimizer_steps != config.final_optimizer_steps
        or state.optimizer_steps - config.parent_optimizer_steps
        != config.tail_optimizer_updates
        or state.cursor.prediction_index != config.final_prediction_position
    ):
        raise ModelDataError(
            "tail stream accounting does not match the approved budget"
        )


def _save_final_model(destination: Path, model: ContextMLP) -> str:
    if destination.exists():
        raise ModelDataError("final tail model artifact already exists")
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise ModelDataError("final tail model temporary path already exists")
    tensors = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }
    try:
        save_file(tensors, str(temporary))
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return digest


def _status_payload(
    plan: MLPTailPlan,
    seed: int,
    parent_run_id: str,
    parent_path: Path,
    derived_revision: str,
    status: str,
    state: TrainingState | None,
    metric: NativeMetrics | None,
    model_sha256: str | None,
    failure_reason: str | None,
    started: float,
) -> dict[str, object]:
    if status not in {"running", "passed", "failed"}:
        raise ModelDataError("tail status is invalid")
    elapsed = time.perf_counter() - started
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ModelDataError("tail runtime is invalid")
    predictions = None if state is None else state.predictions_seen
    tail_predictions = (
        None
        if predictions is None
        else predictions - plan.config.parent_prediction_position
    )
    online_loss = None if state is None else state.training_loss_numerator
    return {
        "schema_version": 1,
        "status": status,
        "run_id": plan.run_id,
        "arm": plan.arm,
        "device": "cpu",
        "seed": seed,
        "exploratory_only": True,
        "tail_contract_identifier": plan.config.contract_identifier,
        "tail_config_sha256": tail_config_sha256(plan.config_path),
        "base_contract_identifier": plan.config.base_contract_identifier,
        "base_config_sha256": config_sha256(plan.base_config_path),
        "week2_source_identity": {
            "model_data_config_sha256": plan.base_config.model_data_config_sha256,
            "model_data_registry_relative_path": plan.base_config.model_data_registry_relative_path,
            "model_data_registry_sha256": plan.base_config.model_data_registry_sha256,
            "training_stream_report_relative_path": plan.base_config.training_stream_report_relative_path,
            "training_stream_report_sha256": plan.base_config.training_stream_report_sha256,
        },
        "parent_code_revision": plan.config.parent_code_revision,
        "derived_code_revision": derived_revision,
        "parent_checkpoint_path": str(parent_path),
        "parent_run_id": parent_run_id,
        "parent_metadata_sha256": plan.config.parent_pin(seed).metadata_sha256,
        "parent_tensor_sha256": plan.config.parent_pin(seed).tensor_sha256,
        "parent_cursor": {
            "prediction_index": plan.config.parent_cursor_prediction_index,
            "protein_index": plan.config.parent_cursor_protein_index,
            "within_protein_target_offset": plan.config.parent_cursor_within_protein_target_offset,
        },
        "schedule": schedule_provenance(plan.arm, plan.config),
        "start_prediction_position": plan.config.parent_prediction_position,
        "final_prediction_position": predictions,
        "start_optimizer_steps": plan.config.parent_optimizer_steps,
        "final_optimizer_steps": None if state is None else state.optimizer_steps,
        "tail_optimizer_updates": None
        if state is None
        else state.optimizer_steps - plan.config.parent_optimizer_steps,
        "tail_online_loss_numerator": online_loss,
        "tail_online_cross_entropy": None
        if tail_predictions in (None, 0)
        else online_loss / tail_predictions,
        "runtime_seconds": elapsed,
        "throughput_tail_predictions_per_second": None
        if tail_predictions is None or elapsed == 0
        else tail_predictions / elapsed,
        "parameter_count": plan.base_config.parameter_count,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "network_requests_made": 0,
        "final_model_file": "final_model.safetensors" if model_sha256 else None,
        "final_model_sha256": model_sha256,
        "final_native_validation": None
        if metric is None
        else {
            "token_count": metric.predictions,
            "nll_numerator": metric.nll_numerator,
            "correct_predictions": metric.correct_predictions,
            "cross_entropy": metric.cross_entropy,
            "accuracy": metric.accuracy,
        },
        "failure_reason": failure_reason,
    }


def _write_status(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _completion_payload(
    plan: MLPTailPlan,
    seed: int,
    state: TrainingState,
    metric: NativeMetrics,
    started: float,
) -> dict[str, object]:
    runtime = time.perf_counter() - started
    if not math.isfinite(runtime) or runtime < 0:
        raise ModelDataError("tail runtime is invalid")
    return {
        "arm": plan.arm,
        "seed": seed,
        "final_cross_entropy": metric.cross_entropy,
        "final_accuracy": metric.accuracy,
        "runtime_seconds": runtime,
        "tail_optimizer_updates": state.optimizer_steps
        - plan.config.parent_optimizer_steps,
    }


def _notify(
    callback: ProgressCallback | None, event: str, payload: dict[str, object]
) -> None:
    if callback is not None:
        callback(event, payload)
