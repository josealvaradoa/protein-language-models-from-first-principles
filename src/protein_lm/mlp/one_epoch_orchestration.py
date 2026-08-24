"""Operator-gated, non-resumable Week 3 one-epoch continuation diagnostic."""

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
from protein_lm.mlp.one_epoch_config import (
    OneEpochContinuationConfig,
    ParentPin,
    config_sha256 as continuation_config_sha256,
    load_one_epoch_config,
)
from protein_lm.mlp.one_epoch_training import train_continuation_batch
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
from protein_lm.mlp.training import TrainingState, new_optimizer


_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")
CollectionLoader = Callable[[Path, ModelDataCollection], Iterable[ProteinSequence]]
ProgressCallback = Callable[[str, dict[str, object]], None]


@dataclass(frozen=True)
class OneEpochContinuationPlan:
    config: OneEpochContinuationConfig
    config_path: Path
    base_config: MLPTrainingConfig
    base_config_path: Path
    run_id: str
    destination: Path


def preflight(root: Path, run_id: str) -> OneEpochContinuationPlan:
    """Load reviewed config bytes only, without inspecting operational state."""

    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ModelDataError(
            "run identifier must be 3-64 lowercase letters, digits, or hyphens"
        )
    config_path = root / "experiments/week_03/mlp_one_epoch_continuation_v1.toml"
    config = load_one_epoch_config(config_path)
    base_config_path = root / config.base_config_relative_path
    base_config = load_config(base_config_path)
    if (
        config_sha256(base_config_path) != config.base_config_sha256
        or base_config.contract_identifier != config.base_contract_identifier
        or base_config.batch_size != config.batch_size
        or base_config.parameter_count != 274_293
    ):
        raise ModelDataError("one-epoch continuation base configuration is invalid")
    return OneEpochContinuationPlan(
        config=config,
        config_path=config_path,
        base_config=base_config,
        base_config_path=base_config_path,
        run_id=run_id,
        destination=root / config.output_relative_root / run_id,
    )


def execute_continuation(
    *,
    root: Path,
    plan: OneEpochContinuationPlan,
    seed: int,
    device_name: str,
    loader: CollectionLoader = load_collection,
    code_revision: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Continue one approved 100M CPU parent through the first epoch exactly once."""

    if device_name != "cpu":
        raise ModelDataError("one-epoch continuation requires explicit CPU execution")
    parent_pin = plan.config.parent_pin(seed)
    if plan.destination.exists():
        raise ModelDataError(
            "continuation destination already exists; runs cannot resume"
        )
    derived_revision = _require_revision(root, code_revision)
    _verify_source_pins(root, plan.base_config)
    _verify_readiness_aggregate(root, plan.config, plan.base_config)
    _require_ignored(root, plan.destination)
    plan.destination.mkdir(parents=True)
    status_path = plan.destination / "run_status.json"
    parent_path = _parent_checkpoint_path(
        root,
        plan.base_config,
        parent_pin.run_id,
        plan.config.parent_prediction_position,
    )
    started = time.perf_counter()
    state: TrainingState | None = None
    milestones: list[dict[str, object]] = []
    model_sha256: str | None = None
    try:
        _write_status(
            status_path,
            _status_payload(
                plan,
                seed,
                parent_pin,
                parent_path,
                derived_revision,
                "running",
                state,
                milestones,
                model_sha256,
                None,
                started,
            ),
        )
        _verify_parent_bytes(parent_path, parent_pin)
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
        _validate_parent_state(state, optimizer, plan.config)
        # This numerator belongs exclusively to the continuation diagnostic.
        state.training_loss_numerator = 0.0
        training = loader(root, ModelDataCollection.FAMILY_AWARE_TRAINING)
        native_validation: tuple[ProteinSequence, ...] | None = None
        for batch in iter_context_batches(
            training,
            namespace=plan.base_config.training_namespace,
            base_seed=plan.base_config.stream_base_seed,
            prediction_budget=plan.config.final_prediction_position,
            batch_size=plan.config.batch_size,
            event_predictions=plan.config.milestone_predictions,
            cursor=state.cursor,
            context_length=plan.base_config.context_length,
        ):
            train_continuation_batch(model, optimizer, batch, state, plan.config)
            if state.predictions_seen in plan.config.milestone_predictions:
                if native_validation is None:
                    native_validation = tuple(
                        loader(root, ModelDataCollection.FAMILY_AWARE_NATIVE_VALIDATION)
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
                _validate_native_metric(metric, plan.config)
                milestones.append(_metric_payload(state.predictions_seen, metric))
                _write_status(
                    status_path,
                    _status_payload(
                        plan,
                        seed,
                        parent_pin,
                        parent_path,
                        derived_revision,
                        "running",
                        state,
                        milestones,
                        model_sha256,
                        None,
                        started,
                    ),
                )
                _notify(progress_callback, "milestone", milestones[-1])
        _validate_final_state(state, milestones, plan.config)
        _verify_parent_bytes(parent_path, parent_pin)
        model_sha256 = _save_final_model(
            plan.destination / "final_model.safetensors", model
        )
        _write_status(
            status_path,
            _status_payload(
                plan,
                seed,
                parent_pin,
                parent_path,
                derived_revision,
                "passed",
                state,
                milestones,
                model_sha256,
                None,
                started,
            ),
        )
        _notify(
            progress_callback,
            "completed",
            _completion_payload(plan, seed, state, milestones[-1], started),
        )
    except Exception as error:
        _write_status(
            status_path,
            _status_payload(
                plan,
                seed,
                parent_pin,
                parent_path,
                derived_revision,
                "failed",
                state,
                milestones,
                model_sha256,
                str(error),
                started,
            ),
        )
        if isinstance(error, ModelDataError):
            raise
        raise ModelDataError(f"one-epoch continuation failed: {error}") from error
    return plan.destination


def _parent_checkpoint_path(
    root: Path,
    base_config: MLPTrainingConfig,
    parent_run_id: str,
    parent_prediction_position: int,
) -> Path:
    return (
        root
        / base_config.output_relative_root
        / parent_run_id
        / f"checkpoint-{parent_prediction_position}"
    )


def _verify_readiness_aggregate(
    root: Path,
    config: OneEpochContinuationConfig,
    base_config: MLPTrainingConfig,
) -> None:
    """Confirm the pinned Week 2 evidence proves this endpoint is one epoch."""

    report_path = root / config.readiness_report_relative_path
    try:
        content = report_path.read_bytes()
    except OSError as error:
        raise ModelDataError("one-epoch readiness evidence is unavailable") from error
    if hashlib.sha256(content).hexdigest() != config.readiness_report_sha256:
        raise ModelDataError("one-epoch readiness report bytes do not match pin")
    try:
        registry = json.loads(
            (root / base_config.model_data_registry_relative_path).read_text(
                encoding="utf-8"
            )
        )
        report = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelDataError(
            "one-epoch readiness evidence is unavailable or malformed"
        ) from error
    if (
        not isinstance(registry, dict)
        or registry.get("readiness")
        != {
            "relative_path": config.readiness_report_relative_path,
            "sha256": config.readiness_report_sha256,
        }
        or not isinstance(report, dict)
        or report.get("scope") != "week_02_model_data_readiness"
        or report.get("candidate_status") != "passed"
        or report.get("network_requests_made") != 0
        or report.get("collection_aggregates", {})
        .get(config.training_collection, {})
        .get("prediction_tokens")
        != config.training_prediction_tokens
        or report.get("collection_aggregates", {})
        .get(config.training_collection, {})
        .get("records")
        != config.training_records
        or report.get("collection_aggregates", {})
        .get(config.native_validation_collection, {})
        .get("prediction_tokens")
        != config.native_validation_prediction_tokens
        or report.get("collection_aggregates", {})
        .get(config.native_validation_collection, {})
        .get("records")
        != config.native_validation_records
    ):
        raise ModelDataError("one-epoch readiness evidence does not match approval")


def _verify_parent_bytes(path: Path, pin: ParentPin) -> None:
    try:
        metadata = hashlib.sha256((path / "checkpoint.json").read_bytes()).hexdigest()
        tensors = hashlib.sha256((path / "model.safetensors").read_bytes()).hexdigest()
    except OSError as error:
        raise ModelDataError(
            "approved continuation parent checkpoint is unavailable"
        ) from error
    if metadata != pin.metadata_sha256 or tensors != pin.tensor_sha256:
        raise ModelDataError(
            "approved continuation parent checkpoint bytes do not match pin"
        )


def _validate_parent_state(
    state: TrainingState,
    optimizer: torch.optim.SGD,
    config: OneEpochContinuationConfig,
) -> None:
    expected_cursor = StreamCursor(
        config.parent_cursor_prediction_index,
        config.parent_cursor_protein_index,
        config.parent_cursor_within_protein_target_offset,
    )
    if (
        state.predictions_seen != config.parent_prediction_position
        or state.optimizer_steps != config.parent_optimizer_steps
        or state.cursor != expected_cursor
        or len(optimizer.param_groups) != 1
        or optimizer.param_groups[0]["lr"] != config.fixed_learning_rate
    ):
        raise ModelDataError(
            "continuation parent identity, state, or active LR is invalid"
        )


def _validate_final_state(
    state: TrainingState,
    milestones: list[dict[str, object]],
    config: OneEpochContinuationConfig,
) -> None:
    if (
        state.predictions_seen != config.final_prediction_position
        or state.optimizer_steps != config.final_optimizer_steps
        or state.optimizer_steps - config.parent_optimizer_steps
        != config.continuation_optimizer_updates
        or state.cursor.prediction_index != config.final_prediction_position
        or state.cursor.protein_index != config.training_records
        or state.cursor.within_protein_target_offset != 0
        or [item["prediction_position"] for item in milestones]
        != list(config.milestone_predictions)
    ):
        raise ModelDataError(
            "continuation stream, cursor, event, or step accounting is invalid"
        )


def _metric_payload(position: int, metric: NativeMetrics) -> dict[str, object]:
    return {
        "prediction_position": position,
        "token_count": metric.predictions,
        "nll_numerator": metric.nll_numerator,
        "correct_predictions": metric.correct_predictions,
        "cross_entropy": metric.cross_entropy,
        "accuracy": metric.accuracy,
    }


def _validate_native_metric(
    metric: NativeMetrics, config: OneEpochContinuationConfig
) -> None:
    if metric.predictions != config.native_validation_prediction_tokens:
        raise ModelDataError("native validation token count does not match approval")


def _save_final_model(destination: Path, model: ContextMLP) -> str:
    if destination.exists():
        raise ModelDataError("final continuation model already exists")
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise ModelDataError("final continuation model temporary path already exists")
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
    plan: OneEpochContinuationPlan,
    seed: int,
    parent_pin: ParentPin,
    parent_path: Path,
    derived_revision: str,
    status: str,
    state: TrainingState | None,
    milestones: list[dict[str, object]],
    model_sha256: str | None,
    failure_reason: str | None,
    started: float,
) -> dict[str, object]:
    if status not in {"running", "passed", "failed"}:
        raise ModelDataError("continuation status is invalid")
    runtime = time.perf_counter() - started
    if not math.isfinite(runtime) or runtime < 0:
        raise ModelDataError("continuation runtime is invalid")
    continuation_predictions = (
        None
        if state is None
        else state.predictions_seen - plan.config.parent_prediction_position
    )
    online_numerator = None if state is None else state.training_loss_numerator
    final_validation = milestones[-1] if milestones else None
    return {
        "schema_version": 1,
        "status": status,
        "exploratory_only": True,
        "non_resumable": True,
        "run_id": plan.run_id,
        "seed": seed,
        "device": "cpu",
        "network_requests_made": 0,
        "contract_identifier": plan.config.contract_identifier,
        "continuation_config_sha256": continuation_config_sha256(plan.config_path),
        "base_contract_identifier": plan.config.base_contract_identifier,
        "base_config_sha256": config_sha256(plan.base_config_path),
        "derived_code_revision": derived_revision,
        "parent": {
            "run_id": parent_pin.run_id,
            "path": str(parent_path),
            "code_revision": plan.config.parent_code_revision,
            "metadata_sha256": parent_pin.metadata_sha256,
            "tensor_sha256": parent_pin.tensor_sha256,
            "prediction_position": plan.config.parent_prediction_position,
            "optimizer_steps": plan.config.parent_optimizer_steps,
            "cursor": {
                "prediction_index": plan.config.parent_cursor_prediction_index,
                "protein_index": plan.config.parent_cursor_protein_index,
                "within_protein_target_offset": plan.config.parent_cursor_within_protein_target_offset,
            },
        },
        "week2_source_identity": {
            "model_data_config_sha256": plan.base_config.model_data_config_sha256,
            "model_data_registry_relative_path": plan.base_config.model_data_registry_relative_path,
            "model_data_registry_sha256": plan.base_config.model_data_registry_sha256,
            "training_stream_report_relative_path": plan.base_config.training_stream_report_relative_path,
            "training_stream_report_sha256": plan.base_config.training_stream_report_sha256,
            "readiness_report_relative_path": plan.config.readiness_report_relative_path,
            "readiness_report_sha256": plan.config.readiness_report_sha256,
            "training_prediction_tokens": plan.config.training_prediction_tokens,
            "training_records": plan.config.training_records,
            "native_validation_prediction_tokens": plan.config.native_validation_prediction_tokens,
            "native_validation_records": plan.config.native_validation_records,
        },
        "schedule": {
            "learning_rate": plan.config.fixed_learning_rate,
            "batch_size": plan.config.batch_size,
            "event_predictions": list(plan.config.milestone_predictions),
            "final_partial_batch_predictions": plan.config.final_partial_batch_predictions,
            "repeat_or_wraparound": False,
        },
        "cursor_step_accounting": {
            "final_prediction_position": None
            if state is None
            else state.predictions_seen,
            "final_optimizer_steps": None if state is None else state.optimizer_steps,
            "continuation_optimizer_updates": None
            if state is None
            else state.optimizer_steps - plan.config.parent_optimizer_steps,
            "final_cursor": None
            if state is None
            else {
                "prediction_index": state.cursor.prediction_index,
                "protein_index": state.cursor.protein_index,
                "within_protein_target_offset": state.cursor.within_protein_target_offset,
            },
        },
        "continuation_online_training": {
            "token_count": continuation_predictions,
            "loss_numerator": online_numerator,
            "cross_entropy": None
            if continuation_predictions in (None, 0)
            else online_numerator / continuation_predictions,
        },
        "native_validation_milestones": milestones,
        "final_model_native_validation": final_validation,
        "runtime_seconds": runtime,
        "throughput_continuation_predictions_per_second": None
        if continuation_predictions is None or runtime == 0
        else continuation_predictions / runtime,
        "parameter_count": plan.base_config.parameter_count,
        "environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "three_seed_decision_rule": {
            "control_mean_native_cross_entropy": plan.config.control_three_seed_mean_native_cross_entropy,
            "minimum_mean_improvement": plan.config.minimum_mean_native_cross_entropy_improvement,
            "useful_mean_native_cross_entropy_at_most": plan.config.useful_three_seed_mean_native_cross_entropy_at_most,
            "per_seed_selection_allowed": False,
            "automatic_decision_generated": False,
        },
        "final_model_file": "final_model.safetensors" if model_sha256 else None,
        "final_model_sha256": model_sha256,
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
    plan: OneEpochContinuationPlan,
    seed: int,
    state: TrainingState,
    final_metric: dict[str, object],
    started: float,
) -> dict[str, object]:
    runtime = time.perf_counter() - started
    if not math.isfinite(runtime) or runtime < 0:
        raise ModelDataError("continuation runtime is invalid")
    return {
        "seed": seed,
        "final_prediction_position": state.predictions_seen,
        "final_cross_entropy": final_metric["cross_entropy"],
        "final_accuracy": final_metric["accuracy"],
        "runtime_seconds": runtime,
        "continuation_optimizer_updates": state.optimizer_steps
        - plan.config.parent_optimizer_steps,
    }


def _notify(
    callback: ProgressCallback | None, event: str, payload: dict[str, object]
) -> None:
    if callback is not None:
        callback(event, payload)
