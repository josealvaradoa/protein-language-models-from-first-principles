"""Strict CPU-only continuation of the approved C=20 capacity-screen parents."""

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
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.data.model_data.loaders import (
    ModelDataCollection,
    ProteinSequence,
    load_collection,
)
from protein_lm.mlp.capacity_screen_config import load_capacity_screen_config
from protein_lm.mlp.capacity_screen_orchestration import _arm_training_config
from protein_lm.mlp.checkpoint import load_checkpoint, save_checkpoint
from protein_lm.mlp.config import MLPTrainingConfig, config_sha256, load_config
from protein_lm.mlp.context20_continuation_config import (
    Context20ContinuationConfig,
    ParentRun,
    config_sha256 as continuation_config_sha256,
    load_context20_continuation_config,
)
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
from protein_lm.mlp.training import (
    TrainingState,
    learning_rate_for,
    new_optimizer,
    train_batch,
)


_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")
CollectionLoader = Callable[[Path, ModelDataCollection], Iterable[ProteinSequence]]
ProgressCallback = Callable[[str, dict[str, object]], None]


@dataclass(frozen=True)
class Context20ContinuationPlan:
    config: Context20ContinuationConfig
    config_path: Path
    base_config: MLPTrainingConfig
    base_config_path: Path
    parent_training_config: MLPTrainingConfig
    capacity_config_path: Path
    run_id: str
    destination: Path

    @property
    def training_config(self) -> MLPTrainingConfig:
        return replace(
            self.base_config,
            contract_identifier=self.config.contract_identifier,
            context_length=self.config.context_length,
            embedding_width=self.config.embedding_width,
            hidden_width=self.config.hidden_width,
            base_learning_rate=self.config.base_learning_rate,
            post_boundary_learning_rate=self.config.post_boundary_learning_rate,
            learning_rate_boundary_predictions=self.config.learning_rate_boundary_predictions,
            prediction_budget=self.config.final_prediction_position,
            milestone_predictions=self.config.historical_milestone_predictions,
            checkpoint_predictions=self.config.historical_checkpoint_predictions,
            output_relative_root=self.config.output_relative_root,
        )


def preflight(root: Path, run_id: str) -> Context20ContinuationPlan:
    """Read only approved configuration bytes. This cannot inspect runtime state."""

    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ModelDataError(
            "run identifier must be 3-64 lowercase letters, digits, or hyphens"
        )
    config_path = root / "experiments/week_03/mlp_context20_100m_continuation_v1.toml"
    config = load_context20_continuation_config(config_path)
    base_config_path = root / config.base_config_relative_path
    capacity_config_path = root / config.capacity_config_relative_path
    base = load_config(base_config_path)
    capacity = load_capacity_screen_config(capacity_config_path)
    if (
        config_sha256(base_config_path) != config.base_config_sha256
        or config_sha256(capacity_config_path) != config.capacity_config_sha256
        or base.contract_identifier != config.base_contract_identifier
        or base.batch_size != config.batch_size
        or base.parameter_count != 274_293
    ):
        raise ModelDataError("context20 continuation configuration lineage is invalid")
    parent = _arm_training_config(base, capacity, capacity.arm("context_20"))
    if (
        parent.parameter_count != config.parameter_count
        or parent.prediction_budget != config.parent_prediction_position
        or parent.expected_optimizer_steps(config.parent_prediction_position)
        != config.parent_optimizer_steps
    ):
        raise ModelDataError(
            "context20 continuation parent training configuration is invalid"
        )
    return Context20ContinuationPlan(
        config,
        config_path,
        base,
        base_config_path,
        parent,
        capacity_config_path,
        run_id,
        root / config.output_relative_root / run_id,
    )


def execute_continuation(
    *,
    root: Path,
    plan: Context20ContinuationPlan,
    seed: int,
    device_name: str,
    loader: CollectionLoader = load_collection,
    code_revision: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Continue exactly one immutable 25M parent, without replaying its stream."""

    plan = _require_canonical_plan(root, plan)
    if device_name != "cpu":
        raise ModelDataError("context20 continuation requires explicit CPU execution")
    parent = plan.config.parent_run(seed)
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
    parent_status_path = _parent_status_path(
        root, plan.parent_training_config, parent.run_id
    )
    parent_checkpoint = _parent_checkpoint_path(
        root,
        plan.parent_training_config,
        parent.run_id,
        plan.config.parent_prediction_position,
    )
    state: TrainingState | None = None
    metrics: list[dict[str, object]] = []
    checkpoints: list[dict[str, object]] = []
    started = time.perf_counter()
    try:
        _write_status(
            status_path,
            _status_payload(
                plan,
                seed,
                parent,
                parent_status_path,
                parent_checkpoint,
                derived_revision,
                "running",
                state,
                metrics,
                checkpoints,
                None,
                started,
            ),
        )
        _verify_parent_status(parent_status_path, parent, plan)
        _verify_parent_checkpoint_bytes(parent_checkpoint, parent)
        model = ContextMLP(plan.parent_training_config, seed, resolve_device("cpu"))
        optimizer = new_optimizer(model, plan.parent_training_config)
        state = load_checkpoint(
            parent_checkpoint,
            model=model,
            optimizer=optimizer,
            config=plan.parent_training_config,
            config_path=plan.capacity_config_path,
            seed=seed,
            run_id=parent.run_id,
            device_name="cpu",
            code_revision=plan.config.parent_code_revision,
        )
        _validate_parent_state(state, optimizer, parent, plan)
        training_config = plan.training_config
        training = loader(root, ModelDataCollection.FAMILY_AWARE_TRAINING)
        native_validation: tuple[ProteinSequence, ...] | None = None
        for batch in iter_context_batches(
            training,
            namespace=training_config.training_namespace,
            base_seed=training_config.stream_base_seed,
            prediction_budget=training_config.prediction_budget,
            batch_size=training_config.batch_size,
            event_predictions=training_config.event_predictions,
            cursor=state.cursor,
            context_length=training_config.context_length,
        ):
            train_batch(model, optimizer, batch, state, training_config)
            if (
                state.predictions_seen
                in plan.config.continuation_evaluation_predictions
            ):
                if native_validation is None:
                    native_validation = tuple(
                        loader(root, ModelDataCollection.FAMILY_AWARE_NATIVE_VALIDATION)
                    )
                metric = evaluate_native(
                    model,
                    iter_native_context_batches(
                        native_validation,
                        namespace=training_config.training_namespace,
                        base_seed=training_config.stream_base_seed,
                        batch_size=training_config.batch_size,
                        context_length=training_config.context_length,
                    ),
                )
                _validate_native_metric(metric, plan.config)
                metrics.append(_metric_payload(state.predictions_seen, metric))
                _notify(progress_callback, "milestone", metrics[-1])
            if (
                state.predictions_seen
                in plan.config.continuation_checkpoint_predictions
            ):
                path = save_checkpoint(
                    plan.destination / f"checkpoint-{state.predictions_seen}",
                    model=model,
                    optimizer=optimizer,
                    state=state,
                    config=training_config,
                    config_path=plan.config_path,
                    seed=seed,
                    run_id=plan.run_id,
                    device_name="cpu",
                    code_revision=derived_revision,
                )
                checkpoints.append(_checkpoint_payload(path, state.predictions_seen))
                _notify(progress_callback, "checkpoint", checkpoints[-1])
            if (
                state.predictions_seen
                in plan.config.continuation_evaluation_predictions
                or state.predictions_seen
                in plan.config.continuation_checkpoint_predictions
            ):
                _write_status(
                    status_path,
                    _status_payload(
                        plan,
                        seed,
                        parent,
                        parent_status_path,
                        parent_checkpoint,
                        derived_revision,
                        "running",
                        state,
                        metrics,
                        checkpoints,
                        None,
                        started,
                    ),
                )
        _validate_final_state(state, metrics, checkpoints, plan.config)
        _verify_parent_status(parent_status_path, parent, plan)
        _verify_parent_checkpoint_bytes(parent_checkpoint, parent)
        _write_status(
            status_path,
            _status_payload(
                plan,
                seed,
                parent,
                parent_status_path,
                parent_checkpoint,
                derived_revision,
                "passed",
                state,
                metrics,
                checkpoints,
                None,
                started,
            ),
        )
        _notify(
            progress_callback,
            "completed",
            _completion_payload(seed, state, metrics[-1], started),
        )
    except Exception as error:
        _write_status(
            status_path,
            _status_payload(
                plan,
                seed,
                parent,
                parent_status_path,
                parent_checkpoint,
                derived_revision,
                "failed",
                state,
                metrics,
                checkpoints,
                str(error),
                started,
            ),
        )
        if isinstance(error, ModelDataError):
            raise
        raise ModelDataError(f"context20 continuation failed: {error}") from error
    return plan.destination


def _require_canonical_plan(
    root: Path, plan: Context20ContinuationPlan
) -> Context20ContinuationPlan:
    """Reject injected plans before they can redirect operational access."""

    if not isinstance(plan, Context20ContinuationPlan):
        raise ModelDataError("context20 continuation execution plan is invalid")
    canonical = preflight(root, plan.run_id)
    if plan != canonical:
        raise ModelDataError(
            "context20 continuation execution plan differs from approval"
        )
    return canonical


def _parent_status_path(root: Path, config: MLPTrainingConfig, run_id: str) -> Path:
    return root / config.output_relative_root / run_id / "run_status.json"


def _parent_checkpoint_path(
    root: Path, config: MLPTrainingConfig, run_id: str, position: int
) -> Path:
    return root / config.output_relative_root / run_id / f"checkpoint-{position}"


def _verify_readiness_aggregate(
    root: Path, config: Context20ContinuationConfig, base: MLPTrainingConfig
) -> None:
    try:
        content = (root / config.readiness_report_relative_path).read_bytes()
    except OSError as error:
        raise ModelDataError(
            "context20 continuation readiness evidence is unavailable or malformed"
        ) from error
    if hashlib.sha256(content).hexdigest() != config.readiness_report_sha256:
        raise ModelDataError(
            "context20 continuation readiness report bytes do not match pin"
        )
    try:
        registry = json.loads(
            (root / base.model_data_registry_relative_path).read_text(encoding="utf-8")
        )
        report = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelDataError(
            "context20 continuation readiness evidence is unavailable or malformed"
        ) from error
    aggregates = (
        report.get("collection_aggregates", {}) if isinstance(report, dict) else {}
    )
    training = (
        aggregates.get(config.training_collection, {})
        if isinstance(aggregates, dict)
        else {}
    )
    native = (
        aggregates.get(config.native_validation_collection, {})
        if isinstance(aggregates, dict)
        else {}
    )
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
        or not isinstance(training, dict)
        or not isinstance(native, dict)
        or training.get("prediction_tokens") != config.training_prediction_tokens
        or training.get("records") != config.training_records
        or native.get("prediction_tokens") != config.native_validation_prediction_tokens
        or native.get("records") != config.native_validation_records
    ):
        raise ModelDataError(
            "context20 continuation readiness evidence does not match approval"
        )


def _verify_parent_status(
    path: Path, parent: ParentRun, plan: Context20ContinuationPlan
) -> None:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ModelDataError(
            "approved context20 parent status is unavailable"
        ) from error
    if hashlib.sha256(content).hexdigest() != parent.run_status_sha256:
        raise ModelDataError("approved context20 parent status bytes do not match pin")
    try:
        status = json.loads(content)
    except json.JSONDecodeError as error:
        raise ModelDataError("approved context20 parent status is malformed") from error
    model = status.get("model", {}) if isinstance(status, dict) else {}
    schedule = status.get("schedule", {}) if isinstance(status, dict) else {}
    accounting = (
        status.get("final_cursor_step_accounting", {})
        if isinstance(status, dict)
        else {}
    )
    cumulative = (
        status.get("cumulative_online_training", {}) if isinstance(status, dict) else {}
    )
    cursor = accounting.get("cursor", {}) if isinstance(accounting, dict) else {}
    if (
        not isinstance(status, dict)
        or status.get("status") != "passed"
        or status.get("arm") != "context_20"
        or status.get("seed") != parent.seed
        or status.get("run_id") != parent.run_id
        or status.get("derived_code_revision") != plan.config.parent_code_revision
        or model
        != {
            "activation": "tanh",
            "context_length": 20,
            "context_vocab_size": 21,
            "embedding_width": 32,
            "hidden_width": 800,
            "initialization": plan.base_config.initialization,
            "parameter_count": 530293,
            "target_vocab_size": 21,
            "tensor_dtype": "float32",
        }
        or not isinstance(schedule, dict)
        or schedule.get("optimizer") != "SGD"
        or schedule.get("learning_rate") != 0.1
        or schedule.get("momentum") != 0.0
        or schedule.get("weight_decay") != 0.0
        or schedule.get("batch_size") != 1024
        or schedule.get("event_predictions")
        != [1_000_000, 5_000_000, 10_000_000, 25_000_000]
        or not isinstance(accounting, dict)
        or accounting.get("prediction_position")
        != plan.config.parent_prediction_position
        or accounting.get("optimizer_steps") != plan.config.parent_optimizer_steps
        or accounting.get("active_learning_rate") != 0.1
        or cursor
        != {
            "prediction_index": 25_000_000,
            "protein_index": 67_233,
            "within_protein_target_offset": 99,
        }
        or not isinstance(cumulative, dict)
        or cumulative.get("prediction_count") != 25_000_000
        or cumulative.get("loss_numerator") != parent.training_loss_numerator
    ):
        raise ModelDataError("approved context20 parent status lineage is invalid")


def _verify_parent_checkpoint_bytes(path: Path, parent: ParentRun) -> None:
    try:
        metadata = hashlib.sha256((path / "checkpoint.json").read_bytes()).hexdigest()
        tensors = hashlib.sha256((path / "model.safetensors").read_bytes()).hexdigest()
    except OSError as error:
        raise ModelDataError(
            "approved context20 parent checkpoint is unavailable"
        ) from error
    if metadata != parent.metadata_sha256 or tensors != parent.tensor_sha256:
        raise ModelDataError(
            "approved context20 parent checkpoint bytes do not match pin"
        )


def _validate_parent_state(
    state: TrainingState,
    optimizer: torch.optim.SGD,
    parent: ParentRun,
    plan: Context20ContinuationPlan,
) -> None:
    expected = StreamCursor(
        plan.config.parent_cursor_prediction_index,
        plan.config.parent_cursor_protein_index,
        plan.config.parent_cursor_within_protein_target_offset,
    )
    if (
        state.predictions_seen != plan.config.parent_prediction_position
        or state.optimizer_steps != plan.config.parent_optimizer_steps
        or state.cursor != expected
        or state.training_loss_numerator != parent.training_loss_numerator
        or len(optimizer.param_groups) != 1
        or optimizer.param_groups[0]["lr"] != 0.1
    ):
        raise ModelDataError(
            "context20 parent state, loss lineage, or active LR is invalid"
        )


def _validate_native_metric(
    metric: NativeMetrics, config: Context20ContinuationConfig
) -> None:
    if metric.predictions != config.native_validation_prediction_tokens:
        raise ModelDataError("native validation token count does not match approval")


def _validate_final_state(
    state: TrainingState | None,
    metrics: list[dict[str, object]],
    checkpoints: list[dict[str, object]],
    config: Context20ContinuationConfig,
) -> None:
    expected_cursor = StreamCursor(
        config.final_cursor_prediction_index,
        config.final_cursor_protein_index,
        config.final_cursor_within_protein_target_offset,
    )
    if state is None or (
        state.predictions_seen != config.final_prediction_position
        or state.optimizer_steps != config.final_optimizer_steps
        or state.cursor != expected_cursor
        or state.optimizer_steps - config.parent_optimizer_steps
        != config.continuation_optimizer_updates
        or [item["prediction_position"] for item in metrics]
        != list(config.continuation_evaluation_predictions)
        or [item["prediction_position"] for item in checkpoints]
        != list(config.continuation_checkpoint_predictions)
    ):
        raise ModelDataError(
            "context20 continuation stream, event, cursor, or step accounting is invalid"
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


def _checkpoint_payload(path: Path, position: int) -> dict[str, object]:
    try:
        metadata = hashlib.sha256((path / "checkpoint.json").read_bytes()).hexdigest()
        tensors = hashlib.sha256((path / "model.safetensors").read_bytes()).hexdigest()
    except OSError as error:
        raise ModelDataError("continuation checkpoint is unreadable") from error
    return {
        "prediction_position": position,
        "relative_path": path.name,
        "metadata_sha256": metadata,
        "tensor_sha256": tensors,
    }


def _status_payload(
    plan: Context20ContinuationPlan,
    seed: int,
    parent: ParentRun,
    parent_status: Path,
    parent_checkpoint: Path,
    derived_revision: str,
    status: str,
    state: TrainingState | None,
    metrics: list[dict[str, object]],
    checkpoints: list[dict[str, object]],
    failure_reason: str | None,
    started: float,
) -> dict[str, object]:
    runtime = time.perf_counter() - started
    if (
        status not in {"running", "passed", "failed"}
        or not math.isfinite(runtime)
        or runtime < 0
    ):
        raise ModelDataError("context20 continuation status accounting is invalid")
    total_predictions = None if state is None else state.predictions_seen
    cumulative_loss = None if state is None else state.training_loss_numerator
    continuation_predictions = (
        None
        if state is None
        else state.predictions_seen - plan.config.parent_prediction_position
    )
    continuation_loss = (
        None
        if state is None
        else state.training_loss_numerator - parent.training_loss_numerator
    )
    return {
        "schema_version": 1,
        "status": status,
        "exploratory_only": True,
        "non_resumable": True,
        "automatic_selection_generated": False,
        "automatic_report_generated": False,
        "run_id": plan.run_id,
        "seed": seed,
        "device": "cpu",
        "network_requests_made": 0,
        "contract_identifier": plan.config.contract_identifier,
        "continuation_config_sha256": continuation_config_sha256(plan.config_path),
        "base_config_sha256": config_sha256(plan.base_config_path),
        "derived_code_revision": derived_revision,
        "parent": {
            "run_id": parent.run_id,
            "run_status_path": str(parent_status),
            "checkpoint_path": str(parent_checkpoint),
            "run_status_sha256": parent.run_status_sha256,
            "metadata_sha256": parent.metadata_sha256,
            "tensor_sha256": parent.tensor_sha256,
            "code_revision": plan.config.parent_code_revision,
            "prediction_position": plan.config.parent_prediction_position,
            "optimizer_steps": plan.config.parent_optimizer_steps,
            "training_loss_numerator": parent.training_loss_numerator,
            "cursor": {
                "prediction_index": plan.config.parent_cursor_prediction_index,
                "protein_index": plan.config.parent_cursor_protein_index,
                "within_protein_target_offset": plan.config.parent_cursor_within_protein_target_offset,
            },
        },
        "model": {
            "context_length": 20,
            "embedding_width": 32,
            "hidden_width": 800,
            "parameter_count": 530_293,
            "tensor_dtype": "float32",
            "activation": plan.base_config.activation,
            "initialization": plan.base_config.initialization,
        },
        "schedule": {
            "optimizer": "SGD",
            "momentum": 0.0,
            "weight_decay": 0.0,
            "batch_size": 1024,
            "historical_milestone_predictions": list(
                plan.config.historical_milestone_predictions
            ),
            "historical_checkpoint_predictions": list(
                plan.config.historical_checkpoint_predictions
            ),
            "continuation_evaluation_predictions": list(
                plan.config.continuation_evaluation_predictions
            ),
            "continuation_checkpoint_predictions": list(
                plan.config.continuation_checkpoint_predictions
            ),
            "learning_rate_before_90000000": 0.1,
            "learning_rate_from_90000000": 0.01,
            "repeat_or_wraparound": False,
        },
        "training_loss_lineage": {
            "total_prediction_count": total_predictions,
            "cumulative_loss_numerator_including_parent": cumulative_loss,
            "cumulative_cross_entropy_including_parent": None
            if total_predictions in (None, 0)
            else cumulative_loss / total_predictions,
            "continuation_prediction_count": continuation_predictions,
            "continuation_only_loss_numerator": continuation_loss,
            "continuation_only_cross_entropy": None
            if continuation_predictions in (None, 0)
            else continuation_loss / continuation_predictions,
        },
        "native_validation_milestones": metrics,
        "checkpoints": checkpoints,
        "final_cursor_step_accounting": None
        if state is None
        else {
            "prediction_position": state.predictions_seen,
            "optimizer_steps": state.optimizer_steps,
            "continuation_optimizer_updates": state.optimizer_steps
            - plan.config.parent_optimizer_steps,
            "cursor": {
                "prediction_index": state.cursor.prediction_index,
                "protein_index": state.cursor.protein_index,
                "within_protein_target_offset": state.cursor.within_protein_target_offset,
            },
            "active_learning_rate": learning_rate_for(
                state.predictions_seen, plan.training_config
            ),
        },
        "week2_source_identity": {
            "model_data_config_sha256": plan.base_config.model_data_config_sha256,
            "model_data_registry_sha256": plan.base_config.model_data_registry_sha256,
            "training_stream_report_sha256": plan.base_config.training_stream_report_sha256,
            "readiness_report_sha256": plan.config.readiness_report_sha256,
            "training_prediction_tokens": plan.config.training_prediction_tokens,
            "native_validation_prediction_tokens": plan.config.native_validation_prediction_tokens,
        },
        "original_100m_control_provenance": [
            {
                "seed": item.seed,
                "run_id": item.run_id,
                "run_status_sha256": item.run_status_sha256,
                "native_cross_entropy": item.native_cross_entropy,
                "native_accuracy": item.native_accuracy,
                "native_nll_numerator": item.native_nll_numerator,
                "native_correct_predictions": item.native_correct_predictions,
            }
            for item in plan.config.control_runs
        ],
        "three_seed_decision_rule": {
            "selection_metric": "100M three-seed mean native cross entropy only",
            "control_mean_native_cross_entropy": plan.config.control_three_seed_mean_native_cross_entropy,
            "control_sample_standard_deviation": plan.config.control_three_seed_sample_standard_deviation,
            "minimum_mean_improvement": plan.config.minimum_mean_native_cross_entropy_improvement,
            "qualifying_mean_native_cross_entropy_at_most": plan.config.qualifying_mean_native_cross_entropy_at_most,
            "per_seed_selection_allowed": False,
            "automatic_decision_generated": False,
        },
        "runtime_seconds": runtime,
        "throughput_continuation_predictions_per_second": None
        if continuation_predictions is None or runtime == 0
        else continuation_predictions / runtime,
        "environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
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
    seed: int, state: TrainingState, metric: dict[str, object], started: float
) -> dict[str, object]:
    runtime = time.perf_counter() - started
    if not math.isfinite(runtime) or runtime < 0:
        raise ModelDataError("context20 continuation runtime is invalid")
    return {
        "seed": seed,
        "prediction_position": state.predictions_seen,
        "optimizer_steps": state.optimizer_steps,
        "native_cross_entropy": metric["cross_entropy"],
        "native_accuracy": metric["accuracy"],
        "runtime_seconds": runtime,
    }


def _notify(
    callback: ProgressCallback | None, event: str, payload: dict[str, object]
) -> None:
    if callback is not None:
        callback(event, payload)
