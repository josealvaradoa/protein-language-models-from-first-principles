"""CPU-only, non-resumable execution for the Week 3 capacity screen."""

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
from protein_lm.mlp.capacity_screen_config import (
    CapacityArm,
    MLPCapacityScreenConfig,
    config_sha256 as screen_config_sha256,
    load_capacity_screen_config,
)
from protein_lm.mlp.checkpoint import save_checkpoint
from protein_lm.mlp.config import MLPTrainingConfig, config_sha256, load_config
from protein_lm.mlp.metrics import NativeMetrics, evaluate_native
from protein_lm.mlp.model import ContextMLP, resolve_device
from protein_lm.mlp.orchestration import (
    _require_ignored,
    _require_revision,
    _verify_source_pins,
)
from protein_lm.mlp.stream import iter_context_batches, iter_native_context_batches
from protein_lm.mlp.training import TrainingState, new_optimizer, train_batch


_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")
CollectionLoader = Callable[[Path, ModelDataCollection], Iterable[ProteinSequence]]
ProgressCallback = Callable[[str, dict[str, object]], None]


@dataclass(frozen=True)
class CapacityScreenPlan:
    config: MLPCapacityScreenConfig
    config_path: Path
    base_config: MLPTrainingConfig
    base_config_path: Path
    arm: CapacityArm
    run_id: str
    destination: Path

    @property
    def training_config(self) -> MLPTrainingConfig:
        """Expose the exact derived config required for later checkpoint loading."""

        return _arm_training_config(self.base_config, self.config, self.arm)


def preflight(root: Path, run_id: str, arm_name: str) -> CapacityScreenPlan:
    """Read reviewed configs only. No corpus, readiness, output, git, or device access."""

    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ModelDataError(
            "run identifier must be 3-64 lowercase letters, digits, or hyphens"
        )
    config_path = root / "experiments/week_03/mlp_capacity_screen_v1.toml"
    config = load_capacity_screen_config(config_path)
    arm = config.arm(arm_name)
    base_config_path = root / config.base_config_relative_path
    base_config = load_config(base_config_path)
    if (
        config_sha256(base_config_path) != config.base_config_sha256
        or base_config.contract_identifier != config.base_contract_identifier
        or base_config.parameter_count != 274_293
        or base_config.batch_size != config.batch_size
    ):
        raise ModelDataError("capacity-screen base configuration identity is invalid")
    return CapacityScreenPlan(
        config,
        config_path,
        base_config,
        base_config_path,
        arm,
        run_id,
        root / config.output_relative_root / run_id,
    )


def execute_screen(
    *,
    root: Path,
    plan: CapacityScreenPlan,
    seed: int,
    device_name: str,
    loader: CollectionLoader = load_collection,
    code_revision: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Execute exactly one new arm/seed 25M screen run, never resuming it."""

    if device_name != "cpu":
        raise ModelDataError("capacity screen requires explicit CPU execution")
    try:
        approved_arm = plan.config.arm(plan.arm.name)
    except (AttributeError, ModelDataError) as error:
        raise ModelDataError("capacity-screen plan arm is not approved") from error
    if plan.arm != approved_arm:
        raise ModelDataError("capacity-screen plan arm differs from strict approval")
    if seed not in plan.config.run_seeds:
        raise ModelDataError("capacity-screen seed is not approved")
    if plan.destination.exists():
        raise ModelDataError(
            "capacity-screen destination already exists; a new run ID is required"
        )
    derived_revision = _require_revision(root, code_revision)
    _verify_source_pins(root, plan.base_config)
    _verify_readiness_aggregate(root, plan.config, plan.base_config)
    _require_ignored(root, plan.destination)
    plan.destination.mkdir(parents=True)
    status_path = plan.destination / "run_status.json"
    training_config = plan.training_config
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
                training_config,
                derived_revision,
                "running",
                state,
                metrics,
                checkpoints,
                None,
                started,
            ),
        )
        device = resolve_device("cpu")
        model = ContextMLP(training_config, seed, device)
        optimizer = new_optimizer(model, training_config)
        training = loader(root, ModelDataCollection.FAMILY_AWARE_TRAINING)
        native_validation: tuple[ProteinSequence, ...] | None = None
        state = TrainingState()
        for batch in iter_context_batches(
            training,
            namespace=training_config.training_namespace,
            base_seed=training_config.stream_base_seed,
            prediction_budget=training_config.prediction_budget,
            batch_size=training_config.batch_size,
            event_predictions=training_config.event_predictions,
            context_length=training_config.context_length,
        ):
            train_batch(model, optimizer, batch, state, training_config)
            if state.predictions_seen not in plan.config.event_predictions:
                continue
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
            checkpoint = save_checkpoint(
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
            checkpoints.append(_checkpoint_payload(checkpoint, state.predictions_seen))
            _write_status(
                status_path,
                _status_payload(
                    plan,
                    seed,
                    training_config,
                    derived_revision,
                    "running",
                    state,
                    metrics,
                    checkpoints,
                    None,
                    started,
                ),
            )
            _notify(progress_callback, "milestone", metrics[-1])
            _notify(progress_callback, "checkpoint", checkpoints[-1])
        _validate_final_state(state, metrics, checkpoints, plan.config)
        _write_status(
            status_path,
            _status_payload(
                plan,
                seed,
                training_config,
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
                training_config,
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
        raise ModelDataError(f"capacity screen failed: {error}") from error
    return plan.destination


def _arm_training_config(
    base: MLPTrainingConfig,
    screen: MLPCapacityScreenConfig,
    arm: CapacityArm,
) -> MLPTrainingConfig:
    """Keep primary mechanics intact while changing only the approved allocation axis."""

    return replace(
        base,
        contract_identifier=screen.contract_identifier,
        context_length=arm.context_length,
        embedding_width=arm.embedding_width,
        hidden_width=arm.hidden_width,
        base_learning_rate=screen.fixed_learning_rate,
        post_boundary_learning_rate=screen.fixed_learning_rate,
        learning_rate_boundary_predictions=screen.prediction_budget,
        momentum=screen.momentum,
        weight_decay=screen.weight_decay,
        prediction_budget=screen.prediction_budget,
        milestone_predictions=screen.event_predictions,
        checkpoint_predictions=screen.event_predictions,
        output_relative_root=screen.output_relative_root,
    )


def _verify_readiness_aggregate(
    root: Path, config: MLPCapacityScreenConfig, base: MLPTrainingConfig
) -> None:
    """Pin both readiness bytes and the exact training/native populations."""

    report_path = root / config.readiness_report_relative_path
    try:
        content = report_path.read_bytes()
    except OSError as error:
        raise ModelDataError(
            "capacity-screen readiness evidence is unavailable"
        ) from error
    if hashlib.sha256(content).hexdigest() != config.readiness_report_sha256:
        raise ModelDataError("capacity-screen readiness report bytes do not match pin")
    try:
        registry = json.loads(
            (root / base.model_data_registry_relative_path).read_text(encoding="utf-8")
        )
        report = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelDataError(
            "capacity-screen readiness evidence is unavailable or malformed"
        ) from error
    if not isinstance(report, dict):
        raise ModelDataError("capacity-screen readiness evidence is not an object")
    aggregates = report.get("collection_aggregates", {})
    if not isinstance(aggregates, dict):
        raise ModelDataError("capacity-screen readiness aggregates are not an object")
    training_aggregate = aggregates.get(config.training_collection, {})
    native_aggregate = aggregates.get(config.native_validation_collection, {})
    if (
        not isinstance(registry, dict)
        or not isinstance(training_aggregate, dict)
        or not isinstance(native_aggregate, dict)
        or registry.get("readiness")
        != {
            "relative_path": config.readiness_report_relative_path,
            "sha256": config.readiness_report_sha256,
        }
        or report.get("scope") != "week_02_model_data_readiness"
        or report.get("candidate_status") != "passed"
        or report.get("network_requests_made") != 0
        or training_aggregate.get("prediction_tokens")
        != config.training_prediction_tokens
        or training_aggregate.get("records") != config.training_records
        or native_aggregate.get("prediction_tokens")
        != config.native_validation_prediction_tokens
        or native_aggregate.get("records") != config.native_validation_records
    ):
        raise ModelDataError(
            "capacity-screen readiness evidence does not match approval"
        )


def _validate_native_metric(
    metric: NativeMetrics, config: MLPCapacityScreenConfig
) -> None:
    if metric.predictions != config.native_validation_prediction_tokens:
        raise ModelDataError(
            "native validation token count does not match capacity-screen approval"
        )


def _validate_final_state(
    state: TrainingState | None,
    metrics: list[dict[str, object]],
    checkpoints: list[dict[str, object]],
    config: MLPCapacityScreenConfig,
) -> None:
    control_cursor = config.control_runs[0]
    if state is None or (
        state.predictions_seen != config.prediction_budget
        or state.optimizer_steps
        != config.expected_optimizer_steps(config.prediction_budget)
        or state.cursor.prediction_index != config.prediction_budget
        or state.cursor.protein_index != control_cursor.cursor_protein_index
        or state.cursor.within_protein_target_offset
        != control_cursor.cursor_within_protein_target_offset
        or [item["prediction_position"] for item in metrics]
        != list(config.event_predictions)
        or [item["prediction_position"] for item in checkpoints]
        != list(config.event_predictions)
    ):
        raise ModelDataError(
            "capacity-screen stream, event, cursor, or step accounting is invalid"
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


def _checkpoint_payload(path: Path, prediction_position: int) -> dict[str, object]:
    try:
        metadata = hashlib.sha256((path / "checkpoint.json").read_bytes()).hexdigest()
        tensors = hashlib.sha256((path / "model.safetensors").read_bytes()).hexdigest()
    except OSError as error:
        raise ModelDataError("capacity-screen checkpoint is unreadable") from error
    return {
        "prediction_position": prediction_position,
        "relative_path": path.name,
        "metadata_sha256": metadata,
        "tensor_sha256": tensors,
    }


def _status_payload(
    plan: CapacityScreenPlan,
    seed: int,
    training_config: MLPTrainingConfig,
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
        raise ModelDataError("capacity-screen status accounting is invalid")
    predictions = None if state is None else state.predictions_seen
    online_numerator = None if state is None else state.training_loss_numerator
    return {
        "schema_version": 1,
        "status": status,
        "exploratory_only": True,
        "automatic_selection_generated": False,
        "automatic_report_generated": False,
        "non_resumable": True,
        "run_id": plan.run_id,
        "arm": plan.arm.name,
        "seed": seed,
        "device": "cpu",
        "network_requests_made": 0,
        "contract_identifier": plan.config.contract_identifier,
        "screen_config_sha256": screen_config_sha256(plan.config_path),
        "base_contract_identifier": plan.config.base_contract_identifier,
        "base_config_sha256": config_sha256(plan.base_config_path),
        "derived_code_revision": derived_revision,
        "model": {
            "context_length": plan.arm.context_length,
            "embedding_width": plan.arm.embedding_width,
            "hidden_width": plan.arm.hidden_width,
            "context_vocab_size": training_config.context_vocab_size,
            "target_vocab_size": training_config.target_vocab_size,
            "parameter_count": training_config.parameter_count,
            "activation": training_config.activation,
            "initialization": training_config.initialization,
            "tensor_dtype": training_config.tensor_dtype,
        },
        "week2_source_identity": {
            "model_data_config_sha256": training_config.model_data_config_sha256,
            "model_data_registry_relative_path": training_config.model_data_registry_relative_path,
            "model_data_registry_sha256": training_config.model_data_registry_sha256,
            "training_stream_report_relative_path": training_config.training_stream_report_relative_path,
            "training_stream_report_sha256": training_config.training_stream_report_sha256,
            "readiness_report_relative_path": plan.config.readiness_report_relative_path,
            "readiness_report_sha256": plan.config.readiness_report_sha256,
            "training_prediction_tokens": plan.config.training_prediction_tokens,
            "training_records": plan.config.training_records,
            "native_validation_prediction_tokens": plan.config.native_validation_prediction_tokens,
            "native_validation_records": plan.config.native_validation_records,
        },
        "schedule": {
            "optimizer": "SGD",
            "learning_rate": plan.config.fixed_learning_rate,
            "momentum": plan.config.momentum,
            "weight_decay": plan.config.weight_decay,
            "batch_size": plan.config.batch_size,
            "event_predictions": list(plan.config.event_predictions),
            "repeat_or_wraparound": False,
        },
        "cumulative_online_training": {
            "prediction_count": predictions,
            "loss_numerator": online_numerator,
            "cumulative_online_cross_entropy": None
            if predictions in (None, 0)
            else online_numerator / predictions,
        },
        "native_validation_milestones": metrics,
        "checkpoints": checkpoints,
        "final_cursor_step_accounting": None
        if state is None
        else {
            "prediction_position": state.predictions_seen,
            "optimizer_steps": state.optimizer_steps,
            "cursor": {
                "prediction_index": state.cursor.prediction_index,
                "protein_index": state.cursor.protein_index,
                "within_protein_target_offset": state.cursor.within_protein_target_offset,
            },
            "active_learning_rate": plan.config.fixed_learning_rate,
        },
        "control_provenance": {
            "control_code_revision": plan.config.control_code_revision,
            "control_runs": [
                {
                    "seed": run.seed,
                    "run_id": run.run_id,
                    "run_status_sha256": run.run_status_sha256,
                    "native_cross_entropy": run.native_cross_entropy,
                    "native_accuracy": run.native_accuracy,
                    "native_nll_numerator": run.native_nll_numerator,
                    "native_correct_predictions": run.native_correct_predictions,
                    "optimizer_steps": run.optimizer_steps,
                    "cursor": {
                        "prediction_index": run.cursor_prediction_index,
                        "protein_index": run.cursor_protein_index,
                        "within_protein_target_offset": run.cursor_within_protein_target_offset,
                    },
                    "active_learning_rate": run.active_learning_rate,
                }
                for run in plan.config.control_runs
            ],
        },
        "three_seed_decision_rule": {
            "selection_metric": "25M three-seed mean native cross entropy",
            "control_mean_native_cross_entropy": plan.config.control_three_seed_mean_native_cross_entropy,
            "control_sample_standard_deviation": plan.config.control_three_seed_sample_standard_deviation,
            "minimum_mean_improvement": plan.config.minimum_mean_native_cross_entropy_improvement,
            "qualifying_mean_native_cross_entropy_at_most": plan.config.qualifying_mean_native_cross_entropy_at_most,
            "among_qualifying_select_lowest_mean_only": True,
            "per_seed_selection_allowed": False,
            "no_qualifying_arm_action": "stop",
            "selection_or_report_is_not_automatic": True,
        },
        "runtime_seconds": runtime,
        "throughput_predictions_per_second": None
        if predictions is None or runtime == 0
        else predictions / runtime,
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
    seed: int, state: TrainingState, final_metric: dict[str, object], started: float
) -> dict[str, object]:
    runtime = time.perf_counter() - started
    if not math.isfinite(runtime) or runtime < 0:
        raise ModelDataError("capacity-screen completion runtime is invalid")
    return {
        "seed": seed,
        "prediction_position": state.predictions_seen,
        "optimizer_steps": state.optimizer_steps,
        "native_cross_entropy": final_metric["cross_entropy"],
        "native_accuracy": final_metric["accuracy"],
        "runtime_seconds": runtime,
    }


def _notify(
    callback: ProgressCallback | None, event: str, payload: dict[str, object]
) -> None:
    if callback is not None:
        callback(event, payload)
