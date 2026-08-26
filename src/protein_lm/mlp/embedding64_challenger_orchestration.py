"""Strict CPU-only E=64 challenger continuation from immutable 25M parents."""

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
from protein_lm.mlp.embedding64_challenger_config import (
    Embedding64ChallengerConfig,
    ParentRun,
    config_sha256 as challenger_config_sha256,
    load_embedding64_challenger_config,
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
class Embedding64ChallengerPlan:
    config: Embedding64ChallengerConfig
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


def preflight(root: Path, run_id: str) -> Embedding64ChallengerPlan:
    """Read only approved configs. It does not access runs, data, output, or git."""

    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ModelDataError(
            "run identifier must be 3-64 lowercase letters, digits, or hyphens"
        )
    config_path = root / "experiments/week_03/mlp_embedding64_100m_challenger_v1.toml"
    config = load_embedding64_challenger_config(config_path)
    base_path = root / config.base_config_relative_path
    capacity_path = root / config.capacity_config_relative_path
    base, capacity = load_config(base_path), load_capacity_screen_config(capacity_path)
    if (
        config_sha256(base_path) != config.base_config_sha256
        or config_sha256(capacity_path) != config.capacity_config_sha256
        or base.contract_identifier != config.base_contract_identifier
        or base.batch_size != config.batch_size
        or base.parameter_count != 274293
    ):
        raise ModelDataError("embedding64 challenger configuration lineage is invalid")
    parent = _arm_training_config(base, capacity, capacity.arm("embedding_64"))
    if (
        parent.parameter_count != config.parameter_count
        or parent.prediction_budget != config.parent_prediction_position
        or parent.expected_optimizer_steps(config.parent_prediction_position)
        != config.parent_optimizer_steps
    ):
        raise ModelDataError(
            "embedding64 challenger parent training configuration is invalid"
        )
    return Embedding64ChallengerPlan(
        config,
        config_path,
        base,
        base_path,
        parent,
        capacity_path,
        run_id,
        root / config.output_relative_root / run_id,
    )


def execute_challenger(
    *,
    root: Path,
    plan: Embedding64ChallengerPlan,
    seed: int,
    device_name: str,
    loader: CollectionLoader = load_collection,
    code_revision: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Continue one approved E=64 parent exactly once, without replaying 0-25M."""

    plan = _require_canonical_plan(root, plan)
    if device_name != "cpu":
        raise ModelDataError("embedding64 challenger requires explicit CPU execution")
    parent = plan.config.parent_run(seed)
    if plan.destination.exists():
        raise ModelDataError(
            "challenger destination already exists; runs cannot resume"
        )
    revision = _require_revision(root, code_revision)
    _verify_source_pins(root, plan.base_config)
    _verify_readiness_aggregate(root, plan.config, plan.base_config)
    _require_ignored(root, plan.destination)
    parent_status = _parent_status_path(
        root, plan.parent_training_config, parent.run_id
    )
    parent_checkpoint = _parent_checkpoint_path(
        root,
        plan.parent_training_config,
        parent.run_id,
        plan.config.parent_prediction_position,
    )
    plan.destination.mkdir(parents=True)
    status_path = plan.destination / "run_status.json"
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
                parent_status,
                parent_checkpoint,
                revision,
                "running",
                state,
                metrics,
                checkpoints,
                None,
                started,
            ),
        )
        _verify_parent_status(parent_status, parent, plan)
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
        config = plan.training_config
        training = loader(root, ModelDataCollection.FAMILY_AWARE_TRAINING)
        native: tuple[ProteinSequence, ...] | None = None
        for batch in iter_context_batches(
            training,
            namespace=config.training_namespace,
            base_seed=config.stream_base_seed,
            prediction_budget=config.prediction_budget,
            batch_size=config.batch_size,
            event_predictions=config.event_predictions,
            cursor=state.cursor,
            context_length=config.context_length,
        ):
            train_batch(model, optimizer, batch, state, config)
            if (
                state.predictions_seen
                in plan.config.continuation_evaluation_predictions
            ):
                if native is None:
                    native = tuple(
                        loader(root, ModelDataCollection.FAMILY_AWARE_NATIVE_VALIDATION)
                    )
                metric = evaluate_native(
                    model,
                    iter_native_context_batches(
                        native,
                        namespace=config.training_namespace,
                        base_seed=config.stream_base_seed,
                        batch_size=config.batch_size,
                        context_length=config.context_length,
                    ),
                )
                _validate_native_metric(metric, plan.config)
                metrics.append(_metric_payload(state.predictions_seen, metric))
                _notify(progress_callback, "milestone", metrics[-1])
            if (
                state.predictions_seen
                in plan.config.continuation_checkpoint_predictions
            ):
                checkpoint = save_checkpoint(
                    plan.destination / f"checkpoint-{state.predictions_seen}",
                    model=model,
                    optimizer=optimizer,
                    state=state,
                    config=config,
                    config_path=plan.config_path,
                    seed=seed,
                    run_id=plan.run_id,
                    device_name="cpu",
                    code_revision=revision,
                )
                checkpoints.append(
                    _checkpoint_payload(checkpoint, state.predictions_seen)
                )
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
                        parent_status,
                        parent_checkpoint,
                        revision,
                        "running",
                        state,
                        metrics,
                        checkpoints,
                        None,
                        started,
                    ),
                )
        _validate_final_state(state, metrics, checkpoints, plan.config)
        _verify_parent_status(parent_status, parent, plan)
        _verify_parent_checkpoint_bytes(parent_checkpoint, parent)
        _write_status(
            status_path,
            _status_payload(
                plan,
                seed,
                parent,
                parent_status,
                parent_checkpoint,
                revision,
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
                parent_status,
                parent_checkpoint,
                revision,
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
        raise ModelDataError(f"embedding64 challenger failed: {error}") from error
    return plan.destination


def _require_canonical_plan(
    root: Path, plan: Embedding64ChallengerPlan
) -> Embedding64ChallengerPlan:
    if not isinstance(plan, Embedding64ChallengerPlan):
        raise ModelDataError("embedding64 challenger execution plan is invalid")
    canonical = preflight(root, plan.run_id)
    if plan != canonical:
        raise ModelDataError(
            "embedding64 challenger execution plan differs from approval"
        )
    return canonical


def _parent_status_path(root: Path, config: MLPTrainingConfig, run_id: str) -> Path:
    return root / config.output_relative_root / run_id / "run_status.json"


def _parent_checkpoint_path(
    root: Path, config: MLPTrainingConfig, run_id: str, position: int
) -> Path:
    return root / config.output_relative_root / run_id / f"checkpoint-{position}"


def _verify_readiness_aggregate(
    root: Path, config: Embedding64ChallengerConfig, base: MLPTrainingConfig
) -> None:
    try:
        content = (root / config.readiness_report_relative_path).read_bytes()
    except OSError as error:
        raise ModelDataError(
            "embedding64 challenger readiness evidence is unavailable or malformed"
        ) from error
    if hashlib.sha256(content).hexdigest() != config.readiness_report_sha256:
        raise ModelDataError(
            "embedding64 challenger readiness evidence does not match approval"
        )
    try:
        registry = json.loads(
            (root / base.model_data_registry_relative_path).read_text(encoding="utf-8")
        )
        report = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelDataError(
            "embedding64 challenger readiness evidence is unavailable or malformed"
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
        or training.get("prediction_tokens") != config.training_prediction_tokens
        or training.get("records") != config.training_records
        or not isinstance(native, dict)
        or native.get("prediction_tokens") != config.native_validation_prediction_tokens
        or native.get("records") != config.native_validation_records
    ):
        raise ModelDataError(
            "embedding64 challenger readiness evidence does not match approval"
        )


def _verify_parent_status(
    path: Path, parent: ParentRun, plan: Embedding64ChallengerPlan
) -> None:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ModelDataError(
            "approved embedding64 parent status is unavailable"
        ) from error
    if hashlib.sha256(content).hexdigest() != parent.run_status_sha256:
        raise ModelDataError(
            "approved embedding64 parent status bytes do not match pin"
        )
    try:
        status = json.loads(content)
    except json.JSONDecodeError as error:
        raise ModelDataError(
            "approved embedding64 parent status is malformed"
        ) from error
    if not isinstance(status, dict):
        raise ModelDataError("approved embedding64 parent status is not an object")
    model, schedule = status.get("model", {}), status.get("schedule", {})
    accounting, cumulative = (
        status.get("final_cursor_step_accounting", {}),
        status.get("cumulative_online_training", {}),
    )
    cursor = accounting.get("cursor", {}) if isinstance(accounting, dict) else {}
    if (
        status.get("status") != "passed"
        or status.get("arm") != "embedding_64"
        or status.get("seed") != parent.seed
        or status.get("run_id") != parent.run_id
        or status.get("derived_code_revision") != plan.config.parent_code_revision
        or model
        != {
            "activation": "tanh",
            "context_length": 10,
            "context_vocab_size": 21,
            "embedding_width": 64,
            "hidden_width": 800,
            "initialization": plan.base_config.initialization,
            "parameter_count": 530965,
            "target_vocab_size": 21,
            "tensor_dtype": "float32",
        }
        or not isinstance(schedule, dict)
        or schedule.get("optimizer") != "SGD"
        or schedule.get("learning_rate") != 0.1
        or schedule.get("momentum") != 0.0
        or schedule.get("weight_decay") != 0.0
        or schedule.get("batch_size") != 1024
        or schedule.get("event_predictions") != [1000000, 5000000, 10000000, 25000000]
        or not isinstance(accounting, dict)
        or accounting.get("prediction_position") != 25000000
        or accounting.get("optimizer_steps") != 24416
        or accounting.get("active_learning_rate") != 0.1
        or cursor
        != {
            "prediction_index": 25000000,
            "protein_index": 67233,
            "within_protein_target_offset": 99,
        }
        or not isinstance(cumulative, dict)
        or cumulative.get("prediction_count") != 25000000
        or cumulative.get("loss_numerator") != parent.training_loss_numerator
    ):
        raise ModelDataError("approved embedding64 parent status lineage is invalid")


def _verify_parent_checkpoint_bytes(path: Path, parent: ParentRun) -> None:
    try:
        metadata = hashlib.sha256((path / "checkpoint.json").read_bytes()).hexdigest()
        tensors = hashlib.sha256((path / "model.safetensors").read_bytes()).hexdigest()
    except OSError as error:
        raise ModelDataError(
            "approved embedding64 parent checkpoint is unavailable"
        ) from error
    if metadata != parent.metadata_sha256 or tensors != parent.tensor_sha256:
        raise ModelDataError(
            "approved embedding64 parent checkpoint bytes do not match pin"
        )


def _validate_parent_state(
    state: TrainingState,
    optimizer: torch.optim.SGD,
    parent: ParentRun,
    plan: Embedding64ChallengerPlan,
) -> None:
    expected = StreamCursor(
        plan.config.parent_cursor_prediction_index,
        plan.config.parent_cursor_protein_index,
        plan.config.parent_cursor_within_protein_target_offset,
    )
    if (
        state.predictions_seen != 25000000
        or state.optimizer_steps != 24416
        or state.cursor != expected
        or state.training_loss_numerator != parent.training_loss_numerator
        or len(optimizer.param_groups) != 1
        or optimizer.param_groups[0]["lr"] != 0.1
    ):
        raise ModelDataError(
            "embedding64 parent state, loss lineage, or active LR is invalid"
        )


def _validate_native_metric(
    metric: NativeMetrics, config: Embedding64ChallengerConfig
) -> None:
    if metric.predictions != config.native_validation_prediction_tokens:
        raise ModelDataError(
            "native validation token count does not match challenger approval"
        )


def _validate_final_state(
    state: TrainingState | None,
    metrics: list[dict[str, object]],
    checkpoints: list[dict[str, object]],
    config: Embedding64ChallengerConfig,
) -> None:
    if (
        state is None
        or state.predictions_seen != config.final_prediction_position
        or state.optimizer_steps != config.final_optimizer_steps
        or state.cursor
        != StreamCursor(
            config.final_cursor_prediction_index,
            config.final_cursor_protein_index,
            config.final_cursor_within_protein_target_offset,
        )
        or [item["prediction_position"] for item in metrics]
        != list(config.continuation_evaluation_predictions)
        or [item["prediction_position"] for item in checkpoints]
        != list(config.continuation_checkpoint_predictions)
    ):
        raise ModelDataError(
            "embedding64 challenger stream, event, cursor, or step accounting is invalid"
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
        raise ModelDataError("challenger checkpoint is unreadable") from error
    return {
        "prediction_position": position,
        "relative_path": path.name,
        "metadata_sha256": metadata,
        "tensor_sha256": tensors,
    }


def _status_payload(
    plan: Embedding64ChallengerPlan,
    seed: int,
    parent: ParentRun,
    parent_status: Path,
    parent_checkpoint: Path,
    revision: str,
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
        raise ModelDataError("embedding64 challenger status accounting is invalid")
    config = plan.config
    prediction_count = None if state is None else state.predictions_seen
    loss = None if state is None else state.training_loss_numerator
    return {
        "schema_version": 1,
        "status": status,
        "exploratory_only": True,
        "automatic_selection_generated": False,
        "automatic_report_generated": False,
        "non_resumable": True,
        "run_id": plan.run_id,
        "arm": "embedding_64",
        "seed": seed,
        "device": "cpu",
        "network_requests_made": 0,
        "contract_identifier": config.contract_identifier,
        "challenger_config_sha256": challenger_config_sha256(plan.config_path),
        "base_contract_identifier": config.base_contract_identifier,
        "base_config_sha256": config_sha256(plan.base_config_path),
        "capacity_config_sha256": config.capacity_config_sha256,
        "derived_code_revision": revision,
        "model": {
            "context_length": 10,
            "embedding_width": 64,
            "hidden_width": 800,
            "context_vocab_size": 21,
            "target_vocab_size": 21,
            "parameter_count": 530965,
            "activation": plan.base_config.activation,
            "initialization": plan.base_config.initialization,
            "tensor_dtype": "float32",
        },
        "schedule": {
            "optimizer": "SGD",
            "learning_rate_before_90000000": 0.1,
            "learning_rate_from_90000000": 0.01,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "batch_size": 1024,
            "historical_milestone_predictions": list(
                config.historical_milestone_predictions
            ),
            "historical_checkpoint_predictions": list(
                config.historical_checkpoint_predictions
            ),
            "continuation_evaluation_predictions": list(
                config.continuation_evaluation_predictions
            ),
            "continuation_checkpoint_predictions": list(
                config.continuation_checkpoint_predictions
            ),
            "continuation_optimizer_updates": config.continuation_optimizer_updates,
            "repeat_or_wraparound": False,
        },
        "parent_provenance": {
            "run_id": parent.run_id,
            "run_status_relative_path": str(
                parent_status.relative_to(plan.destination.parents[4])
            ),
            "run_status_sha256": parent.run_status_sha256,
            "checkpoint_relative_path": str(
                parent_checkpoint.relative_to(plan.destination.parents[4])
            ),
            "checkpoint_metadata_sha256": parent.metadata_sha256,
            "checkpoint_tensor_sha256": parent.tensor_sha256,
            "code_revision": config.parent_code_revision,
        },
        "reference_context20_provenance": {
            "contract_identifier": config.reference_context20_contract_identifier,
            "code_revision": config.reference_context20_code_revision,
            "parameter_count": config.reference_context20_parameter_count,
            "three_seed_mean_native_cross_entropy": config.reference_context20_three_seed_mean_native_cross_entropy,
            "three_seed_sample_standard_deviation": config.reference_context20_three_seed_sample_standard_deviation,
            "minimum_material_mean_native_cross_entropy_gap": config.minimum_material_mean_native_cross_entropy_gap,
            "runs": [
                {
                    "seed": item.seed,
                    "run_id": item.run_id,
                    "run_status_sha256": item.run_status_sha256,
                    "native_cross_entropy": item.native_cross_entropy,
                    "native_accuracy": item.native_accuracy,
                    "native_nll_numerator": item.native_nll_numerator,
                    "native_correct_predictions": item.native_correct_predictions,
                }
                for item in config.reference_context20_runs
            ],
        },
        "challenger_selection_provenance": {
            "scope": config.challenger_selection_scope,
            "basis": config.challenger_selection_basis,
            "open_model_selection": False,
            "embedding64_25m_three_seed_mean_native_cross_entropy": config.embedding64_25m_three_seed_mean_native_cross_entropy,
            "hidden1600_25m_three_seed_mean_native_cross_entropy": config.hidden1600_25m_three_seed_mean_native_cross_entropy,
        },
        "three_seed_interpretation_rule": {
            "metric": "100M three-seed mean native cross entropy",
            "delta_definition": "embedding64_mean_native_cross_entropy - context20_mean_native_cross_entropy",
            "material_gap": config.minimum_material_mean_native_cross_entropy_gap,
            "context20_materially_better_if_embedding64_mean_at_or_above": config.context20_materially_better_if_embedding64_mean_at_or_above,
            "embedding64_materially_better_if_embedding64_mean_at_or_below": config.embedding64_materially_better_if_embedding64_mean_at_or_below,
            "context20_materially_better_boundary_inclusive": True,
            "embedding64_materially_better_boundary_inclusive": True,
            "practical_tie_interval": {
                "lower_exclusive": config.embedding64_materially_better_if_embedding64_mean_at_or_below,
                "upper_exclusive": config.context20_materially_better_if_embedding64_mean_at_or_above,
            },
            "categories": {
                "context20_materially_better": {
                    "embedding64_mean_at_or_above": config.context20_materially_better_if_embedding64_mean_at_or_above,
                    "boundary_inclusive": True,
                },
                "practical_tie": {
                    "lower_exclusive": config.embedding64_materially_better_if_embedding64_mean_at_or_below,
                    "upper_exclusive": config.context20_materially_better_if_embedding64_mean_at_or_above,
                },
                "embedding64_materially_better": {
                    "embedding64_mean_at_or_below": config.embedding64_materially_better_if_embedding64_mean_at_or_below,
                    "boundary_inclusive": True,
                },
            },
            "per_seed_interpretation_generated": False,
            "automatic_decision_generated": False,
            "automatic_report_generated": False,
        },
        "readiness": {
            "relative_path": config.readiness_report_relative_path,
            "sha256": config.readiness_report_sha256,
            "training_prediction_tokens": config.training_prediction_tokens,
            "training_records": config.training_records,
            "native_validation_prediction_tokens": config.native_validation_prediction_tokens,
            "native_validation_records": config.native_validation_records,
        },
        "final_cursor_step_accounting": None
        if state is None
        else {
            "prediction_position": state.predictions_seen,
            "optimizer_steps": state.optimizer_steps,
            "active_learning_rate": learning_rate_for(
                state.predictions_seen, plan.training_config
            ),
            "cursor": {
                "prediction_index": state.cursor.prediction_index,
                "protein_index": state.cursor.protein_index,
                "within_protein_target_offset": state.cursor.within_protein_target_offset,
            },
        },
        "training_loss_lineage": {
            "parent_prediction_count": config.parent_prediction_position,
            "parent_loss_numerator": parent.training_loss_numerator,
            "total_prediction_count": prediction_count,
            "total_loss_numerator": loss,
            "continuation_prediction_count": None
            if prediction_count is None
            else prediction_count - config.parent_prediction_position,
            "continuation_loss_numerator": None
            if loss is None
            else loss - parent.training_loss_numerator,
            "parent_online_cross_entropy": parent.training_loss_numerator
            / config.parent_prediction_position,
            "total_online_cross_entropy": None
            if prediction_count in (None, 0)
            else loss / prediction_count,
            "continuation_online_cross_entropy": None
            if prediction_count in (None, config.parent_prediction_position)
            else (loss - parent.training_loss_numerator)
            / (prediction_count - config.parent_prediction_position),
        },
        "metrics": metrics,
        "checkpoints": checkpoints,
        "runtime": {
            "seconds": runtime,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
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
        if temporary.exists():
            temporary.unlink()
        raise


def _completion_payload(
    seed: int, state: TrainingState, metric: dict[str, object], started: float
) -> dict[str, object]:
    runtime = time.perf_counter() - started
    if not math.isfinite(runtime) or runtime < 0:
        raise ModelDataError("embedding64 challenger completion runtime is invalid")
    return {
        "seed": seed,
        "prediction_position": state.predictions_seen,
        "optimizer_steps": state.optimizer_steps,
        "native_cross_entropy": metric["cross_entropy"],
        "runtime_seconds": runtime,
    }


def _notify(
    callback: ProgressCallback | None, event: str, payload: dict[str, object]
) -> None:
    if callback is not None:
        callback(event, payload)
