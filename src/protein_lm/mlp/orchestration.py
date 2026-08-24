"""Operator-gated Week 3 runs, with durable local progress records only."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import torch

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.data.model_data.loaders import (
    ModelDataCollection,
    ProteinSequence,
    load_collection,
)
from protein_lm.mlp.checkpoint import load_checkpoint, save_checkpoint
from protein_lm.mlp.config import MLPTrainingConfig, config_sha256, load_config
from protein_lm.mlp.metrics import NativeMetrics, evaluate_native
from protein_lm.mlp.model import ContextMLP, resolve_device
from protein_lm.mlp.stream import iter_context_batches, iter_native_context_batches
from protein_lm.mlp.training import TrainingState, new_optimizer, train_batch


_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
CollectionLoader = Callable[[Path, ModelDataCollection], Iterable[ProteinSequence]]
ProgressCallback = Callable[[str, dict[str, object]], None]


@dataclass(frozen=True)
class MLPPlan:
    config: MLPTrainingConfig
    config_path: Path
    run_id: str
    destination: Path


def preflight(root: Path, run_id: str) -> MLPPlan:
    """Read only pinned configuration bytes, without corpus/device/output access."""

    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ModelDataError(
            "run identifier must be 3-64 lowercase letters, digits, or hyphens"
        )
    config_path = root / "experiments/week_03/mlp_training_v1.toml"
    config = load_config(config_path)
    return MLPPlan(
        config, config_path, run_id, root / config.output_relative_root / run_id
    )


def execute_run(
    *,
    root: Path,
    plan: MLPPlan,
    seed: int,
    device_name: str,
    loader: CollectionLoader = load_collection,
    resume_checkpoint: Path | None = None,
    code_revision: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Run/resume one approved seed, reaching only the two family-aware collections."""

    if seed not in plan.config.run_seeds:
        raise ModelDataError("new run seed is not approved")
    revision = _require_revision(root, code_revision)
    _verify_source_pins(root, plan.config)
    device = resolve_device(device_name)
    status_path = plan.destination / "run_status.json"
    if resume_checkpoint is None:
        if plan.destination.exists():
            raise ModelDataError("run destination already exists")
        _require_ignored(root, plan.destination)
        plan.destination.mkdir(parents=True)
        record = _new_record(plan, seed, device_name, revision)
    else:
        if (
            not plan.destination.is_dir()
            or resume_checkpoint.resolve().parent != plan.destination.resolve()
        ):
            raise ModelDataError("resume checkpoint must belong to the named run")
        record = _load_record(status_path, plan, seed, device_name, revision)
        if record["status"] == "passed":
            raise ModelDataError("a passed run cannot be resumed")
        if (
            resume_checkpoint.resolve()
            != _latest_checkpoint(plan.destination, plan.config).resolve()
        ):
            raise ModelDataError(
                "resume checkpoint is not the latest installed checkpoint"
            )
    started = time.perf_counter()
    runtime_base = float(record["cumulative_runtime_seconds"])
    state: TrainingState | None = (
        _state_from_record(record) if resume_checkpoint is not None else None
    )
    metrics = list(record["native_validation"])
    try:
        _write_record(
            status_path, record, "running", state, metrics, runtime_base, started, None
        )
        training = tuple(loader(root, ModelDataCollection.FAMILY_AWARE_TRAINING))
        native_validation = tuple(
            loader(root, ModelDataCollection.FAMILY_AWARE_NATIVE_VALIDATION)
        )
        model = ContextMLP(plan.config, seed, device)
        optimizer = new_optimizer(model, plan.config)
        if resume_checkpoint is not None:
            state = load_checkpoint(
                resume_checkpoint,
                model=model,
                optimizer=optimizer,
                config=plan.config,
                config_path=plan.config_path,
                seed=seed,
                run_id=plan.run_id,
                device_name=device_name,
                code_revision=revision,
            )
            metrics = [
                item
                for item in metrics
                if item["predictions"] <= state.predictions_seen
            ]
            _write_record(
                status_path,
                record,
                "running",
                state,
                metrics,
                runtime_base,
                started,
                None,
            )
        else:
            state = TrainingState()
        for batch in iter_context_batches(
            training,
            namespace=plan.config.training_namespace,
            base_seed=plan.config.stream_base_seed,
            prediction_budget=plan.config.prediction_budget,
            batch_size=plan.config.batch_size,
            event_predictions=plan.config.event_predictions,
            cursor=state.cursor,
            context_length=plan.config.context_length,
        ):
            train_batch(model, optimizer, batch, state, plan.config)
            if state.predictions_seen in plan.config.milestone_predictions:
                metric = evaluate_native(
                    model,
                    iter_native_context_batches(
                        native_validation,
                        namespace=plan.config.training_namespace,
                        base_seed=plan.config.stream_base_seed,
                        batch_size=plan.config.batch_size,
                        context_length=plan.config.context_length,
                    ),
                )
                metrics = _append_metric(metrics, state.predictions_seen, metric)
                _write_record(
                    status_path,
                    record,
                    "running",
                    state,
                    metrics,
                    runtime_base,
                    started,
                    None,
                )
                _notify(progress_callback, "milestone", metrics[-1])
            if state.predictions_seen in plan.config.checkpoint_predictions:
                _write_record(
                    status_path,
                    record,
                    "running",
                    state,
                    metrics,
                    runtime_base,
                    started,
                    None,
                )
                checkpoint = save_checkpoint(
                    plan.destination / f"checkpoint-{state.predictions_seen}",
                    model=model,
                    optimizer=optimizer,
                    state=state,
                    config=plan.config,
                    config_path=plan.config_path,
                    seed=seed,
                    run_id=plan.run_id,
                    device_name=device_name,
                    code_revision=revision,
                )
                _notify(
                    progress_callback,
                    "checkpoint",
                    {"path": str(checkpoint), "predictions": state.predictions_seen},
                )
        if state.predictions_seen != plan.config.prediction_budget:
            raise ModelDataError("run ended before the frozen prediction budget")
        if {item["predictions"] for item in metrics} != set(
            plan.config.milestone_predictions
        ):
            raise ModelDataError(
                "all native-validation milestones must be recorded before passing"
            )
        _write_record(
            status_path, record, "passed", state, metrics, runtime_base, started, None
        )
    except Exception as error:
        _write_record(
            status_path,
            record,
            "failed",
            state,
            metrics,
            runtime_base,
            started,
            str(error),
        )
        if isinstance(error, ModelDataError):
            raise
        raise ModelDataError(f"MLP training failed: {error}") from error
    return plan.destination


def _new_record(
    plan: MLPPlan, seed: int, device: str, revision: str
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "new",
        "run_id": plan.run_id,
        "seed": seed,
        "device": device,
        "code_revision": revision,
        "contract_identifier": plan.config.contract_identifier,
        "config_sha256": config_sha256(plan.config_path),
        "source_identity": _source_identity(plan.config),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "parameter_count": plan.config.parameter_count,
        "cumulative_runtime_seconds": 0.0,
        "native_validation": [],
    }


def _load_record(
    path: Path, plan: MLPPlan, seed: int, device: str, revision: str
) -> dict[str, object]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelDataError("run status is unreadable") from error
    required = set(_new_record(plan, seed, device, revision)) | {
        "predictions_seen",
        "optimizer_steps",
        "training_loss_numerator",
        "cursor",
        "training_cross_entropy",
        "throughput_predictions_per_second",
        "failure_reason",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise ModelDataError("run status schema is invalid")
    identity = _new_record(plan, seed, device, revision)
    for key in identity:
        if (
            key not in {"status", "cumulative_runtime_seconds", "native_validation"}
            and record[key] != identity[key]
        ):
            raise ModelDataError("run status identity differs from requested resume")
    _validate_metrics(record["native_validation"], plan.config)
    _validate_resume_record(record, plan.config)
    return record


def _write_record(
    path: Path,
    record: dict[str, object],
    status: str,
    state: TrainingState | None,
    metrics: list[dict[str, object]],
    runtime_base: float,
    started: float,
    failure_reason: str | None,
) -> None:
    elapsed = time.perf_counter() - started
    if not isinstance(runtime_base, float) or runtime_base < 0:
        raise ModelDataError("run status runtime is invalid")
    predictions = None if state is None else state.predictions_seen
    payload = {
        **record,
        "status": status,
        "cumulative_runtime_seconds": runtime_base + elapsed,
        "native_validation": metrics,
        "predictions_seen": predictions,
        "optimizer_steps": None if state is None else state.optimizer_steps,
        "training_loss_numerator": None
        if state is None
        else state.training_loss_numerator,
        "cursor": None if state is None else _cursor_payload(state),
        "training_cross_entropy": None
        if state is None or predictions == 0
        else state.training_loss_numerator / predictions,
        "throughput_predictions_per_second": None
        if state is None or runtime_base + elapsed <= 0
        else state.predictions_seen / (runtime_base + elapsed),
        "failure_reason": failure_reason,
    }
    record.update(payload)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _state_from_record(record: dict[str, object]) -> TrainingState:
    cursor = record.get("cursor")
    if (
        not isinstance(cursor, dict)
        or set(cursor)
        != {"prediction_index", "protein_index", "within_protein_target_offset"}
        or any(type(value) is not int for value in cursor.values())
        or type(record.get("predictions_seen")) is not int
        or type(record.get("optimizer_steps")) is not int
        or type(record.get("training_loss_numerator")) not in (int, float)
    ):
        raise ModelDataError("resume run status progress is invalid")
    from protein_lm.mlp.stream import StreamCursor

    state = TrainingState(
        record["predictions_seen"],
        record["optimizer_steps"],
        float(record["training_loss_numerator"]),
        StreamCursor(
            cursor["prediction_index"],
            cursor["protein_index"],
            cursor["within_protein_target_offset"],
        ),
    )
    if state.cursor.prediction_index != state.predictions_seen:
        raise ModelDataError("resume run status cursor is invalid")
    return state


def _append_metric(
    records: list[dict[str, object]], predictions: int, metric: NativeMetrics
) -> list[dict[str, object]]:
    if any(item["predictions"] == predictions for item in records):
        return records
    return records + [
        {
            "predictions": predictions,
            "token_count": metric.predictions,
            "nll_numerator": metric.nll_numerator,
            "correct_predictions": metric.correct_predictions,
            "cross_entropy": metric.cross_entropy,
            "accuracy": metric.accuracy,
        }
    ]


def _validate_metrics(value: object, config: MLPTrainingConfig) -> None:
    if not isinstance(value, list):
        raise ModelDataError("run status native-validation records are invalid")
    seen: set[int] = set()
    for item in value:
        if (
            not isinstance(item, dict)
            or set(item)
            != {
                "predictions",
                "token_count",
                "nll_numerator",
                "correct_predictions",
                "cross_entropy",
                "accuracy",
            }
            or type(item["predictions"]) is not int
            or item["predictions"] not in config.milestone_predictions
            or item["predictions"] in seen
        ):
            raise ModelDataError("run status native-validation records are invalid")
        if (
            type(item["token_count"]) is not int
            or item["token_count"] <= 0
            or type(item["correct_predictions"]) is not int
            or not 0 <= item["correct_predictions"] <= item["token_count"]
            or any(
                type(item[name]) not in (int, float) or not math.isfinite(item[name])
                for name in ("nll_numerator", "cross_entropy", "accuracy")
            )
            or item["nll_numerator"] < 0
            or item["cross_entropy"] != item["nll_numerator"] / item["token_count"]
            or item["accuracy"] != item["correct_predictions"] / item["token_count"]
        ):
            raise ModelDataError("run status native-validation arithmetic is invalid")
        seen.add(item["predictions"])
    if [item["predictions"] for item in value] != sorted(seen):
        raise ModelDataError("run status milestones are not ordered")
    if len({item["token_count"] for item in value}) > 1:
        raise ModelDataError("run status native token counts disagree")


def _validate_resume_record(
    record: dict[str, object], config: MLPTrainingConfig
) -> None:
    if record["status"] not in {"running", "failed"}:
        raise ModelDataError("run status is not resumable")
    if (
        type(record["cumulative_runtime_seconds"]) not in (int, float)
        or not math.isfinite(record["cumulative_runtime_seconds"])
        or record["cumulative_runtime_seconds"] < 0
        or record["failure_reason"] is not None
        and not isinstance(record["failure_reason"], str)
    ):
        raise ModelDataError("run status runtime or failure reason is invalid")
    state = _state_from_record(record)
    cumulative_runtime = record["cumulative_runtime_seconds"]
    training_ce = record["training_cross_entropy"]
    throughput = record["throughput_predictions_per_second"]
    if (
        state.predictions_seen <= 0
        or not math.isfinite(state.training_loss_numerator)
        or state.training_loss_numerator < 0
        or state.optimizer_steps
        != config.expected_optimizer_steps(state.predictions_seen)
        or type(training_ce) not in (int, float)
        or not math.isfinite(training_ce)
        or training_ce != state.training_loss_numerator / state.predictions_seen
        or type(throughput) not in (int, float)
        or not math.isfinite(throughput)
        or throughput < 0
        or cumulative_runtime <= 0
        or throughput != state.predictions_seen / cumulative_runtime
        or any(
            item["predictions"] > state.predictions_seen
            for item in record["native_validation"]
        )
    ):
        raise ModelDataError("run status progress accounting is invalid")


def _latest_checkpoint(destination: Path, config: MLPTrainingConfig) -> Path:
    checkpoints = []
    for path in destination.glob("checkpoint-*"):
        try:
            predictions = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        if path.is_dir() and predictions in config.checkpoint_predictions:
            checkpoints.append((predictions, path))
    if not checkpoints:
        raise ModelDataError("run has no installed checkpoint to resume")
    return max(checkpoints)[1]


def _require_revision(root: Path, injected: str | None) -> str:
    if injected is not None:
        if _REVISION.fullmatch(injected) is None:
            raise ModelDataError("injected code revision is invalid")
        return injected
    try:
        status = subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.SubprocessError as error:
        raise ModelDataError("could not establish clean code revision") from error
    if status or _REVISION.fullmatch(revision) is None:
        raise ModelDataError("production execution requires a clean committed revision")
    return revision


def _verify_source_pins(root: Path, config: MLPTrainingConfig) -> None:
    for relative, expected in (
        (config.model_data_registry_relative_path, config.model_data_registry_sha256),
        (
            config.training_stream_report_relative_path,
            config.training_stream_report_sha256,
        ),
        (
            "experiments/week_02/model_data_readiness.toml",
            config.model_data_config_sha256,
        ),
    ):
        try:
            actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        except OSError as error:
            raise ModelDataError("approved Week 2 source pin is unavailable") from error
        if actual != expected:
            raise ModelDataError("approved Week 2 source pin drifted")


def _source_identity(config: MLPTrainingConfig) -> dict[str, str]:
    return {
        "model_data_config_sha256": config.model_data_config_sha256,
        "model_data_registry_relative_path": config.model_data_registry_relative_path,
        "model_data_registry_sha256": config.model_data_registry_sha256,
        "training_stream_report_relative_path": config.training_stream_report_relative_path,
        "training_stream_report_sha256": config.training_stream_report_sha256,
    }


def _cursor_payload(state: TrainingState) -> dict[str, int]:
    return {
        "prediction_index": state.cursor.prediction_index,
        "protein_index": state.cursor.protein_index,
        "within_protein_target_offset": state.cursor.within_protein_target_offset,
    }


def _require_ignored(root: Path, destination: Path) -> None:
    try:
        relative = destination.resolve().relative_to(root.resolve())
        result = subprocess.run(
            ("git", "check-ignore", "--quiet", "--", str(relative)),
            cwd=root,
            check=False,
        )
    except (OSError, ValueError) as error:
        raise ModelDataError("could not prove MLP output path is ignored") from error
    if result.returncode != 0:
        raise ModelDataError("MLP output path is not ignored")


def _notify(
    callback: ProgressCallback | None, event: str, payload: dict[str, object]
) -> None:
    if callback is not None:
        callback(event, payload)
