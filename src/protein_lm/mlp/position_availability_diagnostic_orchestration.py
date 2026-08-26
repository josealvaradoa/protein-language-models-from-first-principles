"""Strict, CPU-only, no-training Week 3 position-availability diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.data.model_data.loaders import (
    ModelDataCollection,
    ProteinSequence,
    load_collection,
)
from protein_lm.mlp.checkpoint import load_checkpoint
from protein_lm.mlp.config import MLPTrainingConfig
from protein_lm.mlp.context20_continuation_config import (
    Context20ContinuationConfig,
    load_context20_continuation_config,
)
from protein_lm.mlp.context20_continuation_orchestration import (
    preflight as context20_preflight,
)
from protein_lm.mlp.embedding64_challenger_config import (
    Embedding64ChallengerConfig,
    load_embedding64_challenger_config,
)
from protein_lm.mlp.embedding64_challenger_orchestration import (
    preflight as embedding64_preflight,
)
from protein_lm.mlp.model import ContextMLP, resolve_device
from protein_lm.mlp.orchestration import (
    _require_ignored,
    _require_revision,
    _verify_source_pins,
)
from protein_lm.mlp.position_availability_diagnostic import (
    BIN_NAMES,
    BinMetrics,
    evaluate_position_availability,
    iter_position_availability_batches,
    overall_metrics,
)
from protein_lm.mlp.position_availability_diagnostic_config import (
    FinalRun,
    PositionAvailabilityDiagnosticConfig,
    config_sha256,
    load_position_availability_diagnostic_config,
)
from protein_lm.mlp.training import new_optimizer


CollectionLoader = Callable[[Path, ModelDataCollection], Iterable[ProteinSequence]]
_CONTEXT20_REVISION = "984551fc6194bb55b55a7a8ecc51dc1be7a661c7"
_EMBEDDING64_REVISION = "d288f33b51a3383b4c3b7cd2a1a952b37f33b2e6"


@dataclass(frozen=True)
class PositionAvailabilityDiagnosticPlan:
    config: PositionAvailabilityDiagnosticConfig
    config_path: Path
    context20_config: Context20ContinuationConfig
    embedding64_config: Embedding64ChallengerConfig
    context20_training_config: MLPTrainingConfig
    embedding64_training_config: MLPTrainingConfig
    run_id: str
    destination: Path


def preflight(root: Path, run_id: str) -> PositionAvailabilityDiagnosticPlan:
    """Read only byte-pinned configurations. This has no operational access."""

    path = root / "experiments/week_03/mlp_position_availability_diagnostic_v1.toml"
    config = load_position_availability_diagnostic_config(path)
    context_path = root / config.context20_config_relative_path
    embedding_path = root / config.embedding64_config_relative_path
    context = load_context20_continuation_config(context_path)
    embedding = load_embedding64_challenger_config(embedding_path)
    context_plan = context20_preflight(root, run_id)
    embedding_plan = embedding64_preflight(root, run_id)
    _validate_paired_training_configs(
        context_plan.training_config, embedding_plan.training_config
    )
    if (
        config_sha256(context_path) != config.context20_config_sha256
        or config_sha256(embedding_path) != config.embedding64_config_sha256
        or context != context_plan.config
        or embedding != embedding_plan.config
        or context.native_validation_collection != config.native_validation_collection
        or embedding.native_validation_collection != config.native_validation_collection
        or context.native_validation_prediction_tokens
        != config.native_validation_prediction_tokens
        or embedding.native_validation_prediction_tokens
        != config.native_validation_prediction_tokens
        or context.native_validation_records != config.native_validation_records
        or embedding.native_validation_records != config.native_validation_records
        or context.batch_size != config.batch_size
        or embedding.batch_size != config.batch_size
    ):
        raise ModelDataError(
            "position-availability diagnostic configuration lineage is invalid"
        )
    return PositionAvailabilityDiagnosticPlan(
        config,
        path,
        context,
        embedding,
        context_plan.training_config,
        embedding_plan.training_config,
        run_id,
        root / config.output_relative_root / run_id,
    )


def _validate_paired_training_configs(
    context: MLPTrainingConfig, embedding: MLPTrainingConfig
) -> None:
    """Fail closed unless both final models consume the same native target order."""

    source_fields = (
        "model_data_config_sha256",
        "model_data_registry_relative_path",
        "model_data_registry_sha256",
        "training_stream_report_relative_path",
        "training_stream_report_sha256",
    )
    shared_fields = ("training_namespace", "stream_base_seed", *source_fields)
    if (
        any(
            getattr(context, field) != getattr(embedding, field)
            for field in shared_fields
        )
        or (context.context_vocab_size, context.target_vocab_size) != (21, 21)
        or (embedding.context_vocab_size, embedding.target_vocab_size) != (21, 21)
        or (
            context.context_length,
            context.embedding_width,
            context.hidden_width,
            context.parameter_count,
        )
        != (20, 32, 800, 530293)
        or (
            embedding.context_length,
            embedding.embedding_width,
            embedding.hidden_width,
            embedding.parameter_count,
        )
        != (10, 64, 800, 530965)
        or context.context_length * context.embedding_width != 640
        or embedding.context_length * embedding.embedding_width != 640
    ):
        raise ModelDataError(
            "position-availability diagnostic paired model invariant is invalid"
        )


def execute_diagnostic(
    *,
    root: Path,
    plan: PositionAvailabilityDiagnosticPlan,
    seed: int,
    device_name: str,
    loader: CollectionLoader = load_collection,
    code_revision: str | None = None,
) -> Path:
    """Evaluate two byte-pinned final checkpoints sequentially, with no training."""

    plan = _require_canonical_plan(root, plan)
    if device_name != "cpu" or plan.config.device != "cpu":
        raise ModelDataError(
            "position-availability diagnostic requires explicit CPU execution"
        )
    if seed not in (20260821, 20260822, 20260823):
        raise ModelDataError("diagnostic seed is not approved")
    if plan.destination.exists():
        raise ModelDataError(
            "diagnostic destination already exists; runs cannot resume"
        )
    revision = _require_revision(root, code_revision)
    _verify_source_pins(root, plan.context20_training_config)
    _verify_source_pins(root, plan.embedding64_training_config)
    _verify_readiness(root, plan)
    _require_ignored(root, plan.destination)
    context_run, embedding_run = (
        plan.config.run("context20", seed),
        plan.config.run("embedding64", seed),
    )
    context_status, context_checkpoint = _source_paths(
        root, plan.context20_training_config.output_relative_root, context_run.run_id
    )
    embedding_status, embedding_checkpoint = _source_paths(
        root,
        plan.embedding64_training_config.output_relative_root,
        embedding_run.run_id,
    )
    _verify_source_artifact(
        context_status,
        context_checkpoint,
        context_run,
        "context20",
        _CONTEXT20_REVISION,
    )
    _verify_source_artifact(
        embedding_status,
        embedding_checkpoint,
        embedding_run,
        "embedding64",
        _EMBEDDING64_REVISION,
    )
    plan.destination.mkdir(parents=True)
    status_path = plan.destination / "run_status.json"
    started = time.perf_counter()
    try:
        native = tuple(loader(root, ModelDataCollection.FAMILY_AWARE_NATIVE_VALIDATION))
        if len(native) != plan.config.native_validation_records:
            raise ModelDataError(
                "native validation record count does not match diagnostic approval"
            )
        target_digest = _target_order_digest(native, plan)
        _write_status(
            status_path,
            _status_payload(
                plan, seed, revision, "running", None, target_digest, None, started
            ),
        )
        context_bins = _evaluate_arm(
            plan, context_run, "context20", context_checkpoint, seed, native
        )
        embedding_bins = _evaluate_arm(
            plan, embedding_run, "embedding64", embedding_checkpoint, seed, native
        )
        _verify_source_artifact(
            context_status,
            context_checkpoint,
            context_run,
            "context20",
            _CONTEXT20_REVISION,
        )
        _verify_source_artifact(
            embedding_status,
            embedding_checkpoint,
            embedding_run,
            "embedding64",
            _EMBEDDING64_REVISION,
        )
        results = _results_payload(context_bins, embedding_bins)
        _write_status(
            status_path,
            _status_payload(
                plan, seed, revision, "passed", results, target_digest, None, started
            ),
        )
    except Exception as error:
        _write_status(
            status_path,
            _status_payload(
                plan, seed, revision, "failed", None, None, str(error), started
            ),
        )
        if isinstance(error, ModelDataError):
            raise
        raise ModelDataError(
            f"position-availability diagnostic failed: {error}"
        ) from error
    return plan.destination


def _require_canonical_plan(
    root: Path, plan: PositionAvailabilityDiagnosticPlan
) -> PositionAvailabilityDiagnosticPlan:
    if not isinstance(plan, PositionAvailabilityDiagnosticPlan):
        raise ModelDataError(
            "position-availability diagnostic execution plan is invalid"
        )
    canonical = preflight(root, plan.run_id)
    if plan != canonical:
        raise ModelDataError(
            "position-availability diagnostic execution plan differs from approval"
        )
    return canonical


def _source_paths(root: Path, output_root: str, run_id: str) -> tuple[Path, Path]:
    base = root / output_root / run_id
    return base / "run_status.json", base / "checkpoint-100000000"


def _verify_source_artifact(
    status_path: Path, checkpoint: Path, run: FinalRun, arm: str, revision: str
) -> None:
    try:
        status_bytes = status_path.read_bytes()
        metadata = (checkpoint / "checkpoint.json").read_bytes()
        tensors = (checkpoint / "model.safetensors").read_bytes()
    except OSError as error:
        raise ModelDataError(f"approved {arm} final source is unavailable") from error
    if (
        hashlib.sha256(status_bytes).hexdigest() != run.run_status_sha256
        or hashlib.sha256(metadata).hexdigest() != run.metadata_sha256
        or hashlib.sha256(tensors).hexdigest() != run.tensor_sha256
    ):
        raise ModelDataError(f"approved {arm} final source bytes do not match pin")
    try:
        status = json.loads(status_bytes)
    except json.JSONDecodeError as error:
        raise ModelDataError(f"approved {arm} final status is malformed") from error
    if (
        not isinstance(status, dict)
        or status.get("status") != "passed"
        or status.get("seed") != run.seed
        or status.get("run_id") != run.run_id
        or status.get("derived_code_revision") != revision
    ):
        raise ModelDataError(f"approved {arm} final status lineage is invalid")


def _verify_readiness(root: Path, plan: PositionAvailabilityDiagnosticPlan) -> None:
    try:
        content = (root / plan.config.readiness_report_relative_path).read_bytes()
    except OSError as error:
        raise ModelDataError(
            "diagnostic readiness evidence is unavailable or malformed"
        ) from error
    if hashlib.sha256(content).hexdigest() != plan.config.readiness_report_sha256:
        raise ModelDataError("diagnostic readiness evidence does not match approval")
    try:
        registry_content = (
            root / plan.context20_training_config.model_data_registry_relative_path
        ).read_bytes()
    except OSError as error:
        raise ModelDataError("diagnostic model-data registry is unavailable") from error
    if (
        hashlib.sha256(registry_content).hexdigest()
        != plan.context20_training_config.model_data_registry_sha256
    ):
        raise ModelDataError(
            "diagnostic model-data registry bytes do not match approval"
        )
    try:
        report = json.loads(content)
        registry = json.loads(registry_content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelDataError(
            "diagnostic readiness evidence is unavailable or malformed"
        ) from error
    native = (
        report.get("collection_aggregates", {}).get(
            plan.config.native_validation_collection, {}
        )
        if isinstance(report, dict)
        else {}
    )
    if (
        not isinstance(registry, dict)
        or registry.get("readiness")
        != {
            "relative_path": plan.config.readiness_report_relative_path,
            "sha256": plan.config.readiness_report_sha256,
        }
        or not isinstance(report, dict)
        or report.get("scope") != "week_02_model_data_readiness"
        or report.get("candidate_status") != "passed"
        or report.get("network_requests_made") != 0
        or not isinstance(native, dict)
        or native.get("prediction_tokens")
        != plan.config.native_validation_prediction_tokens
        or native.get("records") != plan.config.native_validation_records
    ):
        raise ModelDataError("diagnostic readiness evidence does not match approval")


def _target_order_digest(
    proteins: tuple[ProteinSequence, ...], plan: PositionAvailabilityDiagnosticPlan
) -> str:
    digest = hashlib.sha256()
    count = 0
    for batch in iter_position_availability_batches(
        proteins,
        namespace=plan.context20_training_config.training_namespace,
        base_seed=plan.context20_training_config.stream_base_seed,
        batch_size=plan.config.batch_size,
        context_length=1,
    ):
        digest.update(batch.context_batch.targets.numpy().tobytes())
        digest.update(batch.prior_residue_counts.numpy().tobytes())
        count += batch.context_batch.predictions
    if count != plan.config.native_validation_prediction_tokens:
        raise ModelDataError(
            "native validation token count does not match diagnostic approval"
        )
    return digest.hexdigest()


def _evaluate_arm(
    plan: PositionAvailabilityDiagnosticPlan,
    run: FinalRun,
    arm: str,
    checkpoint: Path,
    seed: int,
    native: tuple[ProteinSequence, ...],
) -> dict[str, BinMetrics]:
    training_config = (
        plan.context20_training_config
        if arm == "context20"
        else plan.embedding64_training_config
    )
    config_path = (
        plan.config.context20_config_relative_path
        if arm == "context20"
        else plan.config.embedding64_config_relative_path
    )
    revision = _CONTEXT20_REVISION if arm == "context20" else _EMBEDDING64_REVISION
    model = ContextMLP(training_config, seed, resolve_device("cpu"))
    optimizer = new_optimizer(model, training_config)
    try:
        load_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            config=training_config,
            config_path=plan.config_path.parents[2] / config_path,
            seed=seed,
            run_id=run.run_id,
            device_name="cpu",
            code_revision=revision,
        )
        bins = evaluate_position_availability(
            model,
            iter_position_availability_batches(
                native,
                namespace=training_config.training_namespace,
                base_seed=training_config.stream_base_seed,
                batch_size=plan.config.batch_size,
                context_length=training_config.context_length,
            ),
        )
        _validate_arm_result(bins, run, plan.config)
        return bins
    finally:
        del optimizer
        del model


def _validate_arm_result(
    bins: dict[str, BinMetrics],
    expected: FinalRun,
    config: PositionAvailabilityDiagnosticConfig,
) -> None:
    overall = overall_metrics(bins)
    if (
        overall.token_count != config.native_validation_prediction_tokens
        or overall.correct_predictions != expected.native_correct_predictions
        or abs(overall.nll_numerator - expected.native_nll_numerator)
        > config.overall_metric_absolute_tolerance
        or abs(overall.cross_entropy - expected.native_cross_entropy)
        > config.overall_metric_absolute_tolerance
        or abs(overall.accuracy - expected.native_accuracy)
        > config.overall_metric_absolute_tolerance
    ):
        raise ModelDataError(
            "diagnostic overall metrics do not match pinned final metrics"
        )


def _metric_payload(metric: BinMetrics) -> dict[str, float | int]:
    return {
        "token_count": metric.token_count,
        "nll_numerator": metric.nll_numerator,
        "correct_predictions": metric.correct_predictions,
        "cross_entropy": metric.cross_entropy,
        "accuracy": metric.accuracy,
    }


def _results_payload(
    context_bins: dict[str, BinMetrics], embedding_bins: dict[str, BinMetrics]
) -> dict[str, object]:
    context_overall, embedding_overall = (
        overall_metrics(context_bins),
        overall_metrics(embedding_bins),
    )
    bins = {}
    for name in BIN_NAMES:
        context, embedding = context_bins[name], embedding_bins[name]
        bins[name] = {
            "context20": _metric_payload(context),
            "embedding64": _metric_payload(embedding),
            "embedding64_minus_context20_cross_entropy": embedding.cross_entropy
            - context.cross_entropy,
            "context20_minus_embedding64_accuracy": context.accuracy
            - embedding.accuracy,
        }
    return {
        "bins": bins,
        "overall": {
            "context20": _metric_payload(context_overall),
            "embedding64": _metric_payload(embedding_overall),
            "embedding64_minus_context20_cross_entropy": embedding_overall.cross_entropy
            - context_overall.cross_entropy,
            "context20_minus_embedding64_accuracy": context_overall.accuracy
            - embedding_overall.accuracy,
        },
    }


def _status_payload(
    plan: PositionAvailabilityDiagnosticPlan,
    seed: int,
    revision: str,
    status: str,
    results: dict[str, object] | None,
    target_digest: str | None,
    failure_reason: str | None,
    started: float,
) -> dict[str, object]:
    runtime = time.perf_counter() - started
    if (
        status not in {"running", "passed", "failed"}
        or not math.isfinite(runtime)
        or runtime < 0
    ):
        raise ModelDataError("diagnostic status accounting is invalid")
    return {
        "schema_version": 1,
        "status": status,
        "exploratory_only": True,
        "descriptive_only": True,
        "no_training": True,
        "no_gradients": True,
        "training_predictions": 0,
        "optimizer_steps": 0,
        "backward_passes": 0,
        "sealed_test_accessed": False,
        "models_evaluated_sequentially": True,
        "automatic_selection_generated": False,
        "automatic_report_generated": False,
        "significance_generated": False,
        "non_resumable": True,
        "run_id": plan.run_id,
        "seed": seed,
        "device": "cpu",
        "network_requests_made": 0,
        "contract_identifier": plan.config.contract_identifier,
        "diagnostic_config_sha256": config_sha256(plan.config_path),
        "derived_code_revision": revision,
        "native_validation": {
            "collection": plan.config.native_validation_collection,
            "records": plan.config.native_validation_records,
            "prediction_tokens": plan.config.native_validation_prediction_tokens,
            "ordered_target_and_prior_residue_sha256": target_digest,
        },
        "position_bins": list(BIN_NAMES),
        "comparison_definitions": {
            "embedding64_minus_context20_cross_entropy": "positive values favor C20",
            "context20_minus_embedding64_accuracy": "positive values favor C20",
        },
        "frozen_comparison_provenance": {
            "scope": plan.config.frozen_comparison_scope,
            "context20_mean_native_cross_entropy": plan.config.frozen_context20_mean_native_cross_entropy,
            "context20_sample_standard_deviation": plan.config.frozen_context20_sample_standard_deviation,
            "embedding64_mean_native_cross_entropy": plan.config.frozen_embedding64_mean_native_cross_entropy,
            "embedding64_sample_standard_deviation": plan.config.frozen_embedding64_sample_standard_deviation,
            "embedding64_minus_context20_mean_native_cross_entropy": plan.config.frozen_embedding64_minus_context20_mean_native_cross_entropy,
            "material_gap": plan.config.frozen_material_gap,
            "frozen_category": plan.config.frozen_category,
            "selection_reopened": False,
        },
        "source_provenance": {
            "context20": _run_provenance(
                plan.config.run("context20", seed), _CONTEXT20_REVISION
            ),
            "embedding64": _run_provenance(
                plan.config.run("embedding64", seed), _EMBEDDING64_REVISION
            ),
        },
        "results": results,
        "runtime_seconds": runtime,
        "failure_reason": failure_reason,
    }


def _run_provenance(run: FinalRun, revision: str) -> dict[str, object]:
    return {
        "run_id": run.run_id,
        "run_status_sha256": run.run_status_sha256,
        "checkpoint_metadata_sha256": run.metadata_sha256,
        "checkpoint_tensor_sha256": run.tensor_sha256,
        "code_revision": revision,
        "expected_native_cross_entropy": run.native_cross_entropy,
        "expected_native_accuracy": run.native_accuracy,
        "expected_native_nll_numerator": run.native_nll_numerator,
        "expected_native_correct_predictions": run.native_correct_predictions,
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
