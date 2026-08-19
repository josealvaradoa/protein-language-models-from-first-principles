"""Operator-gated scoring orchestration for the fixed Week 2 evaluation matrix."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from pathlib import Path

import torch

from protein_lm.bigram.candidate import preflight as model_candidate_preflight
from protein_lm.bigram.candidate_validation import validate_candidate
from protein_lm.bigram.evaluation_metrics import score_collection
from protein_lm.bigram.evaluation_models import log_probabilities
from protein_lm.bigram.evaluation_plan import (
    EvaluationPlan,
    clean_revision,
    require_ignored,
    validate_plan,
    verify_candidate_provenance,
)
from protein_lm.bigram.evaluation_reporting import (
    registry_payload,
    run_record,
    write_new_json,
    write_run_record,
)
from protein_lm.bigram.evaluation_results import result_payload, validate_records
from protein_lm.bigram.serialization import load_model_artifacts
from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.data.model_data.loaders import (
    ModelDataCollection,
    ProteinSequence,
    load_collection,
)


CollectionLoader = Callable[[Path, ModelDataCollection], Iterable[ProteinSequence]]
_LOAD_NAMES = {
    ModelDataCollection.RANDOM_NATIVE_VALIDATION: "random_native_validation",
    ModelDataCollection.FAMILY_AWARE_NATIVE_VALIDATION: "family_aware_native_validation",
    ModelDataCollection.SHARED_VALIDATION: "shared_validation",
}
_GATES = {
    "candidate_validation",
    "twelve_principal_records",
    "shared_validation_loaded_once",
    "sealed_test_never_loaded",
    "evaluation_only_no_retraining_or_selection",
    "no_network_requests",
}


def execute_evaluation(
    *,
    root: Path,
    plan: EvaluationPlan,
    loader: CollectionLoader = load_collection,
    code_revision: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Score exactly three permitted collections and preserve terminal evidence once."""

    validate_plan(root, plan)
    if plan.destination.exists():
        raise ModelDataError("evaluation destination already exists")
    revision = code_revision or clean_revision(root)
    require_ignored(root, plan.destination)
    plan.destination.parent.mkdir(parents=True, exist_ok=True)
    plan.destination.mkdir()
    started = time.perf_counter()
    loads = _new_load_accounting()
    gates = _new_gates()
    _write_record(plan, revision, "running", started, None, loads, gates)
    try:
        verify_candidate_provenance(plan)
        candidate_plan = model_candidate_preflight(root, plan.config.model_candidate_id)
        if candidate_plan.destination != plan.model_candidate:
            raise ModelDataError(
                "evaluation candidate path disagrees with the approved candidate plan"
            )
        validate_candidate(plan.model_candidate, candidate_plan)
        gates["candidate_validation"] = True
        models = _load_models(plan.model_candidate)
        records: list[dict[str, object]] = []
        random_native = _load(
            loader, root, ModelDataCollection.RANDOM_NATIVE_VALIDATION, loads, progress
        )
        records.extend(
            _score_arm(
                "random_training",
                "random_native_validation",
                random_native,
                models,
                plan,
            )
        )
        family_native = _load(
            loader,
            root,
            ModelDataCollection.FAMILY_AWARE_NATIVE_VALIDATION,
            loads,
            progress,
        )
        records.extend(
            _score_arm(
                "family_aware_training",
                "family_aware_native_validation",
                family_native,
                models,
                plan,
            )
        )
        shared = _load(
            loader, root, ModelDataCollection.SHARED_VALIDATION, loads, progress
        )
        for arm in plan.config.model_arms:
            records.extend(_score_arm(arm, "shared_validation", shared, models, plan))
        validate_records(records, plan.config)
        gates["twelve_principal_records"] = True
        gates["shared_validation_loaded_once"] = loads["shared_validation"] == 1
        result = result_payload(
            config=plan.config,
            config_sha256=plan.config_sha256,
            evaluation_id=plan.evaluation_id,
            records=records,
        )
        write_new_json(plan.destination / "evaluation.json", result)
        _replace_record(plan, revision, "passed", started, None, loads, gates)
        write_new_json(
            plan.destination / "evaluation_registry.json",
            registry_payload(plan.destination, plan.evaluation_id),
        )
    except Exception as error:
        (plan.destination / "evaluation.json").unlink(missing_ok=True)
        (plan.destination / "evaluation_registry.json").unlink(missing_ok=True)
        _replace_record(plan, revision, "failed", started, str(error), loads, gates)
        if isinstance(error, ModelDataError):
            raise
        raise ModelDataError(f"Week 2 bigram evaluation failed: {error}") from error
    return plan.destination


def _load(
    loader: CollectionLoader,
    root: Path,
    collection: ModelDataCollection,
    loads: dict[str, int],
    progress: Callable[[str], None] | None,
) -> tuple[ProteinSequence, ...]:
    _progress(progress, f"scoring {_LOAD_NAMES[collection].replace('_', ' ')}")
    loads[_LOAD_NAMES[collection]] += 1
    return tuple(loader(root, collection))


def _load_models(candidate: Path) -> dict[tuple[str, str], torch.Tensor]:
    models: dict[tuple[str, str], torch.Tensor] = {}
    for arm in ("random_training", "family_aware_training"):
        for model_type in ("unigram", "count_bigram", "neural_bigram"):
            found, tensor, metadata = load_model_artifacts(
                json_path=candidate / f"{arm}__{model_type}.json",
                safetensors_path=candidate / f"{arm}__{model_type}.safetensors",
            )
            if found != model_type or metadata.get("arm") != arm:
                raise ModelDataError(
                    "evaluation model provenance disagrees with its filename"
                )
            models[(arm, model_type)] = log_probabilities(found, tensor)
    return models


def _score_arm(
    arm: str,
    collection: str,
    proteins: tuple[ProteinSequence, ...],
    models: dict[tuple[str, str], torch.Tensor],
    plan: EvaluationPlan,
) -> list[dict[str, object]]:
    return [
        {
            "model_arm": arm,
            "model_type": model_type,
            "collection": collection,
            "metrics": score_collection(
                proteins, models[(arm, model_type)], plan.config.length_buckets
            ).as_dict(),
        }
        for model_type in plan.config.model_types
    ]


def _new_load_accounting() -> dict[str, int]:
    return {
        "random_native_validation": 0,
        "family_aware_native_validation": 0,
        "shared_validation": 0,
        "shared_sealed_test": 0,
    }


def _new_gates() -> dict[str, bool]:
    return {
        "candidate_validation": False,
        "twelve_principal_records": False,
        "shared_validation_loaded_once": False,
        "sealed_test_never_loaded": True,
        "evaluation_only_no_retraining_or_selection": True,
        "no_network_requests": True,
    }


def _write_record(
    plan: EvaluationPlan,
    revision: str,
    status: str,
    started: float,
    failure_reason: str | None,
    loads: dict[str, int],
    gates: dict[str, bool],
) -> None:
    write_run_record(
        plan.destination / "run_record.json",
        run_record(
            config=plan.config,
            evaluation_id=plan.evaluation_id,
            revision=revision,
            status=status,
            started=started,
            failure_reason=failure_reason,
            configuration_sha256=plan.config_sha256,
            collection_loads=loads,
            hard_gates=gates,
        ),
    )


def _replace_record(
    plan: EvaluationPlan,
    revision: str,
    status: str,
    started: float,
    failure_reason: str | None,
    loads: dict[str, int],
    gates: dict[str, bool],
) -> None:
    _write_record(plan, revision, status, started, failure_reason, loads, gates)


def _progress(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)
