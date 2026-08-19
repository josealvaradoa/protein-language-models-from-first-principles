"""One-pass fitting and model artifact installation for one bigram arm."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from protein_lm.bigram.serialization import write_model_artifacts
from protein_lm.bigram.stream import (
    ArmStreamAudit,
    PairBatch,
    iter_pair_batches,
    new_stream_hasher,
)
from protein_lm.bigram.training import TrainingSettings, fit_batches, new_training_state
from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.data.model_data.loaders import ProteinSequence

if TYPE_CHECKING:
    from protein_lm.bigram.candidate_contract import AuditedArm, CandidatePlan


def fit_arm(
    *,
    proteins: Iterable[ProteinSequence],
    audited_arm: AuditedArm,
    plan: CandidatePlan,
    settings: TrainingSettings,
):
    """Use each batch once for all three models and audit it as it is consumed."""

    state = new_training_state(settings)
    observed = _ObservedStream(
        namespace=audited_arm.namespace,
        hash_domain=plan.stream_config.stream_hash_domain,
        base_seed=plan.training_config.base_seed,
    )
    losses = fit_batches(
        state,
        _observed_batches(
            proteins,
            audited_arm,
            plan.training_config.base_seed,
            settings,
            observed,
        ),
        settings,
    )
    audit = observed.finish()
    expected = ArmStreamAudit(
        namespace=audited_arm.namespace,
        pairs_emitted=settings.prediction_pair_budget,
        proteins_started=audited_arm.proteins_started,
        proteins_completed=audited_arm.proteins_completed,
        final_protein_partial=audited_arm.final_protein_partial,
        context_counts=audited_arm.context_counts,
        target_counts=audited_arm.target_counts,
        stream_sha256=audited_arm.stream_sha256,
    )
    if audit != expected:
        raise ModelDataError("fitted stream does not equal the audited stream commitment")
    return state, losses, audit


def write_arm_artifacts(
    *,
    destination,
    arm: AuditedArm,
    plan: CandidatePlan,
    state,
    code_revision: str,
) -> None:
    """Install exactly three dual serializations for a completed arm."""

    tensors = {
        "unigram": state.unigram_counts,
        "count_bigram": state.count_bigram_counts,
        "neural_bigram": state.neural_weights.detach(),
    }
    for model_type, tensor in tensors.items():
        write_model_artifacts(
            json_path=destination / f"{arm.collection}__{model_type}.json",
            safetensors_path=destination / f"{arm.collection}__{model_type}.safetensors",
            model_type=model_type,  # type: ignore[arg-type]
            tensor=tensor,
            metadata=_artifact_metadata(arm, plan, model_type, code_revision),
        )


def _observed_batches(
    proteins: Iterable[ProteinSequence],
    audited_arm: AuditedArm,
    base_seed: int,
    settings: TrainingSettings,
    observed: "_ObservedStream",
) -> Iterator[PairBatch]:
    for batch in iter_pair_batches(
        proteins,
        namespace=audited_arm.namespace,
        base_seed=base_seed,
        pair_budget=settings.prediction_pair_budget,
        batch_size=settings.batch_size,
    ):
        observed.update(batch)
        yield batch


@dataclass
class _ObservedStream:
    namespace: str
    hash_domain: str
    base_seed: int

    def __post_init__(self) -> None:
        self.hasher = new_stream_hasher(
            self.hash_domain, self.namespace, self.base_seed
        )
        self.contexts: Counter[int] = Counter()
        self.targets: Counter[int] = Counter()
        self.pairs = 0
        self.started = 0
        self.completed = 0
        self.partial = False

    def update(self, batch: PairBatch) -> None:
        self.hasher.update(batch.pair_bytes)
        self.contexts.update(batch.contexts)
        self.targets.update(batch.targets)
        self.pairs += len(batch.contexts)
        self.started += batch.proteins_started
        self.completed += batch.proteins_completed
        self.partial = self.partial or batch.final_protein_partial

    def finish(self) -> ArmStreamAudit:
        return ArmStreamAudit(
            namespace=self.namespace,
            pairs_emitted=self.pairs,
            proteins_started=self.started,
            proteins_completed=self.completed,
            final_protein_partial=self.partial,
            context_counts=tuple(self.contexts[index] for index in range(21)),
            target_counts=tuple(self.targets[index] for index in range(21)),
            stream_sha256=self.hasher.hexdigest(),
        )


def _artifact_metadata(
    arm: AuditedArm, plan: CandidatePlan, model_type: str, code_revision: str
) -> dict[str, object]:
    neural = model_type == "neural_bigram"
    return {
        "arm": arm.collection,
        "model_type": model_type,
        "context_roles": list(plan.training_config.context_roles),
        "target_roles": list(plan.training_config.target_roles),
        "stream_sha256": arm.stream_sha256,
        "config_sha256": plan.training_config_sha256,
        "source_identity": plan.source_identity,
        "code_revision": code_revision,
        "seed": plan.training_config.base_seed,
        "prediction_pair_budget": plan.training_config.prediction_pair_budget,
        "batch_size": plan.training_config.batch_size,
        "batches_consumed": plan.training_config.total_optimizer_steps,
        "optimizer_steps": plan.training_config.total_optimizer_steps if neural else 0,
        "smoothing_alpha": None if neural else plan.training_config.count_smoothing_alpha,
        "initial_weights_sha256": (
            plan.training_config.initial_weights_sha256 if neural else None
        ),
        "optimizer": (
            {
                "name": plan.training_config.optimizer,
                "learning_rate": plan.training_config.learning_rate,
                "momentum": plan.training_config.momentum,
                "weight_decay": plan.training_config.weight_decay,
            }
            if neural
            else None
        ),
    }
