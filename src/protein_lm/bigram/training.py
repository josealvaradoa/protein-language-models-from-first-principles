"""Small, inspectable fitting primitives for the three Week 2 bigram models."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import math

import torch
from torch.nn import functional as functional

from protein_lm.bigram.stream import PairBatch
from protein_lm.data.model_data.contracts import ModelDataError


ROLE_SPACE_SIZE = 21


@dataclass(frozen=True)
class TrainingSettings:
    """Validated execution settings, reusable with small synthetic test budgets."""

    batch_size: int
    prediction_pair_budget: int
    learning_rate: float = 1.0
    momentum: float = 0.0
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (self.batch_size, self.prediction_pair_budget)
        ):
            raise ModelDataError(
                "training batch size and budget must be positive integers"
            )
        numeric_values = (
            self.learning_rate,
            self.momentum,
            self.weight_decay,
        )
        if any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in numeric_values
        ) or (self.learning_rate <= 0 or self.momentum < 0 or self.weight_decay < 0):
            raise ModelDataError("training optimizer settings are invalid")


@dataclass
class BigramTrainingState:
    """Raw count matrices and the one trainable 21 by 21 logit matrix."""

    unigram_counts: torch.Tensor
    count_bigram_counts: torch.Tensor
    neural_weights: torch.Tensor
    optimizer: torch.optim.Optimizer
    pairs_seen: int = 0
    optimizer_steps: int = 0


def new_training_state(settings: TrainingSettings) -> BigramTrainingState:
    """Start the two count models and bias-free neural matrix at their frozen zeros."""

    weights = torch.zeros(
        (ROLE_SPACE_SIZE, ROLE_SPACE_SIZE), dtype=torch.float32, device="cpu"
    ).requires_grad_()
    return BigramTrainingState(
        unigram_counts=torch.zeros(ROLE_SPACE_SIZE, dtype=torch.int64, device="cpu"),
        count_bigram_counts=torch.zeros(
            (ROLE_SPACE_SIZE, ROLE_SPACE_SIZE), dtype=torch.int64, device="cpu"
        ),
        neural_weights=weights,
        optimizer=torch.optim.SGD(
            [weights],
            lr=float(settings.learning_rate),
            momentum=float(settings.momentum),
            weight_decay=float(settings.weight_decay),
        ),
    )


def all_zero_weights_sha256() -> str:
    """Return the checksum of the frozen CPU float32 zero logit matrix."""

    weights = torch.zeros(
        (ROLE_SPACE_SIZE, ROLE_SPACE_SIZE), dtype=torch.float32, device="cpu"
    )
    return hashlib.sha256(weights.numpy().tobytes()).hexdigest()


def update_models_from_batch(state: BigramTrainingState, batch: PairBatch) -> float:
    """Apply one shared batch to both raw counts and one neural SGD step.

    ``weights[contexts]`` selects the same rows as a conceptual one-hot context
    matrix multiplied by ``weights``. It avoids allocating that sparse one-hot
    matrix while preserving the introductory linear-model derivation.
    """

    _validate_batch(batch)
    contexts = torch.tensor(list(batch.contexts), dtype=torch.int64, device="cpu")
    targets = torch.tensor(list(batch.targets), dtype=torch.int64, device="cpu")
    state.unigram_counts += torch.bincount(targets, minlength=ROLE_SPACE_SIZE)
    state.count_bigram_counts.index_put_(
        (contexts, targets),
        torch.ones_like(targets, dtype=torch.int64),
        accumulate=True,
    )
    state.optimizer.zero_grad(set_to_none=True)
    logits = state.neural_weights[contexts]
    loss = functional.cross_entropy(logits, targets, reduction="mean")
    loss.backward()
    state.optimizer.step()
    state.pairs_seen += len(batch.contexts)
    state.optimizer_steps += 1
    return float(loss.detach())


def fit_batches(
    state: BigramTrainingState,
    batches: Iterable[PairBatch],
    settings: TrainingSettings,
) -> tuple[float, ...]:
    """Fit one fresh state on exactly the configured full-and-final schedule."""

    _validate_fresh_state(state, settings)
    expected_steps = math.ceil(settings.prediction_pair_budget / settings.batch_size)
    losses: list[float] = []
    for index, batch in enumerate(batches):
        if not isinstance(batch, PairBatch):
            raise ModelDataError("training stream yielded an invalid batch")
        if index >= expected_steps:
            raise ModelDataError("training stream has too many batches")
        expected_size = (
            settings.batch_size
            if index < expected_steps - 1
            else settings.prediction_pair_budget
            - settings.batch_size * (expected_steps - 1)
        )
        if len(batch.contexts) != expected_size:
            raise ModelDataError("training batch does not match the fixed schedule")
        losses.append(update_models_from_batch(state, batch))
    if (
        len(losses) != expected_steps
        or state.pairs_seen != settings.prediction_pair_budget
        or state.optimizer_steps != expected_steps
    ):
        raise ModelDataError("training stream does not match its configured budget")
    return tuple(losses)


def _validate_batch(batch: PairBatch) -> None:
    if not batch.contexts or len(batch.contexts) != len(batch.targets):
        raise ModelDataError("training batch is empty or has unequal role lengths")
    if max(batch.contexts) >= ROLE_SPACE_SIZE or max(batch.targets) >= ROLE_SPACE_SIZE:
        raise ModelDataError("training batch has a role outside the compact spaces")


def _validate_fresh_state(
    state: BigramTrainingState, settings: TrainingSettings
) -> None:
    """Prevent a rerun or resumed state from masquerading as the planned run."""

    if (
        state.pairs_seen != 0
        or state.optimizer_steps != 0
        or state.optimizer.state
        or state.unigram_counts.device.type != "cpu"
        or state.unigram_counts.dtype != torch.int64
        or tuple(state.unigram_counts.shape) != (ROLE_SPACE_SIZE,)
        or not torch.equal(
            state.unigram_counts,
            torch.zeros(ROLE_SPACE_SIZE, dtype=torch.int64, device="cpu"),
        )
        or state.count_bigram_counts.device.type != "cpu"
        or state.count_bigram_counts.dtype != torch.int64
        or tuple(state.count_bigram_counts.shape) != (ROLE_SPACE_SIZE, ROLE_SPACE_SIZE)
        or not torch.equal(
            state.count_bigram_counts,
            torch.zeros(
                (ROLE_SPACE_SIZE, ROLE_SPACE_SIZE), dtype=torch.int64, device="cpu"
            ),
        )
        or state.neural_weights.device.type != "cpu"
        or state.neural_weights.dtype != torch.float32
        or not state.neural_weights.requires_grad
        or tuple(state.neural_weights.shape) != (ROLE_SPACE_SIZE, ROLE_SPACE_SIZE)
        or not torch.equal(
            state.neural_weights.detach(),
            torch.zeros(
                (ROLE_SPACE_SIZE, ROLE_SPACE_SIZE), dtype=torch.float32, device="cpu"
            ),
        )
    ):
        raise ModelDataError("training state is not a fresh all-zero planned run")
    if (
        not isinstance(state.optimizer, torch.optim.SGD)
        or len(state.optimizer.param_groups) != 1
    ):
        raise ModelDataError("training state optimizer is not the planned SGD run")
    group = state.optimizer.param_groups[0]
    parameters = group["params"]
    if (
        len(parameters) != 1
        or parameters[0] is not state.neural_weights
        or group["lr"] != float(settings.learning_rate)
        or group["momentum"] != float(settings.momentum)
        or group["weight_decay"] != float(settings.weight_decay)
    ):
        raise ModelDataError("training state optimizer is not the planned SGD run")
