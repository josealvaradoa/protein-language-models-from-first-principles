"""Exact event-aware SGD accounting for the frozen Week 3 MLP."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as functional

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.mlp.config import MLPTrainingConfig
from protein_lm.mlp.model import ContextMLP
from protein_lm.mlp.stream import ContextBatch, StreamCursor


@dataclass
class TrainingState:
    predictions_seen: int = 0
    optimizer_steps: int = 0
    training_loss_numerator: float = 0.0
    cursor: StreamCursor = StreamCursor(0, 0, 0)


def new_optimizer(model: ContextMLP, config: MLPTrainingConfig) -> torch.optim.SGD:
    return torch.optim.SGD(
        model.parameters(),
        lr=config.base_learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )


def learning_rate_for(predictions_seen: int, config: MLPTrainingConfig) -> float:
    if type(predictions_seen) is not int or predictions_seen < 0:
        raise ModelDataError("prediction accounting is invalid")
    return (
        config.base_learning_rate
        if predictions_seen < config.learning_rate_boundary_predictions
        else config.post_boundary_learning_rate
    )


def train_batch(
    model: ContextMLP,
    optimizer: torch.optim.SGD,
    batch: ContextBatch,
    state: TrainingState,
    config: MLPTrainingConfig,
) -> float:
    """Apply one scheduled SGD update and assert its exact cursor position."""

    if (
        batch.start_prediction != state.predictions_seen
        or batch.start_cursor != state.cursor
        or batch.predictions <= 0
    ):
        raise ModelDataError("training batch cursor does not match state")
    if state.predictions_seen + batch.predictions > config.prediction_budget:
        raise ModelDataError("training batch exceeds prediction budget")
    expected_lr = learning_rate_for(state.predictions_seen, config)
    if any(group["lr"] != expected_lr for group in optimizer.param_groups):
        for group in optimizer.param_groups:
            group["lr"] = expected_lr
    device = next(model.parameters()).device
    contexts, targets = batch.contexts.to(device), batch.targets.to(device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(contexts)
    loss = functional.cross_entropy(logits, targets, reduction="mean")
    if not torch.isfinite(loss):
        raise ModelDataError("training produced nonfinite loss")
    loss.backward()
    optimizer.step()
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ModelDataError("optimizer step produced nonfinite model parameters")
    loss_value = float(loss.detach().cpu())
    state.training_loss_numerator += loss_value * batch.predictions
    state.predictions_seen += batch.predictions
    state.optimizer_steps += 1
    state.cursor = batch.end_cursor
    if (
        state.predictions_seen in config.event_predictions
        or state.predictions_seen == config.learning_rate_boundary_predictions
    ):
        if state.predictions_seen not in config.event_predictions:
            raise ModelDataError("learning-rate boundary was not a stream event")
    return loss_value


def assert_optimizer_contract(
    optimizer: torch.optim.Optimizer, config: MLPTrainingConfig
) -> None:
    if not isinstance(optimizer, torch.optim.SGD) or len(optimizer.param_groups) != 1:
        raise ModelDataError("optimizer is not the approved SGD form")
    group = optimizer.param_groups[0]
    if group["momentum"] != 0.0 or group["weight_decay"] != 0.0 or optimizer.state:
        raise ModelDataError(
            "checkpoint optimizer state differs from zero-momentum SGD"
        )
