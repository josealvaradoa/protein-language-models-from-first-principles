"""Additive SGD scheduling for the Week 3 exploratory tail arms."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as functional

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.mlp.model import ContextMLP
from protein_lm.mlp.stream import ContextBatch
from protein_lm.mlp.tail_config import MLPTailConfig
from protein_lm.mlp.training import TrainingState


def tail_learning_rate(arm: str, start_prediction: int, config: MLPTailConfig) -> float:
    """Return the LR for a complete inherited batch, never splitting it."""

    if arm not in config.approved_arms:
        raise ModelDataError("tail arm is not approved")
    if (
        type(start_prediction) is not int
        or not config.parent_prediction_position
        <= start_prediction
        < config.final_prediction_position
        or (start_prediction - config.parent_prediction_position) % config.batch_size
        != 0
    ):
        raise ModelDataError("tail batch start is not an inherited batch boundary")
    if arm == "staged_97m_003":
        return (
            config.base_tail_learning_rate
            if start_prediction < config.staged_boundary_prediction
            else config.staged_lower_learning_rate
        )
    if start_prediction == config.parent_prediction_position:
        return config.base_tail_learning_rate
    progress = (start_prediction - config.parent_prediction_position) / (
        config.final_prediction_position - config.parent_prediction_position
    )
    return config.cosine_endpoint_learning_rate + 0.5 * (
        config.base_tail_learning_rate - config.cosine_endpoint_learning_rate
    ) * (1 + math.cos(math.pi * progress))


def staged_effective_boundary(config: MLPTailConfig) -> int:
    """First inherited normal batch start at or after 97M."""

    offset = config.staged_boundary_prediction - config.parent_prediction_position
    batches_before = math.ceil(offset / config.batch_size)
    return config.parent_prediction_position + batches_before * config.batch_size


def tail_last_applied_learning_rate(arm: str, config: MLPTailConfig) -> float:
    remainder = (
        config.final_prediction_position - config.parent_prediction_position
    ) % config.batch_size
    last_size = remainder or config.batch_size
    last_start = config.final_prediction_position - last_size
    return tail_learning_rate(arm, last_start, config)


def schedule_provenance(arm: str, config: MLPTailConfig) -> dict[str, object]:
    """Return only the schedule facts that apply to one approved arm."""

    if arm == "staged_97m_003":
        return {
            "arm": arm,
            "declared_boundary_prediction": config.staged_boundary_prediction,
            "effective_lower_lr_start_prediction": staged_effective_boundary(config),
            "initial_learning_rate": config.base_tail_learning_rate,
            "lower_learning_rate": config.staged_lower_learning_rate,
            "last_applied_learning_rate": tail_last_applied_learning_rate(arm, config),
        }
    if arm == "cosine_90m_100m_001":
        return {
            "arm": arm,
            "formula_identifier": "cosine_90m_100m_001_v1",
            "start_prediction": config.parent_prediction_position,
            "final_prediction": config.final_prediction_position,
            "start_learning_rate": config.base_tail_learning_rate,
            "mathematical_endpoint_learning_rate": config.cosine_endpoint_learning_rate,
            "last_applied_learning_rate": tail_last_applied_learning_rate(arm, config),
        }
    raise ModelDataError("tail arm is not approved")


def train_tail_batch(
    model: ContextMLP,
    optimizer: torch.optim.SGD,
    batch: ContextBatch,
    state: TrainingState,
    config: MLPTailConfig,
    arm: str,
) -> float:
    """Apply one CPU float32 inherited batch under an approved tail schedule."""

    if (
        batch.start_prediction != state.predictions_seen
        or batch.start_cursor != state.cursor
        or batch.predictions <= 0
        or state.predictions_seen < config.parent_prediction_position
        or state.predictions_seen + batch.predictions > config.final_prediction_position
    ):
        raise ModelDataError("tail batch cursor does not match state")
    expected_lr = tail_learning_rate(arm, state.predictions_seen, config)
    if len(optimizer.param_groups) != 1:
        raise ModelDataError("tail optimizer has an invalid parameter-group count")
    optimizer.param_groups[0]["lr"] = expected_lr
    device = next(model.parameters()).device
    contexts, targets = batch.contexts.to(device), batch.targets.to(device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(contexts)
    loss = functional.cross_entropy(logits, targets, reduction="mean")
    if not torch.isfinite(loss):
        raise ModelDataError("tail training produced nonfinite loss")
    loss.backward()
    optimizer.step()
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ModelDataError("tail optimizer step produced nonfinite model parameters")
    loss_value = float(loss.detach().cpu())
    state.training_loss_numerator += loss_value * batch.predictions
    state.predictions_seen += batch.predictions
    state.optimizer_steps += 1
    state.cursor = batch.end_cursor
    return loss_value
