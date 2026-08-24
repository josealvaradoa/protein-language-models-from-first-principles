"""Fixed-LR update mechanics for the one-epoch continuation diagnostic."""

from __future__ import annotations

import torch
from torch.nn import functional as functional

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.mlp.model import ContextMLP
from protein_lm.mlp.one_epoch_config import OneEpochContinuationConfig
from protein_lm.mlp.stream import ContextBatch
from protein_lm.mlp.training import TrainingState


def train_continuation_batch(
    model: ContextMLP,
    optimizer: torch.optim.SGD,
    batch: ContextBatch,
    state: TrainingState,
    config: OneEpochContinuationConfig,
) -> float:
    """Apply one exact CPU float32 continuation batch at the fixed 0.01 LR."""

    if (
        batch.start_prediction != state.predictions_seen
        or batch.start_cursor != state.cursor
        or batch.predictions <= 0
        or state.predictions_seen < config.parent_prediction_position
        or state.predictions_seen + batch.predictions > config.final_prediction_position
    ):
        raise ModelDataError("continuation batch cursor does not match state")
    expected_predictions = min(
        config.batch_size,
        config.final_prediction_position - state.predictions_seen,
    )
    if batch.predictions != expected_predictions:
        raise ModelDataError(
            "continuation events must preserve full batches before the final batch"
        )
    if len(optimizer.param_groups) != 1:
        raise ModelDataError("continuation optimizer has an invalid parameter group")
    optimizer.param_groups[0]["lr"] = config.fixed_learning_rate
    device = next(model.parameters()).device
    contexts, targets = batch.contexts.to(device), batch.targets.to(device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(contexts)
    loss = functional.cross_entropy(logits, targets, reduction="mean")
    if not torch.isfinite(loss):
        raise ModelDataError("continuation training produced nonfinite loss")
    loss.backward()
    optimizer.step()
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ModelDataError("continuation optimizer produced nonfinite parameters")
    loss_value = float(loss.detach().cpu())
    state.training_loss_numerator += loss_value * batch.predictions
    state.predictions_seen += batch.predictions
    state.optimizer_steps += 1
    state.cursor = batch.end_cursor
    return loss_value
