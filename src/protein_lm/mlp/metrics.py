"""Streaming, token-weighted native-validation metrics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch.nn import functional as functional

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.mlp.model import ContextMLP
from protein_lm.mlp.stream import ContextBatch


@dataclass(frozen=True)
class NativeMetrics:
    predictions: int
    nll_numerator: float
    correct_predictions: int

    @property
    def cross_entropy(self) -> float:
        return self.nll_numerator / self.predictions

    @property
    def accuracy(self) -> float:
        return self.correct_predictions / self.predictions


@torch.no_grad()
def evaluate_native(
    model: ContextMLP, batches: Iterable[ContextBatch]
) -> NativeMetrics:
    """Evaluate incrementally, weighting each batch by its actual token count."""

    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_correct = 0
    total_predictions = 0
    try:
        for batch in batches:
            if (
                not isinstance(batch, ContextBatch)
                or batch.contexts.dtype != torch.int64
                or batch.targets.dtype != torch.int64
                or batch.contexts.ndim != 2
                or batch.targets.ndim != 1
                or batch.contexts.shape[0] != batch.targets.shape[0]
                or batch.predictions <= 0
            ):
                raise ModelDataError("native validation batch is invalid")
            contexts = batch.contexts.to(next(model.parameters()).device)
            targets = batch.targets.to(next(model.parameters()).device)
            logits = model(contexts)
            losses = functional.cross_entropy(logits, targets, reduction="none")
            if not torch.isfinite(losses).all():
                raise ModelDataError("native validation produced nonfinite loss")
            total_nll += float(losses.detach().cpu().to(torch.float64).sum())
            total_correct += int((logits.argmax(dim=1) == targets).sum().cpu())
            total_predictions += batch.predictions
    finally:
        model.train(was_training)
    if total_predictions == 0:
        raise ModelDataError("native validation stream is empty")
    return NativeMetrics(total_predictions, total_nll, total_correct)
