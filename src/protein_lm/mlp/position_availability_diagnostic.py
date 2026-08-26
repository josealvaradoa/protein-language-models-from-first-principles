"""No-training native-validation accounting by available real-residue count."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch
from torch.nn import functional as functional

from protein_lm.bigram.stream import ordered_proteins
from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.data.model_data.loaders import ProteinSequence
from protein_lm.mlp.model import ContextMLP
from protein_lm.mlp.stream import (
    ContextBatch,
    StreamCursor,
    _iter_protein_context_pairs,
)


BIN_NAMES = (
    "available_prior_residues_0_10",
    "available_prior_residues_11_19",
    "available_prior_residues_20_plus",
)


@dataclass(frozen=True)
class PositionAvailabilityBatch:
    context_batch: ContextBatch
    prior_residue_counts: torch.Tensor


@dataclass(frozen=True)
class BinMetrics:
    token_count: int
    nll_numerator: float
    correct_predictions: int

    @property
    def cross_entropy(self) -> float:
        return self.nll_numerator / self.token_count

    @property
    def accuracy(self) -> float:
        return self.correct_predictions / self.token_count


def position_availability_bin(prior_residues: int) -> str:
    """Classify a target by real residues available before it, including EOS."""

    if type(prior_residues) is not int or prior_residues < 0:
        raise ModelDataError("available prior-residue count is invalid")
    if prior_residues <= 10:
        return BIN_NAMES[0]
    if prior_residues <= 19:
        return BIN_NAMES[1]
    return BIN_NAMES[2]


def iter_position_availability_batches(
    proteins: Iterable[ProteinSequence],
    *,
    namespace: str,
    base_seed: int,
    batch_size: int,
    context_length: int,
) -> Iterator[PositionAvailabilityBatch]:
    """Yield the exact native target order plus a bin key for every target once."""

    if type(batch_size) is not int or batch_size <= 0:
        raise ModelDataError("diagnostic batch size is invalid")
    ordered = ordered_proteins(proteins, namespace, base_seed)
    contexts: list[int] = []
    targets: list[int] = []
    priors: list[int] = []
    prediction_index = 0
    start = StreamCursor(0, 0, 0)
    for protein_index, protein in enumerate(ordered):
        for context, target, next_offset in _iter_protein_context_pairs(
            protein.sequence, context_length, 0
        ):
            contexts.extend(context)
            targets.append(target)
            priors.append(next_offset - 1)
            prediction_index += 1
            end = (
                StreamCursor(prediction_index, protein_index + 1, 0)
                if next_offset == protein.biological_length + 1
                else StreamCursor(prediction_index, protein_index, next_offset)
            )
            if len(targets) == batch_size:
                yield _batch(contexts, targets, priors, start, end, context_length)
                contexts, targets, priors, start = [], [], [], end
    if targets:
        terminal = StreamCursor(prediction_index, len(ordered), 0)
        yield _batch(contexts, targets, priors, start, terminal, context_length)
    if prediction_index == 0:
        raise ModelDataError("native validation collection is empty")


def _batch(
    contexts: list[int],
    targets: list[int],
    priors: list[int],
    start: StreamCursor,
    end: StreamCursor,
    context_length: int,
) -> PositionAvailabilityBatch:
    return PositionAvailabilityBatch(
        ContextBatch(
            torch.tensor(contexts, dtype=torch.int64).reshape(-1, context_length),
            torch.tensor(targets, dtype=torch.int64),
            start,
            end,
        ),
        torch.tensor(priors, dtype=torch.int64),
    )


@torch.no_grad()
def evaluate_position_availability(
    model: ContextMLP, batches: Iterable[PositionAvailabilityBatch]
) -> dict[str, BinMetrics]:
    """Evaluate exactly once with no gradients and a token-conserving bin split."""

    was_training = model.training
    model.eval()
    totals = {name: [0, 0.0, 0] for name in BIN_NAMES}
    try:
        for item in batches:
            if not isinstance(item, PositionAvailabilityBatch):
                raise ModelDataError("position-availability batch is invalid")
            batch, priors = item.context_batch, item.prior_residue_counts
            if (
                not isinstance(batch, ContextBatch)
                or batch.contexts.dtype != torch.int64
                or batch.targets.dtype != torch.int64
                or batch.contexts.ndim != 2
                or batch.targets.ndim != 1
                or batch.contexts.shape[0] != batch.targets.shape[0]
                or priors.dtype != torch.int64
                or priors.ndim != 1
                or priors.numel() != batch.predictions
                or batch.predictions <= 0
            ):
                raise ModelDataError(
                    "position-availability batch accounting is invalid"
                )
            device = next(model.parameters()).device
            logits = model(batch.contexts.to(device))
            losses = functional.cross_entropy(
                logits, batch.targets.to(device), reduction="none"
            )
            if not torch.isfinite(losses).all():
                raise ModelDataError("position-availability produced nonfinite loss")
            correct = logits.argmax(dim=1).eq(batch.targets.to(device))
            masks = {
                BIN_NAMES[0]: priors <= 10,
                BIN_NAMES[1]: (priors >= 11) & (priors <= 19),
                BIN_NAMES[2]: priors >= 20,
            }
            if sum(int(mask.sum()) for mask in masks.values()) != batch.predictions:
                raise ModelDataError(
                    "position-availability batch bins do not cover once"
                )
            for name, mask in masks.items():
                count = int(mask.sum())
                if count == 0:
                    continue
                selected = mask.to(device)
                totals[name][0] += count
                totals[name][1] += float(
                    losses[selected].detach().cpu().to(torch.float64).sum()
                )
                totals[name][2] += int(correct[selected].sum().detach().cpu())
    finally:
        model.train(was_training)
    if sum(value[0] for value in totals.values()) == 0:
        raise ModelDataError("position-availability stream is empty")
    return {name: BinMetrics(*values) for name, values in totals.items()}


def overall_metrics(bins: dict[str, BinMetrics]) -> BinMetrics:
    if tuple(bins) != BIN_NAMES or any(
        metric.token_count <= 0 for metric in bins.values()
    ):
        raise ModelDataError("position-availability bins are incomplete")
    return BinMetrics(
        sum(metric.token_count for metric in bins.values()),
        sum(metric.nll_numerator for metric in bins.values()),
        sum(metric.correct_predictions for metric in bins.values()),
    )
