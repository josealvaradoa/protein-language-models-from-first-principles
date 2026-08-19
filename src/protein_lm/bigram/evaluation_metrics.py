"""Float64 metrics for one model on one boundary-safe protein collection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from protein_lm.bigram.stream import protein_pair_bytes
from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.data.model_data.loaders import ProteinSequence


@dataclass(frozen=True)
class AggregateMetrics:
    """Aggregate accounting, including enough parts to independently recompute CE."""

    token_count: int
    protein_count: int
    total_nll: float
    correct_tokens: int
    cross_entropy: float
    accuracy: float
    median_per_protein_nll: float
    median_lower_per_protein_nll: float
    median_upper_per_protein_nll: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "token_count": self.token_count,
            "protein_count": self.protein_count,
            "total_nll": self.total_nll,
            "correct_tokens": self.correct_tokens,
            "cross_entropy": self.cross_entropy,
            "accuracy": self.accuracy,
            "median_per_protein_nll": self.median_per_protein_nll,
            "median_lower_per_protein_nll": self.median_lower_per_protein_nll,
            "median_upper_per_protein_nll": self.median_upper_per_protein_nll,
        }


@dataclass(frozen=True)
class CollectionMetrics:
    """Overall and frozen biological-length-bucket results for one principal record."""

    overall: AggregateMetrics
    buckets: dict[str, AggregateMetrics]

    def as_dict(self) -> dict[str, object]:
        return {
            "overall": self.overall.as_dict(),
            "length_buckets": {
                name: value.as_dict() for name, value in self.buckets.items()
            },
        }


def score_collection(
    proteins: tuple[ProteinSequence, ...] | list[ProteinSequence],
    log_probabilities: torch.Tensor,
    length_buckets: tuple[str, ...],
) -> CollectionMetrics:
    """Score complete proteins only, including BOS and EOS but no cross-protein pair.

    ``log_probabilities`` is precomputed in float64.  The loop therefore only
    gathers values and accumulates Python floats, avoiding lower-precision
    reduction surprises in the reported metrics.
    """

    _validate_log_probabilities(log_probabilities)
    if not proteins:
        raise ModelDataError("evaluation collection is empty")
    if not length_buckets or len(set(length_buckets)) != len(length_buckets):
        raise ModelDataError("evaluation length buckets are invalid")
    accumulators = {name: _Accumulator() for name in length_buckets}
    total = _Accumulator()
    for protein in proteins:
        if protein.length_bucket not in accumulators:
            raise ModelDataError(
                "protein is outside the frozen evaluation length buckets"
            )
        _, contexts, targets = protein_pair_bytes(protein.sequence)
        nll, correct = _score_pairs(contexts, targets, log_probabilities)
        per_protein_nll = nll / len(targets)
        total.add(len(targets), nll, correct, per_protein_nll)
        accumulators[protein.length_bucket].add(
            len(targets), nll, correct, per_protein_nll
        )
    return CollectionMetrics(
        overall=total.finish(),
        buckets={
            name: accumulator.finish() for name, accumulator in accumulators.items()
        },
    )


class _Accumulator:
    def __init__(self) -> None:
        self.tokens = 0
        self.proteins = 0
        self.total_nll = 0.0
        self.correct = 0
        self.per_protein_nll: list[float] = []

    def add(
        self, tokens: int, nll: float, correct: int, per_protein_nll: float
    ) -> None:
        self.tokens += tokens
        self.proteins += 1
        self.total_nll += nll
        self.correct += correct
        self.per_protein_nll.append(per_protein_nll)

    def finish(self) -> AggregateMetrics:
        if self.tokens <= 0 or self.proteins <= 0 or not math.isfinite(self.total_nll):
            raise ModelDataError("evaluation metric accounting is invalid")
        lower, upper = _median_bounds(self.per_protein_nll)
        return AggregateMetrics(
            token_count=self.tokens,
            protein_count=self.proteins,
            total_nll=self.total_nll,
            correct_tokens=self.correct,
            cross_entropy=self.total_nll / self.tokens,
            accuracy=self.correct / self.tokens,
            median_per_protein_nll=(lower + upper) / 2.0,
            median_lower_per_protein_nll=lower,
            median_upper_per_protein_nll=upper,
        )


def _score_pairs(
    contexts: bytes, targets: bytes, log_probabilities: torch.Tensor
) -> tuple[float, int]:
    values = log_probabilities[torch.tensor(list(contexts), dtype=torch.int64)]
    target_tensor = torch.tensor(list(targets), dtype=torch.int64)
    selected = values.gather(1, target_tensor.unsqueeze(1)).squeeze(1)
    nll = -float(selected.sum(dtype=torch.float64).item())
    # torch.argmax returns the first maximum, but write the frozen tie rule here.
    predicted = torch.argmax(values, dim=1)
    correct = int((predicted == target_tensor).sum().item())
    return nll, correct


def _median(values: list[float]) -> float:
    """Keep the teaching helper while the output carries the two exact bounds."""

    lower, upper = _median_bounds(values)
    return (lower + upper) / 2.0


def _median_bounds(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ModelDataError("cannot calculate a median without proteins")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle], ordered[middle]
    return ordered[middle - 1], ordered[middle]


def _validate_log_probabilities(value: torch.Tensor) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float64
        or tuple(value.shape) != (21, 21)
        or not torch.isfinite(value).all().item()
    ):
        raise ModelDataError(
            "evaluation log probabilities must be finite CPU float64 21 by 21 values"
        )
