"""Probability-table construction for the three already-fitted Week 2 models."""

from __future__ import annotations

import torch

from protein_lm.bigram.serialization import ModelType
from protein_lm.data.model_data.contracts import ModelDataError


def log_probabilities(model_type: ModelType, parameters: torch.Tensor) -> torch.Tensor:
    """Convert frozen artifacts into one float64 context-by-target log-probability table.

    Counts use add-one smoothing over all 21 valid targets.  Neural parameters
    are frozen logits and receive only a float64 log-softmax, never an update.
    """

    if model_type == "unigram":
        _require(parameters, (21,), torch.int64)
        counts = parameters.to(dtype=torch.float64)
        row = torch.log((counts + 1.0) / (counts.sum() + 21.0))
        return row.repeat(21, 1)
    if model_type == "count_bigram":
        _require(parameters, (21, 21), torch.int64)
        counts = parameters.to(dtype=torch.float64)
        return torch.log((counts + 1.0) / (counts.sum(dim=1, keepdim=True) + 21.0))
    if model_type == "neural_bigram":
        _require(parameters, (21, 21), torch.float32)
        return torch.log_softmax(parameters.to(dtype=torch.float64), dim=1)
    raise ModelDataError("evaluation model type is invalid")


def _require(
    parameters: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    if (
        not isinstance(parameters, torch.Tensor)
        or parameters.device.type != "cpu"
        or parameters.dtype != dtype
        or tuple(parameters.shape) != shape
        or (dtype.is_floating_point and not torch.isfinite(parameters).all().item())
        or (not dtype.is_floating_point and torch.any(parameters < 0).item())
    ):
        raise ModelDataError(
            "evaluation model parameters have an invalid type, shape, or value"
        )
