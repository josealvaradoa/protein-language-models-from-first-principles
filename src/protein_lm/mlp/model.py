"""The exact Week 3 learned-context embedding MLP."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.mlp.config import MLPTrainingConfig


class ContextMLP(nn.Module):
    """Embedding, flatten, affine+tanh, affine logits. Nothing is implicit."""

    def __init__(
        self, config: MLPTrainingConfig, seed: int, device: torch.device
    ) -> None:
        super().__init__()
        if type(seed) is not int or seed not in config.run_seeds:
            raise ModelDataError("MLP seed is not an approved isolated run seed")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        # CPU creation fixes the random stream across supported execution devices.
        self.embedding = nn.Parameter(
            torch.empty(
                (config.context_vocab_size, config.embedding_width),
                device="cpu",
                dtype=torch.float32,
            )
        )
        self.w1 = nn.Parameter(
            torch.empty(
                (config.context_length * config.embedding_width, config.hidden_width),
                device="cpu",
                dtype=torch.float32,
            )
        )
        self.b1 = nn.Parameter(
            torch.empty(config.hidden_width, device="cpu", dtype=torch.float32)
        )
        self.w2 = nn.Parameter(
            torch.empty(
                (config.hidden_width, config.target_vocab_size),
                device="cpu",
                dtype=torch.float32,
            )
        )
        self.b2 = nn.Parameter(
            torch.empty(config.target_vocab_size, device="cpu", dtype=torch.float32)
        )
        with torch.no_grad():
            self.embedding.copy_(
                torch.randn(
                    self.embedding.shape, generator=generator, dtype=torch.float32
                )
                * config.embedding_width**-0.5
            )
            self.w1.copy_(
                torch.randn(self.w1.shape, generator=generator, dtype=torch.float32)
                * (config.context_length * config.embedding_width) ** -0.5
            )
            self.b1.zero_()
            self.w2.copy_(
                torch.randn(self.w2.shape, generator=generator, dtype=torch.float32)
                * config.hidden_width**-0.5
            )
            self.b2.zero_()
        self.to(device)

    def forward(self, contexts: torch.Tensor) -> torch.Tensor:
        embedded = functional.embedding(contexts, self.embedding)
        hidden = torch.tanh(embedded.flatten(1) @ self.w1 + self.b1)
        return hidden @ self.w2 + self.b2


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def resolve_device(name: str) -> torch.device:
    """Resolve only an explicitly requested CPU or available MPS device."""

    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise ModelDataError("MPS was requested but is unavailable")
        return torch.device("mps")
    raise ModelDataError("device must be explicitly cpu or mps")
