"""Disposable synthetic workload used by the device-envelope benchmark."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from protein_lm.benchmarks.config import (
    FEED_FORWARD_MULTIPLIER,
    RESIDUE_TOKEN_MIN,
    BenchmarkConfig,
)


class NonFiniteLossError(RuntimeError):
    """Signal a checked loss that is not finite."""


class NonFiniteGradientError(RuntimeError):
    """Signal a checked gradient that is missing or not finite."""


class SyntheticAttentionBlock(torch.nn.Module):
    """One bidirectional attention and feed-forward residual block."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.attention_norm = torch.nn.LayerNorm(width)
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=width,
            num_heads=heads,
            dropout=0.0,
            batch_first=True,
        )
        self.feed_forward_norm = torch.nn.LayerNorm(width)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(width, width * FEED_FORWARD_MULTIPLIER),
            torch.nn.GELU(),
            torch.nn.Linear(width * FEED_FORWARD_MULTIPLIER, width),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply generic bidirectional attention without an attention mask."""
        normalized = self.attention_norm(hidden_states)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        hidden_states = hidden_states + attended
        return hidden_states + self.feed_forward(self.feed_forward_norm(hidden_states))


class SyntheticDeviceWorkload(torch.nn.Module):
    """Disposable model that supplies the frozen training-shaped computation."""

    def __init__(self, config: BenchmarkConfig) -> None:
        super().__init__()
        self.token_embedding = torch.nn.Embedding(config.vocabulary_size, config.width)
        self.position_embedding = torch.nn.Embedding(
            config.sequence_length,
            config.width,
        )
        self.blocks = torch.nn.ModuleList(
            SyntheticAttentionBlock(config.width, config.heads)
            for _ in range(config.layers)
        )
        self.output_norm = torch.nn.LayerNorm(config.width)
        self.output_projection = torch.nn.Linear(config.width, config.vocabulary_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return one vocabulary logit vector for every token position."""
        positions = torch.arange(token_ids.size(1), device=token_ids.device)
        hidden_states = self.token_embedding(token_ids) + self.position_embedding(
            positions
        )
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.output_projection(self.output_norm(hidden_states))


def create_synthetic_token_tensors(
    config: BenchmarkConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic canonical-residue inputs and targets only."""
    positions = torch.arange(config.tokens_per_step, dtype=torch.long)
    token_ids = (positions * 7 + config.seed) % 20 + RESIDUE_TOKEN_MIN
    target_ids = (positions * 11 + config.seed + 3) % 20 + RESIDUE_TOKEN_MIN
    shape = (config.batch_size, config.sequence_length)
    return token_ids.reshape(shape).to(device), target_ids.reshape(shape).to(device)


def run_one_training_step(
    model: SyntheticDeviceWorkload,
    optimizer: torch.optim.Optimizer,
    token_ids: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Size:
    """Exercise forward, backward, AdamW, and explicit gradient clearing once."""
    output = model(token_ids)
    loss = functional.cross_entropy(
        output.reshape(-1, output.size(-1)),
        target_ids.reshape(-1),
    )
    if not torch.isfinite(loss).item():
        raise NonFiniteLossError("cross-entropy loss is not finite")

    loss.backward()
    gradient_checks = []
    for parameter in model.parameters():
        if parameter.grad is None:
            raise NonFiniteGradientError("a model gradient is missing")
        gradient_checks.append(torch.isfinite(parameter.grad).all())

    if not gradient_checks or not torch.stack(gradient_checks).all().item():
        raise NonFiniteGradientError("a model gradient is not finite")

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return output.shape
