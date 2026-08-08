"""Frozen settings for the Week 1 synthetic device-envelope benchmark."""

from __future__ import annotations

from dataclasses import dataclass


VOCABULARY_SIZE = 24
RESIDUE_TOKEN_MIN = 4
RESIDUE_TOKEN_MAX = 23
BENCHMARK_SEED = 20260807
FEED_FORWARD_MULTIPLIER = 4
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 3
MEASURED_STEPS = 10


@dataclass(frozen=True)
class BenchmarkConfig:
    """One complete synthetic training-shaped workload configuration."""

    identifier: str
    batch_size: int
    sequence_length: int
    width: int
    layers: int
    heads: int
    warmup_steps: int = WARMUP_STEPS
    measured_steps: int = MEASURED_STEPS
    vocabulary_size: int = VOCABULARY_SIZE
    seed: int = BENCHMARK_SEED

    def __post_init__(self) -> None:
        positive_fields = (
            "batch_size",
            "sequence_length",
            "width",
            "layers",
            "heads",
            "warmup_steps",
            "measured_steps",
            "vocabulary_size",
        )
        for field_name in positive_fields:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")

        if self.width % self.heads != 0:
            raise ValueError("width must be divisible by heads")

        if self.vocabulary_size < RESIDUE_TOKEN_MAX + 1:
            raise ValueError("vocabulary_size must contain canonical residue token IDs")

    @property
    def tokens_per_step(self) -> int:
        """Return the number of unpadded positions processed in one step."""
        return self.batch_size * self.sequence_length

    @property
    def values_per_head(self) -> int:
        """Return the attention value width implied by the configuration."""
        return self.width // self.heads

    @property
    def feed_forward_width(self) -> int:
        """Return the fixed four-times-width feed-forward dimension."""
        return self.width * FEED_FORWARD_MULTIPLIER

    def as_dict(self) -> dict[str, int | str | float]:
        """Return the complete configuration in a JSON-ready form."""
        return {
            "identifier": self.identifier,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "tokens_per_step": self.tokens_per_step,
            "width": self.width,
            "layers": self.layers,
            "heads": self.heads,
            "values_per_head": self.values_per_head,
            "feed_forward_multiplier": FEED_FORWARD_MULTIPLIER,
            "feed_forward_width": self.feed_forward_width,
            "vocabulary_size": self.vocabulary_size,
            "residue_token_range": f"{RESIDUE_TOKEN_MIN}-{RESIDUE_TOKEN_MAX}",
            "seed": self.seed,
            "dropout": 0.0,
            "activation": "GELU",
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": self.warmup_steps,
            "measured_steps": self.measured_steps,
            "gradient_clearing": "set_to_none=True",
            "precision": "float32",
        }


BENCHMARK_CANDIDATES: dict[str, BenchmarkConfig] = {
    "A": BenchmarkConfig(
        identifier="A",
        batch_size=8,
        sequence_length=128,
        width=128,
        layers=2,
        heads=4,
    ),
    "B": BenchmarkConfig(
        identifier="B",
        batch_size=8,
        sequence_length=256,
        width=256,
        layers=4,
        heads=8,
    ),
    "C": BenchmarkConfig(
        identifier="C",
        batch_size=4,
        sequence_length=512,
        width=384,
        layers=6,
        heads=8,
    ),
    "D": BenchmarkConfig(
        identifier="D",
        batch_size=2,
        sequence_length=1024,
        width=384,
        layers=6,
        heads=8,
    ),
}
