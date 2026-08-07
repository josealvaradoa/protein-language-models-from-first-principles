import pytest

from protein_lm.benchmarks.config import (
    BENCHMARK_CANDIDATES,
    BenchmarkConfig,
)


def test_approved_candidates_match_the_frozen_device_envelope() -> None:
    expected = {
        "A": (8, 128, 128, 2, 4, 32),
        "B": (8, 256, 256, 4, 8, 32),
        "C": (4, 512, 384, 6, 8, 48),
        "D": (2, 1024, 384, 6, 8, 48),
    }

    assert set(BENCHMARK_CANDIDATES) == set(expected)
    for identifier, candidate in BENCHMARK_CANDIDATES.items():
        assert (
            candidate.batch_size,
            candidate.sequence_length,
            candidate.width,
            candidate.layers,
            candidate.heads,
            candidate.values_per_head,
        ) == expected[identifier]
        assert candidate.as_dict()["warmup_steps"] == 3
        assert candidate.as_dict()["measured_steps"] == 10
        assert candidate.as_dict()["precision"] == "float32"
        assert candidate.as_dict()["gradient_clearing"] == "set_to_none=True"


def test_config_rejects_attention_width_that_cannot_be_divided_into_heads() -> None:
    with pytest.raises(ValueError, match="width must be divisible by heads"):
        BenchmarkConfig(
            identifier="test",
            batch_size=1,
            sequence_length=8,
            width=10,
            layers=1,
            heads=4,
        )
