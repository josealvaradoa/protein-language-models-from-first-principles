import math

import pytest
import torch

from protein_lm.synthetic.collation import SyntheticProteinBatch
from protein_lm.synthetic.smoke import (
    INITIAL_PARAMETER,
    LEARNING_RATE,
    create_synthetic_smoke_batch,
    run_synthetic_smoke,
)


RELATIVE_TOLERANCE = 1e-6
ABSOLUTE_TOLERANCE = 1e-7


def _batch_tensors(batch: SyntheticProteinBatch) -> tuple[torch.Tensor, ...]:
    return (
        batch.token_ids,
        batch.biological_lengths,
        batch.non_padding_mask,
        batch.residue_coordinates,
        batch.residue_mask,
        batch.protein_starts,
        batch.protein_ends,
    )


def test_smoke_batch_has_the_frozen_fixture_geometry() -> None:
    batch = create_synthetic_smoke_batch()

    assert batch.partition == "synthetic"
    assert batch.token_ids.shape == (8, 8)
    assert batch.token_ids.numel() == 64
    assert batch.non_padding_mask.sum().item() == 49
    assert batch.residue_mask.sum().item() == 33


def test_cpu_smoke_exercises_the_masked_training_interface() -> None:
    result = run_synthetic_smoke(
        create_synthetic_smoke_batch(),
        device="cpu",
    )

    expected_first_loss = result.residue_positions * INITIAL_PARAMETER**2
    expected_first_gradient = 2 * result.residue_positions * INITIAL_PARAMETER
    expected_parameter = INITIAL_PARAMETER - LEARNING_RATE * expected_first_gradient
    expected_second_loss = result.residue_positions * expected_parameter**2
    expected_second_gradient = 2 * result.residue_positions * expected_parameter

    assert result.device == "cpu"
    assert result.batch_shape == (8, 8)
    assert result.output_shape == result.batch_shape
    assert result.total_positions == 64
    assert result.non_padding_positions == 49
    assert result.residue_positions == 33
    assert result.residue_positions < result.non_padding_positions
    assert result.non_padding_positions < result.total_positions

    assert math.isfinite(result.first_loss)
    assert math.isfinite(result.first_gradient)
    assert result.first_loss == pytest.approx(
        expected_first_loss,
        rel=RELATIVE_TOLERANCE,
        abs=ABSOLUTE_TOLERANCE,
    )
    assert result.first_gradient == pytest.approx(
        expected_first_gradient,
        rel=RELATIVE_TOLERANCE,
        abs=ABSOLUTE_TOLERANCE,
    )

    assert result.initial_parameter == INITIAL_PARAMETER
    assert result.updated_parameter != result.initial_parameter
    assert result.updated_parameter == pytest.approx(
        expected_parameter,
        rel=RELATIVE_TOLERANCE,
        abs=ABSOLUTE_TOLERANCE,
    )

    assert result.gradient_cleared is True
    assert math.isfinite(result.second_loss)
    assert math.isfinite(result.second_gradient)
    assert result.second_loss == pytest.approx(
        expected_second_loss,
        rel=RELATIVE_TOLERANCE,
        abs=ABSOLUTE_TOLERANCE,
    )
    assert result.second_gradient == pytest.approx(
        expected_second_gradient,
        rel=RELATIVE_TOLERANCE,
        abs=ABSOLUTE_TOLERANCE,
    )


def test_recreated_cpu_smokes_are_identical_and_independent() -> None:
    first_batch = create_synthetic_smoke_batch()
    second_batch = create_synthetic_smoke_batch()

    for first_tensor, second_tensor in zip(
        _batch_tensors(first_batch),
        _batch_tensors(second_batch),
        strict=True,
    ):
        assert torch.equal(first_tensor, second_tensor)

    assert first_batch.accessions == second_batch.accessions
    assert first_batch.sequence_sha256s == second_batch.sequence_sha256s

    first_result = run_synthetic_smoke(first_batch, device="cpu")
    second_result = run_synthetic_smoke(second_batch, device="cpu")

    assert first_result.device == second_result.device == "cpu"
    assert first_result.batch_shape == second_result.batch_shape
    assert first_result.output_shape == second_result.output_shape
    assert first_result.total_positions == second_result.total_positions
    assert first_result.non_padding_positions == second_result.non_padding_positions
    assert first_result.residue_positions == second_result.residue_positions
    assert first_result.gradient_cleared is second_result.gradient_cleared is True

    repeated_values = (
        (first_result.initial_parameter, second_result.initial_parameter),
        (first_result.first_loss, second_result.first_loss),
        (first_result.first_gradient, second_result.first_gradient),
        (first_result.updated_parameter, second_result.updated_parameter),
        (first_result.second_loss, second_result.second_loss),
        (first_result.second_gradient, second_result.second_gradient),
    )
    for first_value, second_value in repeated_values:
        assert first_value == pytest.approx(
            second_value,
            rel=RELATIVE_TOLERANCE,
            abs=ABSOLUTE_TOLERANCE,
        )


def test_mps_request_fails_instead_of_silently_falling_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="MPS was requested but is not available"):
        run_synthetic_smoke(
            create_synthetic_smoke_batch(),
            device="mps",
        )
