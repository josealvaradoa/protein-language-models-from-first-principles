"""Verify PyTorch's training interface with one synthetic protein batch."""

from dataclasses import dataclass

import torch

from protein_lm.synthetic.collation import SyntheticProteinBatch
from protein_lm.synthetic.dataset import SyntheticProtein, SyntheticProteinDataset
from protein_lm.synthetic.loader import (
    SYNTHETIC_BATCH_SIZE,
    create_synthetic_loader,
)


INITIAL_PARAMETER = 0.5
LEARNING_RATE = 0.01
SMOKE_EPOCH_INDEX = 0

SYNTHETIC_SMOKE_PROTEINS = (
    SyntheticProtein(accession="SYNTH_001", sequence="ACD"),
    SyntheticProtein(accession="SYNTH_002", sequence="WY"),
    SyntheticProtein(accession="SYNTH_003", sequence="MNPQ"),
    SyntheticProtein(accession="SYNTH_004", sequence="RSTV"),
    SyntheticProtein(accession="SYNTH_005", sequence="GHIKL"),
    SyntheticProtein(accession="SYNTH_006", sequence="EFG"),
    SyntheticProtein(accession="SYNTH_007", sequence="ACDEFG"),
    SyntheticProtein(accession="SYNTH_008", sequence="KLMNPQ"),
)


@dataclass(frozen=True)
class SyntheticSmokeResult:
    """Observable results from one synthetic training-interface smoke."""

    device: str
    batch_shape: tuple[int, int]
    output_shape: tuple[int, int]
    total_positions: int
    non_padding_positions: int
    residue_positions: int
    initial_parameter: float
    first_loss: float
    first_gradient: float
    updated_parameter: float
    gradient_cleared: bool
    second_loss: float
    second_gradient: float


def create_synthetic_smoke_batch() -> SyntheticProteinBatch:
    """Create exactly one deterministic epoch-zero synthetic batch."""
    dataset = SyntheticProteinDataset(SYNTHETIC_SMOKE_PROTEINS)

    if len(dataset) != SYNTHETIC_BATCH_SIZE:
        raise RuntimeError(
            "the synthetic smoke fixture must contain exactly one full batch"
        )

    loader = create_synthetic_loader(
        dataset,
        epoch_index=SMOKE_EPOCH_INDEX,
    )

    return next(iter(loader))


def _move_batch_to_device(
    batch: SyntheticProteinBatch,
    device: torch.device,
) -> SyntheticProteinBatch:
    """Move every tensor in a synthetic batch to one PyTorch device."""
    return SyntheticProteinBatch(
        token_ids=batch.token_ids.to(device),
        biological_lengths=batch.biological_lengths.to(device),
        non_padding_mask=batch.non_padding_mask.to(device),
        residue_coordinates=batch.residue_coordinates.to(device),
        residue_mask=batch.residue_mask.to(device),
        protein_starts=batch.protein_starts.to(device),
        protein_ends=batch.protein_ends.to(device),
        accessions=batch.accessions,
        sequence_sha256s=batch.sequence_sha256s,
    )


def run_synthetic_smoke(
    batch: SyntheticProteinBatch,
    *,
    device: str | torch.device,
) -> SyntheticSmokeResult:
    """Run two backward passes around one test-only SGD parameter update."""
    requested_device = torch.device(device)

    if requested_device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")

    device_batch = _move_batch_to_device(batch, requested_device)

    parameter = torch.nn.Parameter(
        torch.tensor(INITIAL_PARAMETER, device=requested_device)
    )
    optimizer = torch.optim.SGD(
        [parameter],
        lr=LEARNING_RATE,
    )

    batch_shape = (
        device_batch.token_ids.size(0),
        device_batch.token_ids.size(1),
    )
    initial_parameter = parameter.detach().item()

    first_output = parameter.expand(batch_shape)

    if first_output.shape != device_batch.token_ids.shape:
        raise RuntimeError("forward output shape does not match the token batch")

    first_loss = first_output[device_batch.residue_mask].square().sum()

    if not torch.isfinite(first_loss).item():
        raise RuntimeError("first loss is not finite")

    first_loss.backward()

    if parameter.grad is None:
        raise RuntimeError("first backward pass did not create a gradient")

    if not torch.isfinite(parameter.grad).item():
        raise RuntimeError("first gradient is not finite")

    first_gradient = parameter.grad.detach().item()

    optimizer.step()
    updated_parameter = parameter.detach().item()

    if updated_parameter == initial_parameter:
        raise RuntimeError("optimizer step did not change the parameter")

    optimizer.zero_grad(set_to_none=True)
    gradient_cleared = parameter.grad is None

    if not gradient_cleared:
        raise RuntimeError("gradient was not cleared before the second backward pass")

    second_output = parameter.expand(batch_shape)
    second_loss = second_output[device_batch.residue_mask].square().sum()

    if not torch.isfinite(second_loss).item():
        raise RuntimeError("second loss is not finite")

    second_loss.backward()

    if parameter.grad is None:
        raise RuntimeError("second backward pass did not create a gradient")

    if not torch.isfinite(parameter.grad).item():
        raise RuntimeError("second gradient is not finite")

    second_gradient = parameter.grad.detach().item()

    return SyntheticSmokeResult(
        device=str(requested_device),
        batch_shape=batch_shape,
        output_shape=(
            first_output.size(0),
            first_output.size(1),
        ),
        total_positions=device_batch.token_ids.numel(),
        non_padding_positions=int(device_batch.non_padding_mask.sum().item()),
        residue_positions=int(device_batch.residue_mask.sum().item()),
        initial_parameter=initial_parameter,
        first_loss=first_loss.detach().item(),
        first_gradient=first_gradient,
        updated_parameter=updated_parameter,
        gradient_cleared=gradient_cleared,
        second_loss=second_loss.detach().item(),
        second_gradient=second_gradient,
    )
