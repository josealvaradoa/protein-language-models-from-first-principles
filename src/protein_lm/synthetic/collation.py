"""Collate synthetic protein items into padded PyTorch batches."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

from protein_lm.synthetic.dataset import SyntheticProteinItem
from protein_lm.tokenization import TokenId, decode


@dataclass(frozen=True)
class SyntheticProteinBatch:
    """One padded, traceable batch of synthetic proteins."""

    token_ids: Tensor
    biological_lengths: Tensor
    non_padding_mask: Tensor
    residue_coordinates: Tensor
    residue_mask: Tensor
    protein_starts: Tensor
    protein_ends: Tensor
    accessions: tuple[str, ...]
    sequence_sha256s: tuple[str, ...]
    partition: str = field(default="synthetic", init=False)


def collate_synthetic_proteins(
    items: Sequence[SyntheticProteinItem],
) -> SyntheticProteinBatch:
    """Right-pad synthetic protein items into one CPU batch."""
    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
        raise TypeError("items must be a sequence of SyntheticProteinItem records")

    if not items:
        raise ValueError("items must not be empty")

    for index, item in enumerate(items):
        if not isinstance(item, SyntheticProteinItem):
            raise TypeError(f"item at index {index} must be a SyntheticProteinItem")
        _validate_item(item, index=index)

    batch_size = len(items)
    maximum_token_length = max(len(item.token_ids) for item in items)

    token_ids = torch.full(
        (batch_size, maximum_token_length),
        fill_value=TokenId.PAD.value,
        dtype=torch.long,
    )
    non_padding_mask = torch.zeros(
        (batch_size, maximum_token_length),
        dtype=torch.bool,
    )
    residue_coordinates = torch.zeros(
        (batch_size, maximum_token_length),
        dtype=torch.long,
    )
    residue_mask = torch.zeros(
        (batch_size, maximum_token_length),
        dtype=torch.bool,
    )
    biological_lengths = torch.empty(batch_size, dtype=torch.long)
    protein_starts = torch.empty(batch_size, dtype=torch.long)
    protein_ends = torch.empty(batch_size, dtype=torch.long)

    for row, item in enumerate(items):
        token_count = len(item.token_ids)
        row_slice = slice(0, token_count)

        token_ids[row, row_slice] = torch.tensor(
            item.token_ids,
            dtype=torch.long,
        )
        non_padding_mask[row, row_slice] = True

        coordinates = tuple(
            0 if coordinate is None else coordinate
            for coordinate in item.residue_coordinates
        )
        residue_coordinates[row, row_slice] = torch.tensor(
            coordinates,
            dtype=torch.long,
        )
        residue_mask[row, row_slice] = torch.tensor(
            tuple(coordinate is not None for coordinate in item.residue_coordinates),
            dtype=torch.bool,
        )

        biological_lengths[row] = item.biological_length
        protein_starts[row] = item.protein_start
        protein_ends[row] = item.protein_end

    return SyntheticProteinBatch(
        token_ids=token_ids,
        biological_lengths=biological_lengths,
        non_padding_mask=non_padding_mask,
        residue_coordinates=residue_coordinates,
        residue_mask=residue_mask,
        protein_starts=protein_starts,
        protein_ends=protein_ends,
        accessions=tuple(item.accession for item in items),
        sequence_sha256s=tuple(item.sequence_sha256 for item in items),
    )


def _validate_item(item: SyntheticProteinItem, *, index: int) -> None:
    """Reject malformed items before their boundaries enter a batch."""
    try:
        decoded_sequence = decode(item.token_ids)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"item at index {index} has invalid token IDs: {error}"
        ) from error

    if len(decoded_sequence) != item.biological_length:
        raise ValueError(
            f"item at index {index} biological length does not match token IDs"
        )

    expected_coordinates = (
        None,
        *range(1, item.biological_length + 1),
        None,
    )
    if item.residue_coordinates != expected_coordinates:
        raise ValueError(f"item at index {index} has misaligned residue coordinates")

    if item.protein_start != 1 or item.protein_end != item.biological_length:
        raise ValueError(f"item at index {index} has invalid full-protein coordinates")
