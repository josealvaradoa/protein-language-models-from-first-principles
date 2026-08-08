"""Residue-mask and pooling invariants for the ESMC smoke."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from protein_lm.external.esmc_contract import ContractValidationError


if TYPE_CHECKING:
    from protein_lm.external.esmc_contract import SyntheticFixture


PAD_POISON_SENTINEL = 1_000_000.0


def build_residue_mask(
    attention_mask: torch.Tensor,
    special_tokens_mask: torch.Tensor,
) -> torch.Tensor:
    """Keep positions that are both attended and non-special residues."""
    return attention_mask.to(torch.bool) & ~special_tokens_mask.to(torch.bool)


def padding_aware_mean_pool(
    hidden_states: torch.Tensor,
    residue_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool true residue positions without reading masked values into sums."""
    counts = residue_mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ContractValidationError("each fixture must contain at least one residue")
    values = torch.where(
        residue_mask.unsqueeze(-1), hidden_states, torch.zeros_like(hidden_states)
    )
    return values.sum(dim=1) / counts.unsqueeze(-1).to(hidden_states.dtype)


def mask_one_residue_per_fixture(
    input_ids: torch.Tensor,
    residue_mask: torch.Tensor,
    fixtures: tuple[SyntheticFixture, ...],
    mask_token_id: int | None,
) -> torch.Tensor:
    """Mask the sole frozen residue coordinate for each synthetic fixture."""
    if mask_token_id is None:
        raise ContractValidationError("the local tokenizer does not define a mask token")
    masked = input_ids.clone()
    for row, fixture in enumerate(fixtures):
        residue_positions = torch.nonzero(residue_mask[row], as_tuple=False).flatten()
        position = residue_positions[fixture.mask_residue_index]
        masked[row, position] = mask_token_id
    return masked


def validate_masked_residue_counts(
    input_ids: torch.Tensor,
    masked_input_ids: torch.Tensor,
    residue_mask: torch.Tensor,
) -> list[int]:
    """Prove that exactly one change per item occurred at a residue position."""
    changed = input_ids != masked_input_ids
    if torch.any(changed & ~residue_mask):
        raise ContractValidationError("a special token or PAD position was masked")
    counts = [int(value) for value in (changed & residue_mask).sum(dim=1).tolist()]
    if counts != [1, 1]:
        raise ContractValidationError("each fixture must contain exactly one masked residue")
    return counts


def padding_poison_invariant(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    residue_mask: torch.Tensor,
) -> bool:
    """Prove that large finite PAD values cannot affect the pooled representation."""
    pooled = padding_aware_mean_pool(hidden_states, residue_mask)
    poisoned = hidden_states.clone()
    poisoned[attention_mask == 0] = PAD_POISON_SENTINEL
    return bool(torch.allclose(pooled, padding_aware_mean_pool(poisoned, residue_mask)))
