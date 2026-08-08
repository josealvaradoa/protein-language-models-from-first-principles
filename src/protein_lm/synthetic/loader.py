"""Build deterministic DataLoaders for synthetic Week 1 fixtures."""

from typing import cast

import torch
from torch.utils.data import DataLoader

from protein_lm.synthetic.collation import (
    SyntheticProteinBatch,
    collate_synthetic_proteins,
)
from protein_lm.synthetic.dataset import SyntheticProteinDataset


SYNTHETIC_BATCH_SIZE = 8
SYNTHETIC_BASE_SEED = 20260727


def create_synthetic_loader(
    dataset: SyntheticProteinDataset,
    *,
    epoch_index: int,
) -> DataLoader[SyntheticProteinBatch]:
    """Create the deterministic synthetic loader for one epoch."""
    if not isinstance(dataset, SyntheticProteinDataset):
        raise TypeError("dataset must be a SyntheticProteinDataset")

    if not isinstance(epoch_index, int) or isinstance(epoch_index, bool):
        raise TypeError("epoch_index must be an integer")

    if epoch_index < 0:
        raise ValueError("epoch_index must not be negative")

    generator = torch.Generator()
    generator.manual_seed(SYNTHETIC_BASE_SEED + epoch_index)

    loader = DataLoader(
        dataset,
        batch_size=SYNTHETIC_BATCH_SIZE,
        shuffle=True,
        generator=generator,
        collate_fn=collate_synthetic_proteins,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )

    return cast(DataLoader[SyntheticProteinBatch], loader)
