"""Synthetic-only PyTorch dataset for Week 1 input validation."""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

from torch.utils.data import Dataset

from protein_lm.tokenization import encode


@dataclass(frozen=True)
class SyntheticProtein:
    accession: str
    sequence: str


@dataclass(frozen=True)
class SyntheticProteinItem:
    accession: str
    sequence_sha256: str
    biological_length: int
    token_ids: tuple[int, ...]
    residue_coordinates: tuple[int | None, ...]
    protein_start: int
    protein_end: int
    partition: str = field(default="synthetic", init=False)


def _build_item(protein: SyntheticProtein) -> SyntheticProteinItem:
    """Validate and tokenize one synthetic protein."""
    if not isinstance(protein.accession, str):
        raise TypeError("accession must be a string")

    if not protein.accession:
        raise ValueError("accession must not be empty")

    token_ids = tuple(encode(protein.sequence))
    biological_length = len(protein.sequence)

    sequence_sha256 = hashlib.sha256(protein.sequence.encode("ascii")).hexdigest()

    residue_coordinates = (
        None,
        *range(1, biological_length + 1),
        None,
    )

    return SyntheticProteinItem(
        accession=protein.accession,
        sequence_sha256=sequence_sha256,
        biological_length=biological_length,
        token_ids=token_ids,
        residue_coordinates=residue_coordinates,
        protein_start=1,
        protein_end=biological_length,
    )


class SyntheticProteinDataset(Dataset[SyntheticProteinItem]):
    """Provide indexed access to validated synthetic protein items."""

    def __init__(self, proteins: Sequence[SyntheticProtein]) -> None:
        if isinstance(proteins, (str, bytes)) or not isinstance(proteins, Sequence):
            raise TypeError("proteins must be a sequence of SyntheticProtein records")

        if not proteins:
            raise ValueError("proteins must not be empty")

        items = []
        seen_accessions = set()

        for index, protein in enumerate(proteins):
            if not isinstance(protein, SyntheticProtein):
                raise TypeError(f"protein at index {index} must be a SyntheticProtein")

            item = _build_item(protein)

            if item.accession in seen_accessions:
                raise ValueError(f"duplicate synthetic accession: {item.accession}")

            seen_accessions.add(item.accession)
            items.append(item)

        self._items = tuple(items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> SyntheticProteinItem:
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("dataset index must be an integer")

        return self._items[index]
