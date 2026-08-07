from dataclasses import replace

import pytest
import torch

from protein_lm.synthetic_collation import collate_synthetic_proteins
from protein_lm.synthetic_dataset import (
    SyntheticProtein,
    SyntheticProteinDataset,
    SyntheticProteinItem,
)


def _items(*proteins: SyntheticProtein) -> tuple[SyntheticProteinItem, ...]:
    dataset = SyntheticProteinDataset(proteins)
    return tuple(dataset[index] for index in range(len(dataset)))


def test_collator_right_pads_tokens_coordinates_and_masks() -> None:
    batch = collate_synthetic_proteins(
        _items(
            SyntheticProtein("SYNTH_001", "ACDY"),
            SyntheticProtein("SYNTH_002", "WY"),
        )
    )

    assert torch.equal(
        batch.token_ids,
        torch.tensor(
            [
                [1, 4, 5, 6, 23, 2],
                [1, 22, 23, 2, 0, 0],
            ]
        ),
    )
    assert torch.equal(
        batch.non_padding_mask,
        torch.tensor(
            [
                [True, True, True, True, True, True],
                [True, True, True, True, False, False],
            ]
        ),
    )
    assert torch.equal(
        batch.residue_coordinates,
        torch.tensor(
            [
                [0, 1, 2, 3, 4, 0],
                [0, 1, 2, 0, 0, 0],
            ]
        ),
    )
    assert torch.equal(
        batch.residue_mask,
        torch.tensor(
            [
                [False, True, True, True, True, False],
                [False, True, True, False, False, False],
            ]
        ),
    )


def test_collator_preserves_shapes_dtypes_and_traceability() -> None:
    items = _items(
        SyntheticProtein("SYNTH_001", "ACDY"),
        SyntheticProtein("SYNTH_002", "WY"),
    )

    batch = collate_synthetic_proteins(items)

    assert batch.token_ids.shape == (2, 6)
    assert batch.biological_lengths.shape == (2,)
    assert batch.non_padding_mask.shape == (2, 6)
    assert batch.residue_coordinates.shape == (2, 6)
    assert batch.residue_mask.shape == (2, 6)
    assert batch.protein_starts.shape == (2,)
    assert batch.protein_ends.shape == (2,)

    assert batch.token_ids.dtype == torch.long
    assert batch.biological_lengths.dtype == torch.long
    assert batch.non_padding_mask.dtype == torch.bool
    assert batch.residue_coordinates.dtype == torch.long
    assert batch.residue_mask.dtype == torch.bool
    assert batch.protein_starts.dtype == torch.long
    assert batch.protein_ends.dtype == torch.long

    assert batch.token_ids.device.type == "cpu"
    assert torch.equal(batch.biological_lengths, torch.tensor([4, 2]))
    assert torch.equal(batch.protein_starts, torch.tensor([1, 1]))
    assert torch.equal(batch.protein_ends, torch.tensor([4, 2]))
    assert batch.accessions == ("SYNTH_001", "SYNTH_002")
    assert batch.sequence_sha256s == tuple(item.sequence_sha256 for item in items)
    assert batch.partition == "synthetic"


def test_collator_does_not_pad_equal_length_items() -> None:
    batch = collate_synthetic_proteins(
        _items(
            SyntheticProtein("SYNTH_001", "ACD"),
            SyntheticProtein("SYNTH_002", "MNP"),
        )
    )

    assert batch.token_ids.shape == (2, 5)
    assert bool(batch.non_padding_mask.all())
    assert not bool((batch.token_ids == 0).any())


def test_collator_is_deterministic_for_identical_items() -> None:
    items = _items(
        SyntheticProtein("SYNTH_001", "ACDY"),
        SyntheticProtein("SYNTH_002", "WY"),
    )

    first = collate_synthetic_proteins(items)
    second = collate_synthetic_proteins(items)

    assert torch.equal(first.token_ids, second.token_ids)
    assert torch.equal(first.non_padding_mask, second.non_padding_mask)
    assert torch.equal(first.residue_coordinates, second.residue_coordinates)
    assert torch.equal(first.residue_mask, second.residue_mask)
    assert first.accessions == second.accessions
    assert first.sequence_sha256s == second.sequence_sha256s


@pytest.mark.parametrize("items", [(), []])
def test_collator_rejects_an_empty_batch(
    items: list[SyntheticProteinItem] | tuple[()],
) -> None:
    with pytest.raises(ValueError, match="items must not be empty"):
        collate_synthetic_proteins(items)


@pytest.mark.parametrize("items", [None, 123, "items", b"items"])
def test_collator_rejects_non_item_sequences(items: object) -> None:
    with pytest.raises(
        TypeError,
        match="items must be a sequence of SyntheticProteinItem records",
    ):
        collate_synthetic_proteins(items)  # type: ignore[arg-type]


def test_collator_rejects_a_non_item_record() -> None:
    item = _items(SyntheticProtein("SYNTH_001", "ACD"))[0]

    with pytest.raises(
        TypeError,
        match="item at index 1 must be a SyntheticProteinItem",
    ):
        collate_synthetic_proteins(
            [item, "not an item"]  # type: ignore[list-item]
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"token_ids": (1, 4, 0, 6, 2)},
            "item at index 0 has invalid token IDs",
        ),
        (
            {"biological_length": 2},
            "item at index 0 biological length does not match token IDs",
        ),
        (
            {"residue_coordinates": (None, 1, 3, 2, None)},
            "item at index 0 has misaligned residue coordinates",
        ),
        (
            {"protein_start": 0},
            "item at index 0 has invalid full-protein coordinates",
        ),
        (
            {"protein_end": 2},
            "item at index 0 has invalid full-protein coordinates",
        ),
    ],
)
def test_collator_rejects_malformed_item_boundaries(
    changes: dict[str, object],
    message: str,
) -> None:
    item = _items(SyntheticProtein("SYNTH_001", "ACD"))[0]
    malformed = replace(item, **changes)

    with pytest.raises(ValueError, match=message):
        collate_synthetic_proteins((malformed,))
