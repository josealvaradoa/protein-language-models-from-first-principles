import pytest
from torch.utils.data import RandomSampler

from protein_lm.synthetic_collation import (
    SyntheticProteinBatch,
    collate_synthetic_proteins,
)
from protein_lm.synthetic_dataset import (
    SyntheticProtein,
    SyntheticProteinDataset,
)
from protein_lm.synthetic_loader import (
    SYNTHETIC_BASE_SEED,
    SYNTHETIC_BATCH_SIZE,
    create_synthetic_loader,
)


def _dataset(size: int = 10) -> SyntheticProteinDataset:
    return SyntheticProteinDataset(
        tuple(
            SyntheticProtein(
                accession=f"SYNTH_{index:03d}",
                sequence="ACD" + ("Y" * index),
            )
            for index in range(size)
        )
    )


def _accession_order(
    dataset: SyntheticProteinDataset,
    *,
    epoch_index: int,
) -> tuple[str, ...]:
    loader = create_synthetic_loader(dataset, epoch_index=epoch_index)
    return tuple(accession for batch in loader for accession in batch.accessions)


def _epoch_snapshot(
    dataset: SyntheticProteinDataset,
    *,
    epoch_index: int,
) -> tuple[object, ...]:
    loader = create_synthetic_loader(dataset, epoch_index=epoch_index)
    return tuple(
        (
            batch.accessions,
            batch.sequence_sha256s,
            batch.token_ids.tolist(),
            batch.non_padding_mask.tolist(),
            batch.residue_coordinates.tolist(),
            batch.residue_mask.tolist(),
        )
        for batch in loader
    )


def test_loader_uses_the_frozen_week_one_settings() -> None:
    loader = create_synthetic_loader(_dataset(), epoch_index=0)

    assert SYNTHETIC_BATCH_SIZE == 8
    assert SYNTHETIC_BASE_SEED == 20260727
    assert loader.batch_size == 8
    assert loader.drop_last is False
    assert loader.num_workers == 0
    assert loader.pin_memory is False
    assert loader.collate_fn is collate_synthetic_proteins
    assert isinstance(loader.sampler, RandomSampler)


def test_loader_keeps_the_smaller_final_batch_and_every_item() -> None:
    dataset = _dataset()
    original_order = tuple(dataset[index].accession for index in range(len(dataset)))

    batches = tuple(create_synthetic_loader(dataset, epoch_index=0))
    observed_order = tuple(
        accession for batch in batches for accession in batch.accessions
    )

    assert all(isinstance(batch, SyntheticProteinBatch) for batch in batches)
    assert [len(batch.accessions) for batch in batches] == [8, 2]
    assert sorted(observed_order) == sorted(original_order)
    assert len(set(observed_order)) == len(dataset)
    assert tuple(dataset[index].accession for index in range(len(dataset))) == (
        original_order
    )


def test_recreated_epoch_has_identical_batches() -> None:
    dataset = _dataset()

    first = _epoch_snapshot(dataset, epoch_index=0)
    second = _epoch_snapshot(dataset, epoch_index=0)

    assert first == second


def test_next_epoch_has_a_different_repeatable_order() -> None:
    dataset = _dataset()

    epoch_zero = _accession_order(dataset, epoch_index=0)
    epoch_one_first = _accession_order(dataset, epoch_index=1)
    epoch_one_second = _accession_order(dataset, epoch_index=1)

    assert epoch_zero != epoch_one_first
    assert epoch_one_first == epoch_one_second


def test_loader_handles_a_dataset_smaller_than_one_batch() -> None:
    batches = tuple(create_synthetic_loader(_dataset(3), epoch_index=0))

    assert len(batches) == 1
    assert len(batches[0].accessions) == 3


@pytest.mark.parametrize("dataset", [None, 123, "dataset", []])
def test_loader_rejects_an_invalid_dataset(dataset: object) -> None:
    with pytest.raises(
        TypeError,
        match="dataset must be a SyntheticProteinDataset",
    ):
        create_synthetic_loader(dataset, epoch_index=0)  # type: ignore[arg-type]


@pytest.mark.parametrize("epoch_index", [None, "0", 0.0, True])
def test_loader_rejects_a_non_integer_epoch(epoch_index: object) -> None:
    with pytest.raises(TypeError, match="epoch_index must be an integer"):
        create_synthetic_loader(
            _dataset(),
            epoch_index=epoch_index,  # type: ignore[arg-type]
        )


def test_loader_rejects_a_negative_epoch() -> None:
    with pytest.raises(ValueError, match="epoch_index must not be negative"):
        create_synthetic_loader(_dataset(), epoch_index=-1)
