import hashlib
from dataclasses import FrozenInstanceError

import pytest
from torch.utils.data import Dataset

from protein_lm.synthetic_dataset import (
    SyntheticProtein,
    SyntheticProteinDataset,
    SyntheticProteinItem,
)


def test_dataset_builds_traceable_items_in_source_order() -> None:
    dataset = SyntheticProteinDataset(
        (
            SyntheticProtein("SYNTH_001", "ACDY"),
            SyntheticProtein("SYNTH_002", "WY"),
        )
    )

    first = dataset[0]
    second = dataset[1]

    assert isinstance(dataset, Dataset)
    assert len(dataset) == 2
    assert first.accession == "SYNTH_001"
    assert first.sequence_sha256 == hashlib.sha256(b"ACDY").hexdigest()
    assert first.partition == "synthetic"
    assert first.biological_length == 4
    assert first.token_ids == (1, 4, 5, 6, 23, 2)
    assert first.residue_coordinates == (None, 1, 2, 3, 4, None)
    assert first.protein_start == 1
    assert first.protein_end == 4
    assert second.accession == "SYNTH_002"
    assert second.token_ids == (1, 22, 23, 2)


def test_dataset_supports_standard_sequence_indexing() -> None:
    dataset = SyntheticProteinDataset(
        (
            SyntheticProtein("SYNTH_001", "ACD"),
            SyntheticProtein("SYNTH_002", "WY"),
        )
    )

    assert dataset[-1].accession == "SYNTH_002"

    with pytest.raises(IndexError):
        dataset[2]


def test_synthetic_items_are_immutable() -> None:
    item = SyntheticProteinDataset((SyntheticProtein("SYNTH_001", "ACD"),))[0]

    with pytest.raises(FrozenInstanceError):
        item.accession = "CHANGED"  # type: ignore[misc]


def test_partition_cannot_be_supplied_by_the_caller() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'partition'"):
        SyntheticProteinItem(
            accession="SYNTH_001",
            sequence_sha256=hashlib.sha256(b"ACD").hexdigest(),
            biological_length=3,
            token_ids=(1, 4, 5, 6, 2),
            residue_coordinates=(None, 1, 2, 3, None),
            protein_start=1,
            protein_end=3,
            partition="training",  # type: ignore[call-arg]
        )


@pytest.mark.parametrize("proteins", [(), []])
def test_dataset_rejects_an_empty_collection(
    proteins: list[SyntheticProtein] | tuple[()],
) -> None:
    with pytest.raises(ValueError, match="proteins must not be empty"):
        SyntheticProteinDataset(proteins)


@pytest.mark.parametrize("proteins", [None, 123, "ACD", b"ACD"])
def test_dataset_rejects_non_record_sequences(proteins: object) -> None:
    with pytest.raises(
        TypeError,
        match="proteins must be a sequence of SyntheticProtein records",
    ):
        SyntheticProteinDataset(proteins)  # type: ignore[arg-type]


def test_dataset_rejects_a_non_protein_record() -> None:
    with pytest.raises(
        TypeError,
        match="protein at index 1 must be a SyntheticProtein",
    ):
        SyntheticProteinDataset(
            [
                SyntheticProtein("SYNTH_001", "ACD"),
                "not a record",  # type: ignore[list-item]
            ]
        )


def test_dataset_rejects_duplicate_accessions() -> None:
    with pytest.raises(
        ValueError,
        match="duplicate synthetic accession: SYNTH_001",
    ):
        SyntheticProteinDataset(
            (
                SyntheticProtein("SYNTH_001", "ACD"),
                SyntheticProtein("SYNTH_001", "WY"),
            )
        )


@pytest.mark.parametrize(
    ("accession", "error_type", "message"),
    [
        ("", ValueError, "accession must not be empty"),
        (123, TypeError, "accession must be a string"),
    ],
)
def test_dataset_rejects_invalid_accessions(
    accession: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        SyntheticProteinDataset(
            (SyntheticProtein(accession, "ACD"),)  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("sequence", "message"),
    [
        ("", "sequence must not be empty"),
        ("acd", "invalid residue 'a' at biological position 1"),
        ("ACX", "invalid residue 'X' at biological position 3"),
        ("A C", "invalid residue ' ' at biological position 2"),
    ],
)
def test_dataset_rejects_invalid_sequences(sequence: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SyntheticProteinDataset((SyntheticProtein("SYNTH_001", sequence),))


@pytest.mark.parametrize("index", [None, "0", 0.0, True])
def test_dataset_rejects_non_integer_indices(index: object) -> None:
    dataset = SyntheticProteinDataset((SyntheticProtein("SYNTH_001", "ACD"),))

    with pytest.raises(TypeError, match="dataset index must be an integer"):
        dataset[index]  # type: ignore[index]
