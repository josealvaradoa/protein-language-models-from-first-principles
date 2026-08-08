import pytest

from protein_lm.tokenization import (
    CANONICAL_AMINO_ACIDS,
    TokenId,
    decode,
    encode,
)


FROZEN_VOCABULARY = {
    "PAD": 0,
    "BOS": 1,
    "EOS": 2,
    "MASK": 3,
    "A": 4,
    "C": 5,
    "D": 6,
    "E": 7,
    "F": 8,
    "G": 9,
    "H": 10,
    "I": 11,
    "K": 12,
    "L": 13,
    "M": 14,
    "N": 15,
    "P": 16,
    "Q": 17,
    "R": 18,
    "S": 19,
    "T": 20,
    "V": 21,
    "W": 22,
    "Y": 23,
}


def test_token_ids_match_frozen_vocabulary() -> None:
    observed = {token.name: token.value for token in TokenId}

    assert observed == FROZEN_VOCABULARY


def test_all_canonical_residues_round_trip() -> None:
    token_ids = encode(CANONICAL_AMINO_ACIDS)

    assert token_ids == [1, *range(4, 24), 2]
    assert decode(token_ids) == CANONICAL_AMINO_ACIDS


@pytest.mark.parametrize(
    ("sequence", "message"),
    [
        ("", "sequence must not be empty"),
        ("acd", "invalid residue 'a' at biological position 1"),
        ("ACX", "invalid residue 'X' at biological position 3"),
        ("A C", "invalid residue ' ' at biological position 2"),
    ],
)
def test_encode_rejects_invalid_sequences(sequence: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        encode(sequence)


@pytest.mark.parametrize("sequence", [None, 123, b"ACD"])
def test_encode_rejects_non_string_inputs(sequence: object) -> None:
    with pytest.raises(TypeError, match="sequence must be a string"):
        encode(sequence)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("token_ids", "message"),
    [
        ([4, 5, 2], "first token must be BOS"),
        ([1, 4, 5], "last token must be EOS"),
        (
            [1, 2],
            "token_ids must contain BOS, at least one residue, and EOS",
        ),
    ],
)
def test_decode_rejects_invalid_boundaries(
    token_ids: list[int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        decode(token_ids)


@pytest.mark.parametrize("token_id", [0, 1, 2, 3, 24, 99])
def test_decode_rejects_non_residue_ids(token_id: int) -> None:
    with pytest.raises(
        ValueError,
        match=rf"token ID {token_id} at position 2 is not a biological residue",
    ):
        decode([1, 4, token_id, 2])


@pytest.mark.parametrize("token_ids", [None, 123, "1452", b"1452"])
def test_decode_rejects_non_sequence_inputs(token_ids: object) -> None:
    with pytest.raises(
        TypeError,
        match="token_ids must be a sequence of integers",
    ):
        decode(token_ids)  # type: ignore[arg-type]


@pytest.mark.parametrize("token_id", ["4", 4.0, True])
def test_decode_rejects_non_integer_token_ids(token_id: object) -> None:
    with pytest.raises(
        TypeError,
        match="token ID at position 1 must be an integer",
    ):
        decode([1, token_id, 2])  # type: ignore[list-item]
