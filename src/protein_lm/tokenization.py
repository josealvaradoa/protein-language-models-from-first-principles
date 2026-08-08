from collections.abc import Sequence
from enum import IntEnum


class TokenId(IntEnum):
    PAD = 0
    BOS = 1
    EOS = 2
    MASK = 3

    A = 4
    C = 5
    D = 6
    E = 7
    F = 8
    G = 9
    H = 10
    I = 11  # noqa: E741
    K = 12
    L = 13
    M = 14
    N = 15
    P = 16
    Q = 17
    R = 18
    S = 19
    T = 20
    V = 21
    W = 22
    Y = 23


CANONICAL_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

_RESIDUE_TO_ID = {residue: TokenId[residue].value for residue in CANONICAL_AMINO_ACIDS}

_ID_TO_RESIDUE = {token_id: residue for residue, token_id in _RESIDUE_TO_ID.items()}


def encode(sequence: str) -> list[int]:
    """Encode one normalized protein sequence with BOS and EOS tokens."""

    if not isinstance(sequence, str):
        raise TypeError("sequence must be a string")

    if not sequence:
        raise ValueError("sequence must not be empty")

    token_ids = [TokenId.BOS.value]

    for position, residue in enumerate(sequence, start=1):
        if residue not in _RESIDUE_TO_ID:
            raise ValueError(
                f"invalid residue {residue!r} at biological position {position}"
            )

        token_ids.append(_RESIDUE_TO_ID[residue])

    token_ids.append(TokenId.EOS.value)
    return token_ids


def decode(token_ids: Sequence[int]) -> str:
    """Decode one BOS/residue/EOS token sequence into a protein sequence."""

    if isinstance(token_ids, (str, bytes)) or not isinstance(token_ids, Sequence):
        raise TypeError("token_ids must be a sequence of integers")

    if len(token_ids) < 3:
        raise ValueError("token_ids must contain BOS, at least one residue, and EOS")

    for token_position, token_id in enumerate(token_ids):
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise TypeError(f"token ID at position {token_position} must be an integer")

    if token_ids[0] != TokenId.BOS.value:
        raise ValueError("first token must be BOS")

    if token_ids[-1] != TokenId.EOS.value:
        raise ValueError("last token must be EOS")

    residues = []

    for token_position, token_id in enumerate(token_ids[1:-1], start=1):
        residue = _ID_TO_RESIDUE.get(token_id)

        if residue is None:
            raise ValueError(
                f"token ID {token_id} at position {token_position} "
                "is not a biological residue"
            )

        residues.append(residue)

    return "".join(residues)
