"""Pure CPU sampling from saved neural bigram logits."""

from __future__ import annotations

import hashlib

import torch

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.tokenization import CANONICAL_AMINO_ACIDS


def derived_seed(base_seed: int, namespace: str, sample_index: int) -> int:
    """Use SHA-256 of UTF-8 `v1\\0base_seed\\0namespace\\0sample_index`.

    The first eight digest bytes are interpreted big-endian, then masked to the
    signed 63-bit range accepted by CPU ``torch.Generator.manual_seed``.
    """
    if type(base_seed) is not int or type(sample_index) is not int or sample_index < 0:
        raise ModelDataError("sampling seed inputs are invalid")
    material = f"v1\0{base_seed}\0{namespace}\0{sample_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def sample_neural_bigram(
    logits: torch.Tensor,
    *,
    base_seed: int,
    namespace: str,
    sample_index: int,
    max_residues: int = 128,
) -> dict[str, object]:
    """Sample at temperature 1.0 from BOS until EOS or the fixed residue cap."""
    if (
        logits.device.type != "cpu"
        or logits.dtype != torch.float32
        or tuple(logits.shape) != (21, 21)
    ):
        raise ModelDataError(
            "neural sampling logits must be a CPU float32 21 by 21 tensor"
        )
    if (
        not torch.isfinite(logits).all().item()
        or not isinstance(namespace, str)
        or not namespace
    ):
        raise ModelDataError("neural sampling logits or namespace are invalid")
    if type(max_residues) is not int or max_residues <= 0:
        raise ModelDataError("sampling residue cap is invalid")
    seed = derived_seed(base_seed, namespace, sample_index)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    context = 0  # BOS is the first context role.
    residues: list[str] = []
    for _ in range(max_residues):
        target = int(
            torch.multinomial(
                torch.softmax(logits[context], dim=0), 1, generator=generator
            ).item()
        )
        if target == 20:  # EOS is the final target role.
            return {
                "sample_index": sample_index,
                "seed": seed,
                "sequence": "".join(residues),
                "residue_length": len(residues),
                "termination_reason": "eos",
            }
        residues.append(CANONICAL_AMINO_ACIDS[target])
        context = target + 1  # target residue 0..19 maps to context role 1..20.
    return {
        "sample_index": sample_index,
        "seed": seed,
        "sequence": "".join(residues),
        "residue_length": len(residues),
        "termination_reason": "max_residues",
    }
