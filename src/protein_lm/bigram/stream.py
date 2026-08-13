"""Exact, protein-major coordinates and hashing for Week 2 bigram streams."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.data.model_data.loaders import ProteinSequence
from protein_lm.tokenization import CANONICAL_AMINO_ACIDS


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BOS_CONTEXT = 0
_EOS_TARGET = 20
_ROLE_SPACE_SIZE = 21
_CANONICAL_SEQUENCE = re.compile(f"[{CANONICAL_AMINO_ACIDS}]+")
_RESIDUE_TARGET_TRANSLATION = bytes.maketrans(
    CANONICAL_AMINO_ACIDS.encode("ascii"), bytes(range(20))
)
_TARGET_CONTEXT_TRANSLATION = bytes.maketrans(bytes(range(20)), bytes(range(1, 21)))


@dataclass(frozen=True)
class ArmStreamAudit:
    """Aggregate-only result for one fully consumed fixed-budget arm."""

    namespace: str
    pairs_emitted: int
    proteins_started: int
    proteins_completed: int
    final_protein_partial: bool
    context_counts: tuple[int, ...]
    target_counts: tuple[int, ...]
    stream_sha256: str


def ordered_proteins(
    proteins: Iterable[ProteinSequence], namespace: str, base_seed: int
) -> tuple[ProteinSequence, ...]:
    """Consume a collection once and return its frozen domain-separated order."""

    if (
        not isinstance(namespace, str)
        or not namespace
        or not isinstance(base_seed, int)
        or isinstance(base_seed, bool)
    ):
        raise ModelDataError("stream namespace or base seed is invalid")
    seen_accessions: set[str] = set()
    seen_hashes: set[str] = set()
    sortable: list[tuple[bytes, str, ProteinSequence]] = []
    for protein in proteins:
        _validate_protein(protein)
        if protein.primary_accession in seen_accessions:
            raise ModelDataError("training collection contains a repeated protein")
        seen_accessions.add(protein.primary_accession)
        if protein.sequence_sha256 in seen_hashes:
            raise ModelDataError(
                "training collection contains duplicate sequence SHA-256 values"
            )
        seen_hashes.add(protein.sequence_sha256)
        order_digest = protein_order_key(
            protein.sequence_sha256, namespace, base_seed
        )
        sortable.append((order_digest, protein.sequence_sha256, protein))
    return tuple(item[2] for item in sorted(sortable, key=lambda item: item[:2]))


def protein_order_key(sequence_sha256: str, namespace: str, base_seed: int) -> bytes:
    """Return the frozen digest used to order one sequence in an arm."""

    if _SHA256.fullmatch(sequence_sha256) is None:
        raise ModelDataError("protein sequence SHA-256 is malformed")
    if not isinstance(namespace, str) or not namespace:
        raise ModelDataError("stream namespace is invalid")
    if not isinstance(base_seed, int) or isinstance(base_seed, bool):
        raise ModelDataError("stream base seed is invalid")
    return hashlib.sha256(
        namespace.encode("utf-8")
        + b"\0"
        + str(base_seed).encode("ascii")
        + b"\0"
        + sequence_sha256.encode("utf-8")
    ).digest()


def audit_stream(
    proteins: Iterable[ProteinSequence],
    *,
    namespace: str,
    base_seed: int,
    pair_budget: int,
    hash_domain: str,
) -> ArmStreamAudit:
    """Audit one exact stream without constructing cross-protein transitions."""

    if (
        not isinstance(pair_budget, int)
        or isinstance(pair_budget, bool)
        or pair_budget <= 0
        or not isinstance(hash_domain, str)
        or not hash_domain
    ):
        raise ModelDataError("stream budget or hash domain is invalid")
    ordered = ordered_proteins(proteins, namespace, base_seed)
    hasher = hashlib.sha256()
    hasher.update(hash_domain.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(namespace.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(str(base_seed).encode("ascii"))
    hasher.update(b"\0")
    contexts = Counter()
    targets = Counter()
    emitted = 0
    started = 0
    completed = 0
    partial = False
    for protein in ordered:
        remaining = pair_budget - emitted
        if remaining == 0:
            break
        packed, context_values, target_values = protein_pair_bytes(protein.sequence)
        pair_count = len(context_values)
        consumed = min(pair_count, remaining)
        hasher.update(packed[: 2 * consumed])
        contexts.update(context_values[:consumed])
        targets.update(target_values[:consumed])
        emitted += consumed
        started += 1
        if consumed == pair_count:
            completed += 1
        else:
            partial = True
            break
    if emitted != pair_budget:
        raise ModelDataError(
            "training collection cannot satisfy the fixed stream budget"
        )
    return ArmStreamAudit(
        namespace=namespace,
        pairs_emitted=emitted,
        proteins_started=started,
        proteins_completed=completed,
        final_protein_partial=partial,
        context_counts=tuple(contexts[index] for index in range(_ROLE_SPACE_SIZE)),
        target_counts=tuple(targets[index] for index in range(_ROLE_SPACE_SIZE)),
        stream_sha256=hasher.hexdigest(),
    )


def protein_pair_bytes(sequence: str) -> tuple[bytes, bytes, bytes]:
    """Return interleaved pair bytes plus role-specific context and target bytes."""

    if not isinstance(sequence, str) or not sequence:
        raise ModelDataError("protein sequence is empty or malformed")
    if _CANONICAL_SEQUENCE.fullmatch(sequence) is None:
        raise ModelDataError("protein sequence contains a noncanonical residue")
    residues = sequence.encode("ascii").translate(_RESIDUE_TARGET_TRANSLATION)
    context_values = bytes((_BOS_CONTEXT,)) + residues.translate(
        _TARGET_CONTEXT_TRANSLATION
    )
    target_values = residues + bytes((_EOS_TARGET,))
    packed = bytearray(2 * len(context_values))
    packed[0::2] = context_values
    packed[1::2] = target_values
    return bytes(packed), context_values, target_values


def _validate_protein(protein: ProteinSequence) -> None:
    if not isinstance(protein, ProteinSequence):
        raise ModelDataError(
            "training stream requires loader-proved ProteinSequence values"
        )
    if _SHA256.fullmatch(protein.sequence_sha256) is None:
        raise ModelDataError("protein sequence SHA-256 is malformed")
    try:
        encoded = protein.sequence.encode("ascii")
    except UnicodeEncodeError as error:
        raise ModelDataError(
            "protein sequence identity disagrees with loader metadata"
        ) from error
    if (
        hashlib.sha256(encoded).hexdigest() != protein.sequence_sha256
        or protein.biological_length != len(protein.sequence)
    ):
        raise ModelDataError("protein sequence identity disagrees with loader metadata")
    protein_pair_bytes(protein.sequence)
