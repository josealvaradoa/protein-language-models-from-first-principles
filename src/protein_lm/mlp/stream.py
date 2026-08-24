"""Bounded C=10 context streams with direct, validated resume cursors."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch

from protein_lm.bigram.stream import ordered_proteins
from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.data.model_data.loaders import ProteinSequence
from protein_lm.tokenization import CANONICAL_AMINO_ACIDS


CONTEXT_LENGTH = 10
BOS_CONTEXT_ID = 0
EOS_TARGET_ID = 20
_RESIDUE_TO_TARGET = {
    residue: index for index, residue in enumerate(CANONICAL_AMINO_ACIDS)
}


@dataclass(frozen=True)
class StreamCursor:
    """Position immediately after a prediction in the frozen protein-major stream."""

    prediction_index: int
    protein_index: int
    within_protein_target_offset: int


@dataclass(frozen=True)
class ContextBatch:
    """One bounded CPU batch and the cursor it reaches."""

    contexts: torch.Tensor
    targets: torch.Tensor
    start_cursor: StreamCursor
    end_cursor: StreamCursor

    @property
    def start_prediction(self) -> int:
        return self.start_cursor.prediction_index

    @property
    def predictions(self) -> int:
        return int(self.targets.numel())


def protein_context_pairs(
    sequence: str, context_length: int = CONTEXT_LENGTH
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return test-sized flattened contexts and residue-plus-EOS targets."""

    pairs = tuple(_iter_protein_context_pairs(sequence, context_length, 0))
    return (
        tuple(value for context, _, _ in pairs for value in context),
        tuple(target for _, target, _ in pairs),
    )


def iter_context_batches(
    proteins: Iterable[ProteinSequence],
    *,
    namespace: str,
    base_seed: int,
    prediction_budget: int,
    batch_size: int,
    event_predictions: Iterable[int] = (),
    cursor: StreamCursor | None = None,
    start_prediction: int | None = None,
    context_length: int = CONTEXT_LENGTH,
) -> Iterator[ContextBatch]:
    """Yield to the budget without replaying pairs before a validated cursor."""

    _validate_request(prediction_budget, batch_size, context_length)
    if start_prediction not in (None, 0):
        raise ModelDataError("nonzero resume requires a complete stream cursor")
    events = _validated_events(event_predictions, prediction_budget)
    ordered = ordered_proteins(proteins, namespace, base_seed)
    active = cursor or StreamCursor(0, 0, 0)
    _validate_cursor(active, ordered, prediction_budget)
    yield from _iter_ordered_batches(
        ordered, prediction_budget, batch_size, events, active, context_length
    )


def iter_native_context_batches(
    proteins: Iterable[ProteinSequence],
    *,
    namespace: str,
    base_seed: int,
    batch_size: int,
    context_length: int = CONTEXT_LENGTH,
) -> Iterator[ContextBatch]:
    """Order native validation once, then stream every target in bounded batches."""

    ordered = ordered_proteins(proteins, namespace, base_seed)
    total = sum(protein.biological_length + 1 for protein in ordered)
    if total <= 0:
        raise ModelDataError("native validation collection is empty")
    yield from _iter_ordered_batches(
        ordered, total, batch_size, (), StreamCursor(0, 0, 0), context_length
    )


def _iter_ordered_batches(
    ordered: tuple[ProteinSequence, ...],
    prediction_budget: int,
    batch_size: int,
    events: tuple[int, ...],
    cursor: StreamCursor,
    context_length: int,
) -> Iterator[ContextBatch]:
    active = cursor
    pending_contexts: list[int] = []
    pending_targets: list[int] = []
    start_cursor = active

    def limit() -> int:
        next_event = next(
            (event for event in events if event > start_cursor.prediction_index),
            prediction_budget,
        )
        return min(
            batch_size,
            prediction_budget - start_cursor.prediction_index,
            next_event - start_cursor.prediction_index,
        )

    def emit() -> ContextBatch:
        nonlocal start_cursor
        result = ContextBatch(
            torch.tensor(pending_contexts, dtype=torch.int64).reshape(
                -1, context_length
            ),
            torch.tensor(pending_targets, dtype=torch.int64),
            start_cursor,
            active,
        )
        pending_contexts.clear()
        pending_targets.clear()
        start_cursor = active
        return result

    for protein_index in range(active.protein_index, len(ordered)):
        protein = ordered[protein_index]
        offset = (
            active.within_protein_target_offset
            if protein_index == active.protein_index
            else 0
        )
        for context, target, next_offset in _iter_protein_context_pairs(
            protein.sequence, context_length, offset
        ):
            if active.prediction_index == prediction_budget:
                break
            pending_contexts.extend(context)
            pending_targets.append(target)
            if next_offset == protein.biological_length + 1:
                active = StreamCursor(active.prediction_index + 1, protein_index + 1, 0)
            else:
                active = StreamCursor(
                    active.prediction_index + 1, protein_index, next_offset
                )
            if len(pending_targets) == limit() or active.prediction_index in events:
                yield emit()
        if active.prediction_index == prediction_budget:
            break
    if active.prediction_index != prediction_budget:
        raise ModelDataError(
            "training collection cannot satisfy the fixed stream budget"
        )
    if pending_targets:
        yield emit()


def _iter_protein_context_pairs(
    sequence: str, context_length: int, offset: int
) -> Iterator[tuple[tuple[int, ...], int, int]]:
    """Start at one offset after reconstructing only its at-most-C prefix."""

    _validate_sequence(sequence)
    if type(offset) is not int or offset < 0 or offset > len(sequence):
        raise ModelDataError("within-protein cursor offset is invalid")
    window = [BOS_CONTEXT_ID] * context_length
    for residue in sequence[max(0, offset - context_length) : offset]:
        window = window[1:] + [_RESIDUE_TO_TARGET[residue] + 1]
    for target_offset in range(offset, len(sequence)):
        target = _RESIDUE_TO_TARGET[sequence[target_offset]]
        yield tuple(window), target, target_offset + 1
        window = window[1:] + [target + 1]
    yield tuple(window), EOS_TARGET_ID, len(sequence) + 1


def _validate_cursor(
    cursor: StreamCursor,
    proteins: tuple[ProteinSequence, ...],
    budget: int,
) -> None:
    if not isinstance(cursor, StreamCursor) or any(
        type(value) is not int
        for value in (
            cursor.prediction_index,
            cursor.protein_index,
            cursor.within_protein_target_offset,
        )
    ):
        raise ModelDataError("stream cursor is invalid")
    if cursor.protein_index < 0 or cursor.protein_index > len(proteins):
        raise ModelDataError("stream cursor protein index is invalid")
    if cursor.protein_index == len(proteins):
        if cursor.within_protein_target_offset != 0:
            raise ModelDataError("terminal stream cursor offset is invalid")
    elif (
        not 0
        <= cursor.within_protein_target_offset
        <= proteins[cursor.protein_index].biological_length
    ):
        raise ModelDataError("stream cursor offset is invalid")
    expected = sum(
        item.biological_length + 1 for item in proteins[: cursor.protein_index]
    )
    expected += cursor.within_protein_target_offset
    if cursor.prediction_index != expected or cursor.prediction_index > budget:
        raise ModelDataError("stream cursor prediction accounting is invalid")


def _validate_sequence(sequence: str) -> None:
    if (
        not isinstance(sequence, str)
        or not sequence
        or any(residue not in _RESIDUE_TO_TARGET for residue in sequence)
    ):
        raise ModelDataError(
            "protein sequence is empty or contains a noncanonical residue"
        )


def _validated_events(events: Iterable[int], budget: int) -> tuple[int, ...]:
    values = tuple(events)
    if any(type(value) is not int or value <= 0 or value > budget for value in values):
        raise ModelDataError("stream event boundary is invalid")
    if tuple(sorted(set(values))) != values:
        raise ModelDataError("stream event boundaries must be ordered and unique")
    return values


def _validate_request(budget: int, batch_size: int, context_length: int) -> None:
    if (
        any(type(value) is not int for value in (budget, batch_size, context_length))
        or budget <= 0
        or batch_size <= 0
        or context_length <= 0
    ):
        raise ModelDataError("context stream budget or batch size is invalid")
