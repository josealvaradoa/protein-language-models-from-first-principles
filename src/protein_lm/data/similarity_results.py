"""Canonical result files and cap convergence for the Task 7 audit."""

from __future__ import annotations

import hashlib
import heapq
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path

from protein_lm.data.similarity_alignment import (
    decode_strict_tsv_line,
    parse_alignment_row,
    parse_canonical_alignment_row,
)
from protein_lm.data.similarity_audit_models import (
    AlignmentRow,
    CanonicalAlignmentEvidence,
    ConvergenceEvidence,
    FileEvidence,
    SequenceMetadata,
)
from protein_lm.data.similarity_audit_policy import SimilarityAuditError

_MERGE_FAN_IN = 64


def canonicalize_mmseqs_tsv(
    raw_path: Path,
    canonical_path: Path,
    *,
    query_metadata: Mapping[str, SequenceMetadata],
    target_metadata: Mapping[str, SequenceMetadata],
    chunk_rows: int,
    resource_guard: Callable[[], None] | None = None,
    delete_raw_after_parse: bool = False,
) -> CanonicalAlignmentEvidence:
    """Strictly parse, type-normalize, and externally sort one raw TSV."""

    if chunk_rows <= 0:
        raise SimilarityAuditError("parser chunk size must be positive")
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    raw_hasher = hashlib.sha256()
    raw_bytes = 0
    raw_rows = 0

    with tempfile.TemporaryDirectory(
        prefix=f".{canonical_path.name}.chunks-",
        dir=canonical_path.parent,
    ) as temporary_directory:
        chunk_directory = Path(temporary_directory)
        chunks: list[Path] = []
        pending: list[bytes] = []
        try:
            source = raw_path.open("rb")
        except OSError as error:
            raise SimilarityAuditError(f"could not open MMseqs2 output: {error}") from error

        with source:
            for line_number, raw_line in enumerate(source, start=1):
                raw_hasher.update(raw_line)
                raw_bytes += len(raw_line)
                raw_rows += 1
                cells = decode_strict_tsv_line(
                    raw_line,
                    context=f"{raw_path.name} line {line_number}",
                ).split("\t")
                row = parse_alignment_row(
                    cells,
                    query_metadata=query_metadata,
                    target_metadata=target_metadata,
                    context=f"{raw_path.name} line {line_number}",
                )
                pending.append(row.canonical_line())
                if len(pending) == chunk_rows:
                    chunks.append(_write_sorted_chunk(chunk_directory, chunks, pending))
                    pending = []
                    if resource_guard is not None:
                        resource_guard()
        if pending:
            chunks.append(_write_sorted_chunk(chunk_directory, chunks, pending))
            if resource_guard is not None:
                resource_guard()

        if raw_path.stat().st_size != raw_bytes:
            raise SimilarityAuditError("MMseqs2 output size changed while parsing")
        if delete_raw_after_parse:
            raw_path.unlink()
        canonical_evidence = _merge_chunks(
            chunks,
            canonical_path,
            resource_guard=resource_guard,
        )

    return CanonicalAlignmentEvidence(
        raw=FileEvidence(raw_rows, raw_bytes, raw_hasher.hexdigest()),
        canonical=canonical_evidence,
    )


def compare_canonical_results(
    left_path: Path,
    right_path: Path,
    *,
    expected_query_ids: Iterable[str],
    left_may_contain_other_queries: bool = False,
) -> tuple[str, ...]:
    """Compare every complete typed row tuple, including empty query results."""

    expected = tuple(sorted(expected_query_ids))
    if len(set(expected)) != len(expected):
        raise SimilarityAuditError("expected query identifiers are not unique")
    expected_set = frozenset(expected)
    left = _GroupCursor(
        _iter_canonical_groups(
            left_path,
            accepted_queries=expected_set,
            allow_other_queries=left_may_contain_other_queries,
        )
    )
    right = _GroupCursor(
        _iter_canonical_groups(
            right_path,
            accepted_queries=expected_set,
            allow_other_queries=False,
        )
    )
    differences = []
    for query in expected:
        left_rows = left.take(query)
        right_rows = right.take(query)
        if tuple(row.typed_tuple() for row in left_rows) != tuple(
            row.typed_tuple() for row in right_rows
        ):
            differences.append(query)
    left.require_exhausted()
    right.require_exhausted()
    return tuple(differences)


def convergence_evidence(
    *,
    expected_query_ids: Iterable[str],
    initial_path: Path,
    comparison_path: Path,
    escalation_path: Path | None,
) -> ConvergenceEvidence:
    """Apply the frozen 1,000 to 10,000 to 100,000 convergence rule."""

    expected = tuple(sorted(expected_query_ids))
    differing = compare_canonical_results(
        initial_path,
        comparison_path,
        expected_query_ids=expected,
    )
    if differing:
        if escalation_path is None:
            raise SimilarityAuditError(
                "staged-cap comparison requires an escalation result"
            )
        final_differences = compare_canonical_results(
            comparison_path,
            escalation_path,
            expected_query_ids=differing,
            left_may_contain_other_queries=True,
        )
    else:
        if escalation_path is not None:
            raise SimilarityAuditError(
                "an escalation result exists when no query required escalation"
            )
        final_differences = ()
    if final_differences:
        raise SimilarityAuditError(
            "staged-cap evidence still differs between 10,000 and 100,000"
        )
    return ConvergenceEvidence(
        expected_queries=len(expected),
        converged_at_comparison_cap=len(expected) - len(differing),
        escalated_queries=len(differing),
        converged_at_escalation_cap=len(differing),
        final_differing_queries=0,
        escalated_query_ids=differing,
    )


def iter_converged_query_rows(
    expected_queries: Sequence[str],
    *,
    comparison_path: Path,
    escalation_path: Path | None,
    escalated_queries: frozenset[str],
) -> Iterator[tuple[str, tuple[AlignmentRow, ...]]]:
    """Yield each query's rows from its final accepted search cap."""

    expected_set = frozenset(expected_queries)
    comparison = _GroupCursor(
        _iter_canonical_groups(
            comparison_path,
            accepted_queries=expected_set,
            allow_other_queries=False,
        )
    )
    escalation = None
    if escalated_queries:
        if escalation_path is None:
            raise SimilarityAuditError("missing final escalation output")
        escalation = _GroupCursor(
            _iter_canonical_groups(
                escalation_path,
                accepted_queries=escalated_queries,
                allow_other_queries=False,
            )
        )
    for query in expected_queries:
        comparison_rows = comparison.take(query)
        if query in escalated_queries:
            if escalation is None:
                raise SimilarityAuditError("missing escalation cursor")
            yield query, escalation.take(query)
        else:
            yield query, comparison_rows
    comparison.require_exhausted()
    if escalation is not None:
        escalation.require_exhausted()


def _write_sorted_chunk(
    directory: Path,
    existing_chunks: Sequence[Path],
    rows: list[bytes],
) -> Path:
    rows.sort()
    path = directory / f"chunk-{len(existing_chunks):06d}.tsv"
    with path.open("wb") as output:
        output.writelines(rows)
    return path


def _merge_chunks(
    chunks: Sequence[Path],
    output_path: Path,
    *,
    resource_guard: Callable[[], None] | None,
) -> FileEvidence:
    current = list(chunks)
    merge_round = 0
    while len(current) > _MERGE_FAN_IN:
        next_round = []
        for group_index, start in enumerate(range(0, len(current), _MERGE_FAN_IN)):
            group = current[start : start + _MERGE_FAN_IN]
            merged_path = group[0].parent / (
                f"merge-{merge_round:03d}-{group_index:06d}.tsv"
            )
            _merge_sorted_files(
                group,
                merged_path,
                resource_guard=resource_guard,
            )
            for path in group:
                path.unlink()
            next_round.append(merged_path)
        current = next_round
        merge_round += 1

    temporary_path = output_path.with_name(f".{output_path.name}.incomplete")
    temporary_path.unlink(missing_ok=True)
    hasher = hashlib.sha256()
    byte_size = 0
    row_count = 0
    previous_pair: tuple[bytes, bytes] | None = None
    handles = []
    try:
        handles = [path.open("rb") for path in current]
        with temporary_path.open("wb") as output:
            for line in heapq.merge(*handles):
                parts = line.split(b"\t", 2)
                if len(parts) != 3:
                    raise SimilarityAuditError("canonical alignment row is malformed")
                pair = (parts[0], parts[1])
                if pair == previous_pair:
                    raise SimilarityAuditError(
                        "MMseqs2 output contains a duplicate query-target pair"
                    )
                previous_pair = pair
                output.write(line)
                hasher.update(line)
                byte_size += len(line)
                row_count += 1
                if resource_guard is not None and row_count % 100_000 == 0:
                    resource_guard()
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        for handle in handles:
            handle.close()
    return FileEvidence(row_count, byte_size, hasher.hexdigest())


def _merge_sorted_files(
    paths: Sequence[Path],
    output_path: Path,
    *,
    resource_guard: Callable[[], None] | None,
) -> None:
    handles = []
    try:
        handles = [path.open("rb") for path in paths]
        with output_path.open("wb") as output:
            for row_count, line in enumerate(heapq.merge(*handles), start=1):
                output.write(line)
                if resource_guard is not None and row_count % 100_000 == 0:
                    resource_guard()
    finally:
        for handle in handles:
            handle.close()


def _iter_canonical_groups(
    path: Path,
    *,
    accepted_queries: frozenset[str],
    allow_other_queries: bool,
) -> Iterator[tuple[str, tuple[AlignmentRow, ...]]]:
    current_query: str | None = None
    current_rows: list[AlignmentRow] = []
    previous_line: bytes | None = None
    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if previous_line is not None and raw_line < previous_line:
                raise SimilarityAuditError(f"{path.name} is not canonically sorted")
            previous_line = raw_line
            cells = decode_strict_tsv_line(
                raw_line,
                context=f"{path.name} line {line_number}",
            ).split("\t")
            query = cells[0]
            if query not in accepted_queries:
                if allow_other_queries:
                    continue
                raise SimilarityAuditError(
                    f"{path.name} contains unexpected query {query!r}"
                )
            row = parse_canonical_alignment_row(
                cells,
                context=f"{path.name} line {line_number}",
            )
            if current_query is None:
                current_query = query
            if query != current_query:
                yield current_query, tuple(current_rows)
                current_query = query
                current_rows = []
            current_rows.append(row)
    if current_query is not None:
        yield current_query, tuple(current_rows)


class _GroupCursor:
    def __init__(self, groups: Iterator[tuple[str, tuple[AlignmentRow, ...]]]) -> None:
        self._groups = groups
        self._current = next(groups, None)

    def take(self, query: str) -> tuple[AlignmentRow, ...]:
        if self._current is None or self._current[0] > query:
            return ()
        if self._current[0] < query:
            raise SimilarityAuditError("canonical query order is inconsistent")
        rows = self._current[1]
        self._current = next(self._groups, None)
        return rows

    def require_exhausted(self) -> None:
        if self._current is not None:
            raise SimilarityAuditError("canonical result contains an extra query")
