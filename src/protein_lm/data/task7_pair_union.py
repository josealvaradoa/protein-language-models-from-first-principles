"""Build the unique prohibited-pair union required by A-004."""

from __future__ import annotations

import heapq
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from protein_lm.data.similarity_alignment import decode_strict_tsv_line
from protein_lm.data.similarity_audit_models import FileEvidence
from protein_lm.data.similarity_audit_policy import SimilarityAuditError
from protein_lm.data.task7_evidence_io import EvidenceWriter


@dataclass(frozen=True)
class PairUnionEvidence:
    """Identity and aggregate counts for one sorted pair union."""

    source_labels: tuple[str, ...]
    unique_pairs: int
    unique_queries: int
    prohibited_pair_file: FileEvidence


@dataclass(frozen=True)
class PairUnionComparison:
    """Additional detected evidence in the staged union."""

    common_pairs: int
    staged_pairs: int
    additional_pairs: int
    common_queries: int
    staged_queries: int
    newly_prohibited_queries: int


def union_prohibited_pairs(
    *,
    source_paths: Mapping[str, Path],
    output_directory: Path,
) -> PairUnionEvidence:
    """Stream sorted pair files into one immutable, duplicate-free union."""

    labels = tuple(sorted(source_paths))
    if not labels or any(not label for label in labels):
        raise SimilarityAuditError("pair union requires named source files")
    paths = tuple(source_paths[label] for label in labels)
    if len(set(paths)) != len(paths):
        raise SimilarityAuditError("pair union source paths must be distinct")

    staging = output_directory.with_name(f".{output_directory.name}.incomplete")
    if output_directory.exists() or staging.exists():
        raise SimilarityAuditError("pair union requires a fresh output directory")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    writer = EvidenceWriter(staging / "prohibited_pairs.tsv")
    unique_pairs = 0
    unique_queries = 0
    previous_pair: tuple[str, str] | None = None
    previous_query: str | None = None

    try:
        readers = tuple(_iter_pair_file(path) for path in paths)
        for pair in heapq.merge(*readers):
            if pair == previous_pair:
                continue
            query, target = pair
            writer.write(f"{query}\t{target}\n".encode("ascii"))
            unique_pairs += 1
            if query != previous_query:
                unique_queries += 1
                previous_query = query
            previous_pair = pair
        pair_evidence = writer.finish()
        staging.replace(output_directory)
    except BaseException:
        writer.abort()
        staging.rmdir()
        raise

    if pair_evidence.row_count != unique_pairs:
        raise SimilarityAuditError("pair union count does not reconcile")
    return PairUnionEvidence(
        source_labels=labels,
        unique_pairs=unique_pairs,
        unique_queries=unique_queries,
        prohibited_pair_file=pair_evidence,
    )


def compare_pair_unions(
    *,
    common_path: Path,
    staged_path: Path,
) -> PairUnionComparison:
    """Require the staged union to contain the common union and count additions."""

    if common_path == staged_path:
        raise SimilarityAuditError("pair-union comparison requires distinct files")
    common_iterator = _iter_pair_file(common_path)
    staged_iterator = _iter_pair_file(staged_path)
    common = next(common_iterator, None)
    staged = next(staged_iterator, None)
    common_pairs = 0
    staged_pairs = 0
    common_queries: set[str] = set()
    staged_queries: set[str] = set()

    while common is not None and staged is not None:
        if common < staged:
            raise SimilarityAuditError("staged union is missing a common-cap pair")
        if staged < common:
            staged_pairs += 1
            staged_queries.add(staged[0])
            staged = next(staged_iterator, None)
            continue
        common_pairs += 1
        staged_pairs += 1
        common_queries.add(common[0])
        staged_queries.add(staged[0])
        common = next(common_iterator, None)
        staged = next(staged_iterator, None)

    if common is not None:
        raise SimilarityAuditError("staged union is missing a common-cap pair")
    if staged is not None:
        staged_pairs += 1
        staged_queries.add(staged[0])
    for pair in staged_iterator:
        staged_pairs += 1
        staged_queries.add(pair[0])

    return PairUnionComparison(
        common_pairs=common_pairs,
        staged_pairs=staged_pairs,
        additional_pairs=staged_pairs - common_pairs,
        common_queries=len(common_queries),
        staged_queries=len(staged_queries),
        newly_prohibited_queries=len(staged_queries - common_queries),
    )


def _iter_pair_file(path: Path) -> Iterator[tuple[str, str]]:
    previous: tuple[str, str] | None = None
    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            cells = decode_strict_tsv_line(
                raw_line,
                context=f"{path.name} line {line_number}",
            ).split("\t")
            if len(cells) != 2 or not all(cells):
                raise SimilarityAuditError("prohibited-pair row is malformed")
            pair = (cells[0], cells[1])
            if previous is not None and pair <= previous:
                raise SimilarityAuditError(
                    "prohibited-pair rows must be unique and sorted"
                )
            previous = pair
            yield pair
