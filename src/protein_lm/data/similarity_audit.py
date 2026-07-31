"""Strict parsing and aggregation for the Week 1 Task 7 MMseqs2 audit."""

from __future__ import annotations

import hashlib
import heapq
import re
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from protein_lm.data.similarity_audit_policy import SimilarityAuditError

ALIGNMENT_FIELDS = (
    "query",
    "target",
    "fident",
    "qcov",
    "tcov",
    "alnlen",
    "qlen",
    "tlen",
    "qstart",
    "qend",
    "tstart",
    "tend",
    "evalue",
    "bits",
)
PROHIBITED_MIN_IDENTITY = Decimal("0.50")
PROHIBITED_MIN_QUERY_COVERAGE = Decimal("0.80")
PROHIBITED_MIN_TARGET_COVERAGE = Decimal("0.80")

CATEGORY_PROHIBITED = "prohibited"
CATEGORY_CLOSEST_PROHIBITED = "closest_match_prohibited"
CATEGORY_GE_50_LOW_COVERAGE = "identity_ge_50_below_bidirectional_coverage"
CATEGORY_40_TO_50 = "identity_40_to_under_50"
CATEGORY_30_TO_40 = "identity_30_to_under_40"
CATEGORY_UNDER_30_OR_NONE = "identity_under_30_or_no_residual_hit"
RESIDUAL_CATEGORIES = (
    CATEGORY_PROHIBITED,
    CATEGORY_GE_50_LOW_COVERAGE,
    CATEGORY_40_TO_50,
    CATEGORY_30_TO_40,
    CATEGORY_UNDER_30_OR_NONE,
)
CLOSEST_RESIDUAL_CATEGORIES = (
    CATEGORY_CLOSEST_PROHIBITED,
    CATEGORY_GE_50_LOW_COVERAGE,
    CATEGORY_40_TO_50,
    CATEGORY_30_TO_40,
    CATEGORY_UNDER_30_OR_NONE,
)

_DECIMAL_PATTERN = re.compile(
    r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$"
)
_POSITIVE_INTEGER_PATTERN = re.compile(r"^[1-9][0-9]*$")
_MERGE_FAN_IN = 64


@dataclass(frozen=True)
class SequenceMetadata:
    """Frozen identity fields for one eligible protein."""

    sequence_sha256: str
    biological_length: int
    uniref50_group: str
    partition: str


@dataclass(frozen=True)
class FileEvidence:
    """Content identity of one local intermediate artifact."""

    row_count: int
    byte_size: int
    sha256: str


@dataclass(frozen=True)
class CanonicalAlignmentEvidence:
    """Raw and canonical identities for one completed MMseqs2 output."""

    raw: FileEvidence
    canonical: FileEvidence


@dataclass(frozen=True)
class AlignmentRow:
    """One fully typed row from the frozen fourteen-column output."""

    query: str
    target: str
    fident: Decimal
    qcov: Decimal
    tcov: Decimal
    alnlen: int
    qlen: int
    tlen: int
    qstart: int
    qend: int
    tstart: int
    tend: int
    evalue: Decimal
    bits: Decimal

    def typed_tuple(self) -> tuple[object, ...]:
        """Return all fields in the exact frozen output order."""

        return tuple(getattr(self, field) for field in ALIGNMENT_FIELDS)

    def canonical_line(self) -> bytes:
        """Serialize typed values so equivalent decimal spellings become equal."""

        values = (
            self.query,
            self.target,
            _canonical_decimal(self.fident),
            _canonical_decimal(self.qcov),
            _canonical_decimal(self.tcov),
            str(self.alnlen),
            str(self.qlen),
            str(self.tlen),
            str(self.qstart),
            str(self.qend),
            str(self.tstart),
            str(self.tend),
            _canonical_decimal(self.evalue),
            _canonical_decimal(self.bits),
        )
        return ("\t".join(values) + "\n").encode("ascii")


@dataclass(frozen=True)
class ConvergenceEvidence:
    """Per-query equality evidence for one search track."""

    expected_queries: int
    converged_at_comparison_cap: int
    escalated_queries: int
    converged_at_escalation_cap: int
    final_differing_queries: int
    escalated_query_ids: tuple[str, ...]


@dataclass(frozen=True)
class AcceptedPassEvidence:
    """Compact local evidence retained after one pass converges."""

    pass_name: str
    accepted_rows: int
    accepted_at_comparison_cap: int
    accepted_at_escalation_cap: int
    returned_pairs: FileEvidence
    prohibited_pairs: FileEvidence
    residual_summaries: FileEvidence | None


def parse_alignment_row(
    cells: Sequence[str],
    *,
    query_metadata: Mapping[str, SequenceMetadata],
    target_metadata: Mapping[str, SequenceMetadata],
    context: str,
) -> AlignmentRow:
    """Parse and validate one raw or canonical MMseqs2 row."""

    if len(cells) != len(ALIGNMENT_FIELDS):
        raise SimilarityAuditError(
            f"{context}: expected {len(ALIGNMENT_FIELDS)} tab-separated fields"
        )
    query, target = cells[0], cells[1]
    _require_visible_ascii(query, f"{context} query")
    _require_visible_ascii(target, f"{context} target")
    if query not in query_metadata:
        raise SimilarityAuditError(f"{context}: unexpected query {query!r}")
    if target not in target_metadata:
        raise SimilarityAuditError(f"{context}: unexpected target {target!r}")
    if query == target:
        raise SimilarityAuditError(f"{context}: query and target are identical")

    fident = _parse_decimal(cells[2], f"{context} fident")
    qcov = _parse_decimal(cells[3], f"{context} qcov")
    tcov = _parse_decimal(cells[4], f"{context} tcov")
    alnlen = _parse_positive_integer(cells[5], f"{context} alnlen")
    qlen = _parse_positive_integer(cells[6], f"{context} qlen")
    tlen = _parse_positive_integer(cells[7], f"{context} tlen")
    qstart = _parse_positive_integer(cells[8], f"{context} qstart")
    qend = _parse_positive_integer(cells[9], f"{context} qend")
    tstart = _parse_positive_integer(cells[10], f"{context} tstart")
    tend = _parse_positive_integer(cells[11], f"{context} tend")
    evalue = _parse_decimal(cells[12], f"{context} evalue")
    bits = _parse_decimal(cells[13], f"{context} bits")

    for name, value in (("fident", fident), ("qcov", qcov), ("tcov", tcov)):
        if value < 0 or value > 1:
            raise SimilarityAuditError(f"{context}: {name} must be between 0 and 1")
    if evalue < 0:
        raise SimilarityAuditError(f"{context}: evalue must not be negative")
    if qlen != query_metadata[query].biological_length:
        raise SimilarityAuditError(f"{context}: qlen differs from Task 4")
    if tlen != target_metadata[target].biological_length:
        raise SimilarityAuditError(f"{context}: tlen differs from Task 4")
    if qstart > qend or qend > qlen:
        raise SimilarityAuditError(f"{context}: query coordinates are invalid")
    if tstart > tend or tend > tlen:
        raise SimilarityAuditError(f"{context}: target coordinates are invalid")

    return AlignmentRow(
        query=query,
        target=target,
        fident=fident,
        qcov=qcov,
        tcov=tcov,
        alnlen=alnlen,
        qlen=qlen,
        tlen=tlen,
        qstart=qstart,
        qend=qend,
        tstart=tstart,
        tend=tend,
        evalue=evalue,
        bits=bits,
    )


def violates_prohibited_boundary(row: AlignmentRow) -> bool:
    """Apply the inclusive project boundary independently of MMseqs2."""

    return (
        row.fident >= PROHIBITED_MIN_IDENTITY
        and row.qcov >= PROHIBITED_MIN_QUERY_COVERAGE
        and row.tcov >= PROHIBITED_MIN_TARGET_COVERAGE
    )


def closest_residual_key(row: AlignmentRow) -> tuple[object, ...]:
    """Return the exact eight-key ordering frozen by W1-D09."""

    return (
        row.fident.copy_negate(),
        min(row.qcov, row.tcov).copy_negate(),
        row.qcov.copy_negate(),
        row.tcov.copy_negate(),
        row.evalue,
        row.bits.copy_negate(),
        -row.alnlen,
        row.target,
    )


def residual_category(row: AlignmentRow | None) -> str:
    """Classify one query by its closest accepted residual hit."""

    if row is None or row.fident < Decimal("0.30"):
        return CATEGORY_UNDER_30_OR_NONE
    if violates_prohibited_boundary(row):
        return CATEGORY_CLOSEST_PROHIBITED
    if row.fident < Decimal("0.40"):
        return CATEGORY_30_TO_40
    if row.fident < Decimal("0.50"):
        return CATEGORY_40_TO_50
    return CATEGORY_GE_50_LOW_COVERAGE


def verify_boundary_fixtures() -> None:
    """Prove just-below, exact, and just-above behavior for all boundaries."""

    query_metadata = {
        "fixture_query": SequenceMetadata("a" * 64, 100, "UniRef50_Q", "validation")
    }
    target_metadata = {
        "fixture_target": SequenceMetadata("b" * 64, 100, "UniRef50_T", "training")
    }
    baseline = {"fident": "0.50", "qcov": "0.80", "tcov": "0.80"}
    fixture_values = (
        ("fident", Decimal("0.499999"), False),
        ("fident", Decimal("0.50"), True),
        ("fident", Decimal("0.500001"), True),
        ("qcov", Decimal("0.799999"), False),
        ("qcov", Decimal("0.80"), True),
        ("qcov", Decimal("0.800001"), True),
        ("tcov", Decimal("0.799999"), False),
        ("tcov", Decimal("0.80"), True),
        ("tcov", Decimal("0.800001"), True),
    )
    for field, value, expected in fixture_values:
        values = {name: str(number) for name, number in baseline.items()}
        values[field] = str(value)
        cells = (
            "fixture_query",
            "fixture_target",
            values["fident"],
            values["qcov"],
            values["tcov"],
            "80",
            "100",
            "100",
            "1",
            "80",
            "1",
            "80",
            "0",
            "100",
        )
        row = parse_alignment_row(
            cells,
            query_metadata=query_metadata,
            target_metadata=target_metadata,
            context=f"boundary fixture {field}={value}",
        )
        if violates_prohibited_boundary(row) is not expected:
            raise SimilarityAuditError(
                f"boundary fixture failed for {field}={value}"
            )


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
                cells = _decode_strict_tsv_line(
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


def compact_converged_results(
    *,
    pass_name: str,
    comparison_path: Path,
    escalation_path: Path | None,
    convergence: ConvergenceEvidence,
    expected_query_ids: Iterable[str],
    output_directory: Path,
    resource_guard: Callable[[], None] | None = None,
) -> AcceptedPassEvidence:
    """Keep pair and closest-hit evidence, then allow full rows to be removed."""

    if pass_name not in {"enforcement", "residual"}:
        raise SimilarityAuditError(f"unknown search pass: {pass_name}")
    expected = tuple(sorted(expected_query_ids))
    escalated = frozenset(convergence.escalated_query_ids)
    output_directory.mkdir(parents=True, exist_ok=True)
    pair_path = output_directory / "returned_pairs.tsv"
    prohibited_path = output_directory / "prohibited_pairs.tsv"
    summary_path = output_directory / "residual_summaries.tsv"

    pair_writer = _AtomicEvidenceWriter(pair_path)
    prohibited_writer = _AtomicEvidenceWriter(prohibited_path)
    summary_writer = (
        _AtomicEvidenceWriter(summary_path) if pass_name == "residual" else None
    )
    accepted_rows = 0
    try:
        for query, rows in _iter_accepted_query_rows(
            expected,
            comparison_path=comparison_path,
            escalation_path=escalation_path,
            escalated_queries=escalated,
        ):
            closest = min(rows, key=closest_residual_key, default=None)
            for row in rows:
                accepted_rows += 1
                pair_writer.write(f"{row.query}\t{row.target}\n".encode("ascii"))
                if violates_prohibited_boundary(row):
                    prohibited_writer.write(
                        f"{row.query}\t{row.target}\n".encode("ascii")
                    )
                if resource_guard is not None and accepted_rows % 100_000 == 0:
                    resource_guard()
            if summary_writer is not None:
                summary_writer.write(_residual_summary_line(query, closest))
        if resource_guard is not None:
            resource_guard()
        pair_evidence = pair_writer.finish()
        prohibited_evidence = prohibited_writer.finish()
        summary_evidence = summary_writer.finish() if summary_writer else None
    except BaseException:
        pair_writer.abort()
        prohibited_writer.abort()
        if summary_writer:
            summary_writer.abort()
        raise

    if pair_evidence.row_count != accepted_rows:
        raise SimilarityAuditError("accepted alignment and pair counts differ")
    return AcceptedPassEvidence(
        pass_name=pass_name,
        accepted_rows=accepted_rows,
        accepted_at_comparison_cap=len(expected) - len(escalated),
        accepted_at_escalation_cap=len(escalated),
        returned_pairs=pair_evidence,
        prohibited_pairs=prohibited_evidence,
        residual_summaries=summary_evidence,
    )


def aggregate_partition_evidence(
    *,
    expected_query_ids: Iterable[str],
    query_metadata: Mapping[str, SequenceMetadata],
    target_metadata: Mapping[str, SequenceMetadata],
    enforcement_directory: Path,
    residual_directory: Path,
) -> dict[str, object]:
    """Combine both passes into aggregate-only evidence for one partition."""

    expected = tuple(sorted(expected_query_ids))
    if set(expected) != set(query_metadata):
        raise SimilarityAuditError("query metadata differs from the held-out universe")

    enforcement_pairs = enforcement_directory / "returned_pairs.tsv"
    residual_pairs = residual_directory / "returned_pairs.tsv"
    unique_returned_pairs = sum(
        1 for _ in _merge_unique_pairs(enforcement_pairs, residual_pairs)
    )

    violating_queries: set[str] = set()
    attribution = Counter(
        {
            "exact_sequence_duplicate": 0,
            "same_uniref50_group": 0,
            "cross_uniref50_group": 0,
        }
    )
    unique_prohibited_pairs = 0
    for query, target in _merge_unique_pairs(
        enforcement_directory / "prohibited_pairs.tsv",
        residual_directory / "prohibited_pairs.tsv",
    ):
        unique_prohibited_pairs += 1
        violating_queries.add(query)
        query_record = query_metadata[query]
        target_record = target_metadata[target]
        if query_record.sequence_sha256 == target_record.sequence_sha256:
            attribution["exact_sequence_duplicate"] += 1
        elif query_record.uniref50_group == target_record.uniref50_group:
            attribution["same_uniref50_group"] += 1
        else:
            attribution["cross_uniref50_group"] += 1

    if sum(attribution.values()) != unique_prohibited_pairs:
        raise SimilarityAuditError("prohibited-pair attribution does not reconcile")
    closest_categories, status_categories = _category_counts(
        residual_directory / "residual_summaries.tsv",
        expected_query_ids=expected,
        violating_queries=violating_queries,
    )
    if sum(closest_categories.values()) != len(expected):
        raise SimilarityAuditError("closest residual categories do not reconcile")
    if sum(status_categories.values()) != len(expected):
        raise SimilarityAuditError("held-out query statuses do not reconcile")

    numerator = len(violating_queries)
    query_hashes = {record.sequence_sha256 for record in query_metadata.values()}
    target_hashes = {record.sequence_sha256 for record in target_metadata.values()}
    query_groups = {record.uniref50_group for record in query_metadata.values()}
    target_groups = {record.uniref50_group for record in target_metadata.values()}
    return {
        "held_out_queries_with_prohibited_match": numerator,
        "held_out_query_count": len(expected),
        "prohibited_query_rate_percent": _percentage(numerator, len(expected)),
        "unique_prohibited_pairs": unique_prohibited_pairs,
        "prohibited_pair_attribution": dict(attribution),
        "enforcement_returned_pairs": _count_pair_rows(enforcement_pairs),
        "residual_returned_pairs": _count_pair_rows(residual_pairs),
        "unique_returned_pair_union": unique_returned_pairs,
        "exact_sequence_hash_crossings_to_training": len(
            query_hashes & target_hashes
        ),
        "uniref50_group_crossings_to_training": len(query_groups & target_groups),
        "closest_residual_categories": closest_categories,
        "held_out_query_status_categories": status_categories,
    }


def _parse_decimal(token: str, context: str) -> Decimal:
    if not _DECIMAL_PATTERN.fullmatch(token):
        raise SimilarityAuditError(f"{context}: malformed decimal")
    try:
        value = Decimal(token)
    except InvalidOperation as error:
        raise SimilarityAuditError(f"{context}: malformed decimal") from error
    if not value.is_finite():
        raise SimilarityAuditError(f"{context}: decimal must be finite")
    return value


def _parse_positive_integer(token: str, context: str) -> int:
    if not _POSITIVE_INTEGER_PATTERN.fullmatch(token):
        raise SimilarityAuditError(f"{context}: expected a positive integer")
    return int(token)


def _canonical_decimal(value: Decimal) -> str:
    if value == 0:
        return "0e0"
    sign, raw_digits, exponent = value.as_tuple()
    digits = list(raw_digits)
    while digits[-1] == 0:
        digits.pop()
        exponent += 1
    coefficient = "".join(str(digit) for digit in digits)
    return f"{'-' if sign else ''}{coefficient}e{exponent}"


def _require_visible_ascii(value: str, context: str) -> None:
    if not value or any(ord(character) < 33 or ord(character) > 126 for character in value):
        raise SimilarityAuditError(f"{context} must contain only visible ASCII")


def _decode_strict_tsv_line(raw_line: bytes, *, context: str) -> str:
    if not raw_line.endswith(b"\n"):
        raise SimilarityAuditError(f"{context}: final line must end with LF")
    if b"\r" in raw_line:
        raise SimilarityAuditError(f"{context}: CR line endings are prohibited")
    if raw_line == b"\n":
        raise SimilarityAuditError(f"{context}: blank rows are prohibited")
    try:
        return raw_line[:-1].decode("utf-8")
    except UnicodeDecodeError as error:
        raise SimilarityAuditError(f"{context}: invalid UTF-8") from error


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
            cells = _decode_strict_tsv_line(
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
            row = _parse_canonical_row(cells, context=f"{path.name} line {line_number}")
            if current_query is None:
                current_query = query
            if query != current_query:
                yield current_query, tuple(current_rows)
                current_query = query
                current_rows = []
            current_rows.append(row)
    if current_query is not None:
        yield current_query, tuple(current_rows)


def _parse_canonical_row(cells: Sequence[str], *, context: str) -> AlignmentRow:
    if len(cells) != len(ALIGNMENT_FIELDS):
        raise SimilarityAuditError(f"{context}: malformed canonical row")
    return AlignmentRow(
        query=cells[0],
        target=cells[1],
        fident=_parse_decimal(cells[2], f"{context} fident"),
        qcov=_parse_decimal(cells[3], f"{context} qcov"),
        tcov=_parse_decimal(cells[4], f"{context} tcov"),
        alnlen=_parse_positive_integer(cells[5], f"{context} alnlen"),
        qlen=_parse_positive_integer(cells[6], f"{context} qlen"),
        tlen=_parse_positive_integer(cells[7], f"{context} tlen"),
        qstart=_parse_positive_integer(cells[8], f"{context} qstart"),
        qend=_parse_positive_integer(cells[9], f"{context} qend"),
        tstart=_parse_positive_integer(cells[10], f"{context} tstart"),
        tend=_parse_positive_integer(cells[11], f"{context} tend"),
        evalue=_parse_decimal(cells[12], f"{context} evalue"),
        bits=_parse_decimal(cells[13], f"{context} bits"),
    )


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


def _iter_accepted_query_rows(
    expected_queries: Sequence[str],
    *,
    comparison_path: Path,
    escalation_path: Path | None,
    escalated_queries: frozenset[str],
) -> Iterator[tuple[str, tuple[AlignmentRow, ...]]]:
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


def _residual_summary_line(query: str, closest: AlignmentRow | None) -> bytes:
    if closest is None:
        return f"{query}\t{CATEGORY_UNDER_30_OR_NONE}\t\t\t\t\n".encode("ascii")
    return (
        f"{query}\t{residual_category(closest)}\t{closest.target}\t"
        f"{_canonical_decimal(closest.fident)}\t"
        f"{_canonical_decimal(closest.qcov)}\t"
        f"{_canonical_decimal(closest.tcov)}\n"
    ).encode("ascii")


class _AtomicEvidenceWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.temporary_path = path.with_name(f".{path.name}.incomplete")
        self.temporary_path.unlink(missing_ok=True)
        self._output = self.temporary_path.open("wb")
        self._hasher = hashlib.sha256()
        self._bytes = 0
        self._rows = 0

    def write(self, content: bytes) -> None:
        if not content.endswith(b"\n"):
            raise SimilarityAuditError("compact evidence rows must end with LF")
        self._output.write(content)
        self._hasher.update(content)
        self._bytes += len(content)
        self._rows += 1

    def finish(self) -> FileEvidence:
        self._output.close()
        self.temporary_path.replace(self.path)
        return FileEvidence(self._rows, self._bytes, self._hasher.hexdigest())

    def abort(self) -> None:
        if not self._output.closed:
            self._output.close()
        self.temporary_path.unlink(missing_ok=True)


def _iter_pair_file(path: Path) -> Iterator[tuple[str, str]]:
    previous: tuple[str, str] | None = None
    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            cells = _decode_strict_tsv_line(
                raw_line,
                context=f"{path.name} line {line_number}",
            ).split("\t")
            if len(cells) != 2:
                raise SimilarityAuditError(f"{path.name} pair row is malformed")
            pair = (cells[0], cells[1])
            if previous is not None and pair <= previous:
                raise SimilarityAuditError(f"{path.name} pair rows are not unique and sorted")
            previous = pair
            yield pair


def _merge_unique_pairs(first_path: Path, second_path: Path) -> Iterator[tuple[str, str]]:
    previous: tuple[str, str] | None = None
    for pair in heapq.merge(_iter_pair_file(first_path), _iter_pair_file(second_path)):
        if pair != previous:
            yield pair
            previous = pair


def _count_pair_rows(path: Path) -> int:
    return sum(1 for _ in _iter_pair_file(path))


def _category_counts(
    path: Path,
    *,
    expected_query_ids: Sequence[str],
    violating_queries: set[str],
) -> tuple[dict[str, int], dict[str, int]]:
    closest_counts = Counter(
        {category: 0 for category in CLOSEST_RESIDUAL_CATEGORIES}
    )
    status_counts = Counter({category: 0 for category in RESIDUAL_CATEGORIES})
    expected = iter(expected_query_ids)
    next_expected = next(expected, None)
    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            cells = _decode_strict_tsv_line(
                raw_line,
                context=f"{path.name} line {line_number}",
            ).split("\t")
            if len(cells) != 6:
                raise SimilarityAuditError("residual summary row is malformed")
            query, category = cells[0], cells[1]
            if query != next_expected:
                raise SimilarityAuditError("residual summary query universe drifted")
            if category not in CLOSEST_RESIDUAL_CATEGORIES:
                raise SimilarityAuditError("residual summary category is unknown")
            closest_counts[category] += 1
            if query in violating_queries:
                status_counts[CATEGORY_PROHIBITED] += 1
            elif category == CATEGORY_CLOSEST_PROHIBITED:
                raise SimilarityAuditError(
                    "a closest prohibited row is absent from the violation union"
                )
            else:
                status_counts[category] += 1
            next_expected = next(expected, None)
    if next_expected is not None:
        raise SimilarityAuditError("residual summary is missing expected queries")
    return dict(closest_counts), dict(status_counts)


def _percentage(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        raise SimilarityAuditError("percentage denominator must be positive")
    return f"{(Decimal(numerator) * 100 / Decimal(denominator)):.6f}"
