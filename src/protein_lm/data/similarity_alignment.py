"""Frozen row contract and scientific rules for the Task 7 audit."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation

from protein_lm.data.similarity_audit_models import (
    ALIGNMENT_FIELDS,
    AlignmentRow,
    SequenceMetadata,
)
from protein_lm.data.similarity_audit_policy import SimilarityAuditError

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


def decode_strict_tsv_line(raw_line: bytes, *, context: str) -> str:
    """Decode one LF-terminated, non-empty UTF-8 TSV row."""

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


def parse_canonical_alignment_row(
    cells: Sequence[str],
    *,
    context: str,
) -> AlignmentRow:
    """Parse one already-validated canonical alignment row."""

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


def _require_visible_ascii(value: str, context: str) -> None:
    if not value or any(ord(character) < 33 or ord(character) > 126 for character in value):
        raise SimilarityAuditError(f"{context} must contain only visible ASCII")
