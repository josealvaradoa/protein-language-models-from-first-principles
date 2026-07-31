"""Immutable records shared by the Task 7 similarity-audit modules."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

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
            canonical_decimal(self.fident),
            canonical_decimal(self.qcov),
            canonical_decimal(self.tcov),
            str(self.alnlen),
            str(self.qlen),
            str(self.tlen),
            str(self.qstart),
            str(self.qend),
            str(self.tstart),
            str(self.tend),
            canonical_decimal(self.evalue),
            canonical_decimal(self.bits),
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


def canonical_decimal(value: Decimal) -> str:
    """Render a Decimal exactly without applying the active context precision."""

    if value == 0:
        return "0e0"
    sign, raw_digits, exponent = value.as_tuple()
    digits = list(raw_digits)
    while digits[-1] == 0:
        digits.pop()
        exponent += 1
    coefficient = "".join(str(digit) for digit in digits)
    return f"{'-' if sign else ''}{coefficient}e{exponent}"
