from dataclasses import replace
from decimal import Decimal

import pytest

from protein_lm.data.similarity_alignment import (
    CATEGORY_30_TO_40,
    CATEGORY_40_TO_50,
    CATEGORY_GE_50_LOW_COVERAGE,
    CATEGORY_UNDER_30_OR_NONE,
    closest_residual_key,
    residual_category,
    verify_boundary_fixtures,
    violates_prohibited_boundary,
)
from protein_lm.data.similarity_audit_models import AlignmentRow


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("fident", "0.499999", False),
        ("fident", "0.50", True),
        ("fident", "0.500001", True),
        ("qcov", "0.799999", False),
        ("qcov", "0.80", True),
        ("qcov", "0.800001", True),
        ("tcov", "0.799999", False),
        ("tcov", "0.80", True),
        ("tcov", "0.800001", True),
    ],
)
def test_inclusive_boundary_fixtures(field: str, value: str, expected: bool) -> None:
    values = {
        "fident": Decimal("0.50"),
        "qcov": Decimal("0.80"),
        "tcov": Decimal("0.80"),
    }
    values[field] = Decimal(value)
    row = AlignmentRow(
        query="Q",
        target="T",
        alnlen=80,
        qlen=100,
        tlen=100,
        qstart=1,
        qend=80,
        tstart=1,
        tend=80,
        evalue=Decimal("0"),
        bits=Decimal("100"),
        **values,
    )
    assert violates_prohibited_boundary(row) is expected
    verify_boundary_fixtures()


def test_boundary_requires_all_three_conditions() -> None:
    row = AlignmentRow(
        query="Q",
        target="T",
        fident=Decimal("0.9"),
        qcov=Decimal("0.9"),
        tcov=Decimal("0.799999"),
        alnlen=80,
        qlen=100,
        tlen=100,
        qstart=1,
        qend=80,
        tstart=1,
        tend=80,
        evalue=Decimal("0"),
        bits=Decimal("100"),
    )
    assert not violates_prohibited_boundary(row)


def test_closest_hit_uses_all_frozen_tie_breakers() -> None:
    base = AlignmentRow(
        query="Q",
        target="T9",
        fident=Decimal("0.40"),
        qcov=Decimal("0.70"),
        tcov=Decimal("0.60"),
        alnlen=60,
        qlen=100,
        tlen=100,
        qstart=1,
        qend=60,
        tstart=1,
        tend=60,
        evalue=Decimal("1e-10"),
        bits=Decimal("50"),
    )
    pairs = [
        (base, replace(base, target="T1", fident=Decimal("0.41"))),
        (base, replace(base, target="T1", tcov=Decimal("0.61"))),
        (base, replace(base, target="T1", qcov=Decimal("0.71"))),
        (
            replace(base, qcov=Decimal("0.60"), tcov=Decimal("0.70")),
            replace(
                base,
                target="T1",
                qcov=Decimal("0.60"),
                tcov=Decimal("0.71"),
            ),
        ),
        (base, replace(base, target="T1", evalue=Decimal("1e-11"))),
        (base, replace(base, target="T1", bits=Decimal("51"))),
        (base, replace(base, target="T1", alnlen=61)),
        (base, replace(base, target="T1")),
    ]
    for loser, winner in pairs:
        assert min((loser, winner), key=closest_residual_key) == winner


def test_closest_hit_preserves_decimal_digits_beyond_context_precision() -> None:
    lower = "0.500000000000000000000000000000001"
    higher = "0.500000000000000000000000000000002"
    lower_row = AlignmentRow(
        query="Q",
        target="T1",
        fident=Decimal(lower),
        qcov=Decimal("0.8"),
        tcov=Decimal("0.8"),
        alnlen=80,
        qlen=100,
        tlen=100,
        qstart=1,
        qend=80,
        tstart=1,
        tend=80,
        evalue=Decimal("1"),
        bits=Decimal("10"),
    )
    higher_row = replace(lower_row, target="T9", fident=Decimal(higher))
    assert min((lower_row, higher_row), key=closest_residual_key) == higher_row


@pytest.mark.parametrize(
    ("identity", "expected"),
    [
        (None, CATEGORY_UNDER_30_OR_NONE),
        ("0.299999", CATEGORY_UNDER_30_OR_NONE),
        ("0.30", CATEGORY_30_TO_40),
        ("0.40", CATEGORY_40_TO_50),
        ("0.50", CATEGORY_GE_50_LOW_COVERAGE),
    ],
)
def test_residual_category_boundaries(identity: str | None, expected: str) -> None:
    row = None
    if identity is not None:
        row = AlignmentRow(
            query="Q",
            target="T",
            fident=Decimal(identity),
            qcov=Decimal("0.5"),
            tcov=Decimal("0.5"),
            alnlen=50,
            qlen=100,
            tlen=100,
            qstart=1,
            qend=50,
            tstart=1,
            tend=50,
            evalue=Decimal("1"),
            bits=Decimal("10"),
        )
    assert residual_category(row) == expected
