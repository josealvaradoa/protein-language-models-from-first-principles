import hashlib
from pathlib import Path

import pytest

from protein_lm.data.similarity_audit_policy import SimilarityAuditError
from protein_lm.data.similarity_results import (
    canonicalize_mmseqs_tsv,
    compare_canonical_results,
    convergence_evidence,
)
from task7_test_support import (
    alignment_tsv_row,
    canonicalize_rows,
    metadata,
    write_raw,
)


def test_canonicalization_ignores_row_order_and_decimal_spelling(
    tmp_path: Path,
) -> None:
    queries = {"Q1": metadata("q1"), "Q2": metadata("q2")}
    targets = {
        "T1": metadata("t1", partition="training"),
        "T2": metadata("t2", partition="training"),
    }
    first = canonicalize_rows(
        tmp_path,
        "first",
        (
            alignment_tsv_row("Q2", "T2", fident="0.500", evalue="1.0e-20"),
            alignment_tsv_row("Q1", "T1", fident="5e-1", evalue="10e-21"),
        ),
        queries,
        targets,
    )
    second = canonicalize_rows(
        tmp_path,
        "second",
        (
            alignment_tsv_row("Q1", "T1", fident="0.5", evalue="1e-20"),
            alignment_tsv_row("Q2", "T2", fident="5e-1", evalue="0.1e-19"),
        ),
        queries,
        targets,
    )
    assert compare_canonical_results(
        first,
        second,
        expected_query_ids=queries,
    ) == ()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"", None),
        (b"\n", "blank rows"),
        (b"Q1\tT1\r\n", "CR line endings"),
        (b"Q1\tT1", "final line"),
        (b"\xff\n", "invalid UTF-8"),
    ],
)
def test_strict_line_contract(
    tmp_path: Path,
    content: bytes,
    message: str | None,
) -> None:
    queries = {"Q1": metadata("q1")}
    targets = {"T1": metadata("t1", partition="training")}
    raw = tmp_path / "raw.tsv"
    raw.write_bytes(content)
    if message is None:
        evidence = canonicalize_mmseqs_tsv(
            raw,
            tmp_path / "canonical.tsv",
            query_metadata=queries,
            target_metadata=targets,
            chunk_rows=10,
        )
        assert evidence.raw.row_count == 0
    else:
        with pytest.raises(SimilarityAuditError, match=message):
            canonicalize_mmseqs_tsv(
                raw,
                tmp_path / "canonical.tsv",
                query_metadata=queries,
                target_metadata=targets,
                chunk_rows=10,
            )


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (alignment_tsv_row(query="UNKNOWN"), "unexpected query"),
        (alignment_tsv_row(target="UNKNOWN"), "unexpected target"),
        (alignment_tsv_row(qlen=99), "qlen differs"),
        (alignment_tsv_row(tlen=99), "tlen differs"),
        (alignment_tsv_row(qstart=81, qend=80), "query coordinates"),
        (alignment_tsv_row(evalue="NaN"), "malformed decimal"),
        (alignment_tsv_row(fident=" 0.5"), "malformed decimal"),
        (alignment_tsv_row(fident="1.01"), "between 0 and 1"),
    ],
)
def test_malformed_alignment_rows_fail(
    tmp_path: Path,
    row: str,
    message: str,
) -> None:
    queries = {"Q1": metadata("q1")}
    targets = {"T1": metadata("t1", partition="training")}
    raw = tmp_path / "raw.tsv"
    write_raw(raw, row)
    with pytest.raises(SimilarityAuditError, match=message):
        canonicalize_mmseqs_tsv(
            raw,
            tmp_path / "canonical.tsv",
            query_metadata=queries,
            target_metadata=targets,
            chunk_rows=10,
        )


def test_duplicate_query_target_pair_fails(tmp_path: Path) -> None:
    queries = {"Q1": metadata("q1")}
    targets = {"T1": metadata("t1", partition="training")}
    raw = tmp_path / "raw.tsv"
    write_raw(raw, alignment_tsv_row(), alignment_tsv_row(bits="101"))
    with pytest.raises(SimilarityAuditError, match="duplicate query-target"):
        canonicalize_mmseqs_tsv(
            raw,
            tmp_path / "canonical.tsv",
            query_metadata=queries,
            target_metadata=targets,
            chunk_rows=1,
        )


def test_external_sort_uses_bounded_multi_pass_merge(tmp_path: Path) -> None:
    queries = {"Q1": metadata("q1")}
    targets = {
        f"T{index:03d}": metadata(f"t{index}", partition="training")
        for index in range(70)
    }
    raw = tmp_path / "many_chunks.raw.tsv"
    write_raw(
        raw,
        *(alignment_tsv_row("Q1", target) for target in reversed(tuple(targets))),
    )
    evidence = canonicalize_mmseqs_tsv(
        raw,
        tmp_path / "many_chunks.canonical.tsv",
        query_metadata=queries,
        target_metadata=targets,
        chunk_rows=1,
    )
    assert evidence.canonical.row_count == 70


def test_raw_output_can_be_discarded_after_its_hash_is_captured(
    tmp_path: Path,
) -> None:
    queries = {"Q1": metadata("q1")}
    targets = {"T1": metadata("t1", partition="training")}
    raw = tmp_path / "discard.raw.tsv"
    write_raw(raw, alignment_tsv_row())
    expected_raw_sha256 = hashlib.sha256(raw.read_bytes()).hexdigest()
    evidence = canonicalize_mmseqs_tsv(
        raw,
        tmp_path / "discard.canonical.tsv",
        query_metadata=queries,
        target_metadata=targets,
        chunk_rows=10,
        delete_raw_after_parse=True,
    )
    assert not raw.exists()
    assert evidence.raw.sha256 == expected_raw_sha256


def test_same_row_count_with_changed_field_is_not_equal(tmp_path: Path) -> None:
    queries = {"Q1": metadata("q1")}
    targets = {"T1": metadata("t1", partition="training")}
    first = canonicalize_rows(
        tmp_path,
        "first",
        (alignment_tsv_row(bits="100"),),
        queries,
        targets,
    )
    second = canonicalize_rows(
        tmp_path,
        "second",
        (alignment_tsv_row(bits="101"),),
        queries,
        targets,
    )
    assert compare_canonical_results(
        first,
        second,
        expected_query_ids=("Q1",),
    ) == ("Q1",)


def test_decimal_digits_beyond_context_precision_remain_exact(tmp_path: Path) -> None:
    queries = {"Q1": metadata("q1")}
    targets = {"T1": metadata("t1", partition="training")}
    lower = "0.500000000000000000000000000000001"
    higher = "0.500000000000000000000000000000002"
    first = canonicalize_rows(
        tmp_path,
        "lower_precision",
        (alignment_tsv_row(fident=lower),),
        queries,
        targets,
    )
    second = canonicalize_rows(
        tmp_path,
        "higher_precision",
        (alignment_tsv_row(fident=higher),),
        queries,
        targets,
    )
    assert compare_canonical_results(
        first,
        second,
        expected_query_ids=("Q1",),
    ) == ("Q1",)


def test_zero_hit_queries_participate_in_equality(tmp_path: Path) -> None:
    queries = {"Q1": metadata("q1"), "Q2": metadata("q2")}
    targets = {"T1": metadata("t1", partition="training")}
    first = canonicalize_rows(
        tmp_path,
        "first",
        (alignment_tsv_row(),),
        queries,
        targets,
    )
    second = canonicalize_rows(
        tmp_path,
        "second",
        (alignment_tsv_row(),),
        queries,
        targets,
    )
    assert compare_canonical_results(
        first,
        second,
        expected_query_ids=queries,
    ) == ()


def test_staged_cap_escalation_converges_or_stops(tmp_path: Path) -> None:
    queries = {"Q1": metadata("q1"), "Q2": metadata("q2")}
    targets = {"T1": metadata("t1", partition="training")}
    initial = canonicalize_rows(tmp_path, "initial", (), queries, targets)
    comparison = canonicalize_rows(
        tmp_path,
        "comparison",
        (alignment_tsv_row(),),
        queries,
        targets,
    )
    escalation = canonicalize_rows(
        tmp_path,
        "escalation",
        (alignment_tsv_row(),),
        {"Q1": queries["Q1"]},
        targets,
    )
    evidence = convergence_evidence(
        expected_query_ids=queries,
        initial_path=initial,
        comparison_path=comparison,
        escalation_path=escalation,
    )
    assert evidence.escalated_query_ids == ("Q1",)
    assert evidence.final_differing_queries == 0

    changed_final = canonicalize_rows(
        tmp_path,
        "changed_final",
        (alignment_tsv_row(bits="101"),),
        {"Q1": queries["Q1"]},
        targets,
    )
    with pytest.raises(SimilarityAuditError, match="still differs"):
        convergence_evidence(
            expected_query_ids=queries,
            initial_path=initial,
            comparison_path=comparison,
            escalation_path=changed_final,
        )
