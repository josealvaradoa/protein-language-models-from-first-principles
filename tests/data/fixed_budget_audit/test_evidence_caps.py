from pathlib import Path

import pytest

from similarity_evidence_test_support import (
    alignment_tsv_row,
    canonicalize_rows,
    metadata,
)
from protein_lm.data.fixed_budget_audit.errors import (
    AuditConfigurationError,
    AuditExecutionError,
    AuditValidationError,
)
from protein_lm.data.fixed_budget_audit.evidence import compare_caps, summarize_cap
from protein_lm.data.similarity_alignment import (
    CATEGORY_30_TO_40,
    CATEGORY_40_TO_50,
    CATEGORY_CLOSEST_PROHIBITED,
    CATEGORY_GE_50_LOW_COVERAGE,
    CATEGORY_UNDER_30_OR_NONE,
)


def test_summarize_cap_writes_small_decision_evidence(tmp_path: Path) -> None:
    queries = {
        "Q1": metadata("q1"),
        "Q2": metadata("q2"),
        "Q3": metadata("q3"),
    }
    targets = {
        "T1": metadata("t1", partition="training"),
        "T2": metadata("t2", partition="training"),
    }
    canonical = canonicalize_rows(
        tmp_path,
        "cap_10000",
        (
            alignment_tsv_row("Q1", "T1"),
            alignment_tsv_row(
                "Q2",
                "T2",
                fident="0.45",
                qcov="0.70",
                tcov="0.70",
            ),
        ),
        queries,
        targets,
    )

    evidence = summarize_cap(
        cap=10_000,
        canonical_path=canonical,
        expected_query_ids=queries,
        output_directory=tmp_path / "summary",
    )

    assert evidence.query_count == 3
    assert evidence.returned_rows == 2
    assert evidence.prohibited_pairs == 1
    assert evidence.prohibited_queries == 1
    assert evidence.closest_categories == {
        CATEGORY_CLOSEST_PROHIBITED: 1,
        CATEGORY_GE_50_LOW_COVERAGE: 0,
        CATEGORY_40_TO_50: 1,
        CATEGORY_30_TO_40: 0,
        CATEGORY_UNDER_30_OR_NONE: 1,
    }
    assert (tmp_path / "summary" / "prohibited_pairs.tsv").read_text() == "Q1\tT1\n"

    original = (tmp_path / "summary" / "query_summaries.tsv").read_bytes()
    with pytest.raises(AuditExecutionError, match="fresh output directory"):
        summarize_cap(
            cap=10_000,
            canonical_path=canonical,
            expected_query_ids=queries,
            output_directory=tmp_path / "summary",
        )
    assert (tmp_path / "summary" / "query_summaries.tsv").read_bytes() == original


def test_compare_caps_accepts_an_escalated_query_subset(tmp_path: Path) -> None:
    queries = {
        "Q1": metadata("q1"),
        "Q2": metadata("q2"),
        "Q3": metadata("q3"),
    }
    targets = {
        "T1": metadata("t1", partition="training"),
        "T2": metadata("t2", partition="training"),
        "T3": metadata("t3", partition="training"),
    }
    common = canonicalize_rows(
        tmp_path,
        "common",
        (
            alignment_tsv_row("Q1", "T1"),
            alignment_tsv_row(
                "Q2",
                "T2",
                fident="0.45",
                qcov="0.70",
                tcov="0.70",
            ),
            alignment_tsv_row(
                "Q3",
                "T3",
                fident="0.35",
                qcov="0.70",
                tcov="0.70",
            ),
        ),
        queries,
        targets,
    )
    escalated_queries = {key: queries[key] for key in ("Q1", "Q2")}
    escalation = canonicalize_rows(
        tmp_path,
        "escalation",
        (
            alignment_tsv_row("Q1", "T1"),
            alignment_tsv_row("Q2", "T2", fident="0.60"),
        ),
        escalated_queries,
        targets,
    )
    common_directory = tmp_path / "common_summary"
    escalation_directory = tmp_path / "escalation_summary"
    summarize_cap(
        cap=10_000,
        canonical_path=common,
        expected_query_ids=queries,
        output_directory=common_directory,
    )
    summarize_cap(
        cap=100_000,
        canonical_path=escalation,
        expected_query_ids=escalated_queries,
        output_directory=escalation_directory,
    )

    comparison = compare_caps(
        baseline_cap=10_000,
        comparison_cap=100_000,
        baseline_canonical_path=common,
        comparison_canonical_path=escalation,
        baseline_summary_path=common_directory / "query_summaries.tsv",
        comparison_summary_path=escalation_directory / "query_summaries.tsv",
        expected_query_ids=escalated_queries,
        baseline_contains_other_queries=True,
    )

    assert comparison.compared_queries == 2
    assert comparison.complete_row_change_query_ids == ("Q2",)
    assert comparison.complete_row_changes == 1
    assert comparison.newly_prohibited_queries == 1
    assert comparison.no_longer_prohibited_queries == 0
    assert comparison.closest_category_changes == 1

    with pytest.raises(AuditConfigurationError, match="frozen A-004 stages"):
        compare_caps(
            baseline_cap=1_000,
            comparison_cap=100_000,
            baseline_canonical_path=common,
            comparison_canonical_path=escalation,
            baseline_summary_path=common_directory / "query_summaries.tsv",
            comparison_summary_path=escalation_directory / "query_summaries.tsv",
            expected_query_ids=escalated_queries,
        )

    escalation_summary = escalation_directory / "query_summaries.tsv"
    corrupted = escalation_summary.read_text().replace("Q2\t1\t", "Q2\t0\t")
    escalation_summary.write_text(corrupted)
    with pytest.raises(AuditValidationError, match="summary values"):
        compare_caps(
            baseline_cap=10_000,
            comparison_cap=100_000,
            baseline_canonical_path=common,
            comparison_canonical_path=escalation,
            baseline_summary_path=common_directory / "query_summaries.tsv",
            comparison_summary_path=escalation_summary,
            expected_query_ids=escalated_queries,
            baseline_contains_other_queries=True,
        )
