import pytest

from protein_lm.data.similarity_alignment import (
    CATEGORY_30_TO_40,
    CATEGORY_40_TO_50,
    CATEGORY_CLOSEST_PROHIBITED,
    CATEGORY_GE_50_LOW_COVERAGE,
    CATEGORY_PROHIBITED,
    CATEGORY_UNDER_30_OR_NONE,
)
from protein_lm.data.similarity_audit_policy import SimilarityAuditError
from protein_lm.data.task7_report import render_task7_report


def test_public_report_guard_and_reconciliation() -> None:
    closest_categories = {
        CATEGORY_CLOSEST_PROHIBITED: 1,
        CATEGORY_GE_50_LOW_COVERAGE: 0,
        CATEGORY_40_TO_50: 0,
        CATEGORY_30_TO_40: 0,
        CATEGORY_UNDER_30_OR_NONE: 1,
    }
    status_categories = {
        CATEGORY_PROHIBITED: 1,
        CATEGORY_GE_50_LOW_COVERAGE: 0,
        CATEGORY_40_TO_50: 0,
        CATEGORY_30_TO_40: 0,
        CATEGORY_UNDER_30_OR_NONE: 1,
    }
    similarity = {
        "held_out_queries_with_prohibited_match": 1,
        "held_out_query_count": 2,
        "prohibited_query_rate_percent": "50.000000",
        "unique_prohibited_pairs": 1,
        "prohibited_pair_attribution": {
            "exact_sequence_duplicate": 0,
            "same_uniref50_group": 0,
            "cross_uniref50_group": 1,
        },
        "enforcement_returned_pairs": 1,
        "residual_returned_pairs": 1,
        "unique_returned_pair_union": 1,
        "closest_residual_categories": closest_categories,
        "held_out_query_status_categories": status_categories,
    }
    balance = {
        "records": 2,
        "record_share_percent": "5.000000",
        "residues": 200,
        "residue_share_percent": "5.000000",
    }
    strategy = {
        "structural_membership": {
            "exact_sequence_hash_crossings": 1,
            "uniref50_group_crossings": 1,
            "retained_records": 100,
            "retained_residues": 10_000,
            "excluded_records": 0,
            "excluded_residues": 0,
            "largest_uniref50_group_records": 5,
            "largest_uniref50_group_residues": 500,
        },
        "partitions": {
            "training": {"balance": {}},
            "validation": {"balance": balance, "similarity": similarity},
            "test": {"balance": balance, "similarity": similarity},
        },
        "overall": {
            "held_out_queries_with_prohibited_match": 2,
            "held_out_query_count": 4,
            "prohibited_query_rate_percent": "50.000000",
            "unique_prohibited_pairs": 2,
            "prohibited_pair_attribution": {
                "exact_sequence_duplicate": 0,
                "same_uniref50_group": 0,
                "cross_uniref50_group": 2,
            },
            "enforcement_returned_pairs": 2,
            "residual_returned_pairs": 2,
            "unique_returned_pair_union": 2,
            "closest_residual_categories": {
                key: value * 2 for key, value in closest_categories.items()
            },
            "held_out_query_status_categories": {
                key: value * 2 for key, value in status_categories.items()
            },
        },
    }
    report = {
        "diagnostic_only": True,
        "diagnostic_audit_authorized": True,
        "candidate_status": "failed_balance",
        "repair_authorized": False,
        "repair_performed": False,
        "selected_split_authorized": False,
        "task8_membership_use_authorized": False,
        "model_use": "prohibited",
        "post_audit_review_required": True,
        "strategies": {"random": strategy, "group_aware": strategy},
    }
    rendered = render_task7_report(report)
    assert "failed-balance" in rendered.markdown_text

    drifted = dict(report, selected_split_authorized=True)
    with pytest.raises(SimilarityAuditError, match="authority guard"):
        render_task7_report(drifted)
