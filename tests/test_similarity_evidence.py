import hashlib
from pathlib import Path

from protein_lm.data.similarity_alignment import (
    CATEGORY_30_TO_40,
    CATEGORY_40_TO_50,
    CATEGORY_CLOSEST_PROHIBITED,
    CATEGORY_GE_50_LOW_COVERAGE,
    CATEGORY_PROHIBITED,
    CATEGORY_UNDER_30_OR_NONE,
)
from protein_lm.data.similarity_audit_models import SequenceMetadata
from protein_lm.data.similarity_evidence import (
    aggregate_partition_evidence,
    compact_converged_results,
)
from protein_lm.data.similarity_results import convergence_evidence
from task7_test_support import alignment_tsv_row, canonicalize_rows, metadata


def test_aggregation_counts_queries_pairs_attribution_and_categories(
    tmp_path: Path,
) -> None:
    shared_digest = hashlib.sha256(b"same").hexdigest()
    queries = {
        "Qexact": SequenceMetadata(shared_digest, 100, "UniRef50_QE", "validation"),
        "Qgroup": metadata("qgroup", group="UniRef50_SHARED"),
        "Qcross": metadata("qcross", group="UniRef50_CROSS_Q"),
        "Qlow": metadata("qlow", group="UniRef50_LOW"),
        "Q45": metadata("q45", group="UniRef50_45"),
        "Q35": metadata("q35", group="UniRef50_35"),
        "Qnone": metadata("qnone", group="UniRef50_NONE"),
    }
    targets = {
        "Texact": SequenceMetadata(shared_digest, 100, "UniRef50_TE", "training"),
        "Tgroup": metadata("tgroup", group="UniRef50_SHARED", partition="training"),
        "Tcross": metadata(
            "tcross",
            group="UniRef50_CROSS_T",
            partition="training",
        ),
        "Tcross2": metadata(
            "tcross2",
            group="UniRef50_CROSS_T2",
            partition="training",
        ),
        "Tlow": metadata("tlow", group="UniRef50_LOW_T", partition="training"),
        "TlowViolation": metadata(
            "tlow_violation",
            group="UniRef50_LOW_VIOLATION",
            partition="training",
        ),
        "T45": metadata("t45", group="UniRef50_45_T", partition="training"),
        "T35": metadata("t35", group="UniRef50_35_T", partition="training"),
    }
    prohibited_rows = (
        alignment_tsv_row("Qexact", "Texact"),
        alignment_tsv_row("Qgroup", "Tgroup", fident="0.70"),
        alignment_tsv_row("Qcross", "Tcross", fident="0.60"),
    )
    residual_rows = prohibited_rows + (
        alignment_tsv_row("Qcross", "Tcross2", fident="0.55"),
        alignment_tsv_row("Qlow", "Tlow", fident="0.90", qcov="0.79"),
        alignment_tsv_row("Qlow", "TlowViolation", fident="0.60"),
        alignment_tsv_row(
            "Q45",
            "T45",
            fident="0.45",
            qcov="0.70",
            tcov="0.70",
        ),
        alignment_tsv_row(
            "Q35",
            "T35",
            fident="0.35",
            qcov="0.70",
            tcov="0.70",
        ),
    )
    enforcement_canonical = canonicalize_rows(
        tmp_path,
        "enforcement",
        prohibited_rows,
        queries,
        targets,
    )
    residual_canonical = canonicalize_rows(
        tmp_path,
        "residual",
        residual_rows,
        queries,
        targets,
    )
    no_escalation = convergence_evidence(
        expected_query_ids=queries,
        initial_path=enforcement_canonical,
        comparison_path=enforcement_canonical,
        escalation_path=None,
    )
    enforcement_directory = tmp_path / "enforcement_compact"
    residual_directory = tmp_path / "residual_compact"
    compact_converged_results(
        pass_name="enforcement",
        comparison_path=enforcement_canonical,
        escalation_path=None,
        convergence=no_escalation,
        expected_query_ids=queries,
        output_directory=enforcement_directory,
    )
    residual_no_escalation = convergence_evidence(
        expected_query_ids=queries,
        initial_path=residual_canonical,
        comparison_path=residual_canonical,
        escalation_path=None,
    )
    compact_converged_results(
        pass_name="residual",
        comparison_path=residual_canonical,
        escalation_path=None,
        convergence=residual_no_escalation,
        expected_query_ids=queries,
        output_directory=residual_directory,
    )
    aggregate = aggregate_partition_evidence(
        expected_query_ids=queries,
        query_metadata=queries,
        target_metadata=targets,
        enforcement_directory=enforcement_directory,
        residual_directory=residual_directory,
    )
    assert aggregate["held_out_queries_with_prohibited_match"] == 4
    assert aggregate["unique_prohibited_pairs"] == 5
    assert aggregate["prohibited_pair_attribution"] == {
        "exact_sequence_duplicate": 1,
        "same_uniref50_group": 1,
        "cross_uniref50_group": 3,
    }
    assert aggregate["exact_sequence_hash_crossings_to_training"] == 1
    assert aggregate["uniref50_group_crossings_to_training"] == 1
    assert aggregate["closest_residual_categories"] == {
        CATEGORY_CLOSEST_PROHIBITED: 3,
        CATEGORY_GE_50_LOW_COVERAGE: 1,
        CATEGORY_40_TO_50: 1,
        CATEGORY_30_TO_40: 1,
        CATEGORY_UNDER_30_OR_NONE: 1,
    }
    assert aggregate["held_out_query_status_categories"] == {
        CATEGORY_PROHIBITED: 4,
        CATEGORY_GE_50_LOW_COVERAGE: 0,
        CATEGORY_40_TO_50: 1,
        CATEGORY_30_TO_40: 1,
        CATEGORY_UNDER_30_OR_NONE: 1,
    }
