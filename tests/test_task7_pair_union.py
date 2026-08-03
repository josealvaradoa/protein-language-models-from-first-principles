from pathlib import Path

import pytest

from protein_lm.data.similarity_audit_policy import SimilarityAuditError
from protein_lm.data.task7_pair_union import (
    compare_pair_unions,
    union_prohibited_pairs,
)


def _write_pairs(path: Path, *pairs: tuple[str, str]) -> None:
    path.write_text("".join(f"{query}\t{target}\n" for query, target in pairs))


def test_union_preserves_every_detected_pair_without_double_counting(
    tmp_path: Path,
) -> None:
    enforcement_1k = tmp_path / "enforcement_1k.tsv"
    enforcement_10k = tmp_path / "enforcement_10k.tsv"
    residual_1k = tmp_path / "residual_1k.tsv"
    residual_10k = tmp_path / "residual_10k.tsv"
    _write_pairs(enforcement_1k, ("Q1", "T1"), ("Q2", "T2"))
    _write_pairs(enforcement_10k, ("Q1", "T1"), ("Q3", "T3"))
    _write_pairs(residual_1k, ("Q2", "T2"), ("Q4", "T4"))
    _write_pairs(residual_10k, ("Q1", "T9"))

    output = tmp_path / "common_10k_union"
    evidence = union_prohibited_pairs(
        source_paths={
            "enforcement_cap_1000": enforcement_1k,
            "enforcement_cap_10000": enforcement_10k,
            "residual_cap_1000": residual_1k,
            "residual_cap_10000": residual_10k,
        },
        output_directory=output,
    )

    assert evidence.unique_pairs == 5
    assert evidence.unique_queries == 4
    assert (output / "prohibited_pairs.tsv").read_text() == (
        "Q1\tT1\nQ1\tT9\nQ2\tT2\nQ3\tT3\nQ4\tT4\n"
    )


def test_union_keeps_a_lower_cap_pair_missing_from_escalation(tmp_path: Path) -> None:
    common = tmp_path / "common.tsv"
    escalation = tmp_path / "escalation.tsv"
    _write_pairs(common, ("Q1", "T1"))
    _write_pairs(escalation, ("Q1", "T2"), ("Q2", "T2"))

    common_output = tmp_path / "common_union"
    union_prohibited_pairs(
        source_paths={"common_cap_10000": common},
        output_directory=common_output,
    )
    staged_output = tmp_path / "staged_union"
    evidence = union_prohibited_pairs(
        source_paths={"common_cap_10000": common, "cap_100000": escalation},
        output_directory=staged_output,
    )

    assert evidence.unique_pairs == 3
    assert evidence.unique_queries == 2
    staged_path = staged_output / "prohibited_pairs.tsv"
    assert staged_path.read_text() == "Q1\tT1\nQ1\tT2\nQ2\tT2\n"

    comparison = compare_pair_unions(
        common_path=common_output / "prohibited_pairs.tsv",
        staged_path=staged_path,
    )
    assert comparison.additional_pairs == 2
    assert comparison.newly_prohibited_queries == 1

    original = staged_path.read_bytes()
    with pytest.raises(SimilarityAuditError, match="fresh output directory"):
        union_prohibited_pairs(
            source_paths={"common_cap_10000": common},
            output_directory=staged_output,
        )
    assert staged_path.read_bytes() == original

    with pytest.raises(SimilarityAuditError, match="missing a common-cap pair"):
        compare_pair_unions(
            common_path=staged_path,
            staged_path=common_output / "prohibited_pairs.tsv",
        )


def test_union_rejects_unsorted_or_duplicate_source_rows(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.tsv"
    _write_pairs(malformed, ("Q2", "T2"), ("Q1", "T1"))

    with pytest.raises(SimilarityAuditError, match="unique and sorted"):
        union_prohibited_pairs(
            source_paths={"malformed": malformed},
            output_directory=tmp_path / "union",
        )
