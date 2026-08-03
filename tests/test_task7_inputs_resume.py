from pathlib import Path

import pytest

from protein_lm.data.similarity_audit_policy import (
    SimilarityAuditError,
    load_similarity_audit_policy,
)
from protein_lm.data.task7_inputs import ensure_materialized_inputs

PROJECT_ROOT = Path(__file__).parents[1]
POLICY_PATH = (
    PROJECT_ROOT / "experiments" / "week_01" / "diagnostic_similarity_audit.toml"
)


def test_materialized_inputs_preserve_unmarked_nonempty_fasta_directory(
    tmp_path: Path,
) -> None:
    partial = tmp_path / "workspace/fastas/random_training.fasta"
    partial.parent.mkdir(parents=True)
    partial.write_text(">T1\nAAAA\n")

    with pytest.raises(SimilarityAuditError, match="lacks its completion marker"):
        ensure_materialized_inputs(
            workspace=tmp_path / "workspace",
            catalog_path=tmp_path / "missing-catalog.tsv",
            manifests={},
            policy=load_similarity_audit_policy(POLICY_PATH),
            fingerprint="synthetic-fingerprint",
        )

    assert partial.read_text() == ">T1\nAAAA\n"
