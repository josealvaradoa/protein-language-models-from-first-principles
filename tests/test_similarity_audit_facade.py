from protein_lm.data import similarity_audit
from protein_lm.data.similarity_alignment import violates_prohibited_boundary
from protein_lm.data.similarity_audit_models import AlignmentRow
from protein_lm.data.similarity_results import canonicalize_mmseqs_tsv


def test_facade_preserves_core_public_imports() -> None:
    assert similarity_audit.AlignmentRow is AlignmentRow
    assert similarity_audit.canonicalize_mmseqs_tsv is canonicalize_mmseqs_tsv
    assert (
        similarity_audit.violates_prohibited_boundary
        is violates_prohibited_boundary
    )
    assert similarity_audit.SimilarityAuditError.__name__ == "SimilarityAuditError"
