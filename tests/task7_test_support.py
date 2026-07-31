"""Small data builders shared by the synthetic Task 7 tests."""

import hashlib
from pathlib import Path

from protein_lm.data.similarity_audit_models import SequenceMetadata
from protein_lm.data.similarity_results import canonicalize_mmseqs_tsv


def metadata(
    digest_seed: str,
    *,
    length: int = 100,
    group: str = "UniRef50_A",
    partition: str = "validation",
) -> SequenceMetadata:
    return SequenceMetadata(
        sequence_sha256=hashlib.sha256(digest_seed.encode()).hexdigest(),
        biological_length=length,
        uniref50_group=group,
        partition=partition,
    )


def alignment_tsv_row(
    query: str = "Q1",
    target: str = "T1",
    *,
    fident: str = "0.50",
    qcov: str = "0.80",
    tcov: str = "0.80",
    alnlen: int = 80,
    qlen: int = 100,
    tlen: int = 100,
    qstart: int = 1,
    qend: int = 80,
    tstart: int = 1,
    tend: int = 80,
    evalue: str = "1e-20",
    bits: str = "100",
) -> str:
    return "\t".join(
        (
            query,
            target,
            fident,
            qcov,
            tcov,
            str(alnlen),
            str(qlen),
            str(tlen),
            str(qstart),
            str(qend),
            str(tstart),
            str(tend),
            evalue,
            bits,
        )
    )


def write_raw(path: Path, *rows: str, final_lf: bool = True) -> None:
    content = "\n".join(rows)
    if rows and final_lf:
        content += "\n"
    path.write_bytes(content.encode("utf-8"))


def canonicalize_rows(
    tmp_path: Path,
    name: str,
    rows: tuple[str, ...],
    queries: dict[str, SequenceMetadata],
    targets: dict[str, SequenceMetadata],
) -> Path:
    raw = tmp_path / f"{name}.raw.tsv"
    canonical = tmp_path / f"{name}.canonical.tsv"
    write_raw(raw, *rows)
    canonicalize_mmseqs_tsv(
        raw,
        canonical,
        query_metadata=queries,
        target_metadata=targets,
        chunk_rows=2,
    )
    return canonical
