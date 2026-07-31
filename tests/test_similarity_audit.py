import hashlib
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from protein_lm.data.similarity_audit import (
    CATEGORY_30_TO_40,
    CATEGORY_40_TO_50,
    CATEGORY_CLOSEST_PROHIBITED,
    CATEGORY_GE_50_LOW_COVERAGE,
    CATEGORY_PROHIBITED,
    CATEGORY_UNDER_30_OR_NONE,
    AlignmentRow,
    SequenceMetadata,
    aggregate_partition_evidence,
    canonicalize_mmseqs_tsv,
    closest_residual_key,
    compact_converged_results,
    compare_canonical_results,
    convergence_evidence,
    residual_category,
    verify_boundary_fixtures,
    violates_prohibited_boundary,
)
from protein_lm.data.similarity_audit_policy import (
    APPROVED_SIMILARITY_AUDIT_CONFIG_SHA256,
    SimilarityAuditError,
    load_similarity_audit_policy,
)
from protein_lm.data.similarity_inputs import (
    iter_one_line_fasta,
    load_strategy_manifest,
    materialize_strategy_fastas,
)
from protein_lm.data.task7_report import render_task7_report

PROJECT_ROOT = Path(__file__).parents[1]
POLICY_PATH = (
    PROJECT_ROOT / "experiments" / "week_01" / "diagnostic_similarity_audit.toml"
)


def _metadata(
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


def _row(
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


def _write_raw(path: Path, *rows: str, final_lf: bool = True) -> None:
    content = "\n".join(rows)
    if rows and final_lf:
        content += "\n"
    path.write_bytes(content.encode("utf-8"))


def _canonicalize(
    tmp_path: Path,
    name: str,
    rows: tuple[str, ...],
    queries: dict[str, SequenceMetadata],
    targets: dict[str, SequenceMetadata],
) -> Path:
    raw = tmp_path / f"{name}.raw.tsv"
    canonical = tmp_path / f"{name}.canonical.tsv"
    _write_raw(raw, *rows)
    canonicalize_mmseqs_tsv(
        raw,
        canonical,
        query_metadata=queries,
        target_metadata=targets,
        chunk_rows=2,
    )
    return canonical


def test_policy_is_byte_pinned_and_rejects_drift(tmp_path: Path) -> None:
    policy = load_similarity_audit_policy(POLICY_PATH)
    assert policy.adjustment_id == "A-003"
    assert policy.repair_authorized is False
    assert policy.model_use == "prohibited"
    assert hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest() == (
        APPROVED_SIMILARITY_AUDIT_CONFIG_SHA256
    )

    drifted = tmp_path / "policy.toml"
    drifted.write_bytes(POLICY_PATH.read_bytes() + b"\n")
    with pytest.raises(SimilarityAuditError, match="approved checksum"):
        load_similarity_audit_policy(drifted)


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


def test_canonicalization_ignores_row_order_and_decimal_spelling(
    tmp_path: Path,
) -> None:
    queries = {"Q1": _metadata("q1"), "Q2": _metadata("q2")}
    targets = {
        "T1": _metadata("t1", partition="training"),
        "T2": _metadata("t2", partition="training"),
    }
    first = _canonicalize(
        tmp_path,
        "first",
        (
            _row("Q2", "T2", fident="0.500", evalue="1.0e-20"),
            _row("Q1", "T1", fident="5e-1", evalue="10e-21"),
        ),
        queries,
        targets,
    )
    second = _canonicalize(
        tmp_path,
        "second",
        (
            _row("Q1", "T1", fident="0.5", evalue="1e-20"),
            _row("Q2", "T2", fident="5e-1", evalue="0.1e-19"),
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
    queries = {"Q1": _metadata("q1")}
    targets = {"T1": _metadata("t1", partition="training")}
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
        (_row(query="UNKNOWN"), "unexpected query"),
        (_row(target="UNKNOWN"), "unexpected target"),
        (_row(qlen=99), "qlen differs"),
        (_row(tlen=99), "tlen differs"),
        (_row(qstart=81, qend=80), "query coordinates"),
        (_row(evalue="NaN"), "malformed decimal"),
        (_row(fident=" 0.5"), "malformed decimal"),
        (_row(fident="1.01"), "between 0 and 1"),
    ],
)
def test_malformed_alignment_rows_fail(
    tmp_path: Path,
    row: str,
    message: str,
) -> None:
    queries = {"Q1": _metadata("q1")}
    targets = {"T1": _metadata("t1", partition="training")}
    raw = tmp_path / "raw.tsv"
    _write_raw(raw, row)
    with pytest.raises(SimilarityAuditError, match=message):
        canonicalize_mmseqs_tsv(
            raw,
            tmp_path / "canonical.tsv",
            query_metadata=queries,
            target_metadata=targets,
            chunk_rows=10,
        )


def test_duplicate_query_target_pair_fails(tmp_path: Path) -> None:
    queries = {"Q1": _metadata("q1")}
    targets = {"T1": _metadata("t1", partition="training")}
    raw = tmp_path / "raw.tsv"
    _write_raw(raw, _row(), _row(bits="101"))
    with pytest.raises(SimilarityAuditError, match="duplicate query-target"):
        canonicalize_mmseqs_tsv(
            raw,
            tmp_path / "canonical.tsv",
            query_metadata=queries,
            target_metadata=targets,
            chunk_rows=1,
        )


def test_external_sort_uses_bounded_multi_pass_merge(tmp_path: Path) -> None:
    queries = {"Q1": _metadata("q1")}
    targets = {
        f"T{index:03d}": _metadata(f"t{index}", partition="training")
        for index in range(70)
    }
    raw = tmp_path / "many_chunks.raw.tsv"
    _write_raw(
        raw,
        *(
            _row("Q1", target)
            for target in reversed(tuple(targets))
        ),
    )
    evidence = canonicalize_mmseqs_tsv(
        raw,
        tmp_path / "many_chunks.canonical.tsv",
        query_metadata=queries,
        target_metadata=targets,
        chunk_rows=1,
    )
    assert evidence.canonical.row_count == 70


def test_raw_output_can_be_discarded_after_its_hash_is_captured(tmp_path: Path) -> None:
    queries = {"Q1": _metadata("q1")}
    targets = {"T1": _metadata("t1", partition="training")}
    raw = tmp_path / "discard.raw.tsv"
    _write_raw(raw, _row())
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
    queries = {"Q1": _metadata("q1")}
    targets = {"T1": _metadata("t1", partition="training")}
    first = _canonicalize(tmp_path, "first", (_row(bits="100"),), queries, targets)
    second = _canonicalize(tmp_path, "second", (_row(bits="101"),), queries, targets)
    assert compare_canonical_results(
        first,
        second,
        expected_query_ids=("Q1",),
    ) == ("Q1",)


def test_decimal_digits_beyond_context_precision_remain_exact(tmp_path: Path) -> None:
    queries = {"Q1": _metadata("q1")}
    targets = {"T1": _metadata("t1", partition="training")}
    lower = "0.500000000000000000000000000000001"
    higher = "0.500000000000000000000000000000002"
    first = _canonicalize(
        tmp_path,
        "lower_precision",
        (_row(fident=lower),),
        queries,
        targets,
    )
    second = _canonicalize(
        tmp_path,
        "higher_precision",
        (_row(fident=higher),),
        queries,
        targets,
    )
    assert compare_canonical_results(
        first,
        second,
        expected_query_ids=("Q1",),
    ) == ("Q1",)

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


def test_zero_hit_queries_participate_in_equality(tmp_path: Path) -> None:
    queries = {"Q1": _metadata("q1"), "Q2": _metadata("q2")}
    targets = {"T1": _metadata("t1", partition="training")}
    first = _canonicalize(tmp_path, "first", (_row(),), queries, targets)
    second = _canonicalize(tmp_path, "second", (_row(),), queries, targets)
    assert compare_canonical_results(
        first,
        second,
        expected_query_ids=queries,
    ) == ()


def test_staged_cap_escalation_converges_or_stops(tmp_path: Path) -> None:
    queries = {"Q1": _metadata("q1"), "Q2": _metadata("q2")}
    targets = {"T1": _metadata("t1", partition="training")}
    initial = _canonicalize(tmp_path, "initial", (), queries, targets)
    comparison = _canonicalize(tmp_path, "comparison", (_row(),), queries, targets)
    escalation = _canonicalize(
        tmp_path,
        "escalation",
        (_row(),),
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

    changed_final = _canonicalize(
        tmp_path,
        "changed_final",
        (_row(bits="101"),),
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


def test_aggregation_counts_queries_pairs_attribution_and_categories(
    tmp_path: Path,
) -> None:
    shared_digest = hashlib.sha256(b"same").hexdigest()
    queries = {
        "Qexact": SequenceMetadata(shared_digest, 100, "UniRef50_QE", "validation"),
        "Qgroup": _metadata("qgroup", group="UniRef50_SHARED"),
        "Qcross": _metadata("qcross", group="UniRef50_CROSS_Q"),
        "Qlow": _metadata("qlow", group="UniRef50_LOW"),
        "Q45": _metadata("q45", group="UniRef50_45"),
        "Q35": _metadata("q35", group="UniRef50_35"),
        "Qnone": _metadata("qnone", group="UniRef50_NONE"),
    }
    targets = {
        "Texact": SequenceMetadata(shared_digest, 100, "UniRef50_TE", "training"),
        "Tgroup": _metadata("tgroup", group="UniRef50_SHARED", partition="training"),
        "Tcross": _metadata("tcross", group="UniRef50_CROSS_T", partition="training"),
        "Tcross2": _metadata("tcross2", group="UniRef50_CROSS_T2", partition="training"),
        "Tlow": _metadata("tlow", group="UniRef50_LOW_T", partition="training"),
        "TlowViolation": _metadata(
            "tlow_violation",
            group="UniRef50_LOW_VIOLATION",
            partition="training",
        ),
        "T45": _metadata("t45", group="UniRef50_45_T", partition="training"),
        "T35": _metadata("t35", group="UniRef50_35_T", partition="training"),
    }
    prohibited_rows = (
        _row("Qexact", "Texact"),
        _row("Qgroup", "Tgroup", fident="0.70"),
        _row("Qcross", "Tcross", fident="0.60"),
    )
    residual_rows = prohibited_rows + (
        _row("Qcross", "Tcross2", fident="0.55"),
        _row("Qlow", "Tlow", fident="0.90", qcov="0.79"),
        _row("Qlow", "TlowViolation", fident="0.60"),
        _row("Q45", "T45", fident="0.45", qcov="0.70", tcov="0.70"),
        _row("Q35", "T35", fident="0.35", qcov="0.70", tcov="0.70"),
    )
    enforcement_canonical = _canonicalize(
        tmp_path,
        "enforcement",
        prohibited_rows,
        queries,
        targets,
    )
    residual_canonical = _canonicalize(
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


def _catalog_row(accession: str, sequence: str, group: str) -> str:
    return "\t".join(
        (
            accession,
            sequence,
            hashlib.sha256(sequence.encode()).hexdigest(),
            str(len(sequence)),
            "false",
            "false",
            "false",
            "false",
            "false",
            "true",
            "",
            group,
            "false",
        )
    )


def _write_manifest_pair(
    tmp_path: Path,
    name: str,
    strategy: str,
    stage: str,
    rows: list[tuple[str, str, str, str]],
) -> tuple[Path, Path]:
    public = tmp_path / f"{name}_public.tsv"
    local = tmp_path / f"{name}_local.tsv"
    public_lines = [
        "primary_accession\tpartition\tsequence_sha256\tbiological_length\tuniref50_group"
    ]
    local_lines = [
        "strategy\tstage\trepair_cycle\tstable_assignment_unit\tpartition_or_exclusion_status\taccession"
    ]
    for accession, sequence, group, partition in rows:
        digest = hashlib.sha256(sequence.encode()).hexdigest()
        public_lines.append(
            f"{accession}\t{partition}\t{digest}\t{len(sequence)}\t{group}"
        )
        unit = accession if strategy == "random" else group
        local_lines.append(
            f"{strategy}\t{stage}\t0\t{unit}\t{partition}\t{accession}"
        )
    public.write_text("\n".join(public_lines) + "\n", encoding="utf-8")
    local.write_text("\n".join(local_lines) + "\n", encoding="utf-8")
    return public, local


def test_manifest_join_materializes_exact_six_fastas(tmp_path: Path) -> None:
    specs = [
        ("A1", "A" * 32, "UniRef50_A", "training"),
        ("A2", "C" * 33, "UniRef50_B", "validation"),
        ("A3", "D" * 34, "UniRef50_C", "test"),
    ]
    random_public, random_local = _write_manifest_pair(
        tmp_path,
        "random",
        "random",
        "diagnostic",
        specs,
    )
    candidate_public, candidate_local = _write_manifest_pair(
        tmp_path,
        "candidate",
        "group_aware",
        "pre_repair",
        specs,
    )
    random_manifest = load_strategy_manifest(
        public_path=random_public,
        local_path=random_local,
        strategy="random",
        stage="diagnostic",
        expected_public_sha256=hashlib.sha256(random_public.read_bytes()).hexdigest(),
        expected_local_sha256=hashlib.sha256(random_local.read_bytes()).hexdigest(),
    )
    candidate_manifest = load_strategy_manifest(
        public_path=candidate_public,
        local_path=candidate_local,
        strategy="group_aware",
        stage="pre_repair",
        expected_public_sha256=hashlib.sha256(candidate_public.read_bytes()).hexdigest(),
        expected_local_sha256=hashlib.sha256(candidate_local.read_bytes()).hexdigest(),
    )
    from protein_lm.data.eligibility import CATALOG_COLUMNS

    catalog = tmp_path / "catalog.tsv"
    catalog.write_text(
        "\t".join(CATALOG_COLUMNS)
        + "\n"
        + "\n".join(_catalog_row(accession, sequence, group) for accession, sequence, group, _ in specs)
        + "\n",
        encoding="utf-8",
    )
    policy = replace(
        load_similarity_audit_policy(POLICY_PATH),
        task4_catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
        task4_catalog_byte_size=catalog.stat().st_size,
        task4_catalog_row_count=len(specs),
        expected_eligible_records=len(specs),
        expected_eligible_residues=sum(len(sequence) for _, sequence, _, _ in specs),
    )
    output = tmp_path / "fastas"
    materialized = materialize_strategy_fastas(
        catalog_path=catalog,
        manifests={"random": random_manifest, "group_aware": candidate_manifest},
        output_directory=output,
        policy=policy,
    )
    assert set(materialized.fastas) == {"random", "group_aware"}
    assert len(list(output.glob("*.fasta"))) == 6
    assert list(iter_one_line_fasta(output / "random_validation.fasta")) == [
        ("A2", "C" * 33)
    ]


def test_manifest_reports_unique_hash_and_group_crossings(tmp_path: Path) -> None:
    specs = [
        ("A1", "A" * 32, "UniRef50_SHARED", "training"),
        ("A2", "A" * 32, "UniRef50_OTHER", "validation"),
        ("A3", "C" * 33, "UniRef50_SHARED", "test"),
    ]
    public, local = _write_manifest_pair(
        tmp_path,
        "crossings",
        "random",
        "diagnostic",
        specs,
    )
    manifest = load_strategy_manifest(
        public_path=public,
        local_path=local,
        strategy="random",
        stage="diagnostic",
        expected_public_sha256=hashlib.sha256(public.read_bytes()).hexdigest(),
        expected_local_sha256=hashlib.sha256(local.read_bytes()).hexdigest(),
    )
    assert manifest.structural_audit.exact_sequence_hash_crossings == 1
    assert manifest.structural_audit.uniref50_group_crossings == 1
    assert manifest.structural_audit.largest_uniref50_group_records == 2


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
