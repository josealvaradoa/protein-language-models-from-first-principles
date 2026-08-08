import json
from pathlib import Path

from protein_lm.data.task2_audit import (
    SourceEvidence,
    build_task2_audit,
    render_task2_audit,
)

FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures" / "week_01" / "corpus_audit"


def _source_evidence() -> dict[str, SourceEvidence]:
    return {
        "swiss_prot_records": SourceEvidence(
            release="fixture",
            filename="uniprot_sprot.dat",
            byte_size=1,
            sha256="a" * 64,
            upstream_checksum_algorithm="md5",
            upstream_checksum="a" * 32,
            license_spdx="CC-BY-4.0",
            retrieval_date="2026-07-28",
            retrieval_method="fixture",
        ),
        "uniref50_membership": SourceEvidence(
            release="fixture",
            filename="idmapping_selected.tab",
            byte_size=1,
            sha256="b" * 64,
            upstream_checksum_algorithm="md5",
            upstream_checksum="b" * 32,
            license_spdx="CC-BY-4.0",
            retrieval_date="2026-07-28",
            retrieval_method="fixture",
        ),
        "proteingym_metadata": SourceEvidence(
            release="fixture",
            filename="DMS_substitutions.csv",
            byte_size=1,
            sha256="c" * 64,
            upstream_checksum_algorithm="git_blob_sha1",
            upstream_checksum="c" * 40,
            license_spdx="MIT",
            retrieval_date="2026-07-28",
            retrieval_method="fixture",
        ),
    }


def _build_fixture_report(*, uniref50_path: Path | None = None):
    return build_task2_audit(
        swiss_prot_path=FIXTURE_DIRECTORY / "uniprot_sprot.dat",
        uniref50_path=(uniref50_path or FIXTURE_DIRECTORY / "idmapping_selected.tab"),
        proteingym_path=FIXTURE_DIRECTORY / "DMS_substitutions.csv",
        sources=_source_evidence(),
        code_revision="fixture-revision",
    )


def test_task2_fixture_report_reconciles_all_three_sources() -> None:
    report = _build_fixture_report()

    assert report.swiss_prot.record_count == 6
    assert report.uniref50.target_accession_count == 6
    assert report.uniref50.mapped_accession_count == 4
    assert report.proteingym_metadata.assay_count == 6
    assert report.proteingym_support.target_location_counts == {
        "in_swiss_prot": 3,
        "outside_swiss_prot": 1,
    }
    assert report.proteingym_support.assay_reference_status_counts == {
        "exact_swiss_prot": 4,
        "mismatched_swiss_prot": 1,
        "outside_swiss_prot": 1,
    }
    assert report.proteingym_support.target_uniref50_status_counts == {
        "mapped_in_swiss_prot": 1,
        "mapped_outside_swiss_prot": 1,
        "missing": 1,
        "blank_group": 1,
        "conflicting": 0,
        "inconsistent_accession": 0,
    }
    assert report.proteingym_support.assay_uniref50_status_counts == {
        "mapped_in_swiss_prot": 3,
        "mapped_outside_swiss_prot": 1,
        "missing": 1,
        "blank_group": 1,
        "conflicting": 0,
        "inconsistent_accession": 0,
    }
    assert report.proteingym_support.reservable_target_count == 2
    assert report.proteingym_support.reservable_assay_count == 4
    assert report.proteingym_support.unique_reservable_family_count == 2


def test_in_corpus_target_uses_primary_accession_before_entry_name(
    tmp_path: Path,
) -> None:
    source = (FIXTURE_DIRECTORY / "idmapping_selected.tab").read_text(encoding="utf-8")
    renamed_mapping = tmp_path / "renamed_entry.tab"
    renamed_mapping.write_text(
        source.replace("P00001\tALPHA_SYNTH", "P00001\tRENAMED_SYNTH"),
        encoding="utf-8",
    )

    report = _build_fixture_report(uniref50_path=renamed_mapping)

    assert (
        report.proteingym_support.target_uniref50_status_counts["mapped_in_swiss_prot"]
        == 1
    )
    assert (
        report.proteingym_support.assay_uniref50_status_counts["mapped_in_swiss_prot"]
        == 3
    )


def test_task2_rendering_is_repeatable_and_aggregate_only() -> None:
    first = render_task2_audit(_build_fixture_report())
    second = render_task2_audit(_build_fixture_report())

    assert first == second
    assert len(first.json_sha256) == 64
    assert json.loads(first.json_text)["scope"] == ("week_01_task_2_aggregate_only")

    for private_value in ("ALPHA_SYNTH", "PG_ALPHA_1", "ACDEF"):
        assert private_value not in first.json_text
        assert private_value not in first.markdown_text
