import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from protein_lm.data import random_split
from protein_lm.data.eligibility import CATALOG_COLUMNS
from protein_lm.data.random_split import (
    APPROVED_RANDOM_SPLIT_CONFIG_SHA256,
    APPROVED_RANDOM_SPLIT_POLICY,
    DiagnosticSplitUseError,
    RandomSplitError,
    assign_partition,
    assignment_payload,
    build_random_diagnostic,
    partition_from_integer,
    require_selected_training_manifest,
    sha256_sidecar,
    validate_task4_report,
)
from protein_lm.data.random_split_policy import load_random_split_policy
from protein_lm.data.task5_report import (
    CompletedPublicArtifact,
    DerivedArtifact,
    Task5Report,
    render_completion_index,
    render_task5_report,
)

PROJECT_ROOT = Path(__file__).parents[1]
POLICY_PATH = PROJECT_ROOT / "experiments" / "week_01" / "random_split.toml"
TASK4_REPORT_PATH = (
    PROJECT_ROOT / "reports" / "week_01" / "task_04_eligible_records.json"
)
FIXTURE_CONFIG_SHA256 = "d" * 64


def _catalog_row(
    accession: str,
    sequence: str,
    group: str,
    *,
    eligible: bool = True,
    reserved: bool = False,
) -> str:
    if eligible:
        flags = ("false",) * 5
        reason = ""
    else:
        flags = ("true", "false", "true", "false", "false")
        reason = "noncanonical_residue"
    values = (
        accession,
        sequence,
        hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        str(len(sequence)),
        *flags,
        "true" if eligible else "false",
        reason,
        group,
        "true" if reserved else "false",
    )
    return "\t".join(values)


def _fixture_rows() -> list[str]:
    return [
        _catalog_row(
            "P00001",
            "A" * 32,
            "UniRef50_SHARED",
            reserved=True,
        ),
        _catalog_row(
            "P00011",
            "A" * 32,
            "UniRef50_SHARED",
            reserved=True,
        ),
        _catalog_row(
            "P00007",
            "C" * 34,
            "UniRef50_OTHER",
        ),
        _catalog_row(
            "P99999",
            "B" * 10,
            "UniRef50_INELIGIBLE",
            eligible=False,
        ),
    ]


def _write_catalog(
    tmp_path: Path,
    rows: list[str],
    *,
    filename: str = "catalog.tsv",
):
    content = "\t".join(CATALOG_COLUMNS) + "\n" + "\n".join(rows) + "\n"
    path = tmp_path / filename
    path.write_bytes(content.encode("utf-8"))
    policy = replace(
        APPROVED_RANDOM_SPLIT_POLICY,
        task4_catalog_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        task4_catalog_byte_size=path.stat().st_size,
        task4_catalog_row_count=len(rows),
        expected_eligible_records=3,
        expected_eligible_residues=98,
        expected_eligible_groups=2,
    )
    return path, policy


def _allow_fixture_policy(
    monkeypatch: pytest.MonkeyPatch,
    policy,
) -> None:
    monkeypatch.setattr(
        random_split,
        "APPROVED_RANDOM_SPLIT_POLICY",
        policy,
    )
    monkeypatch.setattr(
        random_split,
        "APPROVED_RANDOM_SPLIT_CONFIG_SHA256",
        FIXTURE_CONFIG_SHA256,
    )


def _build_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    rows: list[str] | None = None,
    suffix: str = "first",
):
    catalog_path, policy = _write_catalog(
        tmp_path,
        rows or _fixture_rows(),
        filename=f"{suffix}_catalog.tsv",
    )
    _allow_fixture_policy(monkeypatch, policy)
    build = build_random_diagnostic(
        catalog_path=catalog_path,
        local_assignment_output_path=tmp_path / f"{suffix}_local.tsv",
        public_manifest_output_path=tmp_path / f"{suffix}_public.tsv",
        policy=policy,
        policy_sha256=FIXTURE_CONFIG_SHA256,
    )
    return build, policy, catalog_path


def test_policy_bytes_and_values_are_frozen(tmp_path: Path) -> None:
    policy = load_random_split_policy(POLICY_PATH)

    assert policy == APPROVED_RANDOM_SPLIT_POLICY
    assert hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest() == (
        APPROVED_RANDOM_SPLIT_CONFIG_SHA256
    )

    drifted_path = tmp_path / "drifted.toml"
    drifted_path.write_bytes(POLICY_PATH.read_bytes() + b"\n")
    with pytest.raises(RandomSplitError, match="approved checksum"):
        load_random_split_policy(drifted_path)


@pytest.mark.parametrize(
    ("accession", "expected_digest", "expected_partition"),
    [
        (
            "P00001",
            "1d91eccbcfddfbae5e8fd9a7d0271ad197cdbba5d653bfef778698c1cb063b7d",
            "training",
        ),
        (
            "P00011",
            "e7c2a5b0f1475057772839d0f47ca00a4f459f59e090d7cfe785023406d5a5c7",
            "validation",
        ),
        (
            "P00007",
            "f63c0d9675d86e6285d48b074a95f93ebf4fb3e461db45f0160950aff9cb44f8",
            "test",
        ),
    ],
)
def test_fixed_accession_hash_vectors(
    accession: str,
    expected_digest: str,
    expected_partition: str,
) -> None:
    payload = assignment_payload(accession)

    assert payload == (
        b"week1-random-v1\x00" + b"20260727\x00" + accession.encode("ascii")
    )
    assert hashlib.sha256(payload).hexdigest() == expected_digest
    assert assign_partition(accession) == expected_partition


def test_exact_integer_partition_boundaries() -> None:
    hash_space = 1 << 256
    last_training = (9 * hash_space - 1) // 10
    first_validation = last_training + 1
    last_validation = (19 * hash_space - 1) // 20
    first_test = last_validation + 1

    assert partition_from_integer(0) == "training"
    assert partition_from_integer(last_training) == "training"
    assert partition_from_integer(first_validation) == "validation"
    assert partition_from_integer(last_validation) == "validation"
    assert partition_from_integer(first_test) == "test"
    assert partition_from_integer(hash_space - 1) == "test"


def test_assignment_rejects_non_ascii_and_invalid_integers() -> None:
    with pytest.raises(RandomSplitError, match="visible ASCII"):
        assign_partition("PRÖTEIN")
    with pytest.raises(TypeError, match="integer"):
        partition_from_integer(True)
    with pytest.raises(ValueError, match="unsigned SHA-256"):
        partition_from_integer(1 << 256)


def test_fixture_build_is_repeatable_order_independent_and_label_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, _, _ = _build_fixture(
        tmp_path,
        monkeypatch,
        suffix="first",
    )
    second, _, _ = _build_fixture(
        tmp_path,
        monkeypatch,
        rows=list(reversed(_fixture_rows())),
        suffix="second",
    )

    assert first == second
    assert (tmp_path / "first_local.tsv").read_bytes() == (
        tmp_path / "second_local.tsv"
    ).read_bytes()
    assert (tmp_path / "first_public.tsv").read_bytes() == (
        tmp_path / "second_public.tsv"
    ).read_bytes()
    assert first.population.records == 3
    assert first.population.residues == 98
    assert first.population.unique_groups == 2
    assert first.partitions["training"].records == 1
    assert first.partitions["training"].residues == 32
    assert first.partitions["validation"].records == 1
    assert first.partitions["validation"].residues == 32
    assert first.partitions["test"].records == 1
    assert first.partitions["test"].residues == 34
    assert sum(partition.unique_groups for partition in first.partitions.values()) == 3

    public_text = (tmp_path / "first_public.tsv").read_text(encoding="utf-8")
    public_lines = public_text.splitlines()
    assert public_lines[0].split("\t") == [
        "primary_accession",
        "partition",
        "sequence_sha256",
        "biological_length",
        "uniref50_group",
    ]
    assert [line.split("\t")[0] for line in public_lines[1:]] == [
        "P00001",
        "P00007",
        "P00011",
    ]
    assert "P99999" not in public_text
    assert "A" * 32 not in public_text
    assert "proteingym" not in public_text
    assert "\r" not in public_text

    public_rows = {
        columns[0]: columns
        for columns in (line.split("\t") for line in public_lines[1:])
    }
    assert public_rows["P00001"][1] == "training"
    assert public_rows["P00011"][1] == "validation"
    assert public_rows["P00001"][2] == public_rows["P00011"][2]
    assert public_rows["P00001"][4] == public_rows["P00011"][4]

    local_lines = (
        (tmp_path / "first_local.tsv").read_text(encoding="utf-8").splitlines()
    )
    assert local_lines[0].split("\t") == [
        "strategy",
        "stage",
        "repair_cycle",
        "stable_assignment_unit",
        "partition_or_exclusion_status",
        "accession",
    ]
    assert local_lines[1].split("\t") == [
        "random",
        "diagnostic",
        "0",
        "P00001",
        "training",
        "P00001",
    ]


def test_catalog_byte_drift_stops_the_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_path, policy = _write_catalog(tmp_path, _fixture_rows())
    _allow_fixture_policy(monkeypatch, policy)
    catalog_path.write_bytes(catalog_path.read_bytes().replace(b"P99999", b"Q99999"))

    with pytest.raises(RandomSplitError, match="catalog drift"):
        build_random_diagnostic(
            catalog_path=catalog_path,
            local_assignment_output_path=tmp_path / "local.tsv",
            public_manifest_output_path=tmp_path / "public.tsv",
            policy=policy,
            policy_sha256=FIXTURE_CONFIG_SHA256,
        )


def test_task4_report_anchors_are_required() -> None:
    task4_report = json.loads(TASK4_REPORT_PATH.read_bytes())

    sources = validate_task4_report(
        task4_report,
        APPROVED_RANDOM_SPLIT_POLICY,
    )

    assert set(sources) == {
        "proteingym_metadata",
        "swiss_prot_records",
        "uniref50_membership",
    }
    task4_report["population"]["eligible"]["records"] += 1
    with pytest.raises(RandomSplitError, match="eligible records"):
        validate_task4_report(
            task4_report,
            APPROVED_RANDOM_SPLIT_POLICY,
        )


def test_model_use_guard_requires_every_selected_candidate_field() -> None:
    selected_metadata = {
        "strategy": "group_aware",
        "stage": "selected",
        "diagnostic_only": False,
        "selected_for_training": True,
        "model_use": "selected_training_allowed",
    }
    require_selected_training_manifest(selected_metadata)

    for missing_field in selected_metadata:
        incomplete = {
            key: value
            for key, value in selected_metadata.items()
            if key != missing_field
        }
        with pytest.raises(DiagnosticSplitUseError):
            require_selected_training_manifest(incomplete)

    rejected_metadata = (
        {**selected_metadata, "strategy": "random"},
        {**selected_metadata, "strategy": "invented"},
        {**selected_metadata, "stage": "diagnostic"},
        {**selected_metadata, "stage": "invented"},
        {**selected_metadata, "diagnostic_only": True},
        {**selected_metadata, "selected_for_training": False},
        {**selected_metadata, "model_use": "prohibited"},
    )
    for metadata in rejected_metadata:
        with pytest.raises(DiagnosticSplitUseError):
            require_selected_training_manifest(metadata)


def test_report_is_aggregate_only_and_sidecars_are_portable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build, policy, _ = _build_fixture(tmp_path, monkeypatch)
    report = Task5Report(
        schema_version=1,
        scope=policy.scope,
        strategy=policy.strategy,
        stage=policy.stage,
        diagnostic_only=True,
        model_use=policy.model_use,
        selected_for_training=False,
        repeat_verified=True,
        verified_passes=2,
        seed=policy.seed,
        assignment_namespace=policy.assignment_namespace,
        hash_algorithm=policy.hash_algorithm,
        license_spdx=policy.license_spdx,
        code_revision="fixture-revision",
        config_sha256=FIXTURE_CONFIG_SHA256,
        task4_report_sha256=policy.task4_report_sha256,
        task4_policy_sha256=policy.task4_policy_sha256,
        sources={"swiss_prot_records": {"license_spdx": "CC-BY-4.0"}},
        input_catalog=DerivedArtifact(
            relative_path="data/processed/week_01/task_04_record_catalog.tsv",
            row_count=policy.task4_catalog_row_count,
            byte_size=policy.task4_catalog_byte_size,
            sha256=policy.task4_catalog_sha256,
        ),
        population=build.population,
        partitions=build.partitions,
        local_assignments=build.local_assignments,
        public_manifest=build.public_manifest,
    )

    rendered = render_task5_report(report)

    assert (
        rendered.json_sha256
        == hashlib.sha256(rendered.json_text.encode("utf-8")).hexdigest()
    )
    assert asdict(report)["model_use"] == "prohibited"
    for private_value in (
        "P00001",
        "UniRef50_SHARED",
        "A" * 32,
    ):
        assert private_value not in rendered.json_text
        assert private_value not in rendered.markdown_text
    assert sha256_sidecar("manifest.tsv", "a" * 64) == (f"{'a' * 64}  manifest.tsv\n")
    with pytest.raises(RandomSplitError, match="directory"):
        sha256_sidecar("nested/manifest.tsv", "a" * 64)
    with pytest.raises(RandomSplitError, match="malformed"):
        sha256_sidecar("manifest.tsv", "not-a-hash")


def test_public_completion_index_is_deterministic_and_fail_closed() -> None:
    artifacts = (
        CompletedPublicArtifact(
            relative_path="reports/week_01/report.json",
            byte_size=12,
            sha256="b" * 64,
        ),
        CompletedPublicArtifact(
            relative_path="manifests/week_01/manifest.tsv",
            byte_size=34,
            sha256="a" * 64,
        ),
    )
    first = render_completion_index(artifacts)
    second = render_completion_index(tuple(reversed(artifacts)))
    assert first == second
    assert first.endswith("\n")
    assert json.loads(first)["complete"] is True

    with pytest.raises(ValueError, match="repository-relative"):
        render_completion_index(
            (
                CompletedPublicArtifact(
                    relative_path="../outside.tsv",
                    byte_size=1,
                    sha256="a" * 64,
                ),
            )
        )
