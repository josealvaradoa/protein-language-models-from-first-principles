"""Literal configuration pins and MMseqs runner-boundary goldens."""

import hashlib
import tomllib
from pathlib import Path

from a004_workflow_test_support import install_synthetic_workflow
from protein_lm.data.task7_a004_workflow import run_a004_fixed_budget_audit

REPOSITORY = Path(__file__).parents[3]
A004_CONFIG = REPOSITORY / "experiments/week_01/read_only_similarity_audit_a004.toml"
A003_CONFIG = REPOSITORY / "experiments/week_01/diagnostic_similarity_audit.toml"

A004_SHA256 = "3a21edeaf45057a8e50b5643abc14d3b633edb69b66aefd184e5f59963931a04"
A003_SHA256 = "ce767f0ce843e4f40edbcd2f9da6ca4642996046cb4042a2410c27c39cbae742"
A003_MARKER_PINS = {
    "source_fastas_marker_sha256": (
        "5a4b23ba4c0550279967d056077a8c6ce06f40d3e4b58c58cd2b7c1445bde9f0"
    ),
    "source_database_marker_sha256": (
        "f3c4c170239d723ca9cc305041da6d3df30b986a5c76617c0e112a13e6f6eb7f"
    ),
    "1000": "6c90bf2d17b968619a665aa0d98974cdf345de68d1a5a6ba89b906f8715df01f",
    "10000": "9b7544504bf8578e19ea2f4db7a3984f4dc1fc5b36af61f2aed325ae164ed48d",
    "100000": "a02b2374fe0c056a9ca912ca8df8a06ce1efd526fa99c0d1847e215dd925bcc5",
}

FRESH_TRACKS = (
    ("random", "validation", "enforcement"),
    ("random", "test", "enforcement"),
    ("random", "test", "residual"),
    ("group_aware", "validation", "enforcement"),
    ("group_aware", "validation", "residual"),
    ("group_aware", "test", "enforcement"),
    ("group_aware", "test", "residual"),
)


def test_literal_configuration_and_a003_marker_pins() -> None:
    a004_bytes = A004_CONFIG.read_bytes()
    a003_bytes = A003_CONFIG.read_bytes()
    parsed = tomllib.loads(a004_bytes.decode("utf-8"))

    assert hashlib.sha256(a004_bytes).hexdigest() == A004_SHA256
    assert hashlib.sha256(a003_bytes).hexdigest() == A003_SHA256
    assert parsed["source_policy_sha256"] == A003_SHA256
    assert parsed["source_fastas_marker_sha256"] == A003_MARKER_PINS[
        "source_fastas_marker_sha256"
    ]
    assert parsed["source_database_marker_sha256"] == A003_MARKER_PINS[
        "source_database_marker_sha256"
    ]
    assert parsed["source_stage_marker_sha256"] == {
        cap: A003_MARKER_PINS[cap] for cap in ("1000", "10000", "100000")
    }


def test_exact_createdb_and_changed_only_search_runner_sequence(
    monkeypatch,
    tmp_path: Path,
) -> None:
    synthetic = install_synthetic_workflow(
        monkeypatch,
        tmp_path,
        changed_search=True,
    )

    run_a004_fixed_budget_audit(
        project_root=synthetic.project_root,
        config_path=synthetic.config_path,
        search_runner=synthetic.search_runner,
        database_runner=synthetic.database_runner,
        hardware=synthetic.hardware,
    )

    expected_database_calls = [
        (
            _createdb_command(synthetic.workspace, strategy),
            synthetic.project_root,
            synthetic.workspace,
            synthetic.workspace / "logs" / f"a004_createdb_{strategy}.log",
            synthetic.source_policy,
        )
        for strategy in ("random", "group_aware")
    ]
    expected_search_calls = [
        (
            _search_command(
                workspace=synthetic.workspace,
                strategy=strategy,
                partition=partition,
                pass_name=pass_name,
                cap=cap,
            ),
            synthetic.project_root,
            synthetic.workspace,
            (
                synthetic.workspace
                / "tracks"
                / strategy
                / partition
                / pass_name
                / f"cap_{cap}"
                / "command.log"
            ),
            synthetic.source_policy,
        )
        for strategy, partition, pass_name in FRESH_TRACKS
        for cap in (1_000, 10_000, 100_000)
    ]
    observed_database_calls = [
        (call.command, call.project_root, call.workspace, call.log_path, call.policy)
        for call in synthetic.database_calls
    ]
    observed_search_calls = [
        (call.command, call.project_root, call.workspace, call.log_path, call.policy)
        for call in synthetic.search_calls
    ]

    assert observed_database_calls == expected_database_calls
    assert observed_search_calls == expected_search_calls
    assert len(observed_search_calls) == 21
    assert all(
        Path(call.command[index]).is_absolute()
        for call in (*synthetic.database_calls, *synthetic.search_calls)
        for index in (
            (2, 3)
            if call.command[1] == "createdb"
            else (2, 3, 4, 5)
        )
    )


def _createdb_command(workspace: Path, strategy: str) -> tuple[str, ...]:
    return (
        "/opt/homebrew/bin/mmseqs",
        "createdb",
        str(workspace / "fastas" / f"{strategy}_training.fasta"),
        str(workspace / "databases" / f".{strategy}.incomplete" / "target"),
        "--dbtype",
        "1",
        "--shuffle",
        "0",
        "--createdb-mode",
        "0",
        "--threads",
        "10",
    )


def _search_command(
    *,
    workspace: Path,
    strategy: str,
    partition: str,
    pass_name: str,
    cap: int,
) -> tuple[str, ...]:
    pass_directory = workspace / "tracks" / strategy / partition / pass_name
    stage_directory = pass_directory / f"cap_{cap}"
    query_fasta = (
        pass_directory / "escalated_queries.fasta"
        if cap == 100_000
        else workspace / "fastas" / f"{strategy}_{partition}.fasta"
    )
    min_identity, coverage = (
        ("0.5", "0.8") if pass_name == "enforcement" else ("0.3", "0.0")
    )
    return (
        "/opt/homebrew/bin/mmseqs",
        "easy-search",
        str(query_fasta),
        str(workspace / "databases" / strategy / "target"),
        str(stage_directory / "raw.tsv"),
        str(stage_directory / "mmseqs_tmp"),
        "--search-type",
        "1",
        "--alignment-mode",
        "3",
        "--seq-id-mode",
        "0",
        "-s",
        "7.5",
        "-e",
        "10",
        "--mask",
        "0",
        "--comp-bias-corr",
        "0",
        "--max-seqs",
        str(cap),
        "--threads",
        "10",
        "--format-output",
        (
            "query,target,fident,qcov,tcov,alnlen,qlen,tlen,qstart,qend,"
            "tstart,tend,evalue,bits"
        ),
        "--min-seq-id",
        min_identity,
        "-c",
        coverage,
        "--cov-mode",
        "0",
    )
