"""Render aggregate evidence for the Task 5 random diagnostic."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class DerivedArtifact:
    """Identity and size of one deterministic derived artifact."""

    relative_path: str
    row_count: int
    byte_size: int
    sha256: str


@dataclass(frozen=True)
class SplitPopulation:
    """The common eligible population assigned by both split strategies."""

    records: int
    residues: int
    unique_groups: int


@dataclass(frozen=True)
class PartitionAudit:
    """Target and realized size of one diagnostic partition."""

    target_numerator: int
    target_denominator: int
    target_share_percent: str
    records: int
    residues: int
    unique_groups: int
    record_share_percent: str
    residue_share_percent: str
    record_deviation_percentage_points: str
    residue_deviation_percentage_points: str


@dataclass(frozen=True)
class RandomSplitBuild:
    """Evidence returned by one complete manifest build."""

    population: SplitPopulation
    partitions: dict[str, PartitionAudit]
    local_assignments: DerivedArtifact
    public_manifest: DerivedArtifact


@dataclass(frozen=True)
class Task5Report:
    """Public metadata and aggregate evidence for the random diagnostic."""

    schema_version: int
    scope: str
    strategy: str
    stage: str
    diagnostic_only: bool
    model_use: str
    selected_for_training: bool
    repeat_verified: bool
    verified_passes: int
    seed: int
    assignment_namespace: str
    hash_algorithm: str
    license_spdx: str
    code_revision: str
    config_sha256: str
    task4_report_sha256: str
    task4_policy_sha256: str
    sources: dict[str, dict[str, object]]
    input_catalog: DerivedArtifact
    population: SplitPopulation
    partitions: dict[str, PartitionAudit]
    local_assignments: DerivedArtifact
    public_manifest: DerivedArtifact


@dataclass(frozen=True)
class RenderedTask5Report:
    """Byte-stable JSON, Markdown, and canonical JSON digest."""

    json_text: str
    markdown_text: str
    json_sha256: str


@dataclass(frozen=True)
class CompletedPublicArtifact:
    """One public file covered by the last-written completion index."""

    relative_path: str
    byte_size: int
    sha256: str


def render_task5_report(report: Task5Report) -> RenderedTask5Report:
    """Render byte-stable aggregate JSON and Markdown."""

    report_dict = asdict(report)
    json_text = json.dumps(report_dict, indent=2, sort_keys=True) + "\n"
    lines = [
        "# Week 1 Task 5 Random Diagnostic",
        "",
        "This is an intentionally unprotected comparison split. It is prohibited "
        "for model use and is not the selected training split.",
        "",
        "The report contains aggregate evidence and provenance. The separate "
        "public manifest contains approved identifiers and membership fields, "
        "but no sequences, labels, scores, or model results.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {_markdown_cell(metric)} | {_markdown_cell(value)} |"
        for metric, value in _flatten(report_dict)
    )
    lines.append("")
    markdown_text = "\n".join(lines)
    return RenderedTask5Report(
        json_text=json_text,
        markdown_text=markdown_text,
        json_sha256=hashlib.sha256(json_text.encode("utf-8")).hexdigest(),
    )


def render_completion_index(
    artifacts: tuple[CompletedPublicArtifact, ...],
    *,
    scope: str = "week_01_task_05_public_outputs",
) -> str:
    """Render the deterministic marker for one complete public output set."""

    indexed = {}
    for artifact in artifacts:
        path = PurePosixPath(artifact.relative_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != artifact.relative_path
        ):
            raise ValueError("completion artifact path must be repository-relative")
        if artifact.relative_path in indexed:
            raise ValueError("completion artifact paths must be unique")
        if artifact.byte_size < 0:
            raise ValueError("completion artifact byte size must not be negative")
        if not _SHA256_PATTERN.fullmatch(artifact.sha256):
            raise ValueError("completion artifact SHA-256 is malformed")
        indexed[artifact.relative_path] = {
            "byte_size": artifact.byte_size,
            "sha256": artifact.sha256,
        }
    if not indexed:
        raise ValueError("completion index must cover at least one artifact")

    payload = {
        "schema_version": 1,
        "scope": scope,
        "complete": True,
        "artifacts": indexed,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _flatten(
    value: object,
    prefix: tuple[str, ...] = (),
) -> Iterator[tuple[str, object]]:
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            yield from _flatten(value[key], (*prefix, str(key)))
        return
    yield ".".join(prefix), value


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")
