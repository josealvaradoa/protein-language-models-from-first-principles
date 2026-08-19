"""Byte-pinned public-report contract for the completed Week 2 evaluation."""

from __future__ import annotations

import hashlib
import re
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from protein_lm.data.model_data.contracts import ModelDataError


APPROVED_PUBLIC_REPORT_CONFIG_SHA256 = (
    "ad73f29b6769ae430ab7bbfaa6dead1f510714b121babd7ab27b12e52d0e5f54"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class PublicReportConfig:
    """Static publication inputs and the three public output names."""

    schema_version: int
    scope: str
    contract_identifier: str
    source_evaluation_id: str
    source_evaluation_relative_path: str
    source_evaluation_sha256: str
    source_run_record_sha256: str
    source_registry_sha256: str
    source_evaluation_code_revision: str
    source_evaluation_config_sha256: str
    report_json_relative_path: str
    report_markdown_relative_path: str
    report_sha256_relative_path: str
    network_requests_made: int
    publication_scope: str

    @property
    def output_paths(self) -> tuple[str, str, str]:
        return (
            self.report_json_relative_path,
            self.report_markdown_relative_path,
            self.report_sha256_relative_path,
        )


def config_sha256(path: Path) -> str:
    """Return the byte identity carried into the public report."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_public_report_config(path: Path) -> PublicReportConfig:
    """Load only the reviewed report contract and no evaluation evidence."""

    try:
        content = path.read_bytes()
        raw = tomllib.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ModelDataError(
            f"could not load public report configuration: {error}"
        ) from error
    if hashlib.sha256(content).hexdigest() != APPROVED_PUBLIC_REPORT_CONFIG_SHA256:
        raise ModelDataError("public report configuration bytes do not match approval")
    if not isinstance(raw, dict) or set(raw) != {
        field.name for field in fields(PublicReportConfig)
    }:
        raise ModelDataError("public report configuration keys differ from the schema")
    if (
        type(raw.get("schema_version")) is not int
        or type(raw.get("network_requests_made")) is not int
    ):
        raise ModelDataError("public report integer values are invalid")
    for name, value in raw.items():
        if name not in {"schema_version", "network_requests_made"} and (
            not isinstance(value, str) or not value
        ):
            raise ModelDataError(f"{name} must be a nonempty string")
    config = PublicReportConfig(**raw)
    _validate(config)
    return config


def _validate(config: PublicReportConfig) -> None:
    if (
        config.schema_version != 1
        or config.scope != "week_02_bigram_evaluation_publication"
        or config.contract_identifier
        != "2026-08-19-week-02-bigram-evaluation-publication-v1"
        or config.source_evaluation_id != "week2-bigram-eval-v1-001"
        or config.source_evaluation_relative_path
        != "data/processed/week_02/bigram_evaluation_candidates/week2-bigram-eval-v1-001"
        or config.source_evaluation_code_revision
        != "36c8cc9d964421e67bd16e6aa5ebcea14f76c80e"
        or config.source_evaluation_config_sha256
        != "219e7a3bc06a6c227ed27b9b4b7e917083b537bd5ac5d11a7526ee8415c2d97c"
        or config.output_paths
        != (
            "reports/week_02/bigram_evaluation_v1.json",
            "reports/week_02/bigram_evaluation_v1.md",
            "reports/week_02/bigram_evaluation_v1.sha256",
        )
        or config.network_requests_made != 0
        or config.publication_scope != "aggregate_only_no_sequences_no_membership"
    ):
        raise ModelDataError("public report contract identity is not approved")
    if (
        not all(
            _SHA256.fullmatch(value)
            for value in (
                config.source_evaluation_sha256,
                config.source_run_record_sha256,
                config.source_registry_sha256,
                config.source_evaluation_config_sha256,
            )
        )
        or _REVISION.fullmatch(config.source_evaluation_code_revision) is None
    ):
        raise ModelDataError("public report source identity is invalid")
