"""Operator-gated publication from one validated local Week 2 evaluation."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from protein_lm.bigram.evaluation_plan import (
    EvaluationPlan,
    preflight as evaluation_preflight,
)
from protein_lm.bigram.evaluation_validation import validate_evaluation
from protein_lm.bigram.public_report import report_payload, write_evidence
from protein_lm.bigram.public_report_contract import (
    PublicReportConfig,
    load_public_report_config,
)
from protein_lm.data.model_data.contracts import ModelDataError


_REVISION = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class PublicReportPlan:
    """Validated, local-only inputs and destinations for the one publication."""

    root: Path
    config_path: Path
    config: PublicReportConfig
    evaluation_plan: EvaluationPlan
    output_paths: tuple[Path, Path, Path]


def preflight(root: Path) -> PublicReportPlan:
    """Validate byte-pinned source evidence without loading models or collections."""

    config_path = root / "experiments/week_02/bigram_evaluation_publication_v1.toml"
    config = load_public_report_config(config_path)
    source = root / config.source_evaluation_relative_path
    _verify_bytes(source / "evaluation.json", config.source_evaluation_sha256)
    _verify_bytes(source / "run_record.json", config.source_run_record_sha256)
    _verify_bytes(source / "evaluation_registry.json", config.source_registry_sha256)
    evaluation = evaluation_preflight(root, config.source_evaluation_id)
    if (
        evaluation.destination != source
        or evaluation.config_sha256 != config.source_evaluation_config_sha256
    ):
        raise ModelDataError("public report source evaluation plan drifted")
    validate_evaluation(source, evaluation)
    run = _load_json(source / "run_record.json", "source evaluation run record")
    if run.get("code_revision") != config.source_evaluation_code_revision:
        raise ModelDataError("public report source evaluation code revision drifted")
    return PublicReportPlan(
        root=root,
        config_path=config_path,
        config=config,
        evaluation_plan=evaluation,
        output_paths=tuple(root / path for path in config.output_paths),
    )


def execute_publication(root: Path, plan: PublicReportPlan) -> tuple[Path, Path, Path]:
    """Render and install exactly the three new public report artifacts."""

    _validate_plan(root, plan)
    if preflight(root) != plan:
        raise ModelDataError("public report plan or validated source drifted")
    if any(path.exists() for path in plan.output_paths):
        raise ModelDataError("public evaluation report already exists")
    publication_revision = _clean_revision(root)
    source = plan.evaluation_plan.destination
    payload = report_payload(
        config_path=plan.config_path,
        config=plan.config,
        source=_load_json(source / "evaluation.json", "source evaluation"),
        run=_load_json(source / "run_record.json", "source evaluation run record"),
        publication_code_revision=publication_revision,
    )
    write_evidence(plan.output_paths, payload)
    return plan.output_paths


def _validate_plan(root: Path, plan: PublicReportPlan) -> None:
    if not isinstance(plan, PublicReportPlan) or plan.root != root:
        raise ModelDataError("public report plan root is invalid")
    expected = tuple(root / path for path in plan.config.output_paths)
    if (
        plan.output_paths != expected
        or plan.config_path
        != root / "experiments/week_02/bigram_evaluation_publication_v1.toml"
    ):
        raise ModelDataError("public report plan targets drifted")
    if (
        plan.evaluation_plan.destination
        != root / plan.config.source_evaluation_relative_path
    ):
        raise ModelDataError("public report source path drifted")


def _clean_revision(root: Path) -> str:
    try:
        status = subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if status:
            raise ModelDataError(
                "public report publication requires a clean committed revision"
            )
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.SubprocessError as error:
        raise ModelDataError(
            "could not establish public report code revision"
        ) from error
    if _REVISION.fullmatch(revision) is None:
        raise ModelDataError("public report code revision is invalid")
    return revision


def _verify_bytes(path: Path, expected: str) -> None:
    try:
        found = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ModelDataError("public report source evidence is unavailable") from error
    if found != expected:
        raise ModelDataError("public report source evidence hash drifted")


def _load_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("nonfinite")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ModelDataError(f"{label} is malformed") from error
    if not isinstance(value, dict):
        raise ModelDataError(f"{label} must be a JSON object")
    return value
