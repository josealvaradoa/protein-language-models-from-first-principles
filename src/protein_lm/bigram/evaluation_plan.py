"""Pinned provenance, path safety, and repository gates for Week 2 evaluation."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from protein_lm.bigram.evaluation_contract import (
    EvaluationConfig,
    config_sha256,
    load_evaluation_config,
)
from protein_lm.data.model_data.contracts import ModelDataError


_EVALUATION_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class EvaluationPlan:
    """The immutable local targets and byte identities for one execution."""

    root: Path
    evaluation_id: str
    destination: Path
    config: EvaluationConfig
    config_sha256: str
    model_candidate: Path
    model_data_registry: Path


def preflight(root: Path, evaluation_id: str) -> EvaluationPlan:
    """Verify pinned provenance without loading model artifacts or collections."""

    if (
        not isinstance(evaluation_id, str)
        or _EVALUATION_ID.fullmatch(evaluation_id) is None
    ):
        raise ModelDataError(
            "evaluation identifier must be 3-64 lowercase letters, digits, or hyphens"
        )
    config_path = root / "experiments/week_02/bigram_evaluation_v1.toml"
    config = load_evaluation_config(config_path)
    candidate = root / config.model_candidate_relative_path
    registry = root / config.model_data_registry_relative_path
    _verify_bytes(
        candidate / "candidate_registry.json", config.model_candidate_registry_sha256
    )
    _verify_bytes(
        candidate / "run_record.json", config.model_candidate_run_record_sha256
    )
    _verify_bytes(registry, config.model_data_registry_sha256)
    _verify_candidate_data_provenance(candidate / "run_record.json", config)
    return EvaluationPlan(
        root=root,
        evaluation_id=evaluation_id,
        destination=root / config.output_root_relative_path / evaluation_id,
        config=config,
        config_sha256=config_sha256(config_path),
        model_candidate=candidate,
        model_data_registry=registry,
    )


def validate_plan(root: Path, plan: EvaluationPlan) -> None:
    """Reject an injected plan that redirects a candidate read or output write."""

    if not isinstance(plan, EvaluationPlan) or plan.root != root:
        raise ModelDataError("evaluation plan root is invalid")
    if (
        not isinstance(plan.evaluation_id, str)
        or _EVALUATION_ID.fullmatch(plan.evaluation_id) is None
    ):
        raise ModelDataError("evaluation plan identifier is invalid")
    config_path = root / "experiments/week_02/bigram_evaluation_v1.toml"
    approved_output_root = (root / plan.config.output_root_relative_path).resolve()
    try:
        plan.destination.resolve().relative_to(approved_output_root)
    except ValueError as error:
        raise ModelDataError(
            "evaluation plan destination escapes the approved output root"
        ) from error
    if (
        plan.destination
        != root / plan.config.output_root_relative_path / plan.evaluation_id
        or plan.model_candidate != root / plan.config.model_candidate_relative_path
        or plan.model_data_registry
        != root / plan.config.model_data_registry_relative_path
        or plan.config_sha256 != config_sha256(config_path)
    ):
        raise ModelDataError(
            "evaluation plan targets or configuration identity drifted"
        )
    # ``load_evaluation_config`` checks both the byte pin and exact constants.
    if load_evaluation_config(config_path) != plan.config:
        raise ModelDataError("evaluation plan configuration drifted")


def clean_revision(root: Path) -> str:
    """Return HEAD only when execution code is committed and the worktree is clean."""

    try:
        status = subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if status:
            raise ModelDataError("evaluation requires a clean committed revision")
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.SubprocessError as error:
        raise ModelDataError("could not establish evaluation code revision") from error
    if _REVISION.fullmatch(revision) is None:
        raise ModelDataError("evaluation code revision is invalid")
    return revision


def require_ignored(root: Path, path: Path) -> None:
    """Require the new local evidence directory to remain outside version control."""

    try:
        relative = path.resolve().relative_to(root.resolve())
        result = subprocess.run(
            ("git", "check-ignore", "--quiet", "--", str(relative)),
            cwd=root,
            check=False,
        )
    except (OSError, ValueError) as error:
        raise ModelDataError("could not prove evaluation output is ignored") from error
    if result.returncode != 0:
        raise ModelDataError("evaluation output path is not ignored")


def verify_candidate_provenance(plan: EvaluationPlan) -> None:
    """Recheck byte pins and prove the fitted candidate cites the same data registry."""

    _verify_bytes(
        plan.model_candidate / "candidate_registry.json",
        plan.config.model_candidate_registry_sha256,
    )
    record = plan.model_candidate / "run_record.json"
    _verify_bytes(record, plan.config.model_candidate_run_record_sha256)
    _verify_bytes(plan.model_data_registry, plan.config.model_data_registry_sha256)
    _verify_candidate_data_provenance(record, plan.config)


def _verify_bytes(path: Path, expected: str) -> None:
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ModelDataError("approved evaluation provenance is unavailable") from error
    if actual != expected:
        raise ModelDataError("approved evaluation provenance hash drifted")


def _verify_candidate_data_provenance(path: Path, config: EvaluationConfig) -> None:
    try:
        record = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("nonfinite")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ModelDataError(
            "approved model candidate run record is malformed"
        ) from error
    source = record.get("source_identity") if isinstance(record, dict) else None
    if (
        not isinstance(source, dict)
        or source.get("model_data_registry") != config.model_data_registry_sha256
    ):
        raise ModelDataError("model candidate data registry provenance drifted")
