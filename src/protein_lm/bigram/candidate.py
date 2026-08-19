"""Operator-gated installation of one local Week 2 bigram candidate."""

from __future__ import annotations

import re
import subprocess
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from protein_lm.bigram.candidate_contract import (
    AuditedArm,
    CandidatePlan,
    expected_artifact_names,
    preflight,
    validate_candidate_id,
)
from protein_lm.bigram.candidate_fitting import fit_arm, write_arm_artifacts
from protein_lm.bigram.candidate_reporting import (
    arm_record,
    record_payload,
    registry_payload,
    write_new_json,
    write_record,
)
from protein_lm.bigram.training import TrainingSettings
from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.data.model_data.loaders import (
    ModelDataCollection,
    ProteinSequence,
    load_collection,
)


_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
CollectionLoader = Callable[[Path, ModelDataCollection], Iterable[ProteinSequence]]


def create_candidate(
    *,
    root: Path,
    plan: CandidatePlan,
    loader: CollectionLoader = load_collection,
    code_revision: str | None = None,
    settings: TrainingSettings | None = None,
) -> Path:
    """Fit each approved arm once and preserve either a passed or failed candidate."""

    _validate_plan(plan, root)
    if plan.destination.exists():
        raise ModelDataError("candidate destination already exists")
    _require_ignored(root, plan.destination)
    revision = code_revision or _clean_revision(root)
    if _GIT_REVISION.fullmatch(revision) is None:
        raise ModelDataError("candidate code revision is invalid")
    fitting_settings = settings or TrainingSettings(
        batch_size=plan.training_config.batch_size,
        prediction_pair_budget=plan.training_config.prediction_pair_budget,
        learning_rate=plan.training_config.learning_rate,
        momentum=plan.training_config.momentum,
        weight_decay=plan.training_config.weight_decay,
    )
    if (
        fitting_settings.batch_size != plan.training_config.batch_size
        or fitting_settings.prediction_pair_budget
        != plan.training_config.prediction_pair_budget
        or fitting_settings.learning_rate != plan.training_config.learning_rate
        or fitting_settings.momentum != plan.training_config.momentum
        or fitting_settings.weight_decay != plan.training_config.weight_decay
    ):
        raise ModelDataError("candidate settings disagree with the frozen training plan")
    plan.destination.parent.mkdir(parents=True, exist_ok=True)
    plan.destination.mkdir()
    started = time.perf_counter()
    arms: dict[str, object] = {}
    write_record(
        plan.destination,
        record_payload(plan, revision, "running", arms, started, None),
    )
    try:
        for audited_arm in plan.arms:
            proteins = loader(root, ModelDataCollection(audited_arm.collection))
            try:
                state, losses, observed = fit_arm(
                    proteins=proteins,
                    audited_arm=audited_arm,
                    plan=plan,
                    settings=fitting_settings,
                )
            finally:
                del proteins
            arms[audited_arm.collection] = arm_record(
                audited_arm, observed, losses, state
            )
            write_arm_artifacts(
                destination=plan.destination,
                arm=audited_arm,
                plan=plan,
                state=state,
                code_revision=revision,
            )
        registry = registry_payload(
            plan.destination, plan, expected_artifact_names(plan)
        )
        write_new_json(plan.destination / "candidate_registry.json", registry)
        write_record(
            plan.destination,
            record_payload(plan, revision, "passed", arms, started, None),
        )
    except Exception as error:
        write_record(
            plan.destination,
            record_payload(plan, revision, "failed", arms, started, str(error)),
        )
        if isinstance(error, ModelDataError):
            raise
        raise ModelDataError(f"bigram candidate fitting failed: {error}") from error
    return plan.destination


def _validate_plan(plan: CandidatePlan, root: Path) -> None:
    validate_candidate_id(plan.candidate_id)
    try:
        plan.destination.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ModelDataError("candidate destination must remain under the repository root") from error
    if len(plan.arms) != 2 or {arm.collection for arm in plan.arms} != {
        "random_training",
        "family_aware_training",
    }:
        raise ModelDataError("candidate plan must contain only the two approved training arms")


def _require_ignored(root: Path, path: Path) -> None:
    try:
        relative = path.resolve().relative_to(root.resolve())
        result = subprocess.run(
            ("git", "check-ignore", "--quiet", "--", str(relative)),
            cwd=root,
            check=False,
        )
    except (OSError, ValueError) as error:
        raise ModelDataError("could not prove candidate output is ignored") from error
    if result.returncode != 0:
        raise ModelDataError("candidate output path is not ignored")


def _clean_revision(root: Path) -> str:
    try:
        status = subprocess.run(
            ("git", "status", "--porcelain"), cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout
        if status:
            raise ModelDataError("candidate creation requires a clean committed revision")
        return subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except subprocess.SubprocessError as error:
        raise ModelDataError("could not establish candidate code revision") from error


__all__ = (
    "AuditedArm",
    "CandidatePlan",
    "create_candidate",
    "expected_artifact_names",
    "preflight",
)
