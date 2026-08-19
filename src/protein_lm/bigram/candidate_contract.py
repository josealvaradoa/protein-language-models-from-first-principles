"""Static source commitments for a single Week 2 bigram candidate."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from protein_lm.bigram.config import BigramStreamConfig, load_config as load_stream_config
from protein_lm.bigram.training_config import (
    BigramTrainingConfig,
    config_sha256 as training_config_sha256,
    load_training_config,
)
from protein_lm.data.model_data.contracts import ModelDataError


_CANDIDATE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class AuditedArm:
    """The aggregate stream commitment that fitting must reproduce in one pass."""

    collection: str
    namespace: str
    stream_sha256: str
    context_counts: tuple[int, ...]
    target_counts: tuple[int, ...]
    proteins_started: int
    proteins_completed: int
    final_protein_partial: bool


@dataclass(frozen=True)
class CandidatePlan:
    """Static, read-only commitments for one named candidate execution."""

    candidate_id: str
    destination: Path
    training_config: BigramTrainingConfig
    training_config_sha256: str
    stream_config: BigramStreamConfig
    stream_report_sha256: str
    source_identity: dict[str, str]
    arms: tuple[AuditedArm, ...]


def preflight(root: Path, candidate_id: str) -> CandidatePlan:
    """Validate config and aggregate audit commitments without loading a collection."""

    validate_candidate_id(candidate_id)
    training_path = root / "experiments/week_02/bigram_training_v1.toml"
    training_config = load_training_config(training_path)
    stream_path = root / training_config.stream_config_relative_path
    stream_config = load_stream_config(stream_path)
    if (
        training_config.stream_config_sha256
        != hashlib.sha256(stream_path.read_bytes()).hexdigest()
        or training_config.base_seed != stream_config.base_seed
        or training_config.prediction_pair_budget
        != stream_config.prediction_pair_budget
        or training_config.batch_size != stream_config.batch_size
        or training_config.context_roles != stream_config.context_roles
        or training_config.target_roles != stream_config.target_roles
    ):
        raise ModelDataError("bigram training and stream commitments disagree")
    if (
        training_config.stream_report_relative_path
        != stream_config.report_json_relative_path
    ):
        raise ModelDataError("bigram training and stream report paths disagree")
    report_path = root / training_config.stream_report_relative_path
    try:
        report_bytes = report_path.read_bytes()
    except OSError as error:
        raise ModelDataError("audited stream report is unavailable") from error
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    if report_sha256 != training_config.stream_report_sha256:
        raise ModelDataError("audited stream report bytes do not match approval")
    _verify_report_checksum(
        root / stream_config.report_sha256_relative_path,
        report_path.name,
        report_sha256,
    )
    training_sha256 = training_config_sha256(training_path)
    arms, source_identity = _load_audited_arms(
        report_bytes, training_config, stream_config, report_sha256, training_sha256
    )
    return CandidatePlan(
        candidate_id=candidate_id,
        destination=root
        / "data/processed/week_02/bigram_model_candidates"
        / candidate_id,
        training_config=training_config,
        training_config_sha256=training_sha256,
        stream_config=stream_config,
        stream_report_sha256=report_sha256,
        source_identity=source_identity,
        arms=arms,
    )


def expected_artifact_names(plan: CandidatePlan) -> tuple[str, ...]:
    """Return the complete two-arm, three-model, dual-format inventory."""

    return tuple(
        f"{arm.collection}__{model_type}.{format_name}"
        for arm in plan.arms
        for model_type in plan.training_config.model_types
        for format_name in plan.training_config.serialization_formats
    )


def validate_candidate_id(candidate_id: str) -> None:
    """Keep candidate output names local, explicit, and unambiguous."""

    if not isinstance(candidate_id, str) or _CANDIDATE_ID.fullmatch(candidate_id) is None:
        raise ModelDataError("candidate identifier must be 3-64 lowercase letters, digits, or hyphens")


def _load_audited_arms(
    report_bytes: bytes,
    training: BigramTrainingConfig,
    stream: BigramStreamConfig,
    report_sha256: str,
    training_sha256: str,
) -> tuple[tuple[AuditedArm, ...], dict[str, str]]:
    try:
        payload = json.loads(report_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelDataError("audited stream report is malformed") from error
    if (
        not isinstance(payload, dict)
        or payload.get("status") != "passed"
        or payload.get("schema_version") != 1
        or payload.get("scope") != stream.scope
        or payload.get("contract_identifier") != stream.contract_identifier
        or payload.get("network_requests_made") != 0
    ):
        raise ModelDataError("audited stream report is not passed")
    configuration = payload.get("configuration")
    source = payload.get("source")
    arms = payload.get("arms")
    if not isinstance(configuration, dict) or not isinstance(source, dict) or not isinstance(arms, dict):
        raise ModelDataError("audited stream report schema is invalid")
    expected_configuration = {
        "sha256": training.stream_config_sha256,
        "base_seed": training.base_seed,
        "prediction_pair_budget": training.prediction_pair_budget,
        "batch_size": training.batch_size,
        "total_optimizer_steps": training.total_optimizer_steps,
        "context_roles": list(training.context_roles),
        "target_roles": list(training.target_roles),
        "stream_hash_domain": stream.stream_hash_domain,
    }
    if any(configuration.get(key) != value for key, value in expected_configuration.items()):
        raise ModelDataError("audited stream report configuration drifted")
    if set(arms) != set(stream.training_collections):
        raise ModelDataError("audited stream report arms drifted")
    gates = payload.get("hard_gates")
    if not isinstance(gates, dict) or not all(value is True for value in gates.values()):
        raise ModelDataError("audited stream report hard gates are not passed")
    identity = {
        "training_config": training_sha256,
        "stream_config": training.stream_config_sha256,
        "stream_report": report_sha256,
        "model_data_registry": source.get("registry_sha256"),
    }
    if (
        source.get("model_data_contract_identifier")
        != stream.model_data_contract_identifier
        or source.get("model_data_config_relative_path")
        != stream.model_data_config_relative_path
        or source.get("model_data_config_sha256") != stream.model_data_config_sha256
        or source.get("registry_relative_path") != stream.model_data_registry_relative_path
        or source.get("registry_sha256") != stream.model_data_registry_sha256
        or not all(
            isinstance(value, str) and _SHA256.fullmatch(value)
            for value in identity.values()
        )
    ):
        raise ModelDataError("audited stream source identity is invalid")
    result = []
    for collection, namespace in zip(
        stream.training_collections, stream.training_namespaces, strict=True
    ):
        arm = arms[collection]
        if not isinstance(arm, dict):
            raise ModelDataError("audited stream arm is malformed")
        context = _counts(arm.get("context_counts"), training.prediction_pair_budget)
        target = _counts(arm.get("target_counts"), training.prediction_pair_budget)
        digest = arm.get("stream_sha256")
        started = arm.get("proteins_started")
        completed = arm.get("proteins_completed")
        partial = arm.get("final_protein_partial")
        if (
            arm.get("namespace") != namespace
            or arm.get("pairs_emitted") != training.prediction_pair_budget
            or arm.get("source_passes") != 1
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or type(started) is not int
            or type(completed) is not int
            or started < 0
            or completed < 0
            or completed > started
            or type(partial) is not bool
        ):
            raise ModelDataError("audited stream arm commitment is invalid")
        result.append(
            AuditedArm(
                collection, namespace, digest, context, target, started, completed, partial
            )
        )
    return tuple(result), identity  # type: ignore[return-value]


def _counts(value: object, budget: int) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) != 21
        or any(type(item) is not int or item < 0 for item in value)
        or sum(value) != budget
    ):
        raise ModelDataError("audited stream role counts are invalid")
    return tuple(value)


def _verify_report_checksum(path: Path, filename: str, digest: str) -> None:
    try:
        expected = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ModelDataError("audited stream checksum is unavailable") from error
    if expected != f"{digest}  {filename}":
        raise ModelDataError("audited stream checksum drifted")
