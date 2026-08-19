"""Read-only integrity validation for an existing Week 2 bigram candidate."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import torch

from protein_lm.bigram.candidate import CandidatePlan, expected_artifact_names
from protein_lm.bigram.serialization import load_model_artifacts
from protein_lm.data.model_data.contracts import ModelDataError


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_GATES = {
    "only_promoted_training_collections",
    "one_pass_per_arm",
    "exact_audited_streams",
    "six_logical_models",
    "dual_serialization",
}
_ARM_GATES = {
    "one_stream_pass",
    "exact_audited_stream",
    "aggregate_role_counts",
    "exact_pair_budget",
    "batch_and_optimizer_accounting",
}


def validate_candidate(candidate: Path, plan: CandidatePlan) -> dict[str, object]:
    """Prove a passed candidate's complete artifact inventory without loading data."""

    if not candidate.is_dir():
        raise ModelDataError("candidate directory does not exist")
    record = _load_json(candidate / "run_record.json", "candidate run record")
    registry = _load_json(candidate / "candidate_registry.json", "candidate registry")
    _validate_record(record, plan)
    _validate_registry(registry, candidate, plan)
    expected_names = set(expected_artifact_names(plan))
    expected_inventory = expected_names | {"run_record.json", "candidate_registry.json"}
    actual_inventory = {path.name for path in candidate.iterdir()}
    if actual_inventory != expected_inventory:
        raise ModelDataError("candidate artifact inventory is not exact")
    registry_artifacts = registry["artifacts"]
    assert isinstance(registry_artifacts, dict)
    for name in sorted(expected_names):
        _validate_artifact_entry(name, registry_artifacts[name], candidate)
    for arm in plan.arms:
        arm_record = record["arms"][arm.collection]
        assert isinstance(arm_record, dict)
        for model_type in plan.training_config.model_types:
            json_path = candidate / f"{arm.collection}__{model_type}.json"
            safe_path = candidate / f"{arm.collection}__{model_type}.safetensors"
            found_type, tensor, metadata = load_model_artifacts(
                json_path=json_path, safetensors_path=safe_path
            )
            if found_type != model_type:
                raise ModelDataError("candidate model filename and type disagree")
            _validate_model(
                arm.collection,
                model_type,
                tensor,
                metadata,
                arm_record,
                record["code_revision"],
                plan,
            )
    return {
        "candidate_id": plan.candidate_id,
        "status": "passed",
        "logical_model_count": 6,
        "serialization_file_count": 12,
    }


def _validate_record(record: dict[str, object], plan: CandidatePlan) -> None:
    required = {
        "schema_version",
        "scope",
        "contract_identifier",
        "candidate_id",
        "status",
        "hard_gates",
        "training",
        "source_identity",
        "code_revision",
        "arms",
        "runtime_seconds",
        "network_requests_made",
        "failure_reason",
    }
    if set(record) != required or record["status"] != "passed":
        raise ModelDataError("candidate run record is not a passed schema")
    if (
        record["schema_version"] != 1
        or record["scope"] != "week_02_bigram_model_candidate"
        or record["contract_identifier"] != plan.training_config.contract_identifier
        or record["candidate_id"] != plan.candidate_id
        or record["source_identity"] != plan.source_identity
        or record["network_requests_made"] != 0
        or record["failure_reason"] is not None
    ):
        raise ModelDataError("candidate run provenance drifted")
    hard_gates = record["hard_gates"]
    if (
        not isinstance(hard_gates, dict)
        or set(hard_gates) != _RUN_GATES
        or not all(value is True for value in hard_gates.values())
    ):
        raise ModelDataError("candidate hard gates are not passed")
    runtime = record["runtime_seconds"]
    if (
        type(runtime) not in (int, float)
        or not math.isfinite(runtime)
        or runtime < 0
    ):
        raise ModelDataError("candidate runtime accounting is invalid")
    training = record["training"]
    expected_training = {
        "config_sha256": plan.training_config_sha256,
        "stream_config_sha256": plan.training_config.stream_config_sha256,
        "stream_report_sha256": plan.stream_report_sha256,
        "base_seed": plan.training_config.base_seed,
        "prediction_pair_budget": plan.training_config.prediction_pair_budget,
        "batch_size": plan.training_config.batch_size,
        "total_optimizer_steps": plan.training_config.total_optimizer_steps,
        "model_types": list(plan.training_config.model_types),
    }
    if training != expected_training:
        raise ModelDataError("candidate training provenance drifted")
    arms = record["arms"]
    if not isinstance(arms, dict) or set(arms) != {arm.collection for arm in plan.arms}:
        raise ModelDataError("candidate run arms drifted")
    if not isinstance(record["code_revision"], str) or not re.fullmatch(
        r"[0-9a-f]{40}", record["code_revision"]
    ):
        raise ModelDataError("candidate code revision is invalid")


def _validate_registry(
    registry: dict[str, object], candidate: Path, plan: CandidatePlan
) -> None:
    if set(registry) != {
        "schema_version",
        "scope",
        "candidate_id",
        "logical_model_count",
        "serialization_file_count",
        "artifacts",
    }:
        raise ModelDataError("candidate registry schema is invalid")
    if (
        registry["schema_version"] != 1
        or registry["scope"] != "week_02_bigram_model_candidate_registry"
        or registry["candidate_id"] != plan.candidate_id
        or registry["logical_model_count"] != 6
        or registry["serialization_file_count"] != 12
    ):
        raise ModelDataError("candidate registry identity drifted")
    artifacts = registry["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != set(expected_artifact_names(plan)):
        raise ModelDataError("candidate registry artifact inventory drifted")


def _validate_artifact_entry(name: str, entry: object, candidate: Path) -> None:
    if not isinstance(entry, dict) or set(entry) != {
        "arm",
        "model_type",
        "format",
        "byte_size",
        "sha256",
    }:
        raise ModelDataError("candidate registry artifact entry is invalid")
    arm, suffix = name.split("__", maxsplit=1)
    model_type, format_name = suffix.rsplit(".", maxsplit=1)
    if (
        entry["arm"] != arm
        or entry["model_type"] != model_type
        or entry["format"] != format_name
        or type(entry["byte_size"]) is not int
        or entry["byte_size"] <= 0
        or not isinstance(entry["sha256"], str)
        or _SHA256.fullmatch(entry["sha256"]) is None
    ):
        raise ModelDataError("candidate registry artifact identity drifted")
    content = (candidate / name).read_bytes()
    if len(content) != entry["byte_size"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
        raise ModelDataError("candidate artifact bytes drifted")


def _validate_model(
    arm: str,
    model_type: str,
    tensor: torch.Tensor,
    metadata: dict[str, object],
    arm_record: dict[str, object],
    code_revision: object,
    plan: CandidatePlan,
) -> None:
    audited = next(item for item in plan.arms if item.collection == arm)
    expected_metadata = {
        "arm": arm,
        "model_type": model_type,
        "context_roles": list(plan.training_config.context_roles),
        "target_roles": list(plan.training_config.target_roles),
        "stream_sha256": audited.stream_sha256,
        "config_sha256": plan.training_config_sha256,
        "source_identity": plan.source_identity,
        "code_revision": code_revision,
    }
    if metadata.get("arm") != arm or metadata.get("model_type") != model_type:
        raise ModelDataError("candidate model arm identity drifted")
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise ModelDataError("candidate model provenance drifted")
    for key, value in {
        "seed": plan.training_config.base_seed,
        "prediction_pair_budget": plan.training_config.prediction_pair_budget,
        "batch_size": plan.training_config.batch_size,
        "batches_consumed": plan.training_config.total_optimizer_steps,
    }.items():
        if metadata.get(key) != value:
            raise ModelDataError("candidate model accounting drifted")
    stream = arm_record.get("stream")
    accounting = arm_record.get("accounting")
    gates = arm_record.get("hard_gates")
    losses = arm_record.get("losses")
    if not isinstance(stream, dict) or not isinstance(accounting, dict) or not isinstance(gates, dict) or not isinstance(losses, list):
        raise ModelDataError("candidate arm evidence is malformed")
    expected_stream = {
        "namespace": audited.namespace,
        "stream_sha256": audited.stream_sha256,
        "context_counts": list(audited.context_counts),
        "target_counts": list(audited.target_counts),
        "pairs_emitted": plan.training_config.prediction_pair_budget,
        "proteins_started": audited.proteins_started,
        "proteins_completed": audited.proteins_completed,
        "final_protein_partial": audited.final_protein_partial,
    }
    if (
        stream != expected_stream
        or set(gates) != _ARM_GATES
        or not all(value is True for value in gates.values())
        or len(losses) != plan.training_config.total_optimizer_steps
        or any(type(loss) not in (int, float) or not math.isfinite(loss) for loss in losses)
        or accounting
        != {
            "pairs_seen": plan.training_config.prediction_pair_budget,
            "optimizer_steps": plan.training_config.total_optimizer_steps,
            "count_bigram_total": plan.training_config.prediction_pair_budget,
        }
    ):
        raise ModelDataError("candidate arm evidence gates drifted")
    if model_type == "unigram":
        expected = torch.tensor(audited.target_counts, dtype=torch.int64)
        if not torch.equal(tensor, expected):
            raise ModelDataError("unigram tensor does not equal audited target counts")
    elif model_type == "count_bigram":
        expected_contexts = torch.tensor(audited.context_counts, dtype=torch.int64)
        expected_targets = torch.tensor(audited.target_counts, dtype=torch.int64)
        if (
            not torch.equal(tensor.sum(dim=1), expected_contexts)
            or not torch.equal(tensor.sum(dim=0), expected_targets)
        ):
            raise ModelDataError("count bigram margins do not equal audited counts")
    elif metadata.get("optimizer_steps") != plan.training_config.total_optimizer_steps:
        raise ModelDataError("neural optimizer accounting drifted")


def _load_json(path: Path, description: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelDataError(f"{description} is malformed") from error
    if not isinstance(value, dict):
        raise ModelDataError(f"{description} is malformed")
    return value
