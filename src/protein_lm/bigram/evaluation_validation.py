"""Read-only validation for evaluation evidence. This module imports no loader."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from protein_lm.bigram.evaluation_plan import (
    EvaluationPlan,
    verify_candidate_provenance,
)
from protein_lm.bigram.evaluation_results import (
    hypothesis,
    provenance,
    validate_records,
)
from protein_lm.data.model_data.contracts import ModelDataError


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_LOADS = {
    "random_native_validation",
    "family_aware_native_validation",
    "shared_validation",
    "shared_sealed_test",
}
_GATES = {
    "candidate_validation",
    "twelve_principal_records",
    "shared_validation_loaded_once",
    "sealed_test_never_loaded",
    "evaluation_only_no_retraining_or_selection",
    "no_network_requests",
}


def validate_evaluation(candidate: Path, plan: EvaluationPlan) -> dict[str, object]:
    """Check local evidence and the pinned input candidate without collection loads."""

    if candidate != plan.destination or not candidate.is_dir():
        raise ModelDataError("evaluation candidate path is invalid")
    verify_candidate_provenance(plan)
    record = _load_json(candidate / "run_record.json", "evaluation run record")
    _validate_run_record(record, plan)
    if record["status"] == "failed":
        if {path.name for path in candidate.iterdir()} != {"run_record.json"}:
            raise ModelDataError("failed evaluation candidate inventory is invalid")
        return {"evaluation_id": plan.evaluation_id, "status": "failed"}
    if {path.name for path in candidate.iterdir()} != {
        "run_record.json",
        "evaluation.json",
        "evaluation_registry.json",
    }:
        raise ModelDataError("passed evaluation candidate inventory is not exact")
    result = _load_json(candidate / "evaluation.json", "evaluation results")
    _validate_result(result, plan)
    registry = _load_json(candidate / "evaluation_registry.json", "evaluation registry")
    _validate_registry(registry, candidate, plan)
    return {
        "evaluation_id": plan.evaluation_id,
        "status": "passed",
        "principal_record_count": 12,
    }


def _validate_run_record(record: dict[str, object], plan: EvaluationPlan) -> None:
    expected = {
        "schema_version",
        "scope",
        "contract_identifier",
        "evaluation_id",
        "status",
        "code_revision",
        "configuration_sha256",
        "model_candidate",
        "model_data_registry",
        "collection_loads",
        "hard_gates",
        "network_requests_made",
        "runtime_seconds",
        "failure_reason",
    }
    if set(record) != expected or record["status"] not in {"passed", "failed"}:
        raise ModelDataError("evaluation run record schema is invalid")
    if (
        record["schema_version"] != 1
        or record["scope"] != "week_02_bigram_evaluation_candidate"
        or record["contract_identifier"] != plan.config.contract_identifier
        or record["evaluation_id"] != plan.evaluation_id
        or record["configuration_sha256"] != plan.config_sha256
        or record["model_candidate"]
        != provenance(plan.config, plan.config_sha256)["model_candidate"]
        or record["model_data_registry"]
        != provenance(plan.config, plan.config_sha256)["model_data_registry"]
        or record["network_requests_made"] != 0
        or not isinstance(record["code_revision"], str)
        or _REVISION.fullmatch(record["code_revision"]) is None
    ):
        raise ModelDataError("evaluation run record provenance drifted")
    runtime = record["runtime_seconds"]
    if type(runtime) not in (int, float) or not math.isfinite(runtime) or runtime < 0:
        raise ModelDataError("evaluation run record runtime is invalid")
    if (record["status"] == "passed") != (record["failure_reason"] is None):
        raise ModelDataError("evaluation run record failure state is invalid")
    if record["status"] == "failed" and not isinstance(record["failure_reason"], str):
        raise ModelDataError("failed evaluation record lacks its failure reason")
    loads = record["collection_loads"]
    gates = record["hard_gates"]
    if (
        not isinstance(loads, dict)
        or set(loads) != _LOADS
        or any(type(value) is not int or value < 0 for value in loads.values())
    ):
        raise ModelDataError("evaluation collection load accounting is invalid")
    if (
        not isinstance(gates, dict)
        or set(gates) != _GATES
        or any(type(value) is not bool for value in gates.values())
    ):
        raise ModelDataError("evaluation hard gates are invalid")
    if (
        loads["shared_sealed_test"] != 0
        or not gates["sealed_test_never_loaded"]
        or not gates["evaluation_only_no_retraining_or_selection"]
        or not gates["no_network_requests"]
    ):
        raise ModelDataError("evaluation boundary gate is invalid")
    if record["status"] == "passed":
        if loads != {
            "random_native_validation": 1,
            "family_aware_native_validation": 1,
            "shared_validation": 1,
            "shared_sealed_test": 0,
        } or not all(gates.values()):
            raise ModelDataError(
                "passed evaluation load accounting or gates are invalid"
            )
    else:
        if gates["shared_validation_loaded_once"] and loads["shared_validation"] != 1:
            raise ModelDataError(
                "failed evaluation record shared-load gate is inconsistent"
            )
        if gates["twelve_principal_records"] and loads != {
            "random_native_validation": 1,
            "family_aware_native_validation": 1,
            "shared_validation": 1,
            "shared_sealed_test": 0,
        }:
            raise ModelDataError(
                "failed evaluation record twelve-record gate is inconsistent"
            )


def _validate_result(value: dict[str, object], plan: EvaluationPlan) -> None:
    expected = {
        "schema_version",
        "scope",
        "contract_identifier",
        "evaluation_id",
        "metric_dtype",
        "provenance",
        "records",
        "hypothesis",
    }
    if set(value) != expected or (
        value["schema_version"] != 1
        or value["scope"] != "week_02_bigram_evaluation_results"
        or value["contract_identifier"] != plan.config.contract_identifier
        or value["evaluation_id"] != plan.evaluation_id
        or value["metric_dtype"] != "float64"
        or value["provenance"] != provenance(plan.config, plan.config_sha256)
        or not isinstance(value["records"], list)
        or not isinstance(value["hypothesis"], dict)
    ):
        raise ModelDataError("evaluation result schema is invalid")
    records = value["records"]
    if any(not isinstance(item, dict) for item in records):
        raise ModelDataError("evaluation result record is invalid")
    validate_records(records, plan.config)  # type: ignore[arg-type]
    if value["hypothesis"] != hypothesis(records):  # type: ignore[arg-type]
        raise ModelDataError("evaluation optimism arithmetic is inconsistent")


def _validate_registry(
    registry: dict[str, object], candidate: Path, plan: EvaluationPlan
) -> None:
    if set(registry) != {"schema_version", "scope", "evaluation_id", "artifacts"} or (
        registry["schema_version"] != 1
        or registry["scope"] != "week_02_bigram_evaluation_candidate_registry"
        or registry["evaluation_id"] != plan.evaluation_id
        or not isinstance(registry["artifacts"], dict)
        or set(registry["artifacts"]) != {"evaluation.json", "run_record.json"}
    ):
        raise ModelDataError("evaluation registry schema is invalid")
    for name, entry in registry["artifacts"].items():
        if not isinstance(entry, dict) or set(entry) != {"byte_size", "sha256"}:
            raise ModelDataError("evaluation registry artifact schema is invalid")
        content = (candidate / name).read_bytes()
        if (
            type(entry["byte_size"]) is not int
            or entry["byte_size"] != len(content)
            or not isinstance(entry["sha256"], str)
            or _SHA256.fullmatch(entry["sha256"]) is None
            or entry["sha256"] != hashlib.sha256(content).hexdigest()
        ):
            raise ModelDataError("evaluation registry artifact checksum drifted")


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
