"""Aggregate-only candidate records and artifact registry publication."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from protein_lm.bigram.stream import ArmStreamAudit

if TYPE_CHECKING:
    from protein_lm.bigram.candidate import AuditedArm, CandidatePlan


def arm_record(audited: AuditedArm, observed: ArmStreamAudit, losses, state) -> dict[str, object]:
    """Render one arm's losses and aggregate accounting without source membership."""

    return {
        "stream": {
            "namespace": observed.namespace,
            "stream_sha256": observed.stream_sha256,
            "context_counts": list(observed.context_counts),
            "target_counts": list(observed.target_counts),
            "pairs_emitted": observed.pairs_emitted,
            "proteins_started": observed.proteins_started,
            "proteins_completed": observed.proteins_completed,
            "final_protein_partial": observed.final_protein_partial,
        },
        "losses": [float(loss) for loss in losses],
        "accounting": {
            "pairs_seen": state.pairs_seen,
            "optimizer_steps": state.optimizer_steps,
            "count_bigram_total": int(state.count_bigram_counts.sum().item()),
        },
        "hard_gates": {
            "one_stream_pass": True,
            "exact_audited_stream": observed.stream_sha256 == audited.stream_sha256,
            "aggregate_role_counts": (
                observed.context_counts == audited.context_counts
                and observed.target_counts == audited.target_counts
            ),
            "exact_pair_budget": observed.pairs_emitted == state.pairs_seen,
            "batch_and_optimizer_accounting": state.optimizer_steps == len(losses),
        },
    }


def record_payload(
    plan: CandidatePlan,
    code_revision: str,
    status: str,
    arms: dict[str, object],
    started: float,
    failure_reason: str | None,
) -> dict[str, object]:
    """Build the sole run record, including a truthful failure state when needed."""

    return {
        "schema_version": 1,
        "scope": "week_02_bigram_model_candidate",
        "contract_identifier": plan.training_config.contract_identifier,
        "candidate_id": plan.candidate_id,
        "status": status,
        "hard_gates": {
            "only_promoted_training_collections": True,
            "one_pass_per_arm": status == "passed",
            "exact_audited_streams": status == "passed",
            "six_logical_models": status == "passed",
            "dual_serialization": status == "passed",
        },
        "training": {
            "config_sha256": plan.training_config_sha256,
            "stream_config_sha256": plan.training_config.stream_config_sha256,
            "stream_report_sha256": plan.stream_report_sha256,
            "base_seed": plan.training_config.base_seed,
            "prediction_pair_budget": plan.training_config.prediction_pair_budget,
            "batch_size": plan.training_config.batch_size,
            "total_optimizer_steps": plan.training_config.total_optimizer_steps,
            "model_types": list(plan.training_config.model_types),
        },
        "source_identity": plan.source_identity,
        "code_revision": code_revision,
        "arms": arms,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "network_requests_made": 0,
        "failure_reason": failure_reason,
    }


def registry_payload(destination: Path, plan: CandidatePlan, names: tuple[str, ...]) -> dict[str, object]:
    """Hash exactly the expected twelve model files for read-only later validation."""

    artifacts = {}
    for name in names:
        content = (destination / name).read_bytes()
        arm, suffix = name.split("__", maxsplit=1)
        model_type, format_name = suffix.rsplit(".", maxsplit=1)
        artifacts[name] = {
            "arm": arm,
            "model_type": model_type,
            "format": format_name,
            "byte_size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    return {
        "schema_version": 1,
        "scope": "week_02_bigram_model_candidate_registry",
        "candidate_id": plan.candidate_id,
        "logical_model_count": 6,
        "serialization_file_count": 12,
        "artifacts": artifacts,
    }


def write_new_json(path: Path, payload: dict[str, object]) -> None:
    """Install a final candidate artifact only when its name is unused."""

    if path.exists():
        from protein_lm.data.model_data.contracts import ModelDataError

        raise ModelDataError("candidate artifact destination already exists")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_record(destination: Path, payload: dict[str, object]) -> None:
    """Replace the status record atomically so a failure reason is never half-written."""

    temporary = destination / ".run_record.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination / "run_record.json")
