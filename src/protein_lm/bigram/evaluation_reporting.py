"""Immutable JSON evidence for a single Week 2 bigram evaluation candidate."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from protein_lm.bigram.evaluation_contract import EvaluationConfig
from protein_lm.data.model_data.contracts import ModelDataError


def write_new_json(path: Path, payload: dict[str, object]) -> None:
    """Install readable JSON once.  Existing evidence is never overwritten."""

    if path.exists():
        raise ModelDataError("evaluation artifact destination already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(encoded)
        staged = Path(handle.name)
    try:
        os.link(staged, path)
    except OSError as error:
        raise ModelDataError(
            f"could not install evaluation artifact: {error}"
        ) from error
    finally:
        staged.unlink(missing_ok=True)


def write_run_record(path: Path, payload: dict[str, object]) -> None:
    """Atomically replace the one mutable state record during this fresh run."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(encoded)
        staged = Path(handle.name)
    try:
        os.replace(staged, path)
    except OSError as error:
        raise ModelDataError(
            f"could not replace evaluation run record: {error}"
        ) from error
    finally:
        staged.unlink(missing_ok=True)


def run_record(
    *,
    config: EvaluationConfig,
    evaluation_id: str,
    revision: str,
    status: str,
    started: float,
    failure_reason: str | None,
    configuration_sha256: str,
    collection_loads: dict[str, int],
    hard_gates: dict[str, bool],
) -> dict[str, object]:
    """Create a truthful terminal or in-progress execution record."""

    if status not in {"running", "passed", "failed"}:
        raise ModelDataError("evaluation run status is invalid")
    return {
        "schema_version": 1,
        "scope": "week_02_bigram_evaluation_candidate",
        "contract_identifier": config.contract_identifier,
        "evaluation_id": evaluation_id,
        "status": status,
        "code_revision": revision,
        "configuration_sha256": configuration_sha256,
        "model_candidate": {
            "candidate_id": config.model_candidate_id,
            "relative_path": config.model_candidate_relative_path,
            "candidate_registry_sha256": config.model_candidate_registry_sha256,
            "run_record_sha256": config.model_candidate_run_record_sha256,
        },
        "model_data_registry": {
            "relative_path": config.model_data_registry_relative_path,
            "sha256": config.model_data_registry_sha256,
        },
        "collection_loads": collection_loads,
        "hard_gates": hard_gates,
        "network_requests_made": 0,
        "runtime_seconds": time.perf_counter() - started,
        "failure_reason": failure_reason,
    }


def registry_payload(destination: Path, evaluation_id: str) -> dict[str, object]:
    """Checksum the two terminal artifacts after both have been written."""

    names = ("evaluation.json", "run_record.json")
    artifacts: dict[str, object] = {}
    for name in names:
        content = (destination / name).read_bytes()
        artifacts[name] = {
            "byte_size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    return {
        "schema_version": 1,
        "scope": "week_02_bigram_evaluation_candidate_registry",
        "evaluation_id": evaluation_id,
        "artifacts": artifacts,
    }
