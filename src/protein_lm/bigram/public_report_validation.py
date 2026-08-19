"""Read-only validator for the three public Week 2 bigram report artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from protein_lm.bigram.evaluation_results import hypothesis, validate_records
from protein_lm.bigram.public_report import (
    derived_comparisons,
    reject_forbidden_keys,
    render_markdown,
)
from protein_lm.bigram.public_report_publication import PublicReportPlan, preflight
from protein_lm.data.model_data.contracts import ModelDataError


_REVISION = re.compile(r"^[0-9a-f]{40}$")


def validate_public_report(root: Path) -> dict[str, object]:
    """Verify exact public bytes, source lineage, arithmetic, and Markdown rendering."""

    plan = preflight(root)
    json_path, markdown_path, checksum_path = plan.output_paths
    _validate_inventory(plan)
    payload = _load_json(json_path, "public evaluation report")
    _validate_payload(payload, plan)
    expected_markdown = render_markdown(payload)
    try:
        found_markdown = markdown_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ModelDataError("public evaluation Markdown is unavailable") from error
    if found_markdown != expected_markdown:
        raise ModelDataError(
            "public evaluation Markdown does not match deterministic renderer"
        )
    _validate_checksums(json_path, markdown_path, checksum_path)
    return {"status": "passed", "principal_record_count": 12}


def _validate_inventory(plan: PublicReportPlan) -> None:
    parent = plan.output_paths[0].parent
    expected = {path.name for path in plan.output_paths}
    stem = plan.output_paths[0].stem
    found = (
        {path.name for path in parent.iterdir() if path.name.startswith(f"{stem}.")}
        if parent.is_dir()
        else set()
    )
    if found != expected:
        raise ModelDataError("public evaluation report inventory is incomplete")


def _validate_payload(payload: dict[str, object], plan: PublicReportPlan) -> None:
    expected = {
        "schema_version",
        "scope",
        "contract_identifier",
        "status",
        "hard_gates",
        "publication_configuration_sha256",
        "publication_code_revision",
        "source",
        "evaluation_provenance",
        "evaluation_runtime",
        "records",
        "hypothesis",
        "derived_comparisons",
        "network_requests_made",
    }
    if set(payload) != expected or (
        payload["schema_version"] != 1
        or payload["scope"] != "week_02_bigram_evaluation_public_report"
        or payload["contract_identifier"] != plan.config.contract_identifier
        or payload["status"] != "passed"
        or payload["network_requests_made"] != 0
        or not isinstance(payload["publication_code_revision"], str)
        or _REVISION.fullmatch(payload["publication_code_revision"]) is None
    ):
        raise ModelDataError("public evaluation report schema is invalid")
    if payload["hard_gates"] != {
        "validated_source_evaluation": True,
        "exact_twelve_records": True,
        "aggregate_only": True,
        "no_network_requests": True,
        "sealed_test_inaccessible": True,
    }:
        raise ModelDataError("public evaluation report hard gates are invalid")
    if (
        payload["publication_configuration_sha256"]
        != hashlib.sha256(plan.config_path.read_bytes()).hexdigest()
    ):
        raise ModelDataError(
            "public evaluation report configuration provenance drifted"
        )
    if payload["source"] != _source_identity(plan):
        raise ModelDataError("public evaluation report source provenance drifted")
    source = _load_json(
        plan.evaluation_plan.destination / "evaluation.json", "source evaluation"
    )
    run = _load_json(
        plan.evaluation_plan.destination / "run_record.json",
        "source evaluation run record",
    )
    if payload["evaluation_provenance"] != source.get("provenance") or payload[
        "evaluation_runtime"
    ] != {
        "runtime_seconds": run.get("runtime_seconds"),
        "collection_loads": run.get("collection_loads"),
    }:
        raise ModelDataError("public evaluation report copied provenance drifted")
    records = payload["records"]
    if not isinstance(records, list) or not all(
        isinstance(item, dict) for item in records
    ):
        raise ModelDataError("public evaluation report records are invalid")
    validate_records(records, plan.evaluation_plan.config)
    if payload["hypothesis"] != hypothesis(records) or payload[
        "derived_comparisons"
    ] != derived_comparisons(records):
        raise ModelDataError(
            "public evaluation report derived arithmetic is inconsistent"
        )
    reject_forbidden_keys(payload)


def _source_identity(plan: PublicReportPlan) -> dict[str, object]:
    config = plan.config
    return {
        "evaluation_id": config.source_evaluation_id,
        "relative_path": config.source_evaluation_relative_path,
        "evaluation_sha256": config.source_evaluation_sha256,
        "run_record_sha256": config.source_run_record_sha256,
        "registry_sha256": config.source_registry_sha256,
        "code_revision": config.source_evaluation_code_revision,
        "evaluation_configuration_sha256": config.source_evaluation_config_sha256,
    }


def _validate_checksums(
    json_path: Path, markdown_path: Path, checksum_path: Path
) -> None:
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ModelDataError("public evaluation checksum is unavailable") from error
    expected = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in (json_path, markdown_path)
    ]
    if lines != expected:
        raise ModelDataError("public evaluation checksum drifted")


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
