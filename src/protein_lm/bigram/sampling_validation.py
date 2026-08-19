"""Read-only validation for published Week 2 sampling diagnostic evidence."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from protein_lm.bigram.sampling_publication import (
    SamplingPlan,
    preflight,
    regenerate_payload,
)
from protein_lm.bigram.sampling_render import render_markdown
from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.tokenization import CANONICAL_AMINO_ACIDS


_REVISION = re.compile(r"^[0-9a-f]{40}$")


def validate_sampling_diagnostic(root: Path) -> dict[str, object]:
    """Check exact output bytes, contract, and independently regenerated samples."""
    plan = preflight(root)
    _validate_inventory(plan)
    json_path, markdown_path, checksum_path = plan.output_paths
    payload = _load_json(json_path)
    _validate_schema(payload, plan)
    revision = payload["publication_code_revision"]
    assert isinstance(revision, str)
    if payload != regenerate_payload(plan, revision):
        raise ModelDataError("sampling diagnostic samples or provenance drifted")
    try:
        markdown = markdown_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ModelDataError("sampling diagnostic Markdown is unavailable") from error
    if markdown != render_markdown(payload):
        raise ModelDataError(
            "sampling diagnostic Markdown does not match deterministic renderer"
        )
    _validate_checksums(json_path, markdown_path, checksum_path)
    return {"status": "passed", "sample_count": 20}


def _validate_schema(payload: dict[str, object], plan: SamplingPlan) -> None:
    expected = {
        "schema_version",
        "scope",
        "contract_identifier",
        "status",
        "hard_gates",
        "sampling_configuration_sha256",
        "publication_code_revision",
        "runtime",
        "source",
        "sampling",
        "samples",
        "network_requests_made",
    }
    if (
        set(payload) != expected
        or payload["schema_version"] != 1
        or payload["scope"] != "week_02_bigram_sampling_diagnostic"
        or payload["contract_identifier"] != plan.config.contract_identifier
        or payload["status"] != "passed"
        or payload["network_requests_made"] != 0
    ):
        raise ModelDataError("sampling diagnostic schema is invalid")
    if (
        payload["sampling_configuration_sha256"]
        != hashlib.sha256(plan.config_path.read_bytes()).hexdigest()
        or not isinstance(payload["publication_code_revision"], str)
        or _REVISION.fullmatch(payload["publication_code_revision"]) is None
    ):
        raise ModelDataError("sampling diagnostic provenance is invalid")
    if payload["hard_gates"] != {
        "validated_passed_candidate": True,
        "two_neural_artifacts_only": True,
        "twenty_samples_preserved": True,
        "synthetic_nonfunctional_educational_output": True,
        "no_selection_or_biological_claims": True,
        "no_network_requests": True,
    }:
        raise ModelDataError("sampling diagnostic hard gates are invalid")
    if payload["runtime"] != {
        "uv_lock_sha256": plan.config.uv_lock_sha256,
        "torch_version": plan.config.torch_version,
    }:
        raise ModelDataError("sampling diagnostic runtime provenance is invalid")
    samples = payload["samples"]
    if (
        not isinstance(samples, list)
        or len(samples) != 20
        or not all(isinstance(sample, dict) for sample in samples)
    ):
        raise ModelDataError("sampling diagnostic sample inventory is invalid")
    for arm, namespace in zip(plan.config.arms, plan.config.namespaces, strict=True):
        arm_samples = [sample for sample in samples if sample.get("model_arm") == arm]
        if len(arm_samples) != 10 or [
            sample.get("sample_index") for sample in arm_samples
        ] != list(range(10)):
            raise ModelDataError("sampling diagnostic arm sample inventory is invalid")
        for sample in arm_samples:
            sequence = sample.get("sequence")
            if (
                sample.get("namespace") != namespace
                or not isinstance(sequence, str)
                or any(residue not in CANONICAL_AMINO_ACIDS for residue in sequence)
                or sample.get("residue_length") != len(sequence)
                or not 0 <= len(sequence) <= plan.config.max_residues
                or sample.get("termination_reason") not in {"eos", "max_residues"}
                or type(sample.get("seed")) is not int
                or sample["seed"] < 0
                or (
                    sample.get("termination_reason") == "max_residues"
                    and len(sequence) != plan.config.max_residues
                )
                or (
                    sample.get("termination_reason") == "eos"
                    and len(sequence) >= plan.config.max_residues
                )
            ):
                raise ModelDataError("sampling diagnostic sample record is invalid")


def _validate_inventory(plan: SamplingPlan) -> None:
    directory = plan.output_paths[0].parent
    expected = {path.name for path in plan.output_paths}
    found = (
        {
            path.name
            for path in directory.iterdir()
            if path.name.startswith("bigram_sampling_v1.")
        }
        if directory.is_dir()
        else set()
    )
    if found != expected:
        raise ModelDataError("sampling diagnostic inventory is incomplete")


def _validate_checksums(
    json_path: Path, markdown_path: Path, checksum_path: Path
) -> None:
    try:
        found = checksum_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ModelDataError("sampling diagnostic checksum is unavailable") from error
    expected = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in (json_path, markdown_path)
    ]
    if found != expected:
        raise ModelDataError("sampling diagnostic checksum drifted")


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("nonfinite")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ModelDataError("sampling diagnostic JSON is malformed") from error
    if not isinstance(value, dict):
        raise ModelDataError("sampling diagnostic JSON must be an object")
    return value
