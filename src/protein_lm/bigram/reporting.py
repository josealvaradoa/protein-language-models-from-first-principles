"""Aggregate-only, atomic evidence publication for the Week 2 stream audit."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path

from protein_lm.bigram.config import BigramStreamConfig, config_sha256
from protein_lm.bigram.stream import ArmStreamAudit
from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.data.model_data.promotion import PROMOTION_CONTRACT


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")


def report_payload(
    *,
    config_path: Path,
    config: BigramStreamConfig,
    audits: dict[str, ArmStreamAudit],
    code_revision: str,
    runtime_seconds: float,
) -> dict[str, object]:
    """Build public aggregate evidence with no membership or sequence material."""

    if not isinstance(audits, dict) or set(audits) != set(config.training_collections):
        raise ModelDataError(
            "stream audit does not cover exactly the approved collections"
        )
    if (
        not isinstance(code_revision, str)
        or _GIT_REVISION.fullmatch(code_revision) is None
        or type(runtime_seconds) not in (int, float)
        or not math.isfinite(runtime_seconds)
        or runtime_seconds < 0
    ):
        raise ModelDataError("stream audit provenance is invalid")
    _require_batch_arithmetic(config)
    return {
        "schema_version": 1,
        "scope": "week_02_bigram_training_stream",
        "contract_identifier": config.contract_identifier,
        "status": "passed",
        "hard_gates": {
            "exact_pair_budget": True,
            "batch_arithmetic": True,
            "approved_arm_namespaces": True,
            "aggregate_role_counts": True,
            "stream_hash_format": True,
            "protein_consumption_accounting": True,
        },
        "configuration": {
            "sha256": config_sha256(config_path),
            "base_seed": config.base_seed,
            "hash_algorithm": config.hash_algorithm,
            "stream_hash_domain": config.stream_hash_domain,
            "prediction_pair_budget": config.prediction_pair_budget,
            "batch_size": config.batch_size,
            "full_batches": config.full_batches,
            "final_partial_batch_pairs": config.final_partial_batch_pairs,
            "total_optimizer_steps": config.full_batches + 1,
            "context_roles": list(config.context_roles),
            "target_roles": list(config.target_roles),
        },
        "source": {
            "model_data_contract_identifier": config.model_data_contract_identifier,
            "candidate_revision": PROMOTION_CONTRACT.candidate_revision,
            "registry_relative_path": config.model_data_registry_relative_path,
            "registry_sha256": config.model_data_registry_sha256,
            "model_data_config_relative_path": config.model_data_config_relative_path,
            "model_data_config_sha256": config.model_data_config_sha256,
            "training_manifest_identities": {
                item.filename: {
                    "row_count": item.row_count,
                    "byte_size": item.byte_size,
                    "sha256": item.sha256,
                }
                for item in PROMOTION_CONTRACT.artifacts
                if item.filename in {"random_arm.tsv", "family_aware_arm.tsv"}
            },
        },
        "arms": {
            collection: _arm_payload(
                audits[collection],
                namespace=namespace,
                pair_budget=config.prediction_pair_budget,
            )
            for collection, namespace in zip(
                config.training_collections, config.training_namespaces, strict=True
            )
        },
        "code_revision": code_revision,
        "runtime_seconds": round(runtime_seconds, 3),
        "network_requests_made": 0,
    }


def write_evidence(
    paths: tuple[Path, Path, Path], payload: dict[str, object]
) -> None:
    """Install JSON, Markdown, and JSON checksum together without overwrite."""

    if any(path.exists() for path in paths):
        raise ModelDataError("bigram stream evidence already exists")
    json_path, markdown_path, sha_path = paths
    if len({path.parent for path in paths}) != 1:
        raise ModelDataError(
            "bigram stream evidence paths must share one directory"
        )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    markdown_content = render_markdown(payload)
    sha_content = (
        f"{hashlib.sha256(json_content.encode()).hexdigest()}  {json_path.name}\n"
    )
    try:
        with tempfile.TemporaryDirectory(
            prefix=".bigram-stream-", dir=json_path.parent
        ) as temporary:
            stage = Path(temporary)
            staged = (
                stage / json_path.name,
                stage / markdown_path.name,
                stage / sha_path.name,
            )
            staged[0].write_text(json_content, encoding="utf-8")
            staged[1].write_text(markdown_content, encoding="utf-8")
            staged[2].write_text(sha_content, encoding="utf-8")
            for source, destination in zip(staged, paths, strict=True):
                os.replace(source, destination)
    except OSError as error:
        raise ModelDataError(
            f"could not install bigram stream evidence: {error}"
        ) from error


def render_markdown(payload: dict[str, object]) -> str:
    """Render a concise aggregate-only human report from the JSON payload."""

    arms = payload["arms"]
    assert isinstance(arms, dict)
    lines = [
        "# Week 2 Bigram Training Streams v1",
        "",
        "Aggregate-only audit evidence. It excludes accessions, sequences, family identifiers, and membership rows.",
        "",
        f"Code revision: `{payload['code_revision']}`",
        f"Network requests made: `{payload['network_requests_made']}`",
        "",
        "| Collection | Stream SHA-256 | Pairs | Started | Completed | Final partial |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for collection, arm in arms.items():
        assert isinstance(arm, dict)
        lines.append(
            f"| {collection} | `{arm['stream_sha256']}` | {arm['pairs_emitted']} | "
            f"{arm['proteins_started']} | {arm['proteins_completed']} | "
            f"{str(arm['final_protein_partial']).lower()} |"
        )
    lines.extend(
        ("", "The audit made one loader pass per arm and constructed no cross-protein transitions.")
    )
    return "\n".join(lines) + "\n"


def _arm_payload(
    audit: ArmStreamAudit, *, namespace: str, pair_budget: int
) -> dict[str, object]:
    if not isinstance(audit, ArmStreamAudit):
        raise ModelDataError("stream audit record is invalid")
    if (
        audit.namespace != namespace
        or type(audit.pairs_emitted) is not int
        or audit.pairs_emitted != pair_budget
        or not _counts_are_valid(audit.context_counts, pair_budget)
        or not _counts_are_valid(audit.target_counts, pair_budget)
        or not isinstance(audit.stream_sha256, str)
        or _SHA256.fullmatch(audit.stream_sha256) is None
        or not _protein_accounting_is_valid(audit)
    ):
        raise ModelDataError("stream audit aggregate counts do not satisfy the frozen budget")
    return {
        "namespace": audit.namespace,
        "pairs_emitted": audit.pairs_emitted,
        "proteins_started": audit.proteins_started,
        "proteins_completed": audit.proteins_completed,
        "final_protein_partial": audit.final_protein_partial,
        "source_passes": 1,
        "context_counts": list(audit.context_counts),
        "target_counts": list(audit.target_counts),
        "stream_sha256": audit.stream_sha256,
    }


def _require_batch_arithmetic(config: BigramStreamConfig) -> None:
    if (
        config.batch_size * config.full_batches
        + config.final_partial_batch_pairs
        != config.prediction_pair_budget
    ):
        raise ModelDataError("stream audit batch arithmetic is invalid")


def _counts_are_valid(counts: tuple[int, ...], pair_budget: int) -> bool:
    return (
        isinstance(counts, tuple)
        and len(counts) == 21
        and all(type(count) is int and count >= 0 for count in counts)
        and sum(counts) == pair_budget
    )


def _protein_accounting_is_valid(audit: ArmStreamAudit) -> bool:
    if (
        type(audit.proteins_started) is not int
        or type(audit.proteins_completed) is not int
        or type(audit.final_protein_partial) is not bool
        or audit.proteins_started <= 0
        or audit.proteins_completed < 0
        or audit.proteins_completed > audit.proteins_started
    ):
        return False
    if audit.final_protein_partial:
        return audit.proteins_completed == audit.proteins_started - 1
    return audit.proteins_completed == audit.proteins_started
