"""Aggregate-only readiness evidence that deliberately excludes sealed membership."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.data.model_data.validation import ValidationResult


def write_readiness_evidence(
    paths: tuple[Path, Path, Path], result: ValidationResult
) -> None:
    """Write public aggregate evidence once, refusing to overwrite prior evidence."""

    if any(path.exists() for path in paths):
        raise ModelDataError("readiness evidence already exists")
    json_path, markdown_path, sha_path = paths
    if len({path.parent for path in paths}) != 1:
        raise ModelDataError("readiness evidence paths must share one directory")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "scope": "week_02_model_data_readiness",
        "candidate_status": result.status,
        "hard_gates": result.gates,
        "random_arm_uniref50_crossing_groups": result.random_group_crossings,
        "artifacts": result.artifacts,
        "collection_aggregates": result.collection_aggregates,
        "mmseqs2_status": "not_run",
        "network_requests_made": 0,
    }
    json_content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    markdown_content = (
        "# Week 2 Model-Data Readiness v1\n\n"
        f"Candidate status: `{result.status}`\n\n"
        "The report contains aggregate evidence only. Sealed membership identifiers are excluded.\n\n"
        + "\n".join(
            f"- {name}: `{str(passed).lower()}`"
            for name, passed in sorted(result.gates.items())
        )
        + "\n\nMMseqs2 status: `not_run`\nNetwork requests made: `none`\n"
    )
    sha_content = (
        f"{hashlib.sha256(json_content.encode()).hexdigest()}  {json_path.name}\n"
    )
    try:
        with tempfile.TemporaryDirectory(
            prefix=".readiness-", dir=json_path.parent
        ) as temporary:
            staged = Path(temporary)
            staged_json = staged / json_path.name
            staged_markdown = staged / markdown_path.name
            staged_sha = staged / sha_path.name
            staged_json.write_text(json_content, encoding="utf-8")
            staged_markdown.write_text(markdown_content, encoding="utf-8")
            staged_sha.write_text(sha_content, encoding="utf-8")
            os.replace(staged_json, json_path)
            os.replace(staged_markdown, markdown_path)
            os.replace(staged_sha, sha_path)
    except OSError as error:
        raise ModelDataError(
            f"could not install readiness evidence: {error}"
        ) from error
