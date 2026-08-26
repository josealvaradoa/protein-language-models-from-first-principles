"""Read-only verification of the aggregate Week 3 public report."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.mlp.publication_orchestration import preflight, report_payload
from protein_lm.mlp.publication_render import render_markdown


_REVISION = re.compile(r"^[0-9a-f]{40}$")


def validate_public_report(root: Path) -> dict[str, object]:
    """Validate inventory, source bytes, arithmetic, rendering, and PCA summaries."""

    plan = preflight(root)
    json_path, markdown_path, checksum_path = plan.output_paths
    _validate_inventory(plan.output_paths)
    payload = _load_json(json_path)
    revision = payload.get("publication_code_revision")
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ModelDataError("public Week 3 report revision is invalid")
    expected = report_payload(root, plan, revision)
    if payload != expected:
        raise ModelDataError("public Week 3 report payload or arithmetic drifted")
    try:
        markdown = markdown_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ModelDataError("public Week 3 Markdown is unavailable") from error
    if markdown != render_markdown(payload):
        raise ModelDataError("public Week 3 Markdown rendering drifted")
    _validate_checksums(json_path, markdown_path, checksum_path)
    return {"status": "passed", "pca_seed_count": 3, "residue_pair_count": 190}


def _validate_inventory(paths: tuple[Path, Path, Path]) -> None:
    parent = paths[0].parent
    expected = {path.name for path in paths}
    found = {path.name for path in parent.glob("mlp_evaluation_v1.*")} if parent.is_dir() else set()
    if found != expected:
        raise ModelDataError("public Week 3 report inventory is incomplete")


def _validate_checksums(json_path: Path, markdown_path: Path, checksum_path: Path) -> None:
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ModelDataError("public Week 3 checksum is unavailable") from error
    expected = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in (json_path, markdown_path)
    ]
    if lines != expected:
        raise ModelDataError("public Week 3 checksum drifted")


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("nonfinite")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ModelDataError("public Week 3 JSON is malformed") from error
    if not isinstance(value, dict):
        raise ModelDataError("public Week 3 JSON must be an object")
    return value
