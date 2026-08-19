"""Atomic aggregate-only public report installation and checksum writing."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from protein_lm.bigram.public_report_render import render_markdown
from protein_lm.data.model_data.contracts import ModelDataError


def write_evidence(paths: tuple[Path, Path, Path], payload: dict[str, object]) -> None:
    """Install JSON, Markdown, and two-file checksum sidecar without overwrite."""

    if any(path.exists() for path in paths):
        raise ModelDataError("public evaluation report already exists")
    if len({path.parent for path in paths}) != 1:
        raise ModelDataError("public evaluation report paths must share one directory")
    json_path, markdown_path, sha_path = paths
    json_content = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    markdown_content = render_markdown(payload)
    checksum_content = "".join(
        f"{hashlib.sha256(content.encode('utf-8')).hexdigest()}  {path.name}\n"
        for path, content in (
            (json_path, json_content),
            (markdown_path, markdown_content),
        )
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    installed: list[Path] = []
    try:
        with tempfile.TemporaryDirectory(
            prefix=".bigram-public-", dir=json_path.parent
        ) as temporary:
            stage = Path(temporary)
            staged = (
                stage / json_path.name,
                stage / markdown_path.name,
                stage / sha_path.name,
            )
            for path, content in zip(
                staged, (json_content, markdown_content, checksum_content), strict=True
            ):
                path.write_text(content, encoding="utf-8")
            for source, destination in zip(staged, paths, strict=True):
                os.link(source, destination)
                installed.append(destination)
    except OSError as error:
        for path in installed:
            path.unlink(missing_ok=True)
        raise ModelDataError(
            f"could not install public evaluation report: {error}"
        ) from error
