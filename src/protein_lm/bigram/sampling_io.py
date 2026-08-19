"""Atomic installation for the three sampling-diagnostic evidence files."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from protein_lm.bigram.sampling_render import render_markdown
from protein_lm.data.model_data.contracts import ModelDataError


def write_evidence(paths: tuple[Path, Path, Path], payload: dict[str, object]) -> None:
    if any(path.exists() for path in paths):
        raise ModelDataError("sampling diagnostic report already exists")
    if len({path.parent for path in paths}) != 1:
        raise ModelDataError("sampling diagnostic paths must share one directory")
    json_path, markdown_path, checksum_path = paths
    contents = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        render_markdown(payload),
    )
    checksum = "".join(
        f"{hashlib.sha256(content.encode()).hexdigest()}  {path.name}\n"
        for path, content in zip(paths[:2], contents, strict=True)
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    installed: list[Path] = []
    try:
        with tempfile.TemporaryDirectory(
            prefix=".bigram-sampling-", dir=json_path.parent
        ) as temporary:
            staged = tuple(Path(temporary) / path.name for path in paths)
            for path, content in zip(staged, (*contents, checksum), strict=True):
                path.write_text(content, encoding="utf-8")
            for source, destination in zip(staged, paths, strict=True):
                os.link(source, destination)
                installed.append(destination)
    except OSError as error:
        for path in installed:
            path.unlink(missing_ok=True)
        raise ModelDataError(
            f"could not install sampling diagnostic: {error}"
        ) from error
