"""Atomic no-overwrite writing for public Week 3 aggregate evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.mlp.publication_render import render_markdown


def write_evidence(paths: tuple[Path, Path, Path], payload: dict[str, object]) -> None:
    if any(path.exists() for path in paths):
        raise ModelDataError("public Week 3 report already exists")
    parent = paths[0].parent
    if any(path.parent != parent for path in paths) or parent.exists() and not parent.is_dir():
        raise ModelDataError("public Week 3 output paths are invalid")
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / ".mlp_evaluation_v1.tmp"
    if temporary.exists():
        raise ModelDataError("public Week 3 temporary output already exists")
    temporary.mkdir()
    try:
        json_bytes = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        markdown_bytes = render_markdown(payload).encode()
        (temporary / paths[0].name).write_bytes(json_bytes)
        (temporary / paths[1].name).write_bytes(markdown_bytes)
        checksum = "\n".join(
            f"{hashlib.sha256((temporary / path.name).read_bytes()).hexdigest()}  {path.name}"
            for path in paths[:2]
        ) + "\n"
        (temporary / paths[2].name).write_text(checksum, encoding="utf-8")
        installed: list[Path] = []
        for path in paths:
            os.replace(temporary / path.name, path)
            installed.append(path)
        temporary.rmdir()
    except Exception:
        for path in locals().get("installed", []):
            if path.exists():
                path.unlink()
        if temporary.exists():
            for child in temporary.glob("*"):
                child.unlink()
            temporary.rmdir()
        raise
