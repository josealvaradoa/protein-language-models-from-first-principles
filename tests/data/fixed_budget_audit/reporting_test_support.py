"""Independent reporting fixtures shared by focused tests."""

from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).parents[3]
SOURCE_CONFIG = PROJECT_ROOT / "experiments/week_01/diagnostic_similarity_audit.toml"
GOLDEN_MARKDOWN = (
    PROJECT_ROOT / "tests/characterization/fixed_budget_audit/goldens/a004_report.md"
)
_GOLDEN_SUPPORT = runpy.run_path(
    str(PROJECT_ROOT / "tests/characterization/fixed_budget_audit/golden_support.py")
)
FINGERPRINT = cast(str, _GOLDEN_SUPPORT["FINGERPRINT"])


def independent_report_payload(tmp_path: Path) -> dict[str, object]:
    """Build the hand-reviewed payload without production reporting code."""

    golden_audit = _GOLDEN_SUPPORT["GoldenAudit"]
    audit = golden_audit(
        tmp_path / "independent-audit",
        SOURCE_CONFIG.read_bytes(),
    )
    payload = audit._report_payload()
    assert isinstance(payload, dict)
    return payload


def identity(content: bytes) -> dict[str, object]:
    """Compute an expected file identity without production helpers."""

    return {
        "byte_size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def json_bytes(payload: object) -> bytes:
    """Serialize expected checkpoint bytes without production helpers."""

    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
