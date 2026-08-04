"""Exact Markdown rendering contracts for the A-004 report."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from protein_lm.data.fixed_budget_audit.reporting import render_markdown_report
from reporting_test_support import (
    GOLDEN_MARKDOWN,
    independent_report_payload,
)


def test_markdown_matches_independent_golden_bytes(tmp_path: Path) -> None:
    payload = _independent_payload(tmp_path)

    rendered = render_markdown_report(payload).encode("utf-8")

    assert rendered == GOLDEN_MARKDOWN.read_bytes()
    assert rendered.endswith(b"\n")
    assert not rendered.endswith(b"\n\n")


def test_markdown_preserves_payload_record_order(tmp_path: Path) -> None:
    payload = deepcopy(_independent_payload(tmp_path))
    payload["partition_results"].reverse()
    payload["tracks"].reverse()

    lines = render_markdown_report(payload).splitlines()
    held_header = lines.index(
        "| Strategy | Partition | Result | Prohibited queries | Denominator | Rate | Prohibited pairs |"
    )
    cap_header = lines.index(
        "| Strategy | Partition | Pass | Source | Cap | Query scope | Prohibited queries | Denominator | Rate | Prohibited pairs |"
    )

    assert lines[held_header + 2].startswith("| random | validation |")
    assert lines[cap_header + 2].startswith("| random | validation | residual |")


def _independent_payload(tmp_path: Path) -> dict[str, object]:
    return independent_report_payload(tmp_path)
