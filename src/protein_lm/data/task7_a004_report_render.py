"""Render a concise Markdown view of a validated A-004 report."""

from __future__ import annotations

from collections.abc import Mapping

from protein_lm.data.task7_a004_report_payload import COMMON_RESULT, STAGED_RESULT


def render_markdown_report(payload: Mapping[str, object]) -> str:
    """Render held-out, per-cap, sensitivity, and limitation tables."""

    lines = [
        "# A-004 read-only fixed-budget audit",
        "",
        "The common result covers every query through cap 10000. The staged result adds cap 100000 only for queries whose complete rows changed between caps 1000 and 10000.",
        "",
        "## Held-out results",
        "",
        "| Strategy | Partition | Result | Prohibited queries | Denominator | Rate | Prohibited pairs |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for partition in payload["partition_results"]:  # type: ignore[index]
        for name in (COMMON_RESULT, STAGED_RESULT):
            result = partition[name]
            lines.append(
                "| {strategy} | {partition} | `{name}` | {queries} | {denominator} | {percent}% | {pairs} |".format(
                    strategy=partition["strategy"],
                    partition=partition["partition"],
                    name=name,
                    queries=result["prohibited_queries"],
                    denominator=result["denominator"],
                    percent=result["rate"]["percent"],
                    pairs=result["prohibited_pairs"],
                )
            )
    _append_cap_table(lines, payload)
    _append_sensitivity_table(lines, payload)
    _append_staged_additions(lines, payload)
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in payload["limitations"])  # type: ignore[index]
    lines.append("")
    return "\n".join(lines)


def _append_cap_table(lines: list[str], payload: Mapping[str, object]) -> None:
    lines.extend(
        [
            "",
            "## Per-cap evidence",
            "",
            "| Strategy | Partition | Pass | Source | Cap | Query scope | Prohibited queries | Denominator | Rate | Prohibited pairs |",
            "|---|---|---|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for track in payload["tracks"]:  # type: ignore[index]
        for cap, evidence in track["caps"].items():
            rate = evidence["prohibited_query_rate"]
            lines.append(
                "| {strategy} | {partition} | {pass_name} | `{source}` | {cap} | {scope} | {queries} | {denominator} | {percent}% | {pairs} |".format(
                    strategy=track["strategy"],
                    partition=track["partition"],
                    pass_name=track["pass_name"],
                    source=track["source_label"],
                    cap=cap,
                    scope=evidence["query_scope"],
                    queries=evidence["prohibited_queries"],
                    denominator=rate["denominator"],
                    percent=rate["percent"],
                    pairs=evidence["prohibited_pairs"],
                )
            )


def _append_sensitivity_table(lines: list[str], payload: Mapping[str, object]) -> None:
    lines.extend(
        [
            "",
            "## Cap sensitivity",
            "",
            "| Strategy | Partition | Pass | Transition | Compared queries | Complete row changes | Newly prohibited | No longer prohibited | Closest-category changes |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for track in payload["tracks"]:  # type: ignore[index]
        for change in track["cap_sensitivity"]:
            lines.append(
                "| {strategy} | {partition} | {pass_name} | {baseline} to {comparison} | {queries} | {rows} | {new} | {lost} | {closest} |".format(
                    strategy=track["strategy"],
                    partition=track["partition"],
                    pass_name=track["pass_name"],
                    baseline=change["baseline_cap"],
                    comparison=change["comparison_cap"],
                    queries=change["compared_queries"],
                    rows=change["complete_row_changes"],
                    new=change["newly_prohibited_queries"],
                    lost=change["no_longer_prohibited_queries"],
                    closest=change["closest_category_changes"],
                )
            )


def _append_staged_additions(lines: list[str], payload: Mapping[str, object]) -> None:
    lines.extend(
        [
            "",
            "## Staged additions",
            "",
            "| Strategy | Partition | Additional pairs | Newly prohibited queries |",
            "|---|---|---:|---:|",
        ]
    )
    for partition in payload["partition_results"]:  # type: ignore[index]
        additions = partition["staged_additions"]
        lines.append(
            "| {strategy} | {partition} | {pairs} | {queries} |".format(
                strategy=partition["strategy"],
                partition=partition["partition"],
                pairs=additions["additional_pairs"],
                queries=additions["newly_prohibited_queries"],
            )
        )
