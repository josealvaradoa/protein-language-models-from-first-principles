"""Deterministic Markdown rendering for the Week 2 public evaluation report."""

from __future__ import annotations

from protein_lm.data.model_data.contracts import ModelDataError


def render_markdown(payload: dict[str, object]) -> str:
    """Render a factual, aggregate-only public view of the report JSON."""

    records = payload["records"]
    hypothesis_result = payload["hypothesis"]
    derived = payload["derived_comparisons"]
    assert (
        isinstance(records, list)
        and isinstance(hypothesis_result, dict)
        and isinstance(derived, dict)
    )
    lines = [
        "# Week 2 Bigram Evaluation v1",
        "",
        "Aggregate-only evaluation evidence. It excludes sequences, accessions, family identifiers, and membership rows.",
        "",
        "## Overall cross-entropy",
        "",
        "| Arm | Model | Collection | Cross-entropy | Accuracy | Tokens | Proteins |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        metrics = record["metrics"]
        assert isinstance(record, dict) and isinstance(metrics, dict)
        overall = metrics["overall"]
        assert isinstance(overall, dict)
        lines.append(
            f"| {record['model_arm']} | {record['model_type']} | {record['collection']} | {_number(overall['cross_entropy'])} | {_number(overall['accuracy'])} | {overall['token_count']} | {overall['protein_count']} |"
        )
    baseline = derived["week_03_baseline"]
    assert isinstance(baseline, dict)
    lines.extend(_hypothesis_lines(hypothesis_result))
    lines.extend(_bucket_lines(records))
    source = payload["source"]
    runtime = payload["evaluation_runtime"]
    revision = payload["publication_code_revision"]
    assert (
        isinstance(source, dict)
        and isinstance(runtime, dict)
        and isinstance(revision, str)
    )
    lines.extend(
        [
            "",
            "## Week 3 baseline",
            "",
            f"Family-aware neural bigram native CE / accuracy: {_number(baseline['native_cross_entropy'])} / {_number(baseline['native_accuracy'])}",
            f"Family-aware neural bigram shared CE / accuracy: {_number(baseline['shared_cross_entropy'])} / {_number(baseline['shared_accuracy'])}",
            f"Family-aware neural optimism gap: {_number(baseline['optimism_gap'])}",
            "",
            "## Source and checksum provenance",
            "",
            f"Source evaluation: `{source['relative_path']}`",
            f"Evaluation SHA-256: `{source['evaluation_sha256']}`",
            f"Run record SHA-256: `{source['run_record_sha256']}`",
            f"Registry SHA-256: `{source['registry_sha256']}`",
            f"Source evaluation code revision: `{source['code_revision']}`",
            f"Publication code revision: `{revision}`",
            f"Evaluation configuration SHA-256: `{source['evaluation_configuration_sha256']}`",
            f"Evaluation runtime seconds: {_number(runtime['runtime_seconds'])}",
            "",
            "## Limitations",
            "",
            "- This compares complete data-arm policies, not grouping-only causality.",
            "- This report makes no statistical-significance claim.",
            "- Adjacent-residue prediction is not biological understanding or function.",
            "- Shared validation is not a sealed test. The sealed test remained inaccessible.",
            "- This report makes no generated-sequence or function claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def _hypothesis_lines(hypothesis_result: dict[str, object]) -> list[str]:
    return [
        "",
        "## Hypothesis and optimism gaps",
        "",
        f"Random neural optimism gap: {_number(hypothesis_result['random_neural_optimism_gap'])}",
        f"Family-aware neural optimism gap: {_number(hypothesis_result['family_neural_optimism_gap'])}",
        f"Random minus family-aware gap: {_number(hypothesis_result['comparison'])}",
        f"Prospective hypothesis supported: {str(hypothesis_result['supports_hypothesis']).lower()}",
        "",
        "## Family-aware shared-validation length buckets",
        "",
        "| Length bucket | Unigram CE | Count CE | Neural CE | Tokens | Proteins |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]


def _bucket_lines(records: list[object]) -> list[str]:
    lines = []
    for bucket in ("32-127", "128-255", "256-511", "512-1023", "1024-2046"):
        buckets = [
            _family_shared_bucket(records, model, bucket)
            for model in ("unigram", "count_bigram", "neural_bigram")
        ]
        lines.append(
            f"| {bucket} | {_number(buckets[0]['cross_entropy'])} | {_number(buckets[1]['cross_entropy'])} | {_number(buckets[2]['cross_entropy'])} | {buckets[0]['token_count']} | {buckets[0]['protein_count']} |"
        )
    return lines


def _family_shared_bucket(
    records: list[object], model: str, bucket: str
) -> dict[str, object]:
    record = next(
        item
        for item in records
        if isinstance(item, dict)
        and item["model_arm"] == "family_aware_training"
        and item["model_type"] == model
        and item["collection"] == "shared_validation"
    )
    metrics = record["metrics"]
    assert isinstance(metrics, dict)
    by_bucket = metrics["length_buckets"]
    assert isinstance(by_bucket, dict) and isinstance(by_bucket[bucket], dict)
    return by_bucket[bucket]


def _number(value: object) -> str:
    if type(value) not in (int, float):
        raise ModelDataError("public report numeric value is invalid")
    return f"{value:.6f}"
