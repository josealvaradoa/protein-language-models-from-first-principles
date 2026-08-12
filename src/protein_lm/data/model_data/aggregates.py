"""Aggregate-only collection summaries for candidate provenance and readiness."""

from __future__ import annotations

from collections import defaultdict

from protein_lm.data.model_data.contracts import CandidateRecord, ModelDataConfig


def collection_aggregate(
    records: tuple[CandidateRecord, ...] | list[CandidateRecord],
    config: ModelDataConfig,
) -> dict[str, object]:
    """Summarize one collection without retaining any membership identifier."""

    bucket_records = defaultdict(int)
    bucket_predictions = defaultdict(int)
    for record in records:
        bucket_records[record.length_bucket] += 1
        bucket_predictions[record.length_bucket] += record.prediction_tokens
    return {
        "records": len(records),
        "residues": sum(record.biological_length for record in records),
        "prediction_tokens": sum(record.prediction_tokens for record in records),
        "unique_uniref50_groups": len({record.uniref50_group for record in records}),
        "length_buckets": {
            bucket.name: {
                "records": bucket_records[bucket.name],
                "prediction_tokens": bucket_predictions[bucket.name],
            }
            for bucket in config.length_buckets
        },
    }
