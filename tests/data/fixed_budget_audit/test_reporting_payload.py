"""Focused report payload and error-boundary contracts."""

from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import protein_lm.data.fixed_budget_audit.reporting as reporting_module
import pytest
from reporting_test_support import (
    FINGERPRINT,
    independent_report_payload,
)
from protein_lm.data.fixed_budget_audit.errors import (
    AuditConfigurationError,
    AuditValidationError,
)
from protein_lm.data.fixed_budget_audit.config import (
    AuditPass,
    DatasetPartition,
    PairUnionKind,
    QueryScope,
    SplitStrategy,
    TrackOrigin,
)
from protein_lm.data.fixed_budget_audit.reporting import (
    COMMON_RESULT,
    STAGED_RESULT,
    validate_report_payload,
)

PUBLIC_CONTRACT = [
    "COMMON_RESULT",
    "STAGED_RESULT",
    "ReportPublication",
    "ReceiptPublication",
    "CompletionAuthorization",
    "build_report_payload",
    "validate_report_payload",
    "render_markdown_report",
    "publish_a004_report",
    "verify_report_publication",
    "publish_receipt",
    "verify_receipt_publication",
    "publish_completion_marker",
]


def test_reporting_public_surface_and_payload_use_plain_strings(tmp_path: Path) -> None:
    payload = _independent_payload(tmp_path)

    validate_report_payload(payload, fingerprint=FINGERPRINT)

    assert reporting_module.__all__ == PUBLIC_CONTRACT
    assert type(COMMON_RESULT) is str
    assert type(STAGED_RESULT) is str
    assert COMMON_RESULT == "common_all_query_10000"
    assert STAGED_RESULT == "staged_union_with_changed_query_100000"
    for track in payload["tracks"]:
        assert type(track["strategy"]) is str
        assert type(track["partition"]) is str
        assert type(track["pass_name"]) is str
        assert type(track["source_label"]) is str
        assert all(
            type(record["query_scope"]) is str for record in track["caps"].values()
        )


def test_reporting_emission_and_validation_vocabulary_is_enum_derived(
    tmp_path: Path,
) -> None:
    payload = _independent_payload(tmp_path)
    tracks = payload["tracks"]
    partitions = payload["partition_results"]
    semantics = payload["result_semantics"]

    assert reporting_module._STRATEGIES == tuple(item.value for item in SplitStrategy)
    assert reporting_module._PARTITIONS == tuple(
        item.value for item in DatasetPartition
    )
    assert reporting_module._PASSES == tuple(item.value for item in AuditPass)
    assert reporting_module._SOURCES == frozenset(item.value for item in TrackOrigin)
    assert reporting_module._RESULT_NAMES == tuple(item.value for item in PairUnionKind)
    assert reporting_module._QUERY_SCOPES == frozenset(
        item.value for item in QueryScope
    )
    assert COMMON_RESULT == PairUnionKind.COMMON_ALL_QUERY_10000.value
    assert STAGED_RESULT == (PairUnionKind.STAGED_UNION_WITH_CHANGED_QUERY_100000.value)

    assert {track["strategy"] for track in tracks} == {
        item.value for item in SplitStrategy
    }
    assert {track["partition"] for track in tracks} == {
        item.value for item in DatasetPartition
    }
    assert {track["pass_name"] for track in tracks} == {
        item.value for item in AuditPass
    }
    assert {track["source_label"] for track in tracks} == {
        item.value for item in TrackOrigin
    }
    assert {
        record["query_scope"] for track in tracks for record in track["caps"].values()
    } == {item.value for item in QueryScope}
    assert semantics == {
        "common_result_name": PairUnionKind.COMMON_ALL_QUERY_10000.value,
        "staged_result_name": (
            PairUnionKind.STAGED_UNION_WITH_CHANGED_QUERY_100000.value
        ),
        "staged_cap_applies_to_all_queries": False,
        "negative_query_meaning": (
            "no prohibited pair detected through the query's highest executed cap"
        ),
    }
    assert all(
        set(partition).issuperset(item.value for item in PairUnionKind)
        for partition in partitions
    )

    enum_wire_values = {
        item.value
        for enum_type in (SplitStrategy, TrackOrigin, PairUnionKind, QueryScope)
        for item in enum_type
    }
    tree = ast.parse(Path(reporting_module.__file__).read_text(encoding="utf-8"))
    raw_duplicates = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value in enum_wire_values
    }
    assert raw_duplicates == set()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("authority", "authority or schema drifted"),
        ("source", "source label is malformed"),
        ("cap_inventory", "cap inventory is malformed"),
        ("rate", "report rate does not reconcile"),
        ("staged_additions", "staged-addition counts do not reconcile"),
        ("sensitivity", "cap-sensitivity transition drifted"),
        ("forbidden_claim", "forbidden all-query 100k claim"),
    ),
)
def test_payload_reconciliation_uses_validation_error(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    payload = deepcopy(_independent_payload(tmp_path))
    _mutate(payload, mutation)

    with pytest.raises(AuditValidationError, match=message):
        validate_report_payload(payload, fingerprint=FINGERPRINT)


def test_payload_preserves_lower_hardware_error_type(tmp_path: Path) -> None:
    payload = deepcopy(_independent_payload(tmp_path))
    payload["hardware"] = {"platform": "incomplete"}

    with pytest.raises(AuditConfigurationError, match="fields drifted"):
        validate_report_payload(payload, fingerprint=FINGERPRINT)


def _independent_payload(tmp_path: Path) -> dict[str, object]:
    return independent_report_payload(tmp_path)


def _mutate(payload: dict[str, object], mutation: str) -> None:
    tracks = payload["tracks"]
    partitions = payload["partition_results"]
    assert isinstance(tracks, list)
    assert isinstance(partitions, list)
    track = tracks[0]
    partition = partitions[0]
    assert isinstance(track, dict)
    assert isinstance(partition, dict)

    if mutation == "authority":
        payload["scope"] = "drifted"
    elif mutation == "source":
        track["source_label"] = "imported_a003"
    elif mutation == "cap_inventory":
        caps = track["caps"]
        assert isinstance(caps, dict)
        del caps["100000"]
    elif mutation == "rate":
        caps = track["caps"]
        assert isinstance(caps, dict)
        cap = caps["1000"]
        assert isinstance(cap, dict)
        rate = cap["prohibited_query_rate"]
        assert isinstance(rate, dict)
        rate["percent"] = "unexpected"
    elif mutation == "staged_additions":
        additions = partition["staged_additions"]
        assert isinstance(additions, dict)
        additions["additional_pairs"] = 1
    elif mutation == "sensitivity":
        sensitivity = track["cap_sensitivity"]
        assert isinstance(sensitivity, list)
        first = sensitivity[0]
        assert isinstance(first, dict)
        first["compared_queries"] = 2
    elif mutation == "forbidden_claim":
        limitations = payload["limitations"]
        assert isinstance(limitations, list)
        limitations[0] = "all_query_100000"
    else:
        raise AssertionError(f"unknown mutation: {mutation}")
