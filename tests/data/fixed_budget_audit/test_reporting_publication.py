"""Focused report, receipt, and completion publication contracts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import protein_lm.data.fixed_budget_audit.reporting as reporting_module
import protein_lm.data.fixed_budget_audit.validation as validation_module
import pytest
from reporting_test_support import (
    FINGERPRINT,
    GOLDEN_MARKDOWN,
    identity,
    independent_report_payload,
    json_bytes,
)
from protein_lm.data.fixed_budget_audit.errors import (
    AuditPublicationError,
    AuditValidationError,
)
from protein_lm.data.fixed_budget_audit.reporting import (
    CompletionAuthorization,
    ReceiptPublication,
    ReportPublication,
    publish_a004_report,
    publish_completion_marker,
    publish_receipt,
    verify_report_publication,
)
from protein_lm.data.similarity_audit_policy import SimilarityAuditError
from protein_lm.data.fixed_budget_audit.validation import revalidate_before_completion

VALID_HARDWARE = {
    "platform": "synthetic-platform",
    "machine": "synthetic-machine",
    "processor": "synthetic-processor",
    "logical_cpu_count": 8,
}


def test_report_publication_and_resume_match_independent_bytes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    payload = _install_independent_report_payload(monkeypatch, tmp_path)
    workspace = tmp_path / "workspace"

    publication = _publish_report(workspace)

    expected_json = json_bytes(payload)
    expected_markdown = GOLDEN_MARKDOWN.read_bytes()
    expected_marker = json_bytes(
        {
            "schema_version": 1,
            "stage": "a004_report_artifacts",
            "fingerprint": FINGERPRINT,
            "json": identity(expected_json),
            "markdown": identity(expected_markdown),
        }
    )
    assert publication.json_path.read_bytes() == expected_json
    assert publication.markdown_path.read_bytes() == expected_markdown
    assert publication.marker_path.read_bytes() == expected_marker
    assert publication.json_identity == identity(expected_json)
    assert publication.markdown_identity == identity(expected_markdown)
    assert publication.marker_identity == identity(expected_marker)

    writes = Mock(side_effect=AssertionError("resume attempted a JSON write"))
    monkeypatch.setattr(reporting_module, "write_json_atomic", writes)
    resumed = _publish_report(workspace)

    assert resumed == publication
    writes.assert_not_called()
    assert publication.json_path.read_bytes() == expected_json
    assert publication.markdown_path.read_bytes() == expected_markdown
    assert publication.marker_path.read_bytes() == expected_marker


@pytest.mark.parametrize(
    ("state", "message"),
    (
        ("unmarked_output", "output lacks its completion marker"),
        ("staging", "has an unmarked staging directory"),
    ),
)
def test_report_rejects_unsafe_publication_state(
    monkeypatch,
    tmp_path: Path,
    state: str,
    message: str,
) -> None:
    _install_independent_report_payload(monkeypatch, tmp_path)
    workspace = tmp_path / "workspace"
    report_directory = workspace / "evidence/report"
    if state == "unmarked_output":
        report_directory.mkdir(parents=True)
        preserved = report_directory / "preserved.txt"
    else:
        staging = report_directory.with_name(".report.incomplete")
        staging.mkdir(parents=True)
        preserved = staging / "preserved.txt"
    preserved.write_text("keep me")

    with pytest.raises(AuditPublicationError, match=message):
        _publish_report(workspace)

    assert preserved.read_text() == "keep me"


def test_report_payload_identity_reconciliation_uses_validation_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _install_independent_report_payload(monkeypatch, tmp_path)
    publication = _publish_report(tmp_path / "workspace")
    drifted_payload = deepcopy(publication.payload)
    drifted_payload["scope"] = "different expected payload"

    with pytest.raises(AuditValidationError, match="JSON report payload drifted"):
        verify_report_publication(
            replace(publication, payload=drifted_payload),
            fingerprint=FINGERPRINT,
        )


def test_report_verification_preserves_lower_artifact_read_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    publication = _report_publication(tmp_path / "workspace")
    lower_error = SimilarityAuditError("sentinel report marker read failure")
    read_json = Mock(side_effect=lower_error)
    monkeypatch.setattr(reporting_module, "read_json", read_json)

    with pytest.raises(SimilarityAuditError) as raised:
        verify_report_publication(publication, fingerprint=FINGERPRINT)

    assert raised.value is lower_error
    assert type(raised.value) is SimilarityAuditError
    assert str(raised.value) == "sentinel report marker read failure"
    read_json.assert_called_once_with(publication.marker_path)


def test_receipt_normalizes_json_and_rejects_conflicting_resume(
    monkeypatch,
    tmp_path: Path,
) -> None:
    raw_receipt = {
        "schema_version": 1,
        "stage": "a004_import_receipt",
        "fingerprint": FINGERPRINT,
        "nested": {1_000: ("left", "right")},
    }
    receipt_payload = Mock(return_value=raw_receipt)
    monkeypatch.setattr(reporting_module, "_receipt_payload", receipt_payload)
    workspace = tmp_path / "workspace"

    publication = _publish_receipt(workspace)

    expected_payload = {
        "schema_version": 1,
        "stage": "a004_import_receipt",
        "fingerprint": FINGERPRINT,
        "nested": {"1000": ["left", "right"]},
    }
    expected_bytes = json_bytes(expected_payload)
    assert publication.payload == expected_payload
    assert publication.path.read_bytes() == expected_bytes
    assert publication.identity == identity(expected_bytes)

    receipt_payload.return_value = {
        **raw_receipt,
        "nested": {1_000: ("changed",)},
    }
    with pytest.raises(
        AuditPublicationError,
        match="a004_import_receipt identity drifted",
    ):
        _publish_receipt(workspace)

    assert publication.path.read_bytes() == expected_bytes


def test_receipt_publication_preserves_lower_artifact_write_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    raw_receipt = {
        "schema_version": 1,
        "stage": "a004_import_receipt",
        "fingerprint": FINGERPRINT,
    }
    monkeypatch.setattr(
        reporting_module,
        "_receipt_payload",
        Mock(return_value=raw_receipt),
    )
    lower_error = PermissionError("sentinel receipt write failure")
    write_json_atomic = Mock(side_effect=lower_error)
    monkeypatch.setattr(
        reporting_module,
        "write_json_atomic",
        write_json_atomic,
    )
    workspace = tmp_path / "workspace"

    with pytest.raises(PermissionError) as raised:
        _publish_receipt(workspace)

    assert raised.value is lower_error
    assert type(raised.value) is PermissionError
    assert str(raised.value) == "sentinel receipt write failure"
    write_json_atomic.assert_called_once_with(
        workspace / "a004_import_receipt.json",
        raw_receipt,
    )


def test_receipt_reconciliation_uses_validation_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    receipt_payload = Mock(
        side_effect=AssertionError("invalid assignments reached serialization")
    )
    monkeypatch.setattr(reporting_module, "_receipt_payload", receipt_payload)

    with pytest.raises(
        AuditValidationError,
        match="receipt cannot claim changed assignments",
    ):
        _publish_receipt(
            tmp_path / "workspace",
            assignments_before={"task5": {"sha256": "before"}},
            assignments_after={"task5": {"sha256": "after"}},
        )

    receipt_payload.assert_not_called()


def test_completion_publication_authorization_bytes_and_conflict(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    receipt = _receipt_publication(workspace)
    report = _report_publication(workspace)
    authorization = _authorization(receipt, report)
    invalid = replace(authorization, fingerprint="wrong-fingerprint")

    with pytest.raises(
        AuditPublicationError,
        match="completion authorization is invalid",
    ):
        publish_completion_marker(
            workspace=workspace,
            fingerprint=FINGERPRINT,
            receipt=receipt,
            report=report,
            authorization=invalid,
        )
    assert not (workspace / "a004_complete.json").exists()

    completion_path = publish_completion_marker(
        workspace=workspace,
        fingerprint=FINGERPRINT,
        receipt=receipt,
        report=report,
        authorization=authorization,
    )
    expected = json_bytes(
        {
            "schema_version": 1,
            "stage": "a004_workflow_complete",
            "fingerprint": FINGERPRINT,
            "receipt": dict(receipt.identity),
            "report": {
                "json": dict(report.json_identity),
                "markdown": dict(report.markdown_identity),
                "marker": dict(report.marker_identity),
            },
            "model_use": "prohibited",
            "task8_membership_use_authorized": False,
            "diagnostic_assignments_unchanged": True,
        }
    )
    assert completion_path.read_bytes() == expected

    resumed = publish_completion_marker(
        workspace=workspace,
        fingerprint=FINGERPRINT,
        receipt=receipt,
        report=report,
        authorization=authorization,
    )
    assert resumed == completion_path
    assert resumed.read_bytes() == expected

    conflicting_report = replace(
        report,
        marker_identity={"byte_size": 44, "sha256": "9" * 64},
    )
    with pytest.raises(
        AuditPublicationError,
        match="a004_workflow_complete identity drifted",
    ):
        publish_completion_marker(
            workspace=workspace,
            fingerprint=FINGERPRINT,
            receipt=receipt,
            report=conflicting_report,
            authorization=_authorization(receipt, conflicting_report),
        )
    assert completion_path.read_bytes() == expected


def test_final_validation_rejects_receipt_report_identity_drift(
    monkeypatch,
    tmp_path: Path,
) -> None:
    report = _report_publication(tmp_path / "workspace")
    receipt = ReceiptPublication(
        path=tmp_path / "workspace/a004_import_receipt.json",
        payload={"report": {"directory": "wrong-report"}},
        identity={"byte_size": 55, "sha256": "5" * 64},
    )
    monkeypatch.setattr(
        validation_module,
        "verify_report_publication",
        Mock(),
    )
    monkeypatch.setattr(
        validation_module,
        "verify_receipt_publication",
        Mock(),
    )
    monkeypatch.setattr(validation_module, "_revalidate_frozen_sources", Mock())

    with pytest.raises(
        AuditValidationError,
        match="receipt does not identify its report",
    ):
        revalidate_before_completion(
            context=object(),  # type: ignore[arg-type]
            fingerprint=FINGERPRINT,
            databases={},
            tracks={},
            unions={},
            report=report,
            receipt=receipt,
        )


def _install_independent_report_payload(
    monkeypatch,
    tmp_path: Path,
) -> dict[str, object]:
    payload = independent_report_payload(tmp_path)
    monkeypatch.setattr(
        reporting_module,
        "build_report_payload",
        Mock(return_value=payload),
    )
    return payload


def _publish_report(workspace: Path) -> ReportPublication:
    return publish_a004_report(
        workspace=workspace,
        fingerprint=FINGERPRINT,
        policy=object(),  # type: ignore[arg-type]
        hardware={},
        assignment_balances={},
        assignments_unchanged=True,
        tracks={},
        unions={},
    )


def _publish_receipt(
    workspace: Path,
    *,
    assignments_before: dict[str, dict[str, object]] | None = None,
    assignments_after: dict[str, dict[str, object]] | None = None,
) -> ReceiptPublication:
    before = assignments_before or {"task5": {"sha256": "same"}}
    after = assignments_after or before
    return publish_receipt(
        workspace=workspace,
        fingerprint=FINGERPRINT,
        policy=object(),  # type: ignore[arg-type]
        source_policy_path=Path("unused-source-policy"),
        code_revision="b" * 40,
        mmseqs_version="18-test",
        hardware=VALID_HARDWARE,
        assignments_before=before,
        assignments_after=after,
        imported=object(),  # type: ignore[arg-type]
        databases={},
        tracks={},
        unions={},
        report=_report_publication(workspace),
    )


def _report_publication(workspace: Path) -> ReportPublication:
    directory = workspace / "evidence/report"
    return ReportPublication(
        directory=directory,
        json_path=directory / "a004_report.json",
        markdown_path=directory / "a004_report.md",
        marker_path=directory / "complete.json",
        payload={},
        json_identity={"byte_size": 11, "sha256": "1" * 64},
        markdown_identity={"byte_size": 22, "sha256": "2" * 64},
        marker_identity={"byte_size": 33, "sha256": "3" * 64},
    )


def _receipt_publication(workspace: Path) -> ReceiptPublication:
    return ReceiptPublication(
        path=workspace / "a004_import_receipt.json",
        payload={},
        identity={"byte_size": 55, "sha256": "5" * 64},
    )


def _authorization(
    receipt: ReceiptPublication,
    report: ReportPublication,
) -> CompletionAuthorization:
    return CompletionAuthorization(
        fingerprint=FINGERPRINT,
        receipt_identity=dict(receipt.identity),
        report_identities={
            "json": dict(report.json_identity),
            "markdown": dict(report.markdown_identity),
            "marker": dict(report.marker_identity),
        },
    )
