"""Pure fingerprint, hardware, and frozen-input provenance contracts."""

import hashlib
from pathlib import Path
from unittest.mock import Mock

import protein_lm.data.fixed_budget_audit.provenance as provenance_module
import pytest
from protein_lm.data.artifacts import file_identity
from protein_lm.data.fixed_budget_audit.config import load_a004_policy
from protein_lm.data.fixed_budget_audit.errors import (
    AuditConfigurationError,
    SourceEvidenceError,
)
from protein_lm.data.fixed_budget_audit.provenance import (
    HARDWARE_FIELDS,
    a004_fingerprint,
    frozen_assignment_identities,
    hardware_provenance,
    require_same_six_fastas,
    validate_hardware_provenance,
)
from protein_lm.data.similarity_audit_models import FileEvidence
from protein_lm.data.similarity_audit_policy import load_similarity_audit_policy
from protein_lm.data.similarity_fastas import FastaEvidence, MaterializedInputs

PROJECT_ROOT = Path(__file__).parents[3]
A004_POLICY_PATH = (
    PROJECT_ROOT / "experiments/week_01/read_only_similarity_audit_a004.toml"
)
SOURCE_POLICY_PATH = (
    PROJECT_ROOT / "experiments/week_01/diagnostic_similarity_audit.toml"
)
CODE_REVISION = "0123456789abcdef0123456789abcdef01234567"
MMSEQS_VERSION = "18-test"
EXPECTED_FINGERPRINT_PAYLOAD = (
    b'{"a003_code_revision":"ca8940a2557a129da2b76fd2bd4b619a0fff8361",'
    b'"a003_policy_sha256":"ce767f0ce843e4f40edbcd2f9da6ca4642996046cb4042'
    b'a2410c27c39cbae742","a003_run_fingerprint":"865c3eaead1167d6d27469a26f2a3'
    b'add8fb0e1f1e52322353251aef2dea52041","a004_code_revision":"0123456789ab'
    b'cdef0123456789abcdef01234567","a004_policy_sha256":"3a21edeaf45057a8e'
    b'50b5643abc14d3b633edb69b66aefd184e5f59963931a04","mmseqs_version":"18-'
    b'test","task4_catalog_sha256":"7d619d7853eb6165786c0e0aca4f50ed66f5b69dfb'
    b'ed134a81d789d9c6dbcb70","task5_local_assignment_sha256":"3b8687cdd5c747'
    b'7b114fef7f425a68516ffd093f084aefee19d98272f397bfe6","task6_local_assign'
    b'ment_sha256":"7a89c5d432ff9b2a2a787cae9d5584754ec8a027df4118aaedcc2bd'
    b'4c41221d1"}'
)
EXPECTED_FINGERPRINT = (
    "40758fffc26101c9141220c32bde425e4ecffb8d31a761e194ce91842f23eeb4"
)
VALID_HARDWARE = {
    "platform": "synthetic-platform",
    "machine": "synthetic-machine",
    "processor": "synthetic-processor",
    "logical_cpu_count": 8,
}
ASSIGNMENT_NAMES = (
    "task5_public",
    "task5_local",
    "task5_report",
    "task6_public",
    "task6_local",
    "task6_report",
)


def test_fingerprint_matches_independent_literal_payload_and_hash() -> None:
    policy = load_a004_policy(A004_POLICY_PATH)
    source_policy = load_similarity_audit_policy(SOURCE_POLICY_PATH)

    assert len(EXPECTED_FINGERPRINT_PAYLOAD) == 710
    assert hashlib.sha256(EXPECTED_FINGERPRINT_PAYLOAD).hexdigest() == (
        EXPECTED_FINGERPRINT
    )
    assert (
        a004_fingerprint(
            policy=policy,
            source_policy=source_policy,
            code_revision=CODE_REVISION,
            mmseqs_version=MMSEQS_VERSION,
        )
        == EXPECTED_FINGERPRINT
    )

    with pytest.raises(AuditConfigurationError, match="inputs are malformed"):
        a004_fingerprint(
            policy=policy,
            source_policy=source_policy,
            code_revision="",
            mmseqs_version=MMSEQS_VERSION,
        )


def test_hardware_capture_has_exact_order_and_json_types(monkeypatch) -> None:
    monkeypatch.setattr(provenance_module.platform, "platform", lambda: "platform")
    monkeypatch.setattr(provenance_module.platform, "machine", lambda: "machine")
    monkeypatch.setattr(provenance_module.platform, "processor", lambda: "processor")
    monkeypatch.setattr(provenance_module.os, "cpu_count", lambda: 12)

    hardware = hardware_provenance()

    assert tuple(hardware) == HARDWARE_FIELDS
    assert hardware == {
        "platform": "platform",
        "machine": "machine",
        "processor": "processor",
        "logical_cpu_count": 12,
    }


@pytest.mark.parametrize(
    ("hardware", "message"),
    (
        (None, "provenance is malformed"),
        ({"platform": "only-one-field"}, "fields drifted"),
        ({**VALID_HARDWARE, "platform": None}, "provenance is malformed"),
        ({**VALID_HARDWARE, "logical_cpu_count": True}, "CPU count is malformed"),
        ({**VALID_HARDWARE, "logical_cpu_count": 0}, "CPU count is malformed"),
    ),
)
def test_hardware_validation_rejects_malformed_contracts(
    hardware: object,
    message: str,
) -> None:
    with pytest.raises(AuditConfigurationError, match=message):
        validate_hardware_provenance(hardware)  # type: ignore[arg-type]


def test_hardware_validation_accepts_empty_processor() -> None:
    validate_hardware_provenance({**VALID_HARDWARE, "processor": ""})


def test_assignment_identities_use_exact_six_names(tmp_path: Path) -> None:
    paths = {}
    for name in ASSIGNMENT_NAMES:
        path = tmp_path / name
        path.write_text(name)
        paths[name] = path
    paths["unrelated"] = tmp_path / "unrelated"

    identities = frozen_assignment_identities(paths)

    assert tuple(identities) == ASSIGNMENT_NAMES
    assert identities == {name: file_identity(paths[name]) for name in ASSIGNMENT_NAMES}
    with pytest.raises(SourceEvidenceError, match="paths are incomplete"):
        frozen_assignment_identities(
            {name: path for name, path in paths.items() if name != "task6_report"}
        )


def test_assignment_identities_preserve_lower_file_identity_error(
    monkeypatch,
) -> None:
    lower_error = FileNotFoundError("sentinel frozen assignment is unavailable")
    file_identity_mock = Mock(side_effect=lower_error)
    monkeypatch.setattr(provenance_module, "file_identity", file_identity_mock)
    paths = {name: Path(f"synthetic/{name}") for name in ASSIGNMENT_NAMES}

    with pytest.raises(FileNotFoundError) as raised:
        frozen_assignment_identities(paths)

    assert raised.value is lower_error
    assert type(raised.value) is FileNotFoundError
    assert str(raised.value) == "sentinel frozen assignment is unavailable"
    file_identity_mock.assert_called_once_with(paths["task5_public"])


def test_same_six_fastas_rejects_one_source_identity_mismatch() -> None:
    fastas = _fastas()
    inputs = MaterializedInputs(
        catalog=FileEvidence(0, 0, "0" * 64),
        fastas=fastas,
    )
    imported = _ImportedFastas(fastas)

    require_same_six_fastas(inputs, imported)  # type: ignore[arg-type]

    assert imported.calls == [
        ("random", "training"),
        ("random", "validation"),
        ("random", "test"),
        ("group_aware", "training"),
        ("group_aware", "validation"),
        ("group_aware", "test"),
    ]
    drifted = {strategy: dict(partitions) for strategy, partitions in fastas.items()}
    drifted["group_aware"]["test"] = FastaEvidence(1, 4, 8, "f" * 64)
    with pytest.raises(SourceEvidenceError, match="differs from preserved A-003"):
        require_same_six_fastas(
            MaterializedInputs(inputs.catalog, drifted),
            _ImportedFastas(fastas),  # type: ignore[arg-type]
        )

    incomplete = {strategy: dict(partitions) for strategy, partitions in fastas.items()}
    del incomplete["random"]["test"]
    with pytest.raises(SourceEvidenceError, match="differs from preserved A-003"):
        require_same_six_fastas(
            MaterializedInputs(inputs.catalog, incomplete),
            _ImportedFastas(fastas),  # type: ignore[arg-type]
        )


def test_same_six_fastas_preserves_lower_import_error() -> None:
    lower_error = SourceEvidenceError("sentinel preserved A-003 FASTA is unavailable")
    imported = Mock()
    imported.fasta.side_effect = lower_error
    inputs = MaterializedInputs(
        catalog=FileEvidence(0, 0, "0" * 64),
        fastas=_fastas(),
    )

    with pytest.raises(SourceEvidenceError) as raised:
        require_same_six_fastas(inputs, imported)

    assert raised.value is lower_error
    assert type(raised.value) is SourceEvidenceError
    assert str(raised.value) == "sentinel preserved A-003 FASTA is unavailable"
    imported.fasta.assert_called_once_with("random", "training")


def _fastas() -> dict[str, dict[str, FastaEvidence]]:
    return {
        strategy: {
            partition: FastaEvidence(1, 4, 8, character * 64)
            for partition, character in (
                ("training", "1"),
                ("validation", "2"),
                ("test", "3"),
            )
        }
        for strategy in ("random", "group_aware")
    }


class _ImportedFastas:
    def __init__(self, fastas: dict[str, dict[str, FastaEvidence]]) -> None:
        self.fastas = fastas
        self.calls: list[tuple[str, str]] = []

    def fasta(self, strategy: str, partition: str) -> FastaEvidence:
        self.calls.append((strategy, partition))
        return self.fastas[strategy][partition]
