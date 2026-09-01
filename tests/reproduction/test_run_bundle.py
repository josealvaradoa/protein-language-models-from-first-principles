"""Focused synthetic checks for immutable reproduction run-bundle storage."""

from __future__ import annotations

import json
import math
import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from protein_lm.reproduction.run_bundle import RunBundle, RunBundleError, RunStatus
import protein_lm.reproduction.run_bundle as run_bundle_module


RUN_ID = "0123456789abcdef0123456789abcdef"
OTHER_RUN_ID = "fedcba9876543210fedcba9876543210"
CONTRACT = b'contract_identifier = "frozen-contract-v1"\n[policy]\nname = "fixed"\n'
INITIAL_RECORD = {
    "run_id": RUN_ID,
    "contract_identifier": "frozen-contract-v1",
    "status": "running",
    "started_by": "synthetic-test",
}
FINAL_FILENAMES = {
    "contract.toml",
    "run.json",
    "log.txt",
    "metrics.json",
    "comparison.json",
    "provenance.json",
}


def create_bundle(tmp_path: Path, run_id: str = RUN_ID) -> RunBundle:
    record = {**INITIAL_RECORD, "run_id": run_id}
    return RunBundle.create(tmp_path / "bundles", run_id, CONTRACT, record)


def finalize(bundle: RunBundle, status: RunStatus = RunStatus.COMPLETED) -> None:
    bundle.finalize(
        status,
        {"finished_by": "synthetic-test"},
        {"cross_entropy": 2.5},
        {"passed": True},
        {"source": "synthetic"},
    )


def test_create_stages_exact_contract_and_publishes_running_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def recording_replace(source: Path | str, destination: Path | str) -> None:
        published.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(run_bundle_module.os, "replace", recording_replace)
    bundle = create_bundle(tmp_path)

    assert bundle.contract_path.read_bytes() == CONTRACT
    assert json.loads(bundle.run_path.read_text(encoding="utf-8"))["status"] == "running"
    assert bundle.log_path.read_bytes() == b""
    assert not any(path.name.startswith(f".{RUN_ID}.staging-") for path in bundle.root.iterdir())
    assert published[-1][1] == bundle.directory
    with pytest.raises(FrozenInstanceError):
        bundle.run_id = OTHER_RUN_ID  # type: ignore[misc]


@pytest.mark.parametrize("run_id", ["../outside", "not-a-run-id"])
def test_direct_construction_validates_root_and_identity(
    tmp_path: Path, run_id: str
) -> None:
    with pytest.raises(RunBundleError):
        RunBundle(tmp_path, run_id, "frozen-contract-v1")
    with pytest.raises(RunBundleError, match="contract identifier"):
        RunBundle(tmp_path, RUN_ID, " ")
    with pytest.raises(RunBundleError, match="existing real directory"):
        RunBundle(tmp_path / "missing", RUN_ID, "frozen-contract-v1")


@pytest.mark.parametrize(
    "contract",
    [b"", b"not = [valid", b"name = 'missing identifier'\n"],
)
def test_create_rejects_invalid_contract_bytes(tmp_path: Path, contract: bytes) -> None:
    with pytest.raises(RunBundleError, match="contract"):
        RunBundle.create(tmp_path, RUN_ID, contract, INITIAL_RECORD)


@pytest.mark.parametrize(
    "record",
    [
        {**INITIAL_RECORD, "run_id": OTHER_RUN_ID},
        {**INITIAL_RECORD, "contract_identifier": "other"},
        {**INITIAL_RECORD, "status": "completed"},
        ["not", "an", "object"],
        {**INITIAL_RECORD, "value": math.nan},
    ],
)
def test_create_validates_run_record_identity_and_json_safety(
    tmp_path: Path, record: object
) -> None:
    with pytest.raises(RunBundleError):
        RunBundle.create(tmp_path, RUN_ID, CONTRACT, record)  # type: ignore[arg-type]
    assert not (tmp_path / RUN_ID).exists()


@pytest.mark.parametrize(
    "run_id",
    [
        "A123456789abcdef0123456789abcdef",
        "0123456789abcdef0123456789abcde",
        "0123456789abcdef0123456789abcdef/",
        "../123456789abcdef0123456789abcdef",
        "g123456789abcdef0123456789abcdef",
    ],
)
def test_invalid_run_ids_and_traversal_are_rejected(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(RunBundleError, match="run id"):
        RunBundle.create(tmp_path, run_id, CONTRACT, INITIAL_RECORD)


def test_existing_run_is_refused_and_retry_requires_a_different_id(tmp_path: Path) -> None:
    bundle = create_bundle(tmp_path)
    with pytest.raises(RunBundleError, match="already exists"):
        create_bundle(tmp_path)

    retry = create_bundle(tmp_path, OTHER_RUN_ID)
    assert bundle.directory != retry.directory


def test_append_log_persists_text_while_running(tmp_path: Path) -> None:
    bundle = create_bundle(tmp_path)
    bundle.append_log("first line\n")
    bundle.append_log("second line\n")

    assert bundle.log_path.read_text(encoding="utf-8") == "first line\nsecond line\n"


@pytest.mark.parametrize(
    "status",
    [
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
        RunStatus.RUNNER_RESTARTED,
    ],
)
def test_terminal_statuses_publish_complete_inventory(
    tmp_path: Path, status: RunStatus
) -> None:
    bundle = create_bundle(tmp_path)
    finalize(bundle, status)

    assert {path.name for path in bundle.directory.iterdir()} == FINAL_FILENAMES
    assert json.loads(bundle.run_path.read_text(encoding="utf-8"))["status"] == status.value


def test_final_record_is_replaced_last(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = create_bundle(tmp_path)
    destinations: list[str] = []
    real_replace = os.replace

    def recording_replace(source: Path | str, destination: Path | str) -> None:
        destinations.append(Path(destination).name)
        real_replace(source, destination)

    monkeypatch.setattr(run_bundle_module.os, "replace", recording_replace)
    finalize(bundle)

    assert destinations == [
        "metrics.json",
        "comparison.json",
        "provenance.json",
        "run.json",
    ]


def test_final_json_is_deterministic_and_leaves_no_temporary_files(tmp_path: Path) -> None:
    bundle = create_bundle(tmp_path)
    bundle.finalize(
        RunStatus.COMPLETED,
        {"z": 1, "a": 2},
        {"z": 1, "a": 2},
        {"z": 1, "a": 2},
        {"z": 1, "a": 2},
    )

    assert bundle.metrics_path.read_bytes() == b'{"a":2,"z":1}\n'
    assert not list(bundle.directory.glob(".*.tmp"))


@pytest.mark.parametrize(
    "bad_payload",
    [math.nan, ["not", "an", "object"], {"value": math.nan}, {"value": object()}],
)
def test_bad_final_payloads_do_not_partially_finalize(
    tmp_path: Path, bad_payload: object
) -> None:
    bundle = create_bundle(tmp_path)
    with pytest.raises(RunBundleError):
        bundle.finalize(
            RunStatus.COMPLETED,
            {},
            bad_payload,  # type: ignore[arg-type]
            {"passed": True},
            {"source": "synthetic"},
        )

    assert json.loads(bundle.run_path.read_text(encoding="utf-8"))["status"] == "running"
    assert not bundle.metrics_path.exists()
    assert not bundle.comparison_path.exists()
    assert not bundle.provenance_path.exists()


@pytest.mark.parametrize(
    "updates",
    [
        {"run_id": OTHER_RUN_ID},
        {"contract_identifier": "other-contract"},
        {"status": "completed"},
    ],
)
def test_finalization_rejects_reserved_run_record_updates(
    tmp_path: Path, updates: dict[str, object]
) -> None:
    bundle = create_bundle(tmp_path)
    with pytest.raises(RunBundleError):
        bundle.finalize(
            RunStatus.COMPLETED,
            updates,
            {"cross_entropy": 2.5},
            {"passed": True},
            {"source": "synthetic"},
        )

    assert json.loads(bundle.run_path.read_text(encoding="utf-8"))["status"] == "running"
    assert not bundle.metrics_path.exists()


def test_same_value_identity_updates_remain_allowed(tmp_path: Path) -> None:
    bundle = create_bundle(tmp_path)
    bundle.finalize(
        RunStatus.COMPLETED,
        {"run_id": RUN_ID, "contract_identifier": "frozen-contract-v1"},
        {"cross_entropy": 2.5},
        {"passed": True},
        {"source": "synthetic"},
    )

    assert json.loads(bundle.run_path.read_text(encoding="utf-8"))["status"] == "completed"


def test_interrupted_final_record_keeps_running_state_and_retry_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = create_bundle(tmp_path)
    real_replace = os.replace

    def interrupt_final_record(source: Path | str, destination: Path | str) -> None:
        if Path(destination) == bundle.run_path:
            raise OSError("simulated interruption")
        real_replace(source, destination)

    monkeypatch.setattr(run_bundle_module.os, "replace", interrupt_final_record)
    with pytest.raises(RunBundleError, match="run.json"):
        finalize(bundle)

    assert json.loads(bundle.run_path.read_text(encoding="utf-8"))["status"] == "running"
    assert not list(bundle.directory.glob(".*.tmp"))
    monkeypatch.setattr(run_bundle_module.os, "replace", real_replace)
    finalize(bundle)
    assert json.loads(bundle.run_path.read_text(encoding="utf-8"))["status"] == "completed"


def test_terminal_bundles_refuse_all_api_mutation(tmp_path: Path) -> None:
    bundle = create_bundle(tmp_path)
    finalize(bundle)

    with pytest.raises(RunBundleError, match="immutable"):
        bundle.append_log("late entry")
    with pytest.raises(RunBundleError, match="immutable"):
        finalize(bundle)


def test_tampered_or_malformed_run_record_fails_closed(tmp_path: Path) -> None:
    bundle = create_bundle(tmp_path)
    bundle.run_path.write_text("[]", encoding="utf-8")
    with pytest.raises(RunBundleError, match="object"):
        bundle.append_log("entry")

    bundle.run_path.write_text('{"run_id":"wrong"}', encoding="utf-8")
    with pytest.raises(RunBundleError, match="identity"):
        finalize(bundle)


def test_symlink_roots_and_bundle_directories_are_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_root = tmp_path / "linked-root"
    try:
        linked_root.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")
    with pytest.raises(RunBundleError, match="symlink"):
        RunBundle.create(linked_root, RUN_ID, CONTRACT, INITIAL_RECORD)

    bundle = create_bundle(tmp_path)
    bundle.directory.rename(outside / "bundle")
    bundle.directory.symlink_to(outside / "bundle", target_is_directory=True)
    with pytest.raises(RunBundleError, match="unsafe"):
        bundle.append_log("escape")


def test_path_validation_rejects_normalized_and_symlink_escapes(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(RunBundleError, match="escapes"):
        run_bundle_module._assert_child(root, root / ".." / "outside" / "new.json")

    link = root / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")
    with pytest.raises(RunBundleError, match="escapes"):
        run_bundle_module._assert_child(link.parent, link / "new.json")
