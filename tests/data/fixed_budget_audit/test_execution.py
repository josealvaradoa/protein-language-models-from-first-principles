"""Synthetic tests for the fixed-budget operating-system boundary."""

from __future__ import annotations

import builtins
import os
import signal
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import protein_lm.data.fixed_budget_audit.execution as execution_module
from protein_lm.data.fixed_budget_audit.errors import AuditExecutionError
from protein_lm.data.fixed_budget_audit.execution import (
    prove_path_is_ignored,
    prove_path_is_public,
    exclusive_lock,
    require_committed_execution_code,
    require_disk_capacity,
    run_mmseqs_command,
)
from protein_lm.data.similarity_audit_policy import (
    SimilarityAuditPolicy,
    load_similarity_audit_policy,
)

PROJECT_ROOT = Path(__file__).parents[3]
POLICY_PATH = (
    PROJECT_ROOT / "experiments" / "week_01" / "diagnostic_similarity_audit.toml"
)


def test_execution_and_package_import_without_fcntl_in_a_fresh_process() -> None:
    code = """
import builtins
import importlib

attempts = []
original_import = builtins.__import__

def blocked_fcntl(name, *args, **kwargs):
    if name == "fcntl":
        attempts.append(name)
        raise ModuleNotFoundError("No module named 'fcntl'", name="fcntl")
    return original_import(name, *args, **kwargs)

builtins.__import__ = blocked_fcntl
execution = importlib.import_module("protein_lm.data.fixed_budget_audit.execution")
package = importlib.import_module("protein_lm.data.fixed_budget_audit")
assert execution is not None
assert package is not None
try:
    __import__("fcntl")
except ModuleNotFoundError:
    pass
else:
    raise AssertionError("fcntl was not blocked")
assert attempts
"""
    environment = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}

    subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_exclusive_lock_reports_missing_fcntl_without_continuing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing_fcntl = ModuleNotFoundError("No module named 'fcntl'", name="fcntl")
    original_import = builtins.__import__

    def blocked_fcntl(name: str, *args: object, **kwargs: object) -> object:
        if name == "fcntl":
            raise missing_fcntl
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_fcntl)

    with pytest.raises(AuditExecutionError) as error:
        with exclusive_lock(tmp_path / "audit.lock"):
            pytest.fail("exclusive_lock yielded without fcntl")

    assert str(error.value) == (
        "exclusive locking is unsupported on platforms without fcntl"
    )
    assert error.value.__cause__ is missing_fcntl
    assert not (tmp_path / "audit.lock").exists()


def test_exclusive_lock_preserves_nonblocking_unix_locking_and_pid_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[int, int]] = []
    fake_fcntl = SimpleNamespace(LOCK_EX=4, LOCK_NB=8, LOCK_UN=16)

    def fake_flock(file_descriptor: int, operation: int) -> None:
        calls.append((file_descriptor, operation))

    fake_fcntl.flock = fake_flock
    monkeypatch.setattr(execution_module, "_load_fcntl", lambda: fake_fcntl)
    lock_path = tmp_path / "locks" / "audit.lock"

    with exclusive_lock(lock_path):
        assert lock_path.read_bytes() == f"pid={os.getpid()}\n".encode()
        assert calls == [(calls[0][0], fake_fcntl.LOCK_EX | fake_fcntl.LOCK_NB)]

    assert calls == [
        (calls[0][0], fake_fcntl.LOCK_EX | fake_fcntl.LOCK_NB),
        (calls[0][0], fake_fcntl.LOCK_UN),
    ]


def test_exclusive_lock_preserves_contention_error_and_cause(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contention = BlockingIOError("synthetic contention")
    fake_fcntl = SimpleNamespace(LOCK_EX=4, LOCK_NB=8, LOCK_UN=16)

    def raise_contention(file_descriptor: int, operation: int) -> None:
        del file_descriptor, operation
        raise contention

    fake_fcntl.flock = raise_contention
    monkeypatch.setattr(execution_module, "_load_fcntl", lambda: fake_fcntl)

    with pytest.raises(AuditExecutionError) as error:
        with exclusive_lock(tmp_path / "audit.lock"):
            pytest.fail("exclusive_lock yielded despite contention")

    assert str(error.value) == "another Task 7 audit is already running"
    assert error.value.__cause__ is contention


class _FakeProcess:
    def __init__(self, *, return_code: int, running_polls: int = 0) -> None:
        self.pid = 31415
        self.returncode: int | None = None
        self.wait_timeouts: list[float | None] = []
        self._return_code = return_code
        self._running_polls = running_polls

    def poll(self) -> int | None:
        if self._running_polls:
            self._running_polls -= 1
            return None
        self.returncode = self._return_code
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        self.returncode = self._return_code
        return self._return_code


@pytest.fixture
def policy() -> SimilarityAuditPolicy:
    return load_similarity_audit_policy(POLICY_PATH)


def test_runner_records_context_log_and_runtime_with_fake_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    policy: SimilarityAuditPolicy,
) -> None:
    project_root = tmp_path / "repo"
    workspace = project_root / "workspace"
    log_path = workspace / "logs" / "command.log"
    project_root.mkdir()
    observed: dict[str, object] = {}
    process = _FakeProcess(return_code=0, running_polls=1)

    def fake_popen(command: list[str], **kwargs: object) -> _FakeProcess:
        observed["command"] = command
        observed.update(kwargs)
        log = kwargs["stdout"]
        log.write(b"synthetic MMseqs log\n")
        log.flush()
        return process

    disk_checks: list[tuple[Path, SimilarityAuditPolicy]] = []
    sleeps: list[float] = []
    clock = iter((100.0, 100.5, 101.234))
    monkeypatch.setattr(execution_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        execution_module,
        "require_disk_capacity",
        lambda checked_workspace, checked_policy: disk_checks.append(
            (checked_workspace, checked_policy)
        ),
    )
    monkeypatch.setattr(execution_module.time, "sleep", sleeps.append)
    monkeypatch.setattr(execution_module.time, "perf_counter", lambda: next(clock))

    runtime = run_mmseqs_command(
        ("/synthetic/mmseqs", "easy-search"),
        project_root=project_root,
        workspace=workspace,
        log_path=log_path,
        policy=policy,
    )

    assert runtime == "1.234"
    assert log_path.read_bytes() == b"synthetic MMseqs log\n"
    assert observed["command"] == ["/synthetic/mmseqs", "easy-search"]
    assert observed["cwd"] == project_root
    assert observed["stderr"] is subprocess.STDOUT
    assert observed["start_new_session"] is True
    assert observed["stdout"].closed
    assert disk_checks == [(workspace, policy), (workspace, policy)]
    assert sleeps == [policy.disk_check_interval_seconds]


def test_runner_reports_nonzero_exit_with_exact_twenty_line_tail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    policy: SimilarityAuditPolicy,
) -> None:
    process = _FakeProcess(return_code=9)

    def fake_popen(command: list[str], **kwargs: object) -> _FakeProcess:
        del command
        log = kwargs["stdout"]
        for line in range(1, 26):
            log.write(f"line {line}\n".encode())
        log.flush()
        return process

    monkeypatch.setattr(execution_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(execution_module, "require_disk_capacity", lambda *args: None)
    monkeypatch.setattr(execution_module.time, "perf_counter", lambda: 100.0)
    expected_tail = "\n".join(f"line {line}" for line in range(6, 26))

    with pytest.raises(AuditExecutionError) as error:
        run_mmseqs_command(
            ("/synthetic/mmseqs", "createdb"),
            project_root=tmp_path,
            workspace=tmp_path / "workspace",
            log_path=tmp_path / "logs" / "failed.log",
            policy=policy,
        )

    assert str(error.value) == (
        f"MMseqs2 exited with status 9. Log tail:\n{expected_tail}"
    )


def test_runner_terminates_process_group_when_runtime_guard_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    policy: SimilarityAuditPolicy,
) -> None:
    process = _FakeProcess(return_code=0, running_polls=10)
    guard_calls = 0

    def fake_guard(*args: object) -> None:
        nonlocal guard_calls
        del args
        guard_calls += 1
        if guard_calls == 2:
            raise AuditExecutionError("synthetic disk guard")

    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        execution_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(execution_module, "require_disk_capacity", fake_guard)
    monkeypatch.setattr(execution_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        execution_module.os,
        "killpg",
        lambda pid, sent_signal: signals.append((pid, sent_signal)),
    )

    with pytest.raises(AuditExecutionError, match="synthetic disk guard"):
        run_mmseqs_command(
            ("/synthetic/mmseqs", "easy-search"),
            project_root=tmp_path,
            workspace=tmp_path / "workspace",
            log_path=tmp_path / "logs" / "interrupted.log",
            policy=policy,
        )

    assert signals == [(process.pid, signal.SIGTERM)]
    assert process.wait_timeouts == [10]


def test_disk_guards_preserve_fixed_ceiling_and_reserve_messages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    policy: SimilarityAuditPolicy,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "evidence.bin").write_bytes(b"1234")
    monkeypatch.setattr(
        execution_module.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=100),
    )

    with pytest.raises(
        AuditExecutionError,
        match="Task 7 workspace exceeded its fixed 200 GiB ceiling",
    ):
        require_disk_capacity(
            workspace,
            replace(policy, workspace_byte_ceiling=3, free_space_reserve=0),
        )

    monkeypatch.setattr(
        execution_module.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=9),
    )
    with pytest.raises(
        AuditExecutionError,
        match="free disk space fell below the fixed 300 GiB reserve",
    ):
        require_disk_capacity(
            workspace,
            replace(policy, workspace_byte_ceiling=4, free_space_reserve=10),
        )


def test_git_path_proofs_accept_expected_check_ignore_statuses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    private = tmp_path / "data" / "private.tsv"
    public = tmp_path / "reports" / "public.json"
    results = iter((SimpleNamespace(returncode=0), SimpleNamespace(returncode=1)))
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        calls.append((command, kwargs))
        return next(results)

    monkeypatch.setattr(execution_module.subprocess, "run", fake_run)

    prove_path_is_ignored(private, tmp_path)
    prove_path_is_public(public, tmp_path)

    assert calls == [
        (
            ["git", "check-ignore", "--quiet", "--", "data/private.tsv"],
            {"cwd": tmp_path, "check": False},
        ),
        (
            ["git", "check-ignore", "--quiet", "--", "reports/public.json"],
            {"cwd": tmp_path, "check": False},
        ),
    ]


@pytest.mark.parametrize(
    ("proof", "return_code", "message"),
    (
        (
            prove_path_is_ignored,
            1,
            "private path is not ignored by Git: evidence/result.tsv",
        ),
        (
            prove_path_is_public,
            0,
            "public path is unexpectedly ignored: evidence/result.tsv",
        ),
        (
            prove_path_is_public,
            2,
            "could not prove public Git status: evidence/result.tsv",
        ),
    ),
)
def test_git_path_proofs_preserve_failure_contracts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    proof,
    return_code: int,
    message: str,
) -> None:
    monkeypatch.setattr(
        execution_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=return_code),
    )

    with pytest.raises(AuditExecutionError) as error:
        proof(tmp_path / "evidence" / "result.tsv", tmp_path)

    assert str(error.value) == message


def test_dirty_execution_code_uses_execution_error_without_relabeling_git(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[tuple[Path, tuple[str, ...]]] = []

    def fake_git_output(root: Path, *arguments: str) -> str:
        observed.append((root, arguments))
        return " M src/protein_lm/data/example.py"

    monkeypatch.setattr(execution_module, "git_output", fake_git_output)

    with pytest.raises(
        AuditExecutionError,
        match="execution code has uncommitted changes; review and commit it first",
    ):
        require_committed_execution_code(tmp_path)

    assert observed == [
        (
            tmp_path,
            (
                "status",
                "--porcelain",
                "--",
                "src",
                "scripts",
                "experiments",
                "pyproject.toml",
                "uv.lock",
                ".gitignore",
                ".gitattributes",
            ),
        )
    ]
