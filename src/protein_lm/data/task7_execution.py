"""Execution and machine-safety guards for the Task 7 audit."""

from __future__ import annotations

import fcntl
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from protein_lm.data.similarity_audit_policy import (
    SimilarityAuditError,
    SimilarityAuditPolicy,
)


def run_mmseqs_command(
    command: Sequence[str],
    *,
    project_root: Path,
    workspace: Path,
    log_path: Path,
    policy: SimilarityAuditPolicy,
) -> str:
    """Run one MMseqs2 command while enforcing the Task 7 disk guards."""

    require_disk_capacity(workspace, policy)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            list(command),
            cwd=project_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        next_heartbeat = 60.0
        try:
            while process.poll() is None:
                time.sleep(policy.disk_check_interval_seconds)
                require_disk_capacity(workspace, policy)
                elapsed = time.perf_counter() - started
                if elapsed >= next_heartbeat:
                    print(
                        f"MMseqs2 still running after {elapsed:.0f} seconds; "
                        f"log: {log_path}",
                        flush=True,
                    )
                    next_heartbeat += 60.0
        except BaseException:
            _terminate_process_group(process)
            raise
        return_code = process.returncode
    if return_code != 0:
        tail = _log_tail(log_path)
        raise SimilarityAuditError(
            f"MMseqs2 exited with status {return_code}. Log tail:\n{tail}"
        )
    return f"{time.perf_counter() - started:.3f}"


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def verify_mmseqs(policy: SimilarityAuditPolicy, project_root: Path) -> str:
    """Prove that the pinned MMseqs2 binary and version are available."""

    executable = Path(policy.mmseqs_executable)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise SimilarityAuditError(
            f"pinned MMseqs2 executable is unavailable: {executable}"
        )
    result = subprocess.run(
        [str(executable), "version"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    version = result.stdout.strip()
    if version != policy.mmseqs_version:
        raise SimilarityAuditError(
            f"MMseqs2 version is {version!r}, expected {policy.mmseqs_version!r}"
        )
    return version


def require_disk_capacity(workspace: Path, policy: SimilarityAuditPolicy) -> None:
    """Enforce the workspace ceiling and minimum free-space reserve."""

    workspace_size = 0
    for path in workspace.rglob("*"):
        try:
            if path.is_file():
                workspace_size += path.stat().st_size
        except FileNotFoundError:
            continue
    free_bytes = shutil.disk_usage(workspace).free
    if workspace_size > policy.workspace_byte_ceiling:
        raise SimilarityAuditError(
            "Task 7 workspace exceeded its fixed 200 GiB ceiling"
        )
    if free_bytes < policy.free_space_reserve:
        raise SimilarityAuditError(
            "free disk space fell below the fixed 300 GiB reserve"
        )


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """Prevent two Task 7 processes from sharing one workspace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SimilarityAuditError("another Task 7 audit is already running") from error
        lock.seek(0)
        lock.truncate()
        lock.write(f"pid={os.getpid()}\n")
        lock.flush()
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def require_committed_execution_code(project_root: Path) -> None:
    """Reject a run whose executable code has not been committed."""

    status = git_output(
        project_root,
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
    )
    if status:
        raise SimilarityAuditError(
            "execution code has uncommitted changes; review and commit it first"
        )


def prove_path_is_ignored(path: Path, project_root: Path) -> None:
    """Prove that a private artifact path is ignored by Git."""

    relative = path.resolve().relative_to(project_root.resolve())
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative.as_posix()],
        cwd=project_root,
        check=False,
    )
    if result.returncode != 0:
        raise SimilarityAuditError(f"private path is not ignored by Git: {relative}")


def prove_path_is_public(path: Path, project_root: Path) -> None:
    """Prove that a public artifact path is not ignored by Git."""

    relative = path.resolve().relative_to(project_root.resolve())
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative.as_posix()],
        cwd=project_root,
        check=False,
    )
    if result.returncode == 0:
        raise SimilarityAuditError(f"public path is unexpectedly ignored: {relative}")
    if result.returncode != 1:
        raise SimilarityAuditError(f"could not prove public Git status: {relative}")


def git_output(project_root: Path, *arguments: str) -> str:
    """Return stripped output from one checked Git command."""

    result = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _log_tail(path: Path, lines: int = 20) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "<log unavailable>"
    return "\n".join(content[-lines:])
