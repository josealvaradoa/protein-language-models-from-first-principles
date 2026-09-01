"""Small local HTTP API and process runner for reproducible dashboard jobs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
import fcntl
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import subprocess
import tempfile
import threading
from typing import Any, Protocol
from urllib.parse import urlsplit
import uuid

from protein_lm.dashboard.catalog import JobDefinition, catalog_payload, find_job


TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled", "blocked"})
ACTIVE_STATES = frozenset({"queued", "running"})
RUN_ID_PATTERN = re.compile(r"[0-9a-f]{32}\Z")
VALID_TRANSITIONS = {
    "queued": frozenset({"running", "failed", "cancelled"}),
    "running": frozenset({"succeeded", "failed", "cancelled"}),
    "succeeded": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
    "blocked": frozenset(),
}

STATIC_ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/assets/dashboard.css": ("dashboard.css", "text/css; charset=utf-8"),
    "/assets/dashboard.js": ("dashboard.js", "text/javascript; charset=utf-8"),
}


class DashboardError(Exception):
    """A client-safe control-plane failure."""

    def __init__(self, code: str, message: str, status: HTTPStatus) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


class ProcessHandle(Protocol):
    """Minimal subprocess surface the runner needs and tests can fake."""

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...


@dataclass(frozen=True, slots=True)
class GitRevision:
    branch: str
    revision: str


Executor = Callable[[JobDefinition, Path, Mapping[str, str], int], ProcessHandle]
RevisionProvider = Callable[[Path], GitRevision]


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _repository_revision(repository: Path) -> GitRevision:
    def git(*args: str) -> str:
        result = subprocess.run(
            ("git", *args),
            cwd=repository,
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        return GitRevision(
            branch=git("branch", "--show-current") or "detached",
            revision=git("rev-parse", "HEAD"),
        )
    except (OSError, subprocess.CalledProcessError):
        return GitRevision(branch="unknown", revision="unknown")


def _safe_environment(repository: Path) -> dict[str, str]:
    """Keep only command discovery plus project imports for fixed local jobs."""

    source_path = str(repository / "src")
    return {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": source_path,
    }


def _default_executor(
    repository: Path,
) -> Executor:
    def execute(
        job: JobDefinition,
        log_path: Path,
        environment: Mapping[str, str],
        execution_lease: int,
    ) -> ProcessHandle:
        with log_path.open("ab") as log_file:
            return subprocess.Popen(
                job.argv,
                cwd=repository,
                env=dict(environment),
                shell=False,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                pass_fds=(execution_lease,),
            )

    return execute


class DashboardControlPlane:
    """Coordinates fixed jobs, durable run records, and a single active process."""

    def __init__(
        self,
        repository: Path,
        run_root: Path | None = None,
        *,
        executor: Executor | None = None,
        revision_provider: RevisionProvider = _repository_revision,
    ) -> None:
        self.repository = repository.resolve()
        self.run_root = (run_root or self.repository / "runs" / "dashboard").resolve()
        self._executor = executor or _default_executor(self.repository)
        self._revision_provider = revision_provider
        self._csrf_token = secrets.token_urlsafe(32)
        self._lock = threading.RLock()
        self._processes: dict[str, ProcessHandle] = {}
        self._records: dict[str, dict[str, object]] = {}
        self._active_run_id: str | None = None
        self._execution_lease: int | None = None
        self._orphaned_active_run_ids: set[str] = set()
        self._reconcile_orphaned_active_records()

    @property
    def csrf_token(self) -> str:
        return self._csrf_token

    def bootstrap(self) -> dict[str, object]:
        revision = self._revision_provider(self.repository)
        return {
            "csrf_token": self.csrf_token,
            "catalog": catalog_payload(),
            "workspace": {
                "git_branch": revision.branch,
                "git_revision": revision.revision,
                "launch_permitted": _is_launch_branch(revision.branch),
            },
        }

    def catalog(self) -> dict[str, object]:
        return catalog_payload()

    def create_run(self, job_id: str) -> dict[str, object]:
        if not isinstance(job_id, str):
            raise DashboardError("invalid_request", "job_id must be a string", HTTPStatus.BAD_REQUEST)
        job = find_job(job_id)
        if job is None:
            raise DashboardError("unknown_job", "job_id is not allowlisted", HTTPStatus.NOT_FOUND)
        if not job.launchable:
            raise DashboardError(
                "job_unavailable",
                job.reason or "job is unavailable",
                HTTPStatus.CONFLICT,
            )
        with self._lock:
            self._refresh_active_run()
            revision = self._revision_provider(self.repository)
            if not _is_launch_branch(revision.branch):
                raise DashboardError(
                    "protected_branch",
                    "dashboard jobs require a named feature branch",
                    HTTPStatus.CONFLICT,
                )
            if self._active_run_id is not None:
                raise DashboardError(
                    "active_run_exists",
                    "only one dashboard job may run at a time",
                    HTTPStatus.CONFLICT,
                )
            execution_lease = self._try_acquire_execution_lease()
            if execution_lease is None:
                raise DashboardError(
                    "active_run_exists",
                    "only one dashboard job may run at a time",
                    HTTPStatus.CONFLICT,
                )
            self._execution_lease = execution_lease
            run_id, run_directory = self._new_run_directory()
            log_path = run_directory / "log.txt"
            record: dict[str, object] = {
                "run_id": run_id,
                "job_id": job.job_id,
                "stage": job.stage,
                "command_display": shlex.join(job.argv),
                "git_branch": revision.branch,
                "git_revision": revision.revision,
                "created_at": _timestamp(),
                "started_at": None,
                "finished_at": None,
                "exit_code": None,
                "status": "queued",
                "log_path": str(log_path.relative_to(self.run_root)),
            }
            self._records[run_id] = record
            self._persist(record)
            log_path.touch(exist_ok=False)
            self._transition(record, "running")
            record["started_at"] = _timestamp()
            try:
                self._processes[run_id] = self._executor(
                    job,
                    log_path,
                    _safe_environment(self.repository),
                    execution_lease,
                )
            except OSError as error:
                self._transition(record, "failed")
                record["finished_at"] = _timestamp()
                record["failure_reason"] = f"could not start fixed job: {error.__class__.__name__}"
            self._persist(record)
            if record["status"] == "running":
                self._active_run_id = run_id
            else:
                self._release_owned_execution_lease()
            return self._public_record(record)

    def shutdown(self) -> None:
        """Stop this process's child and relinquish this process's lease handle."""

        with self._lock:
            for run_id, process in tuple(self._processes.items()):
                record = self._records.get(run_id)
                if process.poll() is None:
                    process.terminate()
                if record is not None and record["status"] in ACTIVE_STATES:
                    self._transition(record, "cancelled")
                    record["finished_at"] = _timestamp()
                    record["exit_code"] = process.poll()
                    self._persist(record)
            self._processes.clear()
            self._active_run_id = None
            self._release_owned_execution_lease()

    def list_runs(self) -> list[dict[str, object]]:
        with self._lock:
            self._refresh_active_run()
            records = [self._public_record(record) for record in self._stored_records()]
            return sorted(records, key=lambda record: str(record["created_at"]), reverse=True)

    def read_run(self, run_id: str) -> dict[str, object]:
        with self._lock:
            self._refresh_active_run()
            return self._public_record(self._record_for(run_id))

    def read_log(self, run_id: str) -> dict[str, object]:
        with self._lock:
            self._record_for(run_id)
            log_path = self._run_directory(run_id) / "log.txt"
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")
            except FileNotFoundError:
                text = ""
            return {"run_id": run_id, "log": text[-1_000_000:]}

    def cancel_run(self, run_id: str) -> dict[str, object]:
        with self._lock:
            self._refresh_active_run()
            record = self._record_for(run_id)
            if record["status"] not in ACTIVE_STATES:
                raise DashboardError(
                    "not_active",
                    "only a queued or running job can be cancelled",
                    HTTPStatus.CONFLICT,
                )
            if run_id in self._orphaned_active_run_ids:
                raise DashboardError(
                    "runner_restarted",
                    "a prior dashboard process owns this running job",
                    HTTPStatus.CONFLICT,
                )
            process = self._processes.get(run_id)
            if process is not None:
                process.terminate()
            self._transition(record, "cancelled")
            record["finished_at"] = _timestamp()
            if process is not None:
                record["exit_code"] = process.poll()
            self._persist(record)
            return self._public_record(record)

    def _new_run_directory(self) -> tuple[str, Path]:
        self.run_root.mkdir(parents=True, exist_ok=True)
        while True:
            run_id = uuid.uuid4().hex
            directory = self.run_root / run_id
            try:
                directory.mkdir()
            except FileExistsError:
                continue
            return run_id, directory

    def _record_for(self, run_id: str) -> dict[str, object]:
        if not _is_run_id(run_id):
            raise DashboardError("unknown_run", "run was not found", HTTPStatus.NOT_FOUND)
        record = self._records.get(run_id)
        if record is not None:
            if self._valid_record(record, run_id):
                return record
            self._records.pop(run_id, None)
        record_path = self._run_directory(run_id) / "run.json"
        try:
            loaded = json.loads(record_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            raise DashboardError("unknown_run", "run was not found", HTTPStatus.NOT_FOUND) from None
        if not self._valid_record(loaded, run_id):
            raise DashboardError("unknown_run", "run was not found", HTTPStatus.NOT_FOUND)
        self._records[run_id] = loaded
        return loaded

    def _stored_records(self) -> list[dict[str, object]]:
        if not self.run_root.exists():
            return list(self._records.values())
        records: list[dict[str, object]] = []
        for path in self.run_root.glob("*/run.json"):
            try:
                candidate = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            run_id = path.parent.name
            try:
                directory = self._run_directory(run_id)
            except DashboardError:
                continue
            if path.parent != directory:
                continue
            if self._valid_record(candidate, run_id):
                self._records[run_id] = candidate
                records.append(candidate)
        return records

    def _reconcile_orphaned_active_records(self) -> None:
        """Reconcile active records only after the inherited execution lease clears."""

        active = [
            record
            for record in self._stored_records()
            if record["status"] in ACTIVE_STATES
            and str(record["run_id"]) not in self._processes
        ]
        if not active:
            self._orphaned_active_run_ids.clear()
            return
        lease = self._try_acquire_execution_lease()
        if lease is None:
            self._orphaned_active_run_ids = {
                str(record["run_id"]) for record in active
            }
            return
        try:
            for record in active:
                self._transition(record, "failed")
                record["finished_at"] = _timestamp()
                record["failure_reason"] = "runner_restarted"
                self._persist(record)
        finally:
            self._release_execution_lease(lease)
        self._orphaned_active_run_ids.clear()

    def _try_acquire_execution_lease(self) -> int | None:
        self.run_root.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self.run_root / ".execution.lease",
            os.O_CREAT | os.O_RDWR,
            0o600,
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            return None
        return descriptor

    @staticmethod
    def _release_execution_lease(descriptor: int) -> None:
        # Do not explicitly unlock: this descriptor may be shared with an
        # inherited child. Closing only this copy preserves the child lease.
        os.close(descriptor)

    def _release_owned_execution_lease(self) -> None:
        if self._execution_lease is None:
            return
        self._release_execution_lease(self._execution_lease)
        self._execution_lease = None

    def _run_directory(self, run_id: str) -> Path:
        if not _is_run_id(run_id):
            raise DashboardError("unknown_run", "run was not found", HTTPStatus.NOT_FOUND)
        directory = self.run_root / run_id
        try:
            resolved = directory.resolve(strict=False)
        except OSError:
            raise DashboardError("unknown_run", "run was not found", HTTPStatus.NOT_FOUND) from None
        if resolved.parent != self.run_root:
            raise DashboardError("unknown_run", "run was not found", HTTPStatus.NOT_FOUND)
        return directory

    @staticmethod
    def _valid_record(candidate: object, run_id: str) -> bool:
        if not _is_run_id(run_id) or not isinstance(candidate, dict):
            return False
        if candidate.get("run_id") != run_id:
            return False
        required_strings = (
            "job_id",
            "stage",
            "command_display",
            "git_branch",
            "git_revision",
            "created_at",
            "status",
        )
        if any(not isinstance(candidate.get(field), str) for field in required_strings):
            return False
        if candidate["status"] not in VALID_TRANSITIONS:
            return False
        return all(
            candidate.get(field) is None or isinstance(candidate.get(field), str)
            for field in ("started_at", "finished_at")
        ) and (
            candidate.get("exit_code") is None
            or isinstance(candidate.get("exit_code"), int)
        )

    def _refresh_active_run(self) -> None:
        if self._active_run_id is None:
            self._reconcile_orphaned_active_records()
            return
        record = self._records[self._active_run_id]
        process = self._processes.get(self._active_run_id)
        if process is None:
            self._reconcile_orphaned_active_records()
            return
        exit_code = process.poll()
        if exit_code is None:
            return
        if record["status"] != "cancelled":
            self._transition(record, "succeeded" if exit_code == 0 else "failed")
            record["finished_at"] = _timestamp()
        record["exit_code"] = exit_code
        self._persist(record)
        self._processes.pop(self._active_run_id, None)
        self._active_run_id = None
        self._release_owned_execution_lease()
        self._reconcile_orphaned_active_records()

    def _transition(self, record: dict[str, object], destination: str) -> None:
        current = str(record["status"])
        if destination not in VALID_TRANSITIONS[current]:
            raise RuntimeError(f"invalid dashboard state transition: {current} to {destination}")
        record["status"] = destination

    def _persist(self, record: Mapping[str, object]) -> None:
        run_id = record.get("run_id")
        if not isinstance(run_id, str):
            raise DashboardError("unknown_run", "run was not found", HTTPStatus.NOT_FOUND)
        directory = self._run_directory(run_id)
        target = directory / "run.json"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".run.json.",
            suffix=".tmp",
            dir=directory,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2, sort_keys=True)
                handle.write("\n")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    def _public_record(self, record: Mapping[str, object]) -> dict[str, object]:
        run_id = record["run_id"]
        if not isinstance(run_id, str) or not self._valid_record(record, run_id):
            raise DashboardError("unknown_run", "run was not found", HTTPStatus.NOT_FOUND)
        public = dict(record)
        public["log_path"] = f"{run_id}/log.txt"
        return public


def validate_bind_address(host: str) -> None:
    """Permit exactly the local IPv4 loopback address used by this dashboard."""

    if host != "127.0.0.1":
        raise ValueError("dashboard may bind only to 127.0.0.1")


def _is_run_id(run_id: object) -> bool:
    return isinstance(run_id, str) and RUN_ID_PATTERN.fullmatch(run_id) is not None


def _is_launch_branch(branch: str) -> bool:
    return branch.strip() not in {"", "main", "master", "detached", "unknown"}


def _has_expected_host(handler: BaseHTTPRequestHandler) -> bool:
    expected_port = handler.server.server_address[1]
    return handler.headers.get("Host", "") == f"127.0.0.1:{expected_port}"


def _is_same_origin(handler: BaseHTTPRequestHandler) -> bool:
    origin_header = handler.headers.get("Origin", "")
    expected_port = handler.server.server_address[1]
    if not _has_expected_host(handler):
        return False
    try:
        origin = urlsplit(origin_header)
        return (
            origin.scheme == "http"
            and origin.hostname == "127.0.0.1"
            and origin.port == expected_port
            and origin.path == ""
            and origin.query == ""
            and origin.fragment == ""
        )
    except ValueError:
        return False


def make_handler(control: DashboardControlPlane) -> type[BaseHTTPRequestHandler]:
    """Create a handler bound to a single in-process control plane."""

    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "ProteinLMDashboard/1.0"

        def do_GET(self) -> None:
            try:
                self._authorize_host()
                if self.path in STATIC_ASSETS:
                    self._send_static(self.path)
                elif self.path == "/api/bootstrap":
                    self._send_json(HTTPStatus.OK, control.bootstrap())
                elif self.path == "/api/catalog":
                    self._send_json(HTTPStatus.OK, control.catalog())
                elif self.path == "/api/runs":
                    self._send_json(HTTPStatus.OK, {"runs": control.list_runs()})
                elif self.path.startswith("/api/runs/") and self.path.endswith("/log"):
                    run_id = self.path.removeprefix("/api/runs/").removesuffix("/log").rstrip("/")
                    self._send_json(HTTPStatus.OK, control.read_log(run_id))
                elif self.path.startswith("/api/runs/"):
                    run_id = self.path.removeprefix("/api/runs/")
                    self._send_json(HTTPStatus.OK, {"run": control.read_run(run_id)})
                else:
                    self._error("not_found", "endpoint was not found", HTTPStatus.NOT_FOUND)
            except DashboardError as error:
                self._error(error.code, error.message, error.status)

        def do_POST(self) -> None:
            try:
                self._authorize_host()
                self._authorize_mutation()
                if self.path == "/api/runs":
                    body = self._json_body()
                    if set(body) != {"job_id"}:
                        raise DashboardError(
                            "invalid_request",
                            "request must contain only job_id",
                            HTTPStatus.BAD_REQUEST,
                        )
                    self._send_json(HTTPStatus.CREATED, {"run": control.create_run(body["job_id"])})
                elif self.path.startswith("/api/runs/") and self.path.endswith("/cancel"):
                    run_id = self.path.removeprefix("/api/runs/").removesuffix("/cancel").rstrip("/")
                    self._send_json(HTTPStatus.OK, {"run": control.cancel_run(run_id)})
                else:
                    self._error("not_found", "endpoint was not found", HTTPStatus.NOT_FOUND)
            except DashboardError as error:
                self._error(error.code, error.message, error.status)

        def _authorize_mutation(self) -> None:
            if not _is_same_origin(self):
                raise DashboardError("origin_rejected", "same-origin request required", HTTPStatus.FORBIDDEN)
            supplied = self.headers.get("X-CSRF-Token", "")
            if not secrets.compare_digest(supplied, control.csrf_token):
                raise DashboardError("csrf_rejected", "valid CSRF token required", HTTPStatus.FORBIDDEN)

        def _authorize_host(self) -> None:
            if not _has_expected_host(self):
                raise DashboardError("host_rejected", "local Host header required", HTTPStatus.FORBIDDEN)

        def _json_body(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise DashboardError("invalid_request", "invalid content length", HTTPStatus.BAD_REQUEST) from None
            if not 0 < length <= 16_384:
                raise DashboardError("invalid_request", "invalid request body length", HTTPStatus.BAD_REQUEST)
            try:
                decoded = json.loads(self.rfile.read(length))
            except json.JSONDecodeError:
                raise DashboardError("invalid_request", "request body must be JSON", HTTPStatus.BAD_REQUEST) from None
            if not isinstance(decoded, dict):
                raise DashboardError("invalid_request", "request body must be an object", HTTPStatus.BAD_REQUEST)
            return decoded

        def _send_json(self, status: HTTPStatus, payload: Mapping[str, object]) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self._send_security_headers()
            self.end_headers()
            self.wfile.write(body)

        def _send_static(self, request_path: str) -> None:
            asset_name, content_type = STATIC_ASSETS[request_path]
            body = (
                files("protein_lm.dashboard")
                .joinpath("static", asset_name)
                .read_bytes()
            )
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self._send_security_headers()
            self.end_headers()
            self.wfile.write(body)

        def _send_security_headers(self) -> None:
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; base-uri 'none'; form-action 'self'; "
                "frame-ancestors 'none'; object-src 'none'",
            )
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")

        def _error(self, code: str, message: str, status: HTTPStatus) -> None:
            self._send_json(status, {"error": {"code": code, "message": message}})

        def log_message(self, format: str, *args: object) -> None:
            return

    return DashboardHandler


def create_server(
    control: DashboardControlPlane,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    """Build, but do not start, the local-only dashboard HTTP server."""

    validate_bind_address(host)
    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    return ThreadingHTTPServer((host, port), make_handler(control))
