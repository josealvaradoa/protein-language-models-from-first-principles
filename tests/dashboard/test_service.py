"""Synthetic-only coverage for the local reproduction dashboard control plane."""

from __future__ import annotations

from collections.abc import Mapping
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import threading

import pytest

from protein_lm.dashboard.service import (
    DashboardControlPlane,
    DashboardError,
    GitRevision,
    create_server,
    validate_bind_address,
)


class FakeProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.was_terminated = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.was_terminated = True
        self.returncode = -15


class FakeExecutor:
    def __init__(self) -> None:
        self.processes: list[FakeProcess] = []
        self.environments: list[Mapping[str, str]] = []

    def __call__(
        self,
        _job: object,
        log_path: Path,
        environment: Mapping[str, str],
        _execution_lease: int,
    ) -> FakeProcess:
        log_path.write_text("synthetic dashboard log\n", encoding="utf-8")
        self.environments.append(environment)
        process = FakeProcess()
        self.processes.append(process)
        return process


def revision(branch: str = "feat/week-04-reproduction-dashboard"):
    return lambda _repository: GitRevision(branch=branch, revision="abc123")


def control(tmp_path: Path, executor: FakeExecutor | None = None) -> DashboardControlPlane:
    return DashboardControlPlane(
        tmp_path / "repository",
        tmp_path / "runs",
        executor=executor or FakeExecutor(),
        revision_provider=revision(),
    )


def test_catalog_is_allowlisted_and_has_no_test_collection_value(tmp_path: Path) -> None:
    payload = control(tmp_path).catalog()
    jobs = {item["job_id"]: item for item in payload["jobs"]}

    assert set(jobs) == {
        "setup_check",
        "week2_public_report_validation",
        "week3_public_report_validation",
        "week1_audit_reproduction",
        "week2_reevaluation",
        "week2_retraining_refit",
        "week3_reevaluation",
        "week3_retraining",
        "week1_retraining",
    }
    assert jobs["week2_reevaluation"]["reason"] == "reproduction_contract_pending"
    assert jobs["week3_retraining"]["availability"] == "blocked"
    assert jobs["week1_retraining"]["availability"] == "not_applicable"
    assert jobs["setup_check"]["experiment_id"] == "workspace"
    assert jobs["week1_audit_reproduction"]["stage"] == "reevaluate"
    assert "not a model evaluation" in jobs["week1_audit_reproduction"]["description"]
    assert jobs["week2_retraining_refit"]["stage"] == "retrain"
    assert jobs["week3_reevaluation"]["stage"] == "reevaluate"
    assert jobs["setup_check"]["command_display"].endswith("scripts/check_setup.py")
    assert jobs["week2_reevaluation"]["command_display"] is None
    assert jobs["week1_retraining"]["command_display"] is None
    week1 = payload["historical_results"][0]
    assert week1["metrics"]["eligible_protein_count"] == 557718
    assert week1["metrics"]["group_aware_test_strong_overlap_percent"] == 37.399853
    assert "sealed" not in json.dumps(payload).lower()


def test_runner_persists_one_active_job_and_terminal_state(tmp_path: Path) -> None:
    executor = FakeExecutor()
    dashboard = control(tmp_path, executor)

    first = dashboard.create_run("setup_check")
    run_id = str(first["run_id"])
    assert first["status"] == "running"
    assert (tmp_path / "runs" / run_id / "run.json").is_file()
    assert dashboard.read_log(run_id)["log"] == "synthetic dashboard log\n"
    assert executor.environments[0]["PYTHONPATH"].endswith("repository/src")
    assert "HOME" not in executor.environments[0]

    with pytest.raises(DashboardError, match="only one dashboard job"):
        dashboard.create_run("week2_public_report_validation")

    (tmp_path / "runs" / run_id / "run.json.tmp").write_text("stale", encoding="utf-8")
    executor.processes[0].returncode = 0
    assert dashboard.read_run(run_id)["status"] == "succeeded"
    record = json.loads((tmp_path / "runs" / run_id / "run.json").read_text())
    assert record["exit_code"] == 0
    assert record["git_branch"] == "feat/week-04-reproduction-dashboard"


def test_cancel_marks_a_running_job_terminal_and_releases_lock(tmp_path: Path) -> None:
    executor = FakeExecutor()
    dashboard = control(tmp_path, executor)
    running = dashboard.create_run("setup_check")

    cancelled = dashboard.cancel_run(str(running["run_id"]))

    assert cancelled["status"] == "cancelled"
    assert executor.processes[0].was_terminated
    assert dashboard.create_run("week2_public_report_validation")["status"] == "running"


def test_restart_waits_for_an_inherited_lease_before_reconciling_orphan(
    tmp_path: Path,
) -> None:
    executor = FakeExecutor()
    original = control(tmp_path, executor)
    orphan = original.create_run("setup_check")
    run_id = str(orphan["run_id"])
    inherited_lease = os.dup(original._execution_lease)
    original._release_owned_execution_lease()

    restarted = control(tmp_path, FakeExecutor())
    assert restarted.read_run(run_id)["status"] == "running"
    with pytest.raises(DashboardError, match="only one dashboard job"):
        restarted.create_run("week2_public_report_validation")
    with pytest.raises(DashboardError, match="prior dashboard process"):
        restarted.cancel_run(run_id)

    os.close(inherited_lease)
    assert restarted.create_run("week2_public_report_validation")["status"] == "running"
    reconciled = restarted.read_run(run_id)
    assert reconciled["status"] == "failed"
    assert reconciled["failure_reason"] == "runner_restarted"


def test_cancel_refreshes_exited_process_and_shutdown_releases_parent_lease(
    tmp_path: Path,
) -> None:
    executor = FakeExecutor()
    current = control(tmp_path, executor)
    run = current.create_run("setup_check")
    executor.processes[-1].returncode = 0
    with pytest.raises(DashboardError, match="only a queued or running job"):
        current.cancel_run(str(run["run_id"]))
    assert current.read_run(str(run["run_id"]))["status"] == "succeeded"

    active = current.create_run("week2_public_report_validation")
    inherited_lease = os.dup(current._execution_lease)
    current.shutdown()
    assert executor.processes[-1].was_terminated
    assert current.read_run(str(active["run_id"]))["status"] == "cancelled"

    restarted = control(tmp_path, FakeExecutor())
    with pytest.raises(DashboardError, match="only one dashboard job"):
        restarted.create_run("setup_check")
    os.close(inherited_lease)
    assert restarted.create_run("setup_check")["status"] == "running"


def test_run_records_reject_bad_ids_and_derive_log_path_from_run_directory(
    tmp_path: Path,
) -> None:
    executor = FakeExecutor()
    dashboard = control(tmp_path, executor)
    run = dashboard.create_run("setup_check")
    run_id = str(run["run_id"])
    record_path = tmp_path / "runs" / run_id / "run.json"
    tampered = json.loads(record_path.read_text())
    tampered["log_path"] = "../../outside.log"
    record_path.write_text(json.dumps(tampered), encoding="utf-8")
    (tmp_path / "outside.log").write_text("must not be read", encoding="utf-8")

    reloaded = control(tmp_path, executor)
    assert reloaded.read_run(run_id)["log_path"] == f"{run_id}/log.txt"
    assert reloaded.read_log(run_id)["log"] == "synthetic dashboard log\n"
    for invalid_id in ("../outside", "A" * 32, "not-a-run", f"{run_id}0"):
        with pytest.raises(DashboardError, match="run was not found"):
            reloaded.read_run(invalid_id)

    malformed_id = "a" * 32
    malformed_directory = tmp_path / "runs" / malformed_id
    malformed_directory.mkdir()
    (malformed_directory / "run.json").write_text(
        json.dumps({"run_id": "b" * 32}), encoding="utf-8"
    )
    assert malformed_id not in {item["run_id"] for item in reloaded.list_runs()}
    with pytest.raises(DashboardError, match="run was not found"):
        reloaded.read_run(malformed_id)


@pytest.mark.parametrize("branch", ["main", "master", "detached", "", "unknown"])
def test_runner_refuses_non_feature_branches(tmp_path: Path, branch: str) -> None:
    protected = DashboardControlPlane(
        tmp_path / "repository",
        tmp_path / "runs",
        executor=FakeExecutor(),
        revision_provider=revision(branch),
    )

    with pytest.raises(DashboardError, match="named feature branch"):
        protected.create_run("setup_check")
    with pytest.raises(DashboardError, match="reproduction_contract_pending"):
        control(tmp_path / "other").create_run("week3_retraining")


@pytest.mark.parametrize("address", ["0.0.0.0", "localhost", "::1", "192.168.1.10"])
def test_loopback_only_binding_rejects_other_addresses(address: str) -> None:
    validate_bind_address("127.0.0.1")
    with pytest.raises(ValueError, match="127.0.0.1"):
        validate_bind_address(address)


def _request(
    port: int,
    method: str,
    path: str,
    body: object | None = None,
    headers: Mapping[str, str] | None = None,
) -> tuple[int, dict[str, str], dict[str, object]]:
    connection = HTTPConnection("127.0.0.1", port, timeout=2)
    encoded = None if body is None else json.dumps(body).encode("utf-8")
    request_headers = dict(headers or {})
    if encoded is not None:
        request_headers["Content-Type"] = "application/json"
    connection.request(method, path, body=encoded, headers=request_headers)
    response = connection.getresponse()
    payload = json.loads(response.read())
    headers_out = dict(response.getheaders())
    connection.close()
    return response.status, headers_out, payload


def _raw_request(port: int, path: str) -> tuple[int, dict[str, str], bytes]:
    connection = HTTPConnection("127.0.0.1", port, timeout=2)
    connection.request("GET", path)
    response = connection.getresponse()
    body = response.read()
    headers_out = dict(response.getheaders())
    connection.close()
    return response.status, headers_out, body


def test_static_allowlist_has_security_headers_and_no_generic_file_route(tmp_path: Path) -> None:
    server = create_server(control(tmp_path), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        for path, content_type, expected in (
            ("/", "text/html; charset=utf-8", b"Reproduction workbench"),
            ("/assets/dashboard.css", "text/css; charset=utf-8", b"--ink"),
            ("/assets/dashboard.js", "text/javascript; charset=utf-8", b"initialize"),
        ):
            status, headers, body = _raw_request(port, path)
            assert status == 200
            assert headers["Content-Type"] == content_type
            assert headers["Cache-Control"] == "no-store"
            assert headers["X-Content-Type-Options"] == "nosniff"
            assert headers["Referrer-Policy"] == "no-referrer"
            assert headers["X-Frame-Options"] == "DENY"
            assert "default-src 'self'" in headers["Content-Security-Policy"]
            assert expected in body

        status, _, payload = _request(port, "GET", "/src/protein_lm/dashboard/static/index.html")
        assert status == 404
        assert payload["error"]["code"] == "not_found"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_bootstrap_includes_safe_workspace_display_metadata(tmp_path: Path) -> None:
    dashboard = control(tmp_path)

    bootstrap = dashboard.bootstrap()

    assert bootstrap["workspace"] == {
        "git_branch": "feat/week-04-reproduction-dashboard",
        "git_revision": "abc123",
        "launch_permitted": True,
    }
    assert "repository" not in json.dumps(bootstrap["workspace"])
    assert "PATH" not in json.dumps(bootstrap["workspace"])


def test_dashboard_client_keeps_a_local_fixed_command_boundary() -> None:
    static_root = Path(__file__).parents[2] / "src" / "protein_lm" / "dashboard" / "static"
    html = (static_root / "index.html").read_text(encoding="utf-8")
    script = (static_root / "dashboard.js").read_text(encoding="utf-8")

    assert "http://" not in html + script
    assert "https://" not in html + script
    assert "innerHTML" not in script
    assert 'role="status"' in html
    assert 'role="alert"' in html
    assert "<dialog" in html
    assert "JSON.stringify({ job_id: job.job_id })" in script


def test_http_mutations_require_same_origin_host_and_csrf(tmp_path: Path) -> None:
    dashboard = control(tmp_path)
    server = create_server(dashboard, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    origin = f"http://127.0.0.1:{port}"
    try:
        status, _, payload = _request(
            port,
            "GET",
            "/api/bootstrap",
            headers={"Host": "example.invalid"},
        )
        assert status == 403
        assert payload["error"]["code"] == "host_rejected"

        status, headers, bootstrap = _request(port, "GET", "/api/bootstrap")
        assert status == 200
        assert "Access-Control-Allow-Origin" not in headers
        token = str(bootstrap["csrf_token"])

        status, _, payload = _request(port, "POST", "/api/runs", {"job_id": "setup_check"})
        assert status == 403
        assert payload["error"]["code"] == "origin_rejected"

        status, _, payload = _request(
            port,
            "POST",
            "/api/runs",
            {"job_id": "setup_check"},
            {"Origin": origin},
        )
        assert status == 403
        assert payload["error"]["code"] == "csrf_rejected"

        status, _, payload = _request(
            port,
            "POST",
            "/api/runs",
            {"job_id": "setup_check", "argv": ["untrusted"]},
            {"Origin": origin, "X-CSRF-Token": token},
        )
        assert status == 400
        assert payload["error"]["code"] == "invalid_request"

        status, _, payload = _request(
            port,
            "POST",
            "/api/runs",
            {"job_id": "setup_check"},
            {"Origin": origin, "X-CSRF-Token": token},
        )
        assert status == 201
        assert payload["run"]["job_id"] == "setup_check"

        status, _, payload = _request(
            port,
            "POST",
            "/api/runs",
            {"job_id": "setup_check"},
            {"Origin": origin, "Host": "example.invalid", "X-CSRF-Token": token},
        )
        assert status == 403
        assert payload["error"]["code"] == "host_rejected"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
