"""Resume identity and final-gate tamper characterization."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest

import protein_lm.data.fixed_budget_audit.workflow as workflow_module
from a004_workflow_test_support import install_synthetic_workflow
from protein_lm.data.similarity_audit_policy import SimilarityAuditError


@dataclass(frozen=True)
class TamperCase:
    name: str
    relative_path: str
    inventory_addition: bool = False


TAMPER_CASES = (
    TamperCase("database artifact", "a004/databases/random/target"),
    TamperCase("database marker", "a004/databases/random/complete.json"),
    TamperCase(
        "imported canonical",
        "a003/tracks/random/validation/residual/cap_1000/canonical.tsv",
    ),
    TamperCase(
        "executed canonical",
        "a004/tracks/random/test/enforcement/cap_1000/canonical.tsv",
    ),
    TamperCase(
        "search marker",
        "a004/tracks/random/test/enforcement/cap_1000/complete.json",
    ),
    TamperCase(
        "escalation FASTA",
        "a004/tracks/random/test/enforcement/escalated_queries.fasta",
    ),
    TamperCase(
        "escalation marker",
        "a004/tracks/random/test/enforcement/escalated_queries.complete.json",
    ),
    TamperCase(
        "pass marker",
        "a004/tracks/random/test/enforcement/complete.json",
    ),
    TamperCase(
        "pass cap inventory",
        "a004/tracks/random/test/enforcement/cap_999999/sentinel",
        inventory_addition=True,
    ),
    TamperCase(
        "cap summary pair",
        "a004/evidence/executed_a004/random/test/enforcement/cap_1000/"
        "prohibited_pairs.tsv",
    ),
    TamperCase(
        "cap summary query",
        "a004/evidence/executed_a004/random/test/enforcement/cap_1000/"
        "query_summaries.tsv",
    ),
    TamperCase(
        "cap summary marker",
        "a004/evidence/executed_a004/random/test/enforcement/cap_1000/complete.json",
    ),
    TamperCase(
        "pair union TSV",
        "a004/pair_unions/random/test/common_all_query_10000/prohibited_pairs.tsv",
    ),
    TamperCase(
        "pair union marker",
        "a004/pair_unions/random/test/common_all_query_10000/complete.json",
    ),
    TamperCase("report JSON", "a004/evidence/report/a004_report.json"),
    TamperCase("report Markdown", "a004/evidence/report/a004_report.md"),
    TamperCase("report marker", "a004/evidence/report/complete.json"),
    TamperCase("receipt", "a004/a004_import_receipt.json"),
)


def test_complete_two_run_resume_is_byte_identical_and_invokes_no_runner(
    monkeypatch, tmp_path: Path
) -> None:
    synthetic = install_synthetic_workflow(monkeypatch, tmp_path, changed_search=True)
    imported_before = _snapshot(synthetic.source_workspace)

    first = workflow_module.run_fixed_budget_audit(
        project_root=synthetic.project_root,
        config_path=synthetic.config_path,
        search_runner=synthetic.search_runner,
        database_runner=synthetic.database_runner,
        hardware=synthetic.hardware,
    )
    first_artifacts = _snapshot(synthetic.workspace, exclude={"audit.lock"})
    assert imported_before == _snapshot(synthetic.source_workspace)

    calls = {"database": 0, "search": 0}

    def forbidden_database_runner(*args, **kwargs):
        calls["database"] += 1
        raise AssertionError("resumed workflow invoked the database runner")

    def forbidden_search_runner(*args, **kwargs):
        calls["search"] += 1
        raise AssertionError("resumed workflow invoked the search runner")

    second = workflow_module.run_fixed_budget_audit(
        project_root=synthetic.project_root,
        config_path=synthetic.config_path,
        search_runner=forbidden_search_runner,
        database_runner=forbidden_database_runner,
        hardware=synthetic.hardware,
    )

    assert calls == {"database": 0, "search": 0}
    assert second == first
    assert _snapshot(synthetic.workspace, exclude={"audit.lock"}) == first_artifacts
    assert _snapshot(synthetic.source_workspace) == imported_before


@pytest.mark.parametrize("case", TAMPER_CASES, ids=lambda case: case.name)
def test_final_gate_rejects_each_mutated_artifact_and_preserves_mutation(
    monkeypatch,
    tmp_path: Path,
    case: TamperCase,
) -> None:
    synthetic = install_synthetic_workflow(monkeypatch, tmp_path, changed_search=True)
    publish_receipt = workflow_module.publish_receipt
    mutated: dict[str, bytes] = {}

    def publish_then_tamper(**kwargs):
        publication = publish_receipt(**kwargs)
        path = synthetic.project_root / case.relative_path
        if case.inventory_addition:
            path.parent.mkdir()
            content = b"unexpected cap output\n"
            path.write_bytes(content)
        else:
            content = path.read_bytes() + b"TAMPER\n"
            path.write_bytes(content)
        mutated["content"] = content
        return publication

    monkeypatch.setattr(workflow_module, "publish_receipt", publish_then_tamper)

    with pytest.raises(SimilarityAuditError):
        workflow_module.run_fixed_budget_audit(
            project_root=synthetic.project_root,
            config_path=synthetic.config_path,
            search_runner=synthetic.search_runner,
            database_runner=synthetic.database_runner,
            hardware=synthetic.hardware,
        )

    path = synthetic.project_root / case.relative_path
    assert path.read_bytes() == mutated["content"]
    assert not (synthetic.workspace / "a004_complete.json").exists()


def _snapshot(
    root: Path,
    *,
    exclude: set[str] | None = None,
) -> dict[str, tuple[bytes, int, str]]:
    ignored = exclude or set()
    snapshot = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in ignored:
            continue
        content = path.read_bytes()
        snapshot[relative] = (
            content,
            len(content),
            hashlib.sha256(content).hexdigest(),
        )
    return snapshot
