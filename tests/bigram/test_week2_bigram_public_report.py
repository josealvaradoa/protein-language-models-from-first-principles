"""Synthetic tests for the operator-gated Week 2 aggregate public report."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from protein_lm.bigram.evaluation_contract import config_sha256, load_evaluation_config
from protein_lm.bigram.evaluation_plan import EvaluationPlan
from protein_lm.bigram.evaluation_results import result_payload
from protein_lm.bigram.public_report import (
    derived_comparisons,
    reject_forbidden_keys,
    render_markdown,
    report_payload,
    write_evidence,
)
from protein_lm.bigram.public_report_contract import load_public_report_config
from protein_lm.bigram.public_report_publication import (
    PublicReportPlan,
    execute_publication,
)
import protein_lm.bigram.public_report_io as report_io_module
from protein_lm.bigram import public_report_publication, public_report_validation
from protein_lm.data.model_data.contracts import ModelDataError


ROOT = Path(__file__).parents[2]
PUBLIC_CONFIG_PATH = ROOT / "experiments/week_02/bigram_evaluation_publication_v1.toml"
EVALUATION_CONFIG_PATH = ROOT / "experiments/week_02/bigram_evaluation_v1.toml"
PUBLICATION_REVISION = "d" * 40


def _aggregate(ce: float, correct: int) -> dict[str, object]:
    return {
        "token_count": 10,
        "protein_count": 1,
        "total_nll": ce * 10.0,
        "correct_tokens": correct,
        "cross_entropy": ce,
        "accuracy": correct / 10.0,
        "median_per_protein_nll": ce,
        "median_lower_per_protein_nll": ce,
        "median_upper_per_protein_nll": ce,
    }


def _records() -> list[dict[str, object]]:
    config = load_evaluation_config(EVALUATION_CONFIG_PATH)
    records = []
    for arm, collection in (
        ("random_training", "random_native_validation"),
        ("random_training", "shared_validation"),
        ("family_aware_training", "family_aware_native_validation"),
        ("family_aware_training", "shared_validation"),
    ):
        for model, base, correct in (
            ("unigram", 3.0, 3),
            ("count_bigram", 2.5, 4),
            ("neural_bigram", 2.0, 5),
        ):
            offset = 0.1 if collection == "shared_validation" else 0.0
            offset += 0.05 if arm == "family_aware_training" else 0.0
            buckets = {
                bucket: _aggregate(base + offset, correct)
                for bucket in config.length_buckets
            }
            overall = _aggregate(base + offset, correct)
            overall.update(
                {
                    "token_count": 50,
                    "protein_count": 5,
                    "total_nll": (base + offset) * 50.0,
                    "correct_tokens": correct * 5,
                }
            )
            records.append(
                {
                    "model_arm": arm,
                    "model_type": model,
                    "collection": collection,
                    "metrics": {"overall": overall, "length_buckets": buckets},
                }
            )
    return records


def _source_and_plan(
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object], PublicReportPlan]:
    evaluation_config = load_evaluation_config(EVALUATION_CONFIG_PATH)
    source = result_payload(
        config=evaluation_config,
        config_sha256=config_sha256(EVALUATION_CONFIG_PATH),
        evaluation_id="week2-bigram-eval-v1-001",
        records=_records(),
    )
    run = {
        "status": "passed",
        "runtime_seconds": 1.25,
        "collection_loads": {
            "random_native_validation": 1,
            "family_aware_native_validation": 1,
            "shared_validation": 1,
            "shared_sealed_test": 0,
        },
        "hard_gates": {
            "candidate_validation": True,
            "twelve_principal_records": True,
            "shared_validation_loaded_once": True,
            "sealed_test_never_loaded": True,
            "evaluation_only_no_retraining_or_selection": True,
            "no_network_requests": True,
        },
        "network_requests_made": 0,
        "failure_reason": None,
    }
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "evaluation.json").write_text(
        json.dumps(source, allow_nan=False), encoding="utf-8"
    )
    (source_dir / "run_record.json").write_text(json.dumps(run), encoding="utf-8")
    public_config = load_public_report_config(PUBLIC_CONFIG_PATH)
    public_config = replace(
        public_config,
        source_evaluation_relative_path="source",
        source_evaluation_sha256="a" * 64,
        source_run_record_sha256="b" * 64,
        source_registry_sha256="c" * 64,
        report_json_relative_path="report.json",
        report_markdown_relative_path="report.md",
        report_sha256_relative_path="report.sha256",
    )
    evaluation_plan = EvaluationPlan(
        root=tmp_path,
        evaluation_id="week2-bigram-eval-v1-001",
        destination=source_dir,
        config=evaluation_config,
        config_sha256=config_sha256(EVALUATION_CONFIG_PATH),
        model_candidate=tmp_path / "model",
        model_data_registry=tmp_path / "registry",
    )
    config_path = tmp_path / "experiments/week_02/bigram_evaluation_publication_v1.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes(PUBLIC_CONFIG_PATH.read_bytes())
    plan = PublicReportPlan(
        root=tmp_path,
        config_path=config_path,
        config=public_config,
        evaluation_plan=evaluation_plan,
        output_paths=(
            tmp_path / "report.json",
            tmp_path / "report.md",
            tmp_path / "report.sha256",
        ),
    )
    return source, run, plan


def test_payload_has_exact_twelve_records_derived_comparisons_and_stable_renderer(
    tmp_path: Path,
) -> None:
    source, run, plan = _source_and_plan(tmp_path)
    payload = report_payload(
        config_path=plan.config_path,
        config=plan.config,
        source=source,
        run=run,
        publication_code_revision=PUBLICATION_REVISION,
    )
    assert len(payload["records"]) == 12
    assert payload["publication_code_revision"] == PUBLICATION_REVISION
    assert payload["hypothesis"] == source["hypothesis"]
    assert payload["derived_comparisons"] == derived_comparisons(source["records"])  # type: ignore[arg-type]
    assert render_markdown(payload) == render_markdown(payload)
    assert "Family-aware shared-validation length buckets" in render_markdown(payload)


def test_publication_contract_pins_the_real_source_and_output_paths() -> None:
    config = load_public_report_config(PUBLIC_CONFIG_PATH)
    assert config.source_evaluation_sha256 == (
        "b531e45391e4f7e8ae30a031fb0ef8dc14beaca37279d8fbcda6a226344b2bf8"
    )
    assert config.source_evaluation_config_sha256 == (
        "219e7a3bc06a6c227ed27b9b4b7e917083b537bd5ac5d11a7526ee8415c2d97c"
    )
    assert config.output_paths == (
        "reports/week_02/bigram_evaluation_v1.json",
        "reports/week_02/bigram_evaluation_v1.md",
        "reports/week_02/bigram_evaluation_v1.sha256",
    )


@pytest.mark.parametrize("revision", ("A" * 40, "a" * 39, "a" * 41))
def test_payload_rejects_invalid_publication_revision(
    tmp_path: Path, revision: str
) -> None:
    source, run, plan = _source_and_plan(tmp_path)
    with pytest.raises(ModelDataError, match="publication code revision"):
        report_payload(
            config_path=plan.config_path,
            config=plan.config,
            source=source,
            run=run,
            publication_code_revision=revision,
        )


@pytest.mark.parametrize("field", ("status", "hard_gates", "collection_loads"))
def test_payload_rejects_a_source_run_that_is_not_passed(
    tmp_path: Path, field: str
) -> None:
    source, run, plan = _source_and_plan(tmp_path)
    changed = dict(run)
    changed[field] = "failed" if field == "status" else {}
    with pytest.raises(ModelDataError, match="not a passed evaluation"):
        report_payload(
            config_path=plan.config_path,
            config=plan.config,
            source=source,
            run=changed,
            publication_code_revision=PUBLICATION_REVISION,
        )


def test_writer_installs_three_files_with_two_checksum_lines_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    source, run, plan = _source_and_plan(tmp_path)
    payload = report_payload(
        config_path=plan.config_path,
        config=plan.config,
        source=source,
        run=run,
        publication_code_revision=PUBLICATION_REVISION,
    )
    write_evidence(plan.output_paths, payload)
    assert all(path.is_file() for path in plan.output_paths)
    assert len(plan.output_paths[2].read_text(encoding="utf-8").splitlines()) == 2
    with pytest.raises(ModelDataError, match="already exists"):
        write_evidence(plan.output_paths, payload)


def test_writer_rolls_back_if_one_of_three_installations_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, run, plan = _source_and_plan(tmp_path)
    payload = report_payload(
        config_path=plan.config_path,
        config=plan.config,
        source=source,
        run=run,
        publication_code_revision=PUBLICATION_REVISION,
    )
    real_link = report_io_module.os.link
    calls = 0

    def fail_second(source_path: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic install failure")
        real_link(source_path, destination)

    monkeypatch.setattr(report_io_module.os, "link", fail_second)
    with pytest.raises(ModelDataError, match="could not install"):
        write_evidence(plan.output_paths, payload)
    assert not any(path.exists() for path in plan.output_paths)


@pytest.mark.parametrize(
    "tamper", ("nan", "boolean", "forbidden", "derived", "markdown", "checksum")
)
def test_read_only_validator_rejects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    source, run, plan = _source_and_plan(tmp_path)
    payload = report_payload(
        config_path=plan.config_path,
        config=plan.config,
        source=source,
        run=run,
        publication_code_revision=PUBLICATION_REVISION,
    )
    write_evidence(plan.output_paths, payload)
    monkeypatch.setattr(public_report_validation, "preflight", lambda _root: plan)
    if tamper == "nan":
        plan.output_paths[0].write_text('{"total": NaN}', encoding="utf-8")
    elif tamper == "boolean":
        changed = json.loads(plan.output_paths[0].read_text())
        changed["records"][0]["metrics"]["overall"]["token_count"] = True
        plan.output_paths[0].write_text(json.dumps(changed), encoding="utf-8")
    elif tamper == "forbidden":
        changed = json.loads(plan.output_paths[0].read_text())
        changed["sequence"] = "ACD"
        plan.output_paths[0].write_text(json.dumps(changed), encoding="utf-8")
    elif tamper == "derived":
        changed = json.loads(plan.output_paths[0].read_text())
        changed["derived_comparisons"]["week_03_baseline"]["optimism_gap"] = 99.0
        plan.output_paths[0].write_text(json.dumps(changed), encoding="utf-8")
    elif tamper == "markdown":
        plan.output_paths[1].write_text("wrong\n", encoding="utf-8")
    else:
        plan.output_paths[2].write_text("0" * 64 + "  report.json\n", encoding="utf-8")
    with pytest.raises(ModelDataError):
        public_report_validation.validate_public_report(tmp_path)


def test_execute_requires_clean_revision_and_no_flag_cli_does_not_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, run, plan = _source_and_plan(tmp_path)
    monkeypatch.setattr(public_report_publication, "preflight", lambda _root: plan)
    monkeypatch.setattr(
        public_report_publication,
        "_clean_revision",
        lambda _root: (_ for _ in ()).throw(ModelDataError("clean committed")),
    )
    with pytest.raises(ModelDataError, match="clean committed"):
        execute_publication(tmp_path, plan)
    assert not any(path.exists() for path in plan.output_paths)
    script = ROOT / "scripts/publish_week2_bigram_evaluation.py"
    spec = importlib.util.spec_from_file_location("public_report_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "preflight", lambda _root: plan)
    monkeypatch.setattr(sys, "argv", [str(script)])
    assert module.main() == 0
    assert not any(path.exists() for path in plan.output_paths)
    called: list[PublicReportPlan] = []
    monkeypatch.setattr(
        module,
        "execute_publication",
        lambda _root, found: called.append(found),
    )
    monkeypatch.setattr(sys, "argv", [str(script), "--execute-publication"])
    assert module.main() == 0
    assert called == [plan]


def test_execution_captures_the_clean_publication_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source, _run, plan = _source_and_plan(tmp_path)
    executing_revision = "e" * 40
    monkeypatch.setattr(public_report_publication, "preflight", lambda _root: plan)
    monkeypatch.setattr(
        public_report_publication, "_clean_revision", lambda _root: executing_revision
    )
    execute_publication(tmp_path, plan)
    payload = json.loads(plan.output_paths[0].read_text(encoding="utf-8"))
    assert payload["publication_code_revision"] == executing_revision
    assert f"Publication code revision: `{executing_revision}`" in plan.output_paths[
        1
    ].read_text(encoding="utf-8")


def test_execution_refuses_an_existing_output_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source, _run, plan = _source_and_plan(tmp_path)
    plan.output_paths[0].write_text("existing", encoding="utf-8")
    monkeypatch.setattr(public_report_publication, "preflight", lambda _root: plan)
    with pytest.raises(ModelDataError, match="already exists"):
        execute_publication(tmp_path, plan)


def test_read_only_validator_cli_reports_a_validated_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = ROOT / "scripts/validate_week2_bigram_public_report.py"
    spec = importlib.util.spec_from_file_location("public_report_validator_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "validate_public_report",
        lambda _root: {"status": "passed", "principal_record_count": 12},
    )
    assert module.main() == 0


def test_public_modules_do_not_import_model_or_collection_loaders() -> None:
    modules = (
        ROOT / "src/protein_lm/bigram/public_report.py",
        ROOT / "src/protein_lm/bigram/public_report_payload.py",
        ROOT / "src/protein_lm/bigram/public_report_render.py",
        ROOT / "src/protein_lm/bigram/public_report_io.py",
        ROOT / "src/protein_lm/bigram/public_report_publication.py",
        ROOT / "src/protein_lm/bigram/public_report_validation.py",
    )
    forbidden = (
        "evaluation_execution",
        "load_collection",
        "load_model",
        "ModelDataCollection",
    )
    assert all(
        not any(token in module.read_text(encoding="utf-8") for token in forbidden)
        for module in modules
    )


def test_forbidden_membership_keys_are_rejected_recursively() -> None:
    with pytest.raises(ModelDataError, match="forbidden membership"):
        reject_forbidden_keys({"nested": [{"uniref50_group": "private"}]})
