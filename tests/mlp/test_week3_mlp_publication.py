"""Synthetic-only guardrails for Week 3 publication helpers."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.mlp.publication_config import load_publication_config
from protein_lm.mlp.publication_io import write_evidence
from protein_lm.mlp.publication_orchestration import (
    _learning_curve,
    _one_epoch_summary,
    _staged_runtimes,
    _tail_summary,
)
from protein_lm.mlp.publication_payload import (
    capacity_screen,
    cosine_summary,
    deterministic_pca,
    position_summary,
    reject_forbidden_keys,
)
from protein_lm.mlp.publication_render import render_markdown
from protein_lm.mlp.publication_sources import verify_bytes
from protein_lm.mlp.publication_validation import _validate_inventory


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (20260821, 20260822, 20260823)
TOKENS = 1_000_495


def metric(*, tokens: int = TOKENS, ce: float = 2.5, correct: int = 100_000) -> dict[str, object]:
    return {
        "token_count": tokens,
        "nll_numerator": ce * tokens,
        "correct_predictions": correct,
        "cross_entropy": ce,
        "accuracy": correct / tokens,
    }


def position_status(seed: int) -> dict[str, object]:
    counts = (29_095, 23_805, 947_595)
    bins: dict[str, dict[str, object]] = {}
    for index, (name, count) in enumerate(zip((
        "available_prior_residues_0_10",
        "available_prior_residues_11_19",
        "available_prior_residues_20_plus",
    ), counts, strict=True)):
        c20 = metric(tokens=count, ce=2.4 + index / 100, correct=count // 10)
        e64 = metric(tokens=count, ce=2.5 + index / 100, correct=count // 11)
        bins[name] = {
            "context20": c20,
            "embedding64": e64,
            "embedding64_minus_context20_cross_entropy": 0.1,
            "context20_minus_embedding64_accuracy": c20["accuracy"] - e64["accuracy"],
        }
    overall: dict[str, object] = {}
    for arm in ("context20", "embedding64"):
        total_nll = sum(float(bins[name][arm]["nll_numerator"]) for name in bins)
        total_correct = sum(int(bins[name][arm]["correct_predictions"]) for name in bins)
        overall[arm] = metric(tokens=TOKENS, ce=total_nll / TOKENS, correct=total_correct)
    overall["embedding64_minus_context20_cross_entropy"] = (
        overall["embedding64"]["cross_entropy"] - overall["context20"]["cross_entropy"]
    )
    overall["context20_minus_embedding64_accuracy"] = (
        overall["context20"]["accuracy"] - overall["embedding64"]["accuracy"]
    )
    return {
        "status": "passed", "seed": seed, "network_requests_made": 0,
        "no_training": True, "no_gradients": True, "sealed_test_accessed": False,
        "significance_generated": False, "automatic_report_generated": False,
        "automatic_selection_generated": False, "training_predictions": 0,
        "optimizer_steps": 0, "backward_passes": 0,
        "native_validation": {
            "prediction_tokens": TOKENS, "records": 2645,
            "ordered_target_and_prior_residue_sha256": "dfac8b1d5eeceadc0428a39568b43b3563acd4a3fabfaac7cf450f724c613097",
        },
        "results": {"bins": bins, "overall": overall},
    }


def capacity_status(arm: str, seed: int, parameter_count: int) -> dict[str, object]:
    milestones = []
    for position in (1_000_000, 5_000_000, 10_000_000, 25_000_000):
        item = metric()
        item["prediction_position"] = position
        milestones.append(item)
    return {
        "status": "passed", "network_requests_made": 0, "arm": arm,
        "seed": seed, "model": {"parameter_count": parameter_count},
        "runtime_seconds": 1.0, "native_validation_milestones": milestones,
    }


def test_committed_contract_pins_inventory_and_scope() -> None:
    config = load_publication_config(ROOT / "experiments/week_03/mlp_publication_v1.toml")
    assert config.output_paths == (
        "reports/week_03/mlp_evaluation_v1.json",
        "reports/week_03/mlp_evaluation_v1.md",
        "reports/week_03/mlp_evaluation_v1.sha256",
    )
    assert config.publication_scope == "aggregate_only_no_sequences_no_accessions_no_family_ids_no_raw_tensors"
    assert sum(pin.kind == "c10_learning_curve_status" for pin in config.sources) == 3


@pytest.mark.parametrize("replacement", [
    ("[expected]\n", "[expected]\nunknown_metric = 1\n"),
    ('kind = "config"', 'kind = "unknown"'),
    ("reports/week_03/mlp_evaluation_v1.json", "../escape.json"),
])
def test_config_traversal_unknown_inventory_fail_closed(
    tmp_path: Path, replacement: tuple[str, str]
) -> None:
    source, target = replacement
    content = (ROOT / "experiments/week_03/mlp_publication_v1.toml").read_text()
    path = tmp_path / "bad.toml"
    path.write_text(content.replace(source, target, 1))
    with pytest.raises(ModelDataError):
        load_publication_config(path)


def test_config_rejects_duplicate_final_and_diagnostic_paths(tmp_path: Path) -> None:
    content = (ROOT / "experiments/week_03/mlp_publication_v1.toml").read_text()
    path = tmp_path / "duplicate.toml"
    path.write_text(content.replace(
        "week3-context20-100m-seed-20260822-cpu/run_status.json",
        "week3-context20-100m-seed-20260821-cpu/run_status.json",
    ))
    with pytest.raises(ModelDataError):
        load_publication_config(path)


def test_pin_and_path_drift_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_text("x")
    with pytest.raises(ModelDataError):
        verify_bytes(source, "0" * 64, "synthetic")


def test_position_reconstruction_and_all_tamper_gates_fail() -> None:
    statuses = [position_status(seed) for seed in SEEDS]
    assert [row["token_count"] for row in position_summary(statuses)["bins"]] == [29_095, 23_805, 947_595]
    for mutator in (
        lambda value: value[0]["results"]["bins"]["available_prior_residues_0_10"]["context20"].update(token_count=1),
        lambda value: value[0]["results"]["bins"]["available_prior_residues_0_10"]["context20"].update(nll_numerator=1),
        lambda value: value[0]["results"]["overall"].update(embedding64_minus_context20_cross_entropy=0),
        lambda value: value[0].update(no_gradients=False),
    ):
        tampered = copy.deepcopy(statuses)
        mutator(tampered)
        with pytest.raises(ModelDataError):
            position_summary(tampered)


def test_capacity_params_and_metric_arithmetic_fail_closed() -> None:
    statuses = [capacity_status("context_20", seed, 530293) for seed in SEEDS]
    statuses += [capacity_status("embedding_64", seed, 530965) for seed in SEEDS]
    statuses += [capacity_status("hidden_1600", seed, 1_000_000) for seed in SEEDS]
    assert {row["parameter_count"] for row in capacity_screen(statuses)} == {530293, 530965, 1_000_000}
    statuses[0]["native_validation_milestones"][-1]["accuracy"] = 0.9
    with pytest.raises(ModelDataError):
        capacity_screen(statuses)


def test_curve_has_exact_c10_c20_positions_and_rejects_duplicate_seed() -> None:
    c10 = []
    for seed in SEEDS:
        status = capacity_status("c10", seed, 274293)
        status.pop("network_requests_made")
        points = status.pop("native_validation_milestones")
        for position in (50_000_000, 100_000_000):
            item = metric()
            item["prediction_position"] = position
            points.append(item)
        status["native_validation"] = points
        c10.append(status)
    c20 = [capacity_status("context_20", seed, 530293) for seed in SEEDS]
    final = []
    for seed in SEEDS:
        record: dict[str, object] = {"seed": seed, "native_validation_milestones": []}
        for position in (50_000_000, 100_000_000):
            item = metric()
            item["prediction_position"] = position
            record["native_validation_milestones"].append(item)
        final.append(record)
    curve = _learning_curve(c10, c20, final)
    assert [point["prediction_position"] for point in curve["series"][1]["points"]] == [1_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000, 100_000_000]
    c10[2]["seed"] = SEEDS[0]
    with pytest.raises(ModelDataError):
        _learning_curve(c10, c20, final)


def test_tail_one_epoch_and_staged_runtime_checks() -> None:
    tails = []
    for arm in ("cosine_90m_100m_001", "staged_97m_003"):
        for seed in SEEDS:
            tails.append({
                "status": "passed", "network_requests_made": 0, "arm": arm,
                "seed": seed, "exploratory_only": True,
                "start_prediction_position": 90_000_000,
                "final_prediction_position": 100_000_000,
                "start_optimizer_steps": 1, "final_optimizer_steps": 2,
                "tail_optimizer_updates": 1, "final_native_validation": metric(),
            })
    assert len(_tail_summary(tails)) == 2
    tails[0]["seed"] = tails[1]["seed"]
    with pytest.raises(ModelDataError):
        _tail_summary(tails)
    one_epoch = []
    for seed in SEEDS:
        milestones = []
        for position in (124999936, 149999872, 171329454):
            item = metric()
            item["prediction_position"] = position
            milestones.append(item)
        one_epoch.append({"status": "passed", "network_requests_made": 0, "seed": seed, "native_validation_milestones": milestones})
    assert len(_one_epoch_summary(one_epoch)) == 3
    one_epoch[0]["native_validation_milestones"][0]["cross_entropy"] = 0
    with pytest.raises(ModelDataError):
        _one_epoch_summary(one_epoch)
    capacity = [capacity_status(arm, seed, 1) for arm in ("context_20", "embedding_64") for seed in SEEDS]
    final = [{"status": "passed", "network_requests_made": 0, "seed": seed, "runtime_seconds": 2.0} for seed in SEEDS]
    assert len(_staged_runtimes(capacity, final, final)["series"]) == 2
    final[1]["seed"] = final[0]["seed"]
    with pytest.raises(ModelDataError):
        _staged_runtimes(capacity, final, final)


def test_pca_cosines_and_forbidden_keys_fail_closed() -> None:
    values = np.arange(21 * 32, dtype=float).reshape(21, 32)
    assert deterministic_pca(values) == deterministic_pca(values)
    with pytest.raises(ModelDataError, match="zero-norm"):
        cosine_summary({20260821: np.zeros((21, 32)), 20260822: values, 20260823: values})
    with pytest.raises(ModelDataError):
        cosine_summary({20260821: values * np.nan, 20260822: values, 20260823: values})
    with pytest.raises(ModelDataError):
        reject_forbidden_keys({"raw_tensors": []}, ("raw_tensors",))


def test_writer_inventory_and_renderer_boundaries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("protein_lm.mlp.publication_io.render_markdown", lambda _: "# synthetic\n")
    paths = tuple(tmp_path / name for name in ("report.json", "report.md", "report.sha256"))
    write_evidence(paths, {"synthetic": True})
    assert all(path.exists() for path in paths) and not (tmp_path / ".mlp_evaluation_v1.tmp").exists()
    with pytest.raises(ModelDataError):
        write_evidence(paths, {"synthetic": True})
    with pytest.raises(ModelDataError):
        _validate_inventory(tuple(tmp_path / f"mlp_evaluation_v1.{suffix}" for suffix in ("json", "md", "sha256")))
    payload = {
        "final_three_seed_comparison": {"context20": {"aggregate": {"mean_cross_entropy": 2.0, "sample_standard_deviation_cross_entropy": 0.1, "mean_accuracy": 0.2}}, "embedding64_challenger": {"aggregate": {"mean_cross_entropy": 2.1, "sample_standard_deviation_cross_entropy": 0.1, "mean_accuracy": 0.1}, "embedding64_minus_context20_mean_cross_entropy": 0.1}},
        "fixed_budget_baseline_comparison": {"baseline": {"cross_entropy": 2.2, "accuracy": 0.1}},
        "position_availability_diagnostic": {"bins": []},
        "learning_curves": {"series": [{"model": "C10_E32_H800", "points": [{"prediction_position": 100000000, "mean_cross_entropy": 2.3, "sample_standard_deviation_cross_entropy": 0.1, "stage": "original"}]}]},
        "capacity_screen_25m": [{"arm": "hidden_1600", "parameter_count": 1, "mean_cross_entropy": 2.0, "sample_standard_deviation_cross_entropy": 0.1, "stage": "screen"}],
        "observed_staged_cpu_wall_time_seconds": {"series": []},
        "negative_and_exploratory_outcomes": {"lr_tails": [], "one_epoch_continuation": []},
    }
    rendered = render_markdown(payload)
    assert "Parameters" in rendered and "274293" in rendered and "H1600 ended at 25M" in rendered


def test_writer_rolls_back_on_replace_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("protein_lm.mlp.publication_io.render_markdown", lambda _: "# synthetic\n")
    paths = tuple(tmp_path / name for name in ("report.json", "report.md", "report.sha256"))
    original = __import__("os").replace
    calls = 0

    def fail(source: str | Path, destination: str | Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected")
        original(source, destination)

    monkeypatch.setattr("protein_lm.mlp.publication_io.os.replace", fail)
    with pytest.raises(OSError):
        write_evidence(paths, {"synthetic": True})
    assert not any(path.exists() for path in paths)
    assert not (tmp_path / ".mlp_evaluation_v1.tmp").exists()


def test_notebook_is_unexecuted_and_compilable() -> None:
    book = json.loads((ROOT / "notebooks/week_03/week_03_mlp_protein_context.ipynb").read_text())
    identifiers = [cell["id"] for cell in book["cells"]]
    assert len(identifiers) == len(set(identifiers))
    source = "\n".join("".join(cell.get("source", [])) for cell in book["cells"])
    assert all(term in source for term in ("candidate", "plt.plot", "plt.bar", "plt.subplots", "plt.errorbar"))
    for cell in book["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell["source"]), cell["id"], "exec")
