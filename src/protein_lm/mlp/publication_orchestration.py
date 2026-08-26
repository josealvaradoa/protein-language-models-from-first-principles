"""Operator-gated Week 3 aggregate publication orchestration."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.mlp.publication_config import PublicationConfig, config_sha256, load_publication_config
from protein_lm.mlp.publication_io import write_evidence
from protein_lm.mlp.publication_payload import (RESIDUES, _metric, build_payload, capacity_screen, cosine_summary, deterministic_pca, final_metrics, position_summary)
from protein_lm.mlp.publication_sources import final_statuses, load_json, validate_sources


_REVISION = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class PublicationPlan:
    root: Path
    config_path: Path
    config: PublicationConfig
    output_paths: tuple[Path, Path, Path]


def preflight(root: Path) -> PublicationPlan:
    """Validate every byte pin without writing, model init, or collection access."""

    path = root / "experiments/week_03/mlp_publication_v1.toml"
    config = load_publication_config(path)
    validate_sources(root, config)
    return PublicationPlan(root, path, config, tuple(root / value for value in config.output_paths))


def execute_publication(root: Path, plan: PublicationPlan) -> tuple[Path, Path, Path]:
    _validate_plan(root, plan)
    if preflight(root) != plan:
        raise ModelDataError("public Week 3 publication plan drifted")
    if any(path.exists() for path in plan.output_paths):
        raise ModelDataError("public Week 3 report already exists")
    revision = _clean_revision(root)
    payload = _payload(root, plan.config, config_sha256(plan.config_path), revision)
    write_evidence(plan.output_paths, payload)
    return plan.output_paths


def report_payload(root: Path, plan: PublicationPlan, revision: str = "0" * 40) -> dict[str, object]:
    """Build the deterministic payload after a successful byte-only preflight."""

    _validate_plan(root, plan)
    validate_sources(root, plan.config)
    return _payload(root, plan.config, config_sha256(plan.config_path), revision)


def _payload(root: Path, config: PublicationConfig, config_hash: str, revision: str) -> dict[str, object]:
    context20_statuses = [status for _, status in final_statuses(root, config, "context20")]
    embedding64_statuses = [status for _, status in final_statuses(root, config, "embedding64")]
    context20 = final_metrics(context20_statuses, "context20")
    embedding64 = final_metrics(embedding64_statuses, "embedding64")
    def source_by_kind(kind: str) -> list[dict[str, object]]:
        return [
            load_json(root / pin.relative_path, kind)
            for pin in config.sources
            if pin.kind == kind
        ]
    capacity = capacity_screen(source_by_kind("capacity_status"))
    baseline = _week2_baseline(source_by_kind("week2_public_baseline"))
    diagnostic = position_summary([load_json(root / pin.relative_path, "position diagnostic") for pin in config.diagnostic_statuses])
    pca, embeddings = _embedding_diagnostics(root, config)
    payload = build_payload(
        config=config, config_hash=config_hash, revision=revision, context20=context20,
        embedding64=embedding64, baseline=baseline, capacity=capacity, diagnostic=diagnostic, pca=pca,
        cosine=cosine_summary(embeddings), source_pins=_source_pins(config),
    )
    payload["learning_curves"] = _learning_curve(
        source_by_kind("c10_learning_curve_status"), source_by_kind("capacity_status"), context20_statuses
    )
    payload["observed_staged_cpu_wall_time_seconds"] = _staged_runtimes(
        source_by_kind("capacity_status"), context20_statuses, embedding64_statuses
    )
    payload["negative_and_exploratory_outcomes"]["lr_tails"] = _tail_summary(source_by_kind("lr_tail_status"))
    payload["negative_and_exploratory_outcomes"]["one_epoch_continuation"] = _one_epoch_summary(source_by_kind("one_epoch_status"))
    from protein_lm.mlp.publication_payload import reject_forbidden_keys
    reject_forbidden_keys(payload, config.forbidden_public_keys)
    return payload


def _embedding_diagnostics(root: Path, config: PublicationConfig) -> tuple[list[dict[str, object]], dict[int, np.ndarray]]:
    output: list[dict[str, object]] = []
    embeddings: dict[int, np.ndarray] = {}
    for pin in sorted((item for item in config.final_checkpoints if item.arm == "context20"), key=lambda item: item.seed):
        tensors = load_file(str(root / pin.checkpoint_relative_path / "model.safetensors"))
        if set(tensors) != {"embedding", "w1", "b1", "w2", "b2"}:
            raise ModelDataError("final checkpoint tensor inventory drifted")
        embedding = np.asarray(tensors["embedding"])
        embeddings[pin.seed] = embedding
        coordinates = deterministic_pca(embedding)
        output.append({"seed": pin.seed, "coordinates": [{"token": token, "pc1": coordinate[0], "pc2": coordinate[1]} for token, coordinate in zip(("BOS", *RESIDUES), coordinates, strict=True)]})
    return output, embeddings


def _learning_curve(c10: list[dict[str, object]], capacity: list[dict[str, object]], final: list[dict[str, object]]) -> dict[str, object]:
    positions = (1000000, 5000000, 10000000, 25000000, 50000000, 100000000)
    def points(statuses: list[dict[str, object]], field: str, allowed: set[int], label: str) -> dict[int, list[float]]:
        output: dict[int, list[float]] = {}
        seeds = set()
        for status in statuses:
            if status.get("status") != "passed" or status.get("network_requests_made", 0) != 0:
                raise ModelDataError("learning-curve status did not pass")
            seed = status.get("seed")
            if type(seed) is not int or seed in seeds:
                raise ModelDataError("learning-curve seed is duplicated or invalid")
            seeds.add(seed)
            metric_rows = status.get(field)
            if not isinstance(metric_rows, list):
                raise ModelDataError("learning-curve metrics are malformed")
            observed = []
            for item in metric_rows:
                if not isinstance(item, dict):
                    raise ModelDataError("learning-curve metric is malformed")
                position = item.get("prediction_position", item.get("predictions"))
                if type(position) is not int:
                    raise ModelDataError("learning-curve position is malformed")
                observed.append(position)
                if position in allowed:
                    metric = _metric(item)
                    output.setdefault(position, []).append(float(metric["cross_entropy"]))
            if tuple(observed) != tuple(sorted(allowed)):
                raise ModelDataError("learning-curve endpoints drifted")
        if seeds != {20260821, 20260822, 20260823}:
            raise ModelDataError(f"{label} learning-curve seeds are incomplete")
        return output
    c10_values = points(c10, "native_validation", set(positions), "C10")
    c20_screen = points([item for item in capacity if item.get("arm") == "context_20"], "native_validation_milestones", set(positions[:4]), "C20 screen")
    c20_final = {50000000: [], 100000000: []}
    final_seeds = set()
    for record in final:
        seed = record.get("seed")
        if type(seed) is not int or seed in final_seeds:
            raise ModelDataError("C20 continuation seed is duplicated or invalid")
        final_seeds.add(seed)
        metric_rows = record.get("native_validation_milestones")
        if not isinstance(metric_rows, list) or [
            item.get("prediction_position") for item in metric_rows if isinstance(item, dict)
        ] != [50000000, 100000000]:
            raise ModelDataError("C20 continuation endpoints drifted")
        for item in metric_rows:
            if not isinstance(item, dict):
                raise ModelDataError("C20 continuation metric is malformed")
            position = item.get("prediction_position")
            if position in c20_final:
                c20_final[position].append(float(_metric(item)["cross_entropy"]))
    c20_values = {**c20_screen, **c20_final}
    if final_seeds != {20260821, 20260822, 20260823} or set(c10_values) != set(positions) or set(c20_values) != set(positions) or any(len(values) != 3 for values in (*c10_values.values(), *c20_values.values())):
        raise ModelDataError("complete frozen learning curves are incomplete")
    def series(name: str, values: dict[int, list[float]]) -> dict[str, object]:
        return {"model": name, "points": [{"prediction_position": position, "mean_cross_entropy": sum(values[position]) / 3, "sample_standard_deviation_cross_entropy": __import__("statistics").stdev(values[position]), "stage": "original_100m_run" if name == "C10_E32_H800" else ("capacity_screen_25m" if position <= 25000000 else "final_continuation_100m")} for position in positions]}
    return {"series": [series("C10_E32_H800", c10_values), series("C20_E32_H800", c20_values)]}


def _staged_runtimes(capacity: list[dict[str, object]], c20: list[dict[str, object]], e64: list[dict[str, object]]) -> dict[str, object]:
    parents: dict[tuple[str, int], float] = {}
    for item in capacity:
        if item.get("status") != "passed" or item.get("network_requests_made") != 0:
            raise ModelDataError("staged runtime parent did not pass local-only gates")
        arm, seed = item.get("arm"), item.get("seed")
        if not isinstance(arm, str) or type(seed) is not int or (arm, seed) in parents:
            raise ModelDataError("staged runtime parent keys are invalid")
        runtime = _positive_runtime(item.get("runtime_seconds"), "parent")
        parents[(arm, seed)] = runtime
    required = {(arm, seed) for arm in ("context_20", "embedding_64") for seed in (20260821, 20260822, 20260823)}
    if not required <= set(parents):
        raise ModelDataError("staged runtime parent evidence is incomplete")
    def run(name: str, arm: str, statuses: list[dict[str, object]]) -> dict[str, object]:
        values = []
        seeds = set()
        for status in statuses:
            if status.get("status") != "passed" or status.get("network_requests_made") != 0:
                raise ModelDataError("staged runtime tail did not pass local-only gates")
            seed = status.get("seed")
            if type(seed) is not int or seed in seeds or (arm, seed) not in parents:
                raise ModelDataError("staged runtime tail join is invalid")
            seeds.add(seed)
            tail = status.get("runtime_seconds", status.get("runtime", {}).get("seconds"))
            values.append({"seed": seed, "seconds": parents[(arm, seed)] + _positive_runtime(tail, "tail")})
        if seeds != {20260821, 20260822, 20260823}:
            raise ModelDataError("staged runtime tail evidence is incomplete")
        seconds = [item["seconds"] for item in values]
        return {"model": name, "label": "observed staged CPU wall time including harness, evaluation, and checkpoint overhead", "records": values, "mean_seconds": sum(seconds) / 3, "sample_standard_deviation_seconds": __import__("statistics").stdev(seconds)}
    return {"week2_baseline_runtime": "unavailable_not_compared", "series": [run("C20_E32_H800", "context_20", c20), run("C10_E64_H800", "embedding_64", e64)]}


def _tail_summary(statuses: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[float]] = {}
    seeds: dict[str, set[int]] = {}
    for status in statuses:
        if status.get("status") != "passed" or status.get("network_requests_made") != 0:
            raise ModelDataError("LR tail status did not pass")
        arm, seed = status.get("arm"), status.get("seed")
        if not isinstance(arm, str) or type(seed) is not int:
            raise ModelDataError("LR tail identity is invalid")
        seeds.setdefault(arm, set())
        if seed in seeds[arm]:
            raise ModelDataError("LR tail seed is duplicated")
        seeds[arm].add(seed)
        if (
            status.get("exploratory_only") is not True
            or status.get("start_prediction_position") != 90_000_000
            or status.get("final_prediction_position") != 100_000_000
        ):
            raise ModelDataError("LR tail final position drifted")
        start_steps = status.get("start_optimizer_steps")
        final_steps = status.get("final_optimizer_steps")
        updates = status.get("tail_optimizer_updates")
        if (
            type(start_steps) is not int
            or type(final_steps) is not int
            or type(updates) is not int
            or start_steps <= 0
            or final_steps <= start_steps
            or updates != final_steps - start_steps
        ):
            raise ModelDataError("LR tail accounting drifted")
        final = status.get("final_native_validation")
        if not isinstance(final, dict):
            raise ModelDataError("LR tail final metrics are malformed")
        grouped.setdefault(arm, []).append(float(_metric(final)["cross_entropy"]))
    if set(grouped) != {"cosine_90m_100m_001", "staged_97m_003"} or any(len(values) != 3 or seeds[arm] != {20260821, 20260822, 20260823} for arm, values in grouped.items()):
        raise ModelDataError("LR-tail evidence is incomplete")
    return [{"arm": arm, "mean_cross_entropy": sum(values) / len(values), "sample_standard_deviation_cross_entropy": __import__("statistics").stdev(values), "stage": "exploratory_lr_tail_100m"} for arm, values in sorted(grouped.items())]


def _one_epoch_summary(statuses: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: dict[int, list[float]] = {}
    seeds = set()
    for status in statuses:
        if status.get("status") != "passed" or status.get("network_requests_made") != 0:
            raise ModelDataError("one-epoch status did not pass")
        seed = status.get("seed")
        if type(seed) is not int or seed in seeds:
            raise ModelDataError("one-epoch seed is duplicated or invalid")
        seeds.add(seed)
        metric_rows = status.get("native_validation_milestones")
        if not isinstance(metric_rows, list):
            raise ModelDataError("one-epoch metrics are malformed")
        expected_positions = (124999936, 149999872, 171329454)
        if tuple(
            item.get("prediction_position") for item in metric_rows if isinstance(item, dict)
        ) != expected_positions:
            raise ModelDataError("one-epoch endpoints drifted")
        for item in metric_rows:
            if not isinstance(item, dict):
                raise ModelDataError("one-epoch metric is malformed")
            rows.setdefault(_int_position(item.get("prediction_position")), []).append(float(_metric(item)["cross_entropy"]))
    expected = {124999936, 149999872, 171329454}
    if seeds != {20260821, 20260822, 20260823} or set(rows) != expected or any(len(values) != 3 for values in rows.values()):
        raise ModelDataError("one-epoch evidence is incomplete")
    return [{"prediction_position": position, "mean_cross_entropy": sum(values) / len(values), "sample_standard_deviation_cross_entropy": __import__("statistics").stdev(values), "stage": "exploratory_one_epoch_continuation"} for position, values in sorted(rows.items())]


def _week2_baseline(reports: list[dict[str, object]]) -> dict[str, float]:
    if len(reports) != 1 or not isinstance(reports[0].get("records"), list):
        raise ModelDataError("Week 2 public baseline is malformed")
    matches = [
        record for record in reports[0]["records"]
        if isinstance(record, dict)
        and record.get("model_arm") == "family_aware_training"
        and record.get("model_type") == "neural_bigram"
        and record.get("collection") == "family_aware_native_validation"
    ]
    if len(matches) != 1:
        raise ModelDataError("Week 2 public baseline record is missing or ambiguous")
    metrics = matches[0].get("metrics")
    if not isinstance(metrics, dict) or not isinstance(metrics.get("overall"), dict):
        raise ModelDataError("Week 2 public baseline metrics are malformed")
    overall = metrics["overall"]
    metric = _metric(overall)
    return {"cross_entropy": float(metric["cross_entropy"]), "accuracy": float(metric["accuracy"])}


def _positive_runtime(value: object, label: str) -> float:
    if type(value) not in (int, float) or not np.isfinite(float(value)) or float(value) <= 0:
        raise ModelDataError(f"{label} runtime is invalid")
    return float(value)


def _int_position(value: object) -> int:
    if type(value) is not int:
        raise ModelDataError("prediction position is malformed")
    return value


def _source_pins(config: PublicationConfig) -> list[dict[str, str]]:
    pins = [{"kind": pin.kind, "relative_path": pin.relative_path, "sha256": pin.sha256} for pin in config.sources]
    pins.extend({"kind": "position_diagnostic", "relative_path": pin.relative_path, "sha256": pin.sha256} for pin in config.diagnostic_statuses)
    for pin in config.final_checkpoints:
        pins.extend((
            {"kind": f"{pin.arm}_final_status", "relative_path": pin.status_relative_path, "sha256": pin.status_sha256},
            {"kind": f"{pin.arm}_final_checkpoint_metadata", "relative_path": f"{pin.checkpoint_relative_path}/checkpoint.json", "sha256": pin.metadata_sha256},
            {"kind": f"{pin.arm}_final_checkpoint_tensor", "relative_path": f"{pin.checkpoint_relative_path}/model.safetensors", "sha256": pin.tensor_sha256},
        ))
    return pins


def _validate_plan(root: Path, plan: PublicationPlan) -> None:
    if not isinstance(plan, PublicationPlan) or plan.root != root or plan.config_path != root / "experiments/week_03/mlp_publication_v1.toml" or plan.output_paths != tuple(root / value for value in plan.config.output_paths):
        raise ModelDataError("public Week 3 publication plan is invalid")


def _clean_revision(root: Path) -> str:
    try:
        if subprocess.run(("git", "status", "--porcelain"), cwd=root, check=True, capture_output=True, text=True).stdout:
            raise ModelDataError("public Week 3 publication requires a clean committed revision")
        revision = subprocess.run(("git", "rev-parse", "HEAD"), cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    except subprocess.SubprocessError as error:
        raise ModelDataError("could not establish publication revision") from error
    if _REVISION.fullmatch(revision) is None:
        raise ModelDataError("publication revision is invalid")
    return revision
