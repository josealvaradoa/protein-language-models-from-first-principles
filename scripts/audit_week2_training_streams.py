"""Preflight or explicitly audit the two frozen Week 2 training streams."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from protein_lm.bigram.config import load_config
from protein_lm.bigram.reporting import report_payload, write_evidence
from protein_lm.bigram.stream import audit_stream
from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.data.model_data.loaders import ModelDataCollection, load_collection
from protein_lm.data.model_data.workflow import _require_clean_committed_revision


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "experiments/week_02/bigram_training_stream_v1.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit frozen Week 2 bigram training streams.")
    parser.add_argument(
        "--execute-stream-audit",
        action="store_true",
        help="load both approved training collections and write aggregate-only evidence",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    try:
        config = load_config(CONFIG_PATH)
        if not arguments.execute_stream_audit:
            _print_plan(config)
            return 0
        _require_clean_committed_revision(PROJECT_ROOT)
        started = time.perf_counter()
        audits = {
            collection_name: _audit_arm(config, collection_name, namespace)
            for collection_name, namespace in zip(
                config.training_collections, config.training_namespaces, strict=True
            )
        }
        revision = _git_revision(PROJECT_ROOT)
        payload = report_payload(
            config_path=CONFIG_PATH,
            config=config,
            audits=audits,
            code_revision=revision,
            runtime_seconds=time.perf_counter() - started,
        )
        write_evidence(
            tuple(PROJECT_ROOT / path for path in config.output_paths), payload
        )
    except (ModelDataError, OSError, subprocess.SubprocessError) as error:
        print(f"Week 2 bigram stream audit failed: {error}")
        return 1
    print("Week 2 bigram stream audit completed.")
    print("network requests made: none")
    return 0


def _print_plan(config) -> None:
    print(f"contract identifier: {config.contract_identifier}")
    print("collections: " + ", ".join(config.training_collections))
    print(f"prediction pairs per arm: {config.prediction_pair_budget}")
    print(
        f"batches: {config.full_batches} full x {config.batch_size} + "
        f"{config.final_partial_batch_pairs} partial"
    )
    print("execution requires --execute-stream-audit, a clean committed revision, and new output paths")
    print("network requests made: none")


def _audit_arm(config, collection_name: str, namespace: str):
    """Load and release one arm before the other arm's catalog is constructed."""

    proteins = load_collection(PROJECT_ROOT, ModelDataCollection(collection_name))
    try:
        return audit_stream(
            proteins,
            namespace=namespace,
            base_seed=config.base_seed,
            pair_budget=config.prediction_pair_budget,
            hash_domain=config.stream_hash_domain,
            batch_size=config.batch_size,
        )
    finally:
        del proteins


def _git_revision(root: Path) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
