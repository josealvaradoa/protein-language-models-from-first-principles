"""Synthetic acceptance checks for the frozen Week 2 bigram stream audit."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from protein_lm.bigram import config as stream_config
from protein_lm.bigram.config import load_config
from protein_lm.bigram.reporting import report_payload, write_evidence
from protein_lm.bigram.stream import (
    ArmStreamAudit,
    audit_stream,
    new_stream_hasher,
    ordered_proteins,
    protein_order_key,
    protein_pair_bytes,
)
from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.data.model_data.loaders import ProteinSequence


ROOT = Path(__file__).parents[2]
CONFIG_PATH = ROOT / "experiments/week_02/bigram_training_stream_v1.toml"
DOMAIN = "protein-lm/week2/bigram-transition-stream/v1"


def protein(sequence: str, accession: str = "P00001") -> ProteinSequence:
    """Build a loader-shaped synthetic protein without touching a manifest."""

    return ProteinSequence(
        primary_accession=accession,
        sequence=sequence,
        sequence_sha256=hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
        biological_length=len(sequence),
        length_bucket="synthetic",
        uniref50_group="UniRef50_SYNTHETIC",
    )


def test_acd_has_exact_role_specific_coordinates_and_bytes() -> None:
    packed, contexts, targets = protein_pair_bytes("ACD")
    assert contexts == bytes((0, 1, 2, 3))
    assert targets == bytes((0, 1, 2, 20))
    assert packed == bytes((0, 0, 1, 1, 2, 2, 3, 20))


def test_protein_boundaries_restart_at_bos_without_crossing() -> None:
    audit = audit_stream(
        (protein("A", "P00001"), protein("C", "P00002")),
        namespace="synthetic/boundaries/v1",
        base_seed=7,
        pair_budget=4,
        hash_domain=DOMAIN,
    )
    assert audit.pairs_emitted == 4
    assert audit.proteins_started == audit.proteins_completed == 2
    assert not audit.final_protein_partial
    assert audit.context_counts[:4] == (2, 1, 1, 0)
    assert audit.target_counts[:3] == (1, 1, 0)
    assert audit.target_counts[20] == 2


def test_ordering_is_deterministic_and_domain_separated() -> None:
    proteins = (
        protein("A", "P00001"),
        protein("C", "P00002"),
        protein("D", "P00003"),
    )
    first = ordered_proteins(proteins, "synthetic/order/a", 7)
    second = ordered_proteins(proteins, "synthetic/order/a", 7)
    other = ordered_proteins(proteins, "synthetic/order/b", 7)
    assert first == second
    assert [item.sequence_sha256 for item in first] != [
        item.sequence_sha256 for item in other
    ]


def test_ordering_key_has_a_frozen_domain_separated_vector() -> None:
    assert protein_order_key(
        "00e66854ddc46722ac3db985136265f4a24bcbbf0b45103d80cfea510e9217bf",
        "week2/training-stream/random/v1",
        20260812,
    ).hex() == "b074c35b8e6bd8d29e531fbbbe24ef659380047865670b7addaeeb5091cb24c1"


def test_stream_hasher_has_one_canonical_domain_separated_header() -> None:
    hasher = new_stream_hasher(DOMAIN, "synthetic/hash/v1", 7)
    expected = hashlib.sha256(
        b"protein-lm/week2/bigram-transition-stream/v1\0synthetic/hash/v1\0"
        + b"7\0"
    )
    assert hasher.hexdigest() == expected.hexdigest()
    assert new_stream_hasher("other-domain", "synthetic/hash/v1", 7).hexdigest() != hasher.hexdigest()


@pytest.mark.parametrize("hash_domain,namespace,base_seed", (("", "arm", 1), ("domain", "", 1), ("domain", "arm", True)))
def test_stream_hasher_rejects_invalid_header_inputs(
    hash_domain: object, namespace: object, base_seed: object
) -> None:
    with pytest.raises(ModelDataError, match="hash domain"):
        new_stream_hasher(hash_domain, namespace, base_seed)  # type: ignore[arg-type]


def test_stopping_inside_final_protein_is_exact_and_no_second_protein_starts() -> None:
    audit = audit_stream(
        (protein("ACD"), protein("Y", "P00002")),
        namespace="synthetic/partial/v1",
        base_seed=7,
        pair_budget=3,
        hash_domain=DOMAIN,
    )
    expected = hashlib.sha256()
    expected.update(DOMAIN.encode("utf-8"))
    expected.update(b"\0synthetic/partial/v1\0" + b"7\0")
    expected.update(protein_pair_bytes("ACD")[0][:6])
    assert audit.stream_sha256 == expected.hexdigest()
    assert audit.proteins_started == 1
    assert audit.proteins_completed == 0
    assert audit.final_protein_partial
    assert audit.context_counts[:4] == (1, 1, 1, 0)
    assert audit.target_counts[:3] == (1, 1, 1)
    assert audit.target_counts[20] == 0


@pytest.mark.parametrize(
    "proteins",
    (
        (protein("B"),),
        (protein("A", "P00001"), protein("A", "P00002")),
        (protein("A", "P00001"), protein("C", "P00001")),
    ),
)
def test_invalid_residues_and_duplicate_hashes_fail_closed(
    proteins: tuple[ProteinSequence, ...],
) -> None:
    with pytest.raises(ModelDataError):
        audit_stream(
            proteins,
            namespace="synthetic/failure/v1",
            base_seed=7,
            pair_budget=2,
            hash_domain=DOMAIN,
        )


def test_insufficient_corpus_fails_closed() -> None:
    with pytest.raises(ModelDataError, match="cannot satisfy"):
        audit_stream(
            (protein("A"),),
            namespace="synthetic/short/v1",
            base_seed=7,
            pair_budget=3,
            hash_domain=DOMAIN,
        )


def test_batch_config_is_exact_and_public_config_is_byte_pinned() -> None:
    config = load_config(CONFIG_PATH)
    assert (
        config.batch_size * config.full_batches + config.final_partial_batch_pairs
        == 100_000_000
    )
    assert config.full_batches + 1 == 1_526
    assert config.model_data_config_sha256 == (
        "b35ec4003b002a065c29e3c70ee72ff115edafc6645f9370603e7020b4a05f12"
    )
    assert config.model_data_registry_sha256 == (
        "13b8e1b3bb371df46f6d363b20882b91a06dde51c64d39b4e5406e0dc44efb5c"
    )
    assert config.context_roles == (
        "BOS",
        "A",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "V",
        "W",
        "Y",
    )
    assert config.target_roles == (
        "A",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "V",
        "W",
        "Y",
        "EOS",
    )


@pytest.mark.parametrize(
    "original,replacement,expected",
    (
        ("base_seed = 20260812", "base_seed = true", "base_seed must be an integer"),
        ("scope = \"week_02_bigram_training_stream\"", "scope = 1", "scope must be a nonempty string"),
        (
            'target_roles = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "EOS"]',
            'target_roles = ["A", 1]',
            "target_roles must be a list of strings",
        ),
    ),
)
def test_config_loader_rejects_wrong_toml_types_before_dataclass_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    original: str,
    replacement: str,
    expected: str,
) -> None:
    content = CONFIG_PATH.read_text(encoding="utf-8").replace(original, replacement)
    path = tmp_path / "wrong-type.toml"
    path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(
        stream_config,
        "APPROVED_BIGRAM_STREAM_CONFIG_SHA256",
        hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )
    with pytest.raises(ModelDataError, match=expected):
        load_config(path)


def valid_audits(config) -> dict[str, ArmStreamAudit]:
    """Return aggregate-only arm records that satisfy the frozen report gates."""

    return {
        collection: ArmStreamAudit(
            namespace=namespace,
            pairs_emitted=config.prediction_pair_budget,
            proteins_started=2,
            proteins_completed=1,
            final_protein_partial=True,
            context_counts=(config.prediction_pair_budget,) + (0,) * 20,
            target_counts=(config.prediction_pair_budget,) + (0,) * 20,
            stream_sha256="a" * 64,
        )
        for collection, namespace in zip(
            config.training_collections, config.training_namespaces, strict=True
        )
    }


def test_report_is_aggregate_only_atomic_and_refuses_overwrite(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    payload = report_payload(
        config_path=CONFIG_PATH,
        config=config,
        audits=valid_audits(config),
        code_revision="f" * 40,
        runtime_seconds=1.25,
    )
    assert payload["status"] == "passed"
    assert payload["hard_gates"] == {
        "exact_pair_budget": True,
        "batch_arithmetic": True,
        "approved_arm_namespaces": True,
        "aggregate_role_counts": True,
        "stream_hash_format": True,
        "protein_consumption_accounting": True,
    }
    assert payload["runtime_seconds"] == 1.25
    assert isinstance(payload["runtime_seconds"], float)
    paths = tuple(
        tmp_path / name for name in ("audit.json", "audit.md", "audit.sha256")
    )
    write_evidence(paths, payload)
    assert all(path.is_file() for path in paths)
    assert "P00001" not in paths[0].read_text(encoding="utf-8")
    assert hashlib.sha256(paths[0].read_bytes()).hexdigest() in paths[2].read_text(encoding="utf-8")
    with pytest.raises(ModelDataError, match="already exists"):
        write_evidence(paths, payload)


@pytest.mark.parametrize(
    "change,expected",
    (
        (
            lambda audits, config: {
                **audits,
                "random_training": replace(
                    audits["random_training"], namespace="wrong/namespace"
                ),
            },
            "aggregate counts",
        ),
        (
            lambda audits, config: {
                **audits,
                "random_training": replace(
                    audits["random_training"], context_counts=(1,) * 21
                ),
            },
            "aggregate counts",
        ),
        (
            lambda audits, config: {
                **audits,
                "random_training": replace(
                    audits["random_training"], stream_sha256="A" * 64
                ),
            },
            "aggregate counts",
        ),
    ),
)
def test_report_rejects_invalid_arm_namespace_counts_and_hash(
    change, expected: str
) -> None:
    config = load_config(CONFIG_PATH)
    with pytest.raises(ModelDataError, match=expected):
        report_payload(
            config_path=CONFIG_PATH,
            config=config,
            audits=change(valid_audits(config), config),
            code_revision="f" * 40,
            runtime_seconds=1.0,
        )


@pytest.mark.parametrize("runtime", (float("nan"), float("inf"), -0.1, True))
def test_report_rejects_nonfinite_or_invalid_runtime(runtime: float) -> None:
    config = load_config(CONFIG_PATH)
    with pytest.raises(ModelDataError, match="provenance"):
        report_payload(
            config_path=CONFIG_PATH,
            config=config,
            audits=valid_audits(config),
            code_revision="f" * 40,
            runtime_seconds=runtime,
        )


def test_no_flag_preflight_does_not_load_collections_or_create_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = ROOT / "scripts/audit_week2_training_streams.py"
    specification = importlib.util.spec_from_file_location("week2_stream_cli", module_path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    config = load_config(CONFIG_PATH)
    output_bytes = {
        path: (ROOT / path).read_bytes() if (ROOT / path).exists() else None
        for path in config.output_paths
    }
    monkeypatch.setattr(module, "load_collection", lambda *_: pytest.fail("loader called"))
    monkeypatch.setattr(sys, "argv", [str(module_path)])
    assert module.main() == 0
    assert {
        path: (ROOT / path).read_bytes() if (ROOT / path).exists() else None
        for path in config.output_paths
    } == output_bytes
