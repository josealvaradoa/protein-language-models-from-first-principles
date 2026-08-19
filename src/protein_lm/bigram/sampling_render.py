"""Deterministic Markdown rendering for synthetic Week 2 samples."""

from __future__ import annotations


def render_markdown(payload: dict[str, object]) -> str:
    """Render every synthetic sequence and its deterministic termination record."""
    lines = [
        "# Week 2 Bigram Sampling Diagnostic v1",
        "",
        "Synthetic, non-functional educational model output. These sequences are not selected, evaluated, or evidence of biological function, safety, or therapeutic value.",
        "",
        "## Sampling contract",
        "",
        "- Temperature: 1.0",
        "- Top-k: none",
        "- Top-p: none",
        "- Start context: BOS",
        "- Stop: EOS or 128 residues",
        "",
    ]
    samples = payload["samples"]
    assert isinstance(samples, list)
    for arm in ("random_training", "family_aware_training"):
        lines.extend(
            [
                f"## {arm}",
                "",
                "| Index | Seed | Residues | Stop | Synthetic sequence |",
                "| ---: | ---: | ---: | --- | --- |",
            ]
        )
        for sample in (item for item in samples if item["model_arm"] == arm):
            lines.append(
                f"| {sample['sample_index']} | {sample['seed']} | {sample['residue_length']} | {sample['termination_reason']} | `{sample['sequence']}` |"
            )
        lines.append("")
    source = payload["source"]
    runtime = payload["runtime"]
    assert isinstance(source, dict) and isinstance(runtime, dict)
    lines.extend(
        [
            "## Provenance",
            "",
            f"Candidate: `{source['relative_path']}`",
            f"Candidate registry SHA-256: `{source['candidate_registry_sha256']}`",
            f"Candidate run record SHA-256: `{source['candidate_run_record_sha256']}`",
            f"Sampling code revision: `{payload['publication_code_revision']}`",
            f"uv.lock SHA-256: `{runtime['uv_lock_sha256']}`",
            f"Torch version: `{runtime['torch_version']}`",
            "",
            "## Limitations",
            "",
            "- These are stochastic draws from adjacent-residue logits, not functional proteins.",
            "- This diagnostic was not used for model selection or biological claims.",
            "- It does not access datasets, validation, test, or evaluation collections.",
        ]
    )
    return "\n".join(lines) + "\n"
