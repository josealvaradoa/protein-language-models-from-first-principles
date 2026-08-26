"""Deterministic Markdown rendering for the Week 3 aggregate report."""

from __future__ import annotations

from protein_lm.data.model_data.contracts import ModelDataError


def render_markdown(payload: dict[str, object]) -> str:
    try:
        comparison = payload["final_three_seed_comparison"]
        baseline = payload["fixed_budget_baseline_comparison"]
        diagnostic = payload["position_availability_diagnostic"]
        assert isinstance(comparison, dict) and isinstance(baseline, dict)
        assert isinstance(diagnostic, dict)
        c20 = comparison["context20"]
        e64 = comparison["embedding64_challenger"]
        assert isinstance(c20, dict) and isinstance(e64, dict)
        c20_aggregate, e64_aggregate = c20["aggregate"], e64["aggregate"]
        assert isinstance(c20_aggregate, dict) and isinstance(e64_aggregate, dict)
        baseline_row = baseline["baseline"]
        assert isinstance(baseline_row, dict)
        curves = payload["learning_curves"]["series"]
        assert isinstance(curves, list)
        c10 = next(item for item in curves if item["model"] == "C10_E32_H800")
        assert isinstance(c10, dict)
        c10_points = c10["points"]
        assert isinstance(c10_points, list)
        c10_final = next(
            item for item in c10_points if item["prediction_position"] == 100000000
        )
        assert isinstance(c10_final, dict)
    except (AssertionError, KeyError, TypeError) as error:
        raise ModelDataError("public report payload cannot be rendered") from error
    lines = [
        "# Week 3: MLP Protein Context",
        "",
        "## Result",
        "",
        "The fixed-context C=20 MLP reached lower native-validation cross-entropy than the frozen Week 2 family-aware neural bigram at the same 100,000,000-prediction budget. This is validation-only evidence, with no sealed-test access and no significance claim.",
        "",
        "| Model | Parameters | Mean CE | Sample SD | Mean accuracy |",
        "|---|---:|---:|---:|---:|",
        f"| C=10, E=32, H=800 original control | 274293 | {_f(c10_final['mean_cross_entropy'])} | {_f(c10_final['sample_standard_deviation_cross_entropy'])} | n/a |",
        f"| C=20, E=32, H=800 | 530293 | {_f(c20_aggregate['mean_cross_entropy'])} | {_f(c20_aggregate['sample_standard_deviation_cross_entropy'])} | {_f(c20_aggregate['mean_accuracy'])} |",
        f"| C=10, E=64, H=800 challenger | 530965 | {_f(e64_aggregate['mean_cross_entropy'])} | {_f(e64_aggregate['sample_standard_deviation_cross_entropy'])} | {_f(e64_aggregate['mean_accuracy'])} |",
        f"| Week 2 family-aware neural bigram | n/a | {_f(baseline_row['cross_entropy'])} | n/a | {_f(baseline_row['accuracy'])} |",
        "",
        "The E=64 challenger had a matched flattened input width and a similar parameter count. Its mean CE was higher by " + _f(e64["embedding64_minus_context20_mean_cross_entropy"]) + ", exceeding the frozen 0.001 material threshold. This supports context allocation over parameter count within this architecture and contract only.",
        "",
        "## Architecture",
        "",
        "Each context token is looked up in the embedding table. The context embeddings are flattened, passed through one tanh hidden layer, and projected to next-token logits. A causal protein language model is a statistical factorization, not a model of ribosomal residue choice. Ribosomes read mRNA codons, not prior amino-acid residues.",
        "",
        "## Experimental evidence trail",
        "",
        "The report separates the original C10 learning curve, the C20 25M capacity screen and its 100M continuation, the final C20 versus E64 challenger, exploratory LR tails and one-epoch continuation, and the post-freeze position diagnostic. H=1600 appears only in the 25M screen.",
        "",
        "## Runtimes and parameters",
        "",
        "Observed staged CPU wall time includes harness, validation, and checkpoint overhead. It is not pure training time. The Week 2 public baseline does not provide a comparable training runtime and is not compared on runtime.",
        "",
        "## Descriptive Position Diagnostic",
        "",
        "The post-freeze position diagnostic did not train models and did not generate significance evidence. It places each native-validation target by the number of real preceding residues available to the context window.",
        "",
        "| Available prior residues | Targets | C20 mean CE | E64 mean CE | Share of mean-NLL advantage |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in diagnostic["bins"]:
        assert isinstance(row, dict)
        lines.append(f"| {row['bin']} | {row['token_count']} | {_f(row['context20_mean_cross_entropy'])} | {_f(row['embedding64_mean_cross_entropy'])} | {_f(row['share_of_total_mean_nll_advantage'])} |")
    lines.extend(["", "## Exploratory outcomes", "", "LR tails and the one-epoch continuation are recorded as exploratory negative or descriptive outcomes. They do not reopen model selection.", ""])
    lines.extend(["## Learning curves", "", "| Model | Predictions | Mean CE | Sample SD | Stage |", "|---|---:|---:|---:|---|"])
    for series in curves:
        for row in series["points"]:
            lines.append(f"| {series['model']} | {row['prediction_position']} | {_f(row['mean_cross_entropy'])} | {_f(row['sample_standard_deviation_cross_entropy'])} | {row['stage']} |")
    lines.extend(["", "## 25M capacity screen", "", "| Arm | Parameters | Mean CE | Sample SD | Stage |", "|---|---:|---:|---:|---|"])
    for row in payload["capacity_screen_25m"]:
        lines.append(f"| {row['arm']} | {row['parameter_count']} | {_f(row['mean_cross_entropy'])} | {_f(row.get('sample_standard_deviation_cross_entropy', 0))} | {row['stage']} |")
    lines.extend(["", "H1600 ended at 25M and was not run to 100M.", "", "## Observed staged CPU wall time", "", "| Model | Mean seconds | Sample SD |", "|---|---:|---:|"])
    for row in payload["observed_staged_cpu_wall_time_seconds"]["series"]:
        lines.append(f"| {row['model']} | {_f(row['mean_seconds'])} | {_f(row['sample_standard_deviation_seconds'])} |")
    lines.extend(["", "## Exploratory LR tails and one epoch", "", "| Run | Position | Mean CE | Sample SD |", "|---|---:|---:|---:|"])
    for row in payload["negative_and_exploratory_outcomes"]["lr_tails"]:
        lines.append(f"| {row['arm']} | 100000000 | {_f(row['mean_cross_entropy'])} | {_f(row.get('sample_standard_deviation_cross_entropy', 0))} |")
    for row in payload["negative_and_exploratory_outcomes"]["one_epoch_continuation"]:
        lines.append(f"| one_epoch | {row['prediction_position']} | {_f(row['mean_cross_entropy'])} | {_f(row.get('sample_standard_deviation_cross_entropy', 0))} |")
    lines.extend([
        "",
        "Embedding PCA panels and within-seed residue cosine summaries are descriptive only. PCA is centered NumPy SVD run separately for each seed with canonicalized component signs. Its axes are not treated as directly comparable across seeds, and BOS is excluded from residue similarity summaries.",
        "",
        "## Limits",
        "",
        "Remaining cross-entropy can reflect genuine conditional variability and information absent from this model's fixed context, including family, function, global fold, distant residues, and future context. These results do not claim biological mechanism, structure learning, function, or test performance.",
        "",
    ])
    return "\n".join(lines)


def _f(value: object) -> str:
    if not isinstance(value, (int, float)):
        raise ModelDataError("public report numeric field is invalid")
    return f"{value:.6f}"
