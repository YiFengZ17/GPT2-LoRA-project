#!/usr/bin/env python3
"""Build the canonical portable-report artifact from validated experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row: dict, field: str) -> float:
    return float(row[field])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    aggregate = read_csv(args.data_dir / "validated_aggregate.csv")
    runs = read_csv(args.data_dir / "validated_runs.csv")
    quality = read_csv(args.data_dir / "quality_checks.csv")
    summary = json.loads((args.data_dir / "analysis_summary.json").read_text(encoding="utf-8"))
    design = summary["design"]
    seed_text = ", ".join(str(seed) for seed in design["seeds"])
    run_count = len(design["seeds"])
    configuration_count = int(design["configurations"])
    completed_runs = int(design["runs"])

    chart_rows = []
    table_rows = []
    for row in aggregate:
        chart_rows.append(
            {
                "configuration": row["configuration"],
                "test_accuracy_pct": 100 * number(row, "test_accuracy_mean"),
                "test_macro_f1_pct": 100 * number(row, "test_macro_f1_mean"),
                "accuracy_std_pp": 100 * number(row, "test_accuracy_std"),
                "macro_f1_std_pp": 100 * number(row, "test_macro_f1_std"),
                "trainable_fraction_pct": 100 * number(row, "trainable_fraction"),
                "trainable_parameters": int(row["trainable_parameters"]),
                "peak_cuda_memory_mb": number(row, "peak_cuda_memory_mb_mean"),
                "checkpoint_mb": number(row, "best_checkpoint_mb_mean"),
                "train_seconds": number(row, "train_seconds_mean"),
                "generalization_gap_pp": 100 * number(row, "final_generalization_gap_mean"),
                "runs": int(row["runs"]),
            }
        )
        table_rows.append(
            {
                "configuration": row["configuration"],
                "test_accuracy": f"{100 * number(row, 'test_accuracy_mean'):.2f} ± {100 * number(row, 'test_accuracy_std'):.2f}%",
                "test_macro_f1": f"{100 * number(row, 'test_macro_f1_mean'):.2f} ± {100 * number(row, 'test_macro_f1_std'):.2f}%",
                "trainable": f"{int(row['trainable_parameters']):,} ({100 * number(row, 'trainable_fraction'):.3f}%)",
                "peak_vram": f"{number(row, 'peak_cuda_memory_mb_mean') / 1024:.2f} GiB",
                "checkpoint": f"{number(row, 'best_checkpoint_mb_mean'):.2f} MiB",
                "train_time": f"{number(row, 'train_seconds_mean') / 60:.2f} min",
                "final_gap": f"{100 * number(row, 'final_generalization_gap_mean'):.1f} pp",
            }
        )

    r16 = next(row for row in chart_rows if row["configuration"] == "LoRA r=16")
    full = next(row for row in chart_rows if row["configuration"] == "Full")
    frozen = next(row for row in chart_rows if row["configuration"] == "Frozen")
    headline = [
        {
            "r16_test_accuracy": r16["test_accuracy_pct"] / 100,
            "r16_accuracy_std_pp": r16["accuracy_std_pp"],
            "r16_macro_f1": r16["test_macro_f1_pct"] / 100,
            "r16_macro_f1_std_pp": r16["macro_f1_std_pp"],
            "r16_trainable_fraction": r16["trainable_fraction_pct"] / 100,
            "r16_checkpoint_mb": r16["checkpoint_mb"],
        }
    ]
    quality_summary = [
        {
            "completed_runs": len(runs),
            "failed_checks": sum(row["status"] != "PASS" for row in quality),
            "quality_checks": len(quality),
        }
    ]

    generated_at = datetime.now(timezone.utc).isoformat()
    full_best_epochs = sorted(
        (int(row["seed"]), int(row["best_epoch_by_validation_accuracy"]))
        for row in runs
        if row["configuration"] == "Full"
    )
    full_best_epoch_text = ", ".join(
        f"seed {seed}: epoch {epoch}" for seed, epoch in full_best_epochs
    )
    runs_by_configuration_seed = {
        (row["configuration"], int(row["seed"])): row for row in runs
    }

    def positive_pair_count(candidate: str, baseline: str, metric: str) -> int:
        return sum(
            number(runs_by_configuration_seed[(candidate, seed)], metric)
            > number(runs_by_configuration_seed[(baseline, seed)], metric)
            for seed in design["seeds"]
        )

    r16_accuracy_wins = positive_pair_count("LoRA r=16", "Full", "test_accuracy")
    r16_f1_wins = positive_pair_count("LoRA r=16", "Full", "test_macro_f1")
    r4_accuracy_wins = positive_pair_count("LoRA r=4", "Full", "test_accuracy")
    r4_f1_wins = positive_pair_count("LoRA r=4", "Full", "test_macro_f1")
    sources = [
        {
            "id": "validated_aggregate",
            "label": "Validated configuration aggregates",
            "path": "queries/validated_aggregate.sql",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": "SELECT * FROM read_csv_auto('analysis/report-data/validated_aggregate.csv', header = true);",
                "description": "Loads the independently recomputed configuration-level means and sample standard deviations used in report charts and cards.",
                "tables_used": ["analysis/report-data/validated_aggregate.csv"],
            },
        },
        {
            "id": "validated_runs",
            "label": "Validated per-run metrics",
            "path": "queries/validated_runs.sql",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": "SELECT * FROM read_csv_auto('analysis/report-data/validated_runs.csv', header = true);",
                "description": "Loads the 30 cross-checked run-level records used for paired comparisons and checkpoint-selection evidence.",
                "tables_used": ["analysis/report-data/validated_runs.csv"],
            },
        },
        {
            "id": "quality_checks",
            "label": "Experiment record quality checks",
            "path": "queries/quality_checks.sql",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": "SELECT * FROM read_csv_auto('analysis/report-data/quality_checks.csv', header = true);",
                "description": "Loads the reproducible data-quality and consistency check results.",
                "tables_used": ["analysis/report-data/quality_checks.csv"],
            },
        },
    ]

    main = summary["main_result"]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "GPT-2 LoRA on SST-5: Main Study Results",
            "description": f"Technical analysis of {completed_runs} completed runs across frozen, full fine-tuning, and four LoRA ranks.",
            "generatedAt": generated_at,
            "sources": sources,
            "cards": [
                {
                    "id": "r16_accuracy",
                    "description": f"Mean across seeds {seed_text}; chip shows sample standard deviation in percentage points.",
                    "dataset": "headline",
                    "sourceId": "validated_aggregate",
                    "metrics": [
                        {"label": "LoRA r=16 test accuracy", "field": "r16_test_accuracy", "format": "percent"},
                        {"label": "SD (pp)", "field": "r16_accuracy_std_pp", "format": "number"},
                    ],
                },
                {
                    "id": "r16_f1",
                    "description": f"Unweighted mean of class-level F1, averaged over the {run_count} runs.",
                    "dataset": "headline",
                    "sourceId": "validated_aggregate",
                    "metrics": [
                        {"label": "LoRA r=16 test macro-F1", "field": "r16_macro_f1", "format": "percent"},
                        {"label": "SD (pp)", "field": "r16_macro_f1_std_pp", "format": "number"},
                    ],
                },
                {
                    "id": "r16_trainable",
                    "description": "Share of GPT-2 plus classifier parameters updated during LoRA training.",
                    "dataset": "headline",
                    "sourceId": "validated_aggregate",
                    "metrics": [
                        {"label": "Trainable parameter share", "field": "r16_trainable_fraction", "format": "percent"},
                        {"label": "Best checkpoint (MiB)", "field": "r16_checkpoint_mb", "format": "number"},
                    ],
                },
            ],
            "charts": [
                {
                    "id": "test_performance",
                    "title": "Mean test performance by training configuration",
                    "subtitle": f"SST-5 test split; bars show {run_count}-seed means on the same percentage-point scale.",
                    "showDescription": True,
                    "intent": "comparison",
                    "question": "Which configuration gives the strongest held-out accuracy and class-balanced F1?",
                    "rationale": "Grouped bars compare two commensurate percentage metrics across six discrete configurations.",
                    "comparisonContext": {"grain": "configuration", "unit": "percentage points", "baseline": "Frozen and Full"},
                    "type": "bar",
                    "dataset": "configuration_metrics",
                    "sourceId": "validated_aggregate",
                    "encodings": {
                        "x": {"field": "configuration", "type": "nominal", "label": "Configuration"},
                        "y": {
                            "fields": ["test_accuracy_pct", "test_macro_f1_pct"],
                            "type": "quantitative",
                            "format": "number",
                            "label": "Score",
                            "unit": "%",
                        },
                        "tooltip": [
                            {"field": "accuracy_std_pp", "type": "quantitative", "label": "Accuracy SD (pp)"},
                            {"field": "macro_f1_std_pp", "type": "quantitative", "label": "Macro-F1 SD (pp)"},
                            {"field": "runs", "type": "quantitative", "label": "Runs"},
                        ],
                    },
                    "valueFormat": "number",
                    "unit": "%",
                    "layout": "full",
                    "surface": {"legend": {"show": True, "position": "bottom"}, "valueLabels": "auto"},
                },
                {
                    "id": "generalization_gap",
                    "title": "Final-epoch train–validation accuracy gap",
                    "subtitle": f"Mean across {run_count} seeds; a larger positive gap signals stronger overfitting by epoch {design['epochs']}.",
                    "showDescription": True,
                    "intent": "comparison",
                    "question": "How strongly does each configuration overfit by the final epoch?",
                    "rationale": "A single-series bar chart makes the magnitude difference across six configurations directly comparable.",
                    "comparisonContext": {"grain": "configuration", "unit": "percentage points", "baseline": "zero gap"},
                    "type": "bar",
                    "dataset": "configuration_metrics",
                    "sourceId": "validated_aggregate",
                    "encodings": {
                        "x": {"field": "configuration", "type": "nominal", "label": "Configuration"},
                        "y": {"field": "generalization_gap_pp", "type": "quantitative", "format": "number", "label": "Accuracy gap", "unit": "pp"},
                        "tooltip": [{"field": "runs", "type": "quantitative", "label": "Runs"}],
                    },
                    "valueFormat": "number",
                    "unit": "pp",
                    "layout": "full",
                    "referenceLines": [{"value": 0, "label": "No gap"}],
                    "surface": {"legend": {"show": False}, "valueLabels": "auto"},
                },
            ],
            "tables": [
                {
                    "id": "exact_comparison",
                    "title": f"Exact {run_count}-seed configuration summary",
                    "subtitle": "Mean ± sample SD for performance; mean resource values; five epochs per run.",
                    "showDescription": True,
                    "dataset": "exact_table",
                    "sourceId": "validated_aggregate",
                    "defaultSort": {"field": "configuration", "direction": "asc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "configuration", "label": "Configuration", "type": "text"},
                        {"field": "test_accuracy", "label": "Test accuracy", "type": "text"},
                        {"field": "test_macro_f1", "label": "Test macro-F1", "type": "text"},
                        {"field": "trainable", "label": "Trainable params", "type": "text"},
                        {"field": "peak_vram", "label": "Peak VRAM", "type": "text"},
                        {"field": "checkpoint", "label": "Checkpoint", "type": "text"},
                        {"field": "train_time", "label": "Train time", "type": "text"},
                        {"field": "final_gap", "label": "Final gap", "type": "text"},
                    ],
                }
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "layout": "full", "body": "# GPT-2 LoRA on SST-5: Main Study Results"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "validated_aggregate",
                    "body": (
                        "## Technical summary\n\n"
                        f"**LoRA r=16 has the strongest observed test-set tradeoff in this study.** Across {run_count} seeds it reached "
                        f"**{100*r16['test_accuracy_pct']/100:.2f}% test accuracy** and **{r16['test_macro_f1_pct']:.2f}% macro-F1**. "
                        f"Relative to full fine-tuning, that is **+{main['accuracy_delta_vs_full_pp']:.2f} percentage points** in accuracy and "
                        f"**+{main['macro_f1_delta_vs_full_pp']:.2f} points** in macro-F1 while updating only **{r16['trainable_fraction_pct']:.3f}%** of parameters. "
                        f"Validation accuracy favors {main['validation_led_accuracy_choice']}, while validation macro-F1 narrowly favors {main['validation_led_macro_f1_choice']}. "
                        "The broader conclusion that LoRA is competitive and much smaller is stronger than any claim that one rank is universally optimal."
                    ),
                },
                {"id": "headline_metrics", "type": "metric-strip", "layout": "full", "cardIds": ["r16_accuracy", "r16_f1", "r16_trainable"]},
                {
                    "id": "performance_finding",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "validated_aggregate",
                    "body": (
                        "## LoRA improves the performance–efficiency tradeoff\n\n"
                        f"All four LoRA settings outperform the frozen baseline on mean accuracy and macro-F1. LoRA r=16 leads on both headline metrics, "
                        f"beating Frozen by **{main['accuracy_delta_vs_frozen_pp']:.2f} accuracy points** and **{main['macro_f1_delta_vs_frozen_pp']:.2f} macro-F1 points**. "
                        f"Against Full, r=16 wins {r16_accuracy_wins}/{run_count} paired seeds on accuracy but only {r16_f1_wins}/{run_count} on macro-F1; r=4 wins {r4_accuracy_wins}/{run_count} and {r4_f1_wins}/{run_count}, respectively. "
                        "Higher rank is not uniformly better: r=8 trails r=4 on accuracy, so the data supports a useful rank range rather than a monotonic scaling law."
                    ),
                },
                {"id": "performance_chart", "type": "chart", "layout": "full", "chartId": "test_performance"},
                {
                    "id": "efficiency_finding",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "validated_aggregate",
                    "body": (
                        "## Rank 16 keeps nearly all weights frozen\n\n"
                        f"LoRA r=16 trains **{main['trainable_parameter_reduction_vs_full_x']:.1f}× fewer parameters** than full fine-tuning, "
                        f"produces a **{main['checkpoint_reduction_vs_full_x']:.1f}× smaller** best checkpoint, uses **{main['peak_memory_reduction_vs_full_percent']:.1f}% less** peak allocated CUDA memory, "
                        f"and finishes training **{main['train_time_reduction_vs_full_percent']:.1f}% faster** in these concurrent runs. "
                        "The storage and parameter-count advantages are intrinsic; the timing difference is less controlled because runs were collected in two launcher waves with different GPU concurrency."
                    ),
                },
                {"id": "exact_table", "type": "table", "layout": "full", "tableId": "exact_comparison"},
                {
                    "id": "overfit_finding",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "validated_runs",
                    "body": (
                        "## Full fine-tuning overfits sharply by epoch 5\n\n"
                        f"Full fine-tuning ends with a mean train–validation accuracy gap of **{full['generalization_gap_pp']:.1f} points**, compared with "
                        f"**{r16['generalization_gap_pp']:.1f} points** for LoRA r=16 and **{frozen['generalization_gap_pp']:.1f} points** for the frozen encoder. "
                        f"All {run_count} full-tuning runs selected their best checkpoint within the first two epochs by validation accuracy ({full_best_epoch_text}), so later epochs mainly increase memorization rather than validation performance."
                    ),
                },
                {"id": "gap_chart", "type": "chart", "layout": "full", "chartId": "generalization_gap"},
                {
                    "id": "scope",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "validated_runs",
                    "body": (
                        "## Scope and metric definitions\n\n"
                        "The study compares six GPT-2 classification strategies on SST-5: a frozen encoder, full fine-tuning, and LoRA ranks 2, 4, 8, and 16. "
                        f"Each configuration uses seeds {seed_text} for {design['epochs']} epochs with FP16 and effective batch size 8. The fixed splits contain 8,544 training, 1,101 validation, and 2,210 test examples. "
                        "Accuracy measures the fraction of correct test predictions; macro-F1 weights each of the five sentiment classes equally."
                    ),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "quality_checks",
                    "body": (
                        "## Validation and aggregation method\n\n"
                        f"The analysis checked the suite manifest, the complete {configuration_count}×{run_count} configuration–seed grid, unique run names, required result/config/history files, {design['epochs']} recorded epochs, confusion-matrix totals, training-code hashes, and agreement between per-run JSON and summary CSV. "
                        f"All {summary['quality']['checks_total']} checks passed. Configuration means and sample standard deviations were independently recomputed and matched the supplied aggregate files. The reported test metrics come from the checkpoint with maximum validation accuracy."
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## What the experiment does and does not establish\n\n"
                        f"- {run_count} seeds quantify run-to-run variation more credibly than the initial three-seed study, but remain limited for strong inferential claims.\n"
                        "- `deterministic=false` means identical-seed reruns need not be bitwise identical.\n"
                        "- `git_commit=null`; the verified SHA-256 snapshot identifies the matching local training files but does not preserve Git history.\n"
                        "- Two launcher waves used different GPU concurrency, so runtime comparisons remain vulnerable to GPU assignment and other server load.\n"
                        "- Because every configuration was evaluated on the same test split, rank choice should remain validation-led; the highest test mean should not be treated as an independently selected winner."
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## Recommended next steps\n\n"
                        "1. Use **LoRA r=16** for the best-observed-test analysis, but label that rank comparison as post-hoc; keep **Frozen** and **Full** as the two main baselines.\n"
                        f"2. Report all {run_count}-seed means and sample SDs; include the exact comparison table instead of highlighting a single best seed.\n"
                        f"3. Predeclare the validation metric for future model selection: accuracy selects **{main['validation_led_accuracy_choice']}**, while macro-F1 selects **{main['validation_led_macro_f1_choice']}** in these runs.\n"
                        "4. For a stronger confirmatory claim, evaluate the shortlisted rank on a fresh held-out dataset or a preregistered additional-seed batch with deterministic settings where practical.\n"
                        "5. Record a Git commit in future runs and benchmark speed in isolation on one fixed physical GPU if runtime efficiency is part of the formal claim."
                    ),
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## Further questions\n\n"
                        "Would the r=16 advantage persist on a fresh held-out evaluation, and would early stopping at the validation-selected epoch reduce the generalization gap without changing rank ordering? "
                        "A future ablation could also separate rank from learning-rate and dropout choices if those hyperparameters are varied."
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": headline,
                "configuration_metrics": chart_rows,
                "exact_table": table_rows,
                "quality_summary": quality_summary,
            },
            "accessIssues": [],
        },
        "sources": sources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
