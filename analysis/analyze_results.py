#!/usr/bin/env python3
"""Validate and summarize the GPT-2 LoRA main study without external packages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = (
    "validation_accuracy",
    "validation_macro_f1",
    "test_accuracy",
    "test_macro_f1",
    "train_seconds",
    "wall_seconds",
    "peak_cuda_memory_mb",
    "best_checkpoint_mb",
)


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def config_label(mode: str, rank: int | None) -> str:
    return f"LoRA r={rank}" if mode == "lora" else mode.capitalize()


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    manifest = load_json(root / "manifest.json")
    seeds = {int(seed) for seed in manifest["arguments"]["seeds"]}
    ranks = {int(rank) for rank in manifest["arguments"]["ranks"]}
    configs = {("frozen", None), ("full", None)} | {("lora", rank) for rank in ranks}
    expected_run_count = len(seeds) * len(configs)
    expected_epochs = int(manifest["arguments"]["epochs"])
    with (root / "summary.csv").open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    with (root / "aggregate.csv").open(newline="", encoding="utf-8") as handle:
        supplied_aggregate = list(csv.DictReader(handle))

    checks: list[dict] = []

    def check(name: str, passed: bool, evidence: str, severity: str = "critical") -> None:
        checks.append(
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "severity_if_failed": severity,
                "evidence": evidence,
            }
        )

    check(
        "Suite manifest completed",
        manifest.get("status") == "completed",
        f"status={manifest.get('status')}",
    )
    check(
        "Expected run count",
        manifest.get("completed_runs") == expected_run_count and manifest.get("failed_runs") == 0,
        f"completed={manifest.get('completed_runs')}, expected={expected_run_count}, failed={manifest.get('failed_runs')}",
    )
    check(
        "Summary row count",
        len(summary_rows) == expected_run_count,
        f"rows={len(summary_rows)}, expected={expected_run_count}",
    )
    names = [row["name"] for row in summary_rows]
    check(
        "Unique run names",
        len(names) == len(set(names)),
        f"unique={len(set(names))}/{expected_run_count}",
    )

    detail_rows: list[dict] = []
    environments = set()
    config_seed_pairs = set()
    result_errors: list[str] = []
    for source_row in summary_rows:
        name = source_row["name"]
        result_path = root / name / "result.json"
        history_path = root / name / "history.json"
        config_path = root / name / "config.json"
        if not (result_path.exists() and history_path.exists() and config_path.exists()):
            result_errors.append(f"{name}: missing result/history/config")
            continue
        result = load_json(result_path)
        history = load_json(history_path)
        if result.get("status") != "completed":
            result_errors.append(f"{name}: status={result.get('status')}")
        meta = result["metadata"]
        mode = meta["mode"]
        rank = meta.get("rank")
        seed = int(meta["seed"])
        config_seed_pairs.add(((mode, rank), seed))
        env = meta["environment"]
        environments.add((env["python"], env["torch"], env["cuda_runtime"], env["gpu"]))

        expected = {
            "status": result["status"],
            "mode": mode,
            "rank": "" if rank is None else str(rank),
            "seed": str(seed),
            "epochs_completed": str(result["epochs_completed"]),
        }
        for field, value in expected.items():
            if source_row[field] != value:
                result_errors.append(f"{name}: summary {field}={source_row[field]} != result {value}")
        numeric_pairs = {
            "validation_accuracy": result["best_validation"]["accuracy"],
            "validation_macro_f1": result["best_validation"]["macro_f1"],
            "test_accuracy": result["test"]["accuracy"],
            "test_macro_f1": result["test"]["macro_f1"],
            "train_seconds": result["train_seconds"],
            "wall_seconds": result["wall_seconds"],
            "peak_cuda_memory_mb": result["peak_cuda_memory_bytes"] / 1024**2,
            "best_checkpoint_mb": result["best_checkpoint_bytes"] / 1024**2,
        }
        for field, value in numeric_pairs.items():
            if not close(float(source_row[field]), value):
                result_errors.append(f"{name}: summary {field} does not match result.json")

        if len(history) != expected_epochs:
            result_errors.append(f"{name}: history has {len(history)} epochs")
        best_epoch_item = max(history, key=lambda item: item["validation"]["accuracy"])
        if not close(best_epoch_item["validation"]["accuracy"], result["best_validation"]["accuracy"]):
            result_errors.append(f"{name}: saved best validation accuracy mismatch")
        for split in ("best_validation", "test"):
            metrics = result[split]
            if sum(sum(row) for row in metrics["confusion_matrix"]) != metrics["examples"]:
                result_errors.append(f"{name}: {split} confusion matrix total mismatch")

        final_epoch = history[-1]
        detail_rows.append(
            {
                "configuration": config_label(mode, rank),
                "mode": mode,
                "rank": "" if rank is None else rank,
                "seed": seed,
                "best_epoch_by_validation_accuracy": best_epoch_item["epoch"],
                "validation_accuracy": result["best_validation"]["accuracy"],
                "validation_macro_f1": result["best_validation"]["macro_f1"],
                "test_accuracy": result["test"]["accuracy"],
                "test_macro_f1": result["test"]["macro_f1"],
                "final_train_accuracy": final_epoch["train"]["accuracy"],
                "final_validation_accuracy": final_epoch["validation"]["accuracy"],
                "final_generalization_gap": final_epoch["train"]["accuracy"]
                - final_epoch["validation"]["accuracy"],
                "train_seconds": result["train_seconds"],
                "wall_seconds": result["wall_seconds"],
                "peak_cuda_memory_mb": result["peak_cuda_memory_bytes"] / 1024**2,
                "best_checkpoint_mb": result["best_checkpoint_bytes"] / 1024**2,
                "trainable_parameters": meta["trainable_parameters"],
                "total_parameters": meta["total_parameters"],
                "trainable_fraction": meta["trainable_fraction"],
                "physical_gpu": int(source_row["physical_gpu"]),
            }
        )

    check(
        "Per-run files and cross-file consistency",
        not result_errors,
        "all result/config/history files agree with summary.csv" if not result_errors else "; ".join(result_errors),
    )
    expected_pairs = {(config, seed) for config in configs for seed in seeds}
    check(
        f"Complete {len(configs)} x {len(seeds)} design",
        config_seed_pairs == expected_pairs,
        f"observed={len(config_seed_pairs)}, expected={len(expected_pairs)}",
    )
    check(
        "Uniform software and GPU family",
        len(environments) == 1,
        "; ".join("/".join(env) for env in sorted(environments)),
        "high",
    )
    fixed_fields = {
        "precision": {row["precision"] for row in summary_rows},
        "effective_batch_size": {row["effective_batch_size"] for row in summary_rows},
        "epochs_completed": {row["epochs_completed"] for row in summary_rows},
    }
    check(
        "Uniform core training settings",
        all(len(values) == 1 for values in fixed_fields.values()),
        ", ".join(f"{key}={sorted(values)}" for key, values in fixed_fields.items()),
        "high",
    )

    hash_snapshot = root / "code-sha256.txt"
    hash_errors: list[str] = []
    hash_count = 0
    if hash_snapshot.exists():
        project_root = root.parents[1]
        for line in hash_snapshot.read_text(encoding="utf-8").splitlines():
            expected_hash, relative_path = line.split(maxsplit=1)
            source_path = project_root / relative_path.strip()
            hash_count += 1
            if not source_path.exists():
                hash_errors.append(f"missing {relative_path.strip()}")
                continue
            actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                hash_errors.append(f"mismatch {relative_path.strip()}")
    else:
        hash_errors.append("missing code-sha256.txt")
    check(
        "Training code SHA-256 snapshot",
        not hash_errors and hash_count > 0,
        f"{hash_count}/{hash_count} files match the recorded hashes"
        if not hash_errors
        else "; ".join(hash_errors),
        "high",
    )

    grouped: dict[tuple[str, int | None], list[dict]] = defaultdict(list)
    for row in detail_rows:
        grouped[(row["mode"], None if row["rank"] == "" else int(row["rank"]))].append(row)

    supplied_by_config = {
        (row["mode"], None if row["rank"] == "" else int(row["rank"])): row
        for row in supplied_aggregate
    }
    aggregate_errors: list[str] = []
    aggregate_rows: list[dict] = []
    for config in sorted(configs, key=lambda item: (0 if item[0] == "frozen" else 2 if item[0] == "full" else 1, item[1] or 0)):
        values = grouped[config]
        supplied = supplied_by_config[config]
        aggregate = {
            "configuration": config_label(*config),
            "mode": config[0],
            "rank": "" if config[1] is None else config[1],
            "runs": len(values),
            "trainable_parameters": values[0]["trainable_parameters"],
            "trainable_fraction": values[0]["trainable_fraction"],
        }
        for metric in METRICS:
            samples = [float(row[metric]) for row in values]
            mean = statistics.mean(samples)
            std = statistics.stdev(samples)
            aggregate[f"{metric}_mean"] = mean
            aggregate[f"{metric}_std"] = std
            if not close(mean, float(supplied[f"{metric}_mean"])) or not close(
                std, float(supplied[f"{metric}_std"])
            ):
                aggregate_errors.append(f"{aggregate['configuration']}: {metric}")
        aggregate["final_generalization_gap_mean"] = statistics.mean(
            row["final_generalization_gap"] for row in values
        )
        aggregate["best_epoch_median"] = statistics.median(
            row["best_epoch_by_validation_accuracy"] for row in values
        )
        aggregate_rows.append(aggregate)
    check(
        "Aggregate recomputation",
        not aggregate_errors,
        "all supplied means and sample standard deviations reproduced"
        if not aggregate_errors
        else ", ".join(aggregate_errors),
    )

    by_label_seed = {(row["configuration"], row["seed"]): row for row in detail_rows}
    paired_rows = []
    for candidate in ("LoRA r=2", "LoRA r=4", "LoRA r=8", "LoRA r=16", "Full"):
        for baseline in ("Frozen", "Full"):
            if candidate == baseline:
                continue
            for seed in sorted(seeds):
                left = by_label_seed[(candidate, seed)]
                right = by_label_seed[(baseline, seed)]
                paired_rows.append(
                    {
                        "candidate": candidate,
                        "baseline": baseline,
                        "seed": seed,
                        "test_accuracy_delta_pp": 100 * (left["test_accuracy"] - right["test_accuracy"]),
                        "test_macro_f1_delta_pp": 100 * (left["test_macro_f1"] - right["test_macro_f1"]),
                    }
                )

    aggregate_fields = list(aggregate_rows[0])
    detail_fields = list(detail_rows[0])
    write_csv(output / "validated_aggregate.csv", aggregate_rows, aggregate_fields)
    write_csv(output / "validated_runs.csv", detail_rows, detail_fields)
    write_csv(
        output / "paired_differences.csv",
        paired_rows,
        ["candidate", "baseline", "seed", "test_accuracy_delta_pp", "test_macro_f1_delta_pp"],
    )
    write_csv(output / "quality_checks.csv", checks, ["check", "status", "severity_if_failed", "evidence"])

    aggregate_by_label = {row["configuration"]: row for row in aggregate_rows}
    r16 = aggregate_by_label["LoRA r=16"]
    full = aggregate_by_label["Full"]
    frozen = aggregate_by_label["Frozen"]
    lora_rows = [row for row in aggregate_rows if row["mode"] == "lora"]
    best_test = max(lora_rows, key=lambda row: (row["test_accuracy_mean"], row["test_macro_f1_mean"]))
    validation_accuracy_choice = max(lora_rows, key=lambda row: row["validation_accuracy_mean"])
    validation_macro_f1_choice = max(lora_rows, key=lambda row: row["validation_macro_f1_mean"])
    summary = {
        "quality": {
            "checks_passed": sum(row["status"] == "PASS" for row in checks),
            "checks_total": len(checks),
            "all_passed": all(row["status"] == "PASS" for row in checks),
        },
        "design": {
            "runs": expected_run_count,
            "configurations": len(configs),
            "seeds": sorted(seeds),
            "epochs": expected_epochs,
            "dataset_sizes": {"train": 8544, "validation": 1101, "test": 2210},
            "deterministic": manifest["arguments"]["deterministic"],
            "git_commit_recorded": all(
                load_json(path)["metadata"]["environment"].get("git_commit") is not None
                for path in root.glob("*/result.json")
            ),
            "code_hash_snapshot_verified": not hash_errors and hash_count > 0,
        },
        "main_result": {
            "best_observed_test_configuration": best_test["configuration"],
            "validation_led_accuracy_choice": validation_accuracy_choice["configuration"],
            "validation_led_macro_f1_choice": validation_macro_f1_choice["configuration"],
            "test_accuracy_mean": r16["test_accuracy_mean"],
            "test_accuracy_std": r16["test_accuracy_std"],
            "test_macro_f1_mean": r16["test_macro_f1_mean"],
            "test_macro_f1_std": r16["test_macro_f1_std"],
            "accuracy_delta_vs_full_pp": 100 * (r16["test_accuracy_mean"] - full["test_accuracy_mean"]),
            "macro_f1_delta_vs_full_pp": 100 * (r16["test_macro_f1_mean"] - full["test_macro_f1_mean"]),
            "accuracy_delta_vs_frozen_pp": 100 * (r16["test_accuracy_mean"] - frozen["test_accuracy_mean"]),
            "macro_f1_delta_vs_frozen_pp": 100 * (r16["test_macro_f1_mean"] - frozen["test_macro_f1_mean"]),
            "trainable_parameter_reduction_vs_full_x": full["trainable_parameters"] / r16["trainable_parameters"],
            "checkpoint_reduction_vs_full_x": full["best_checkpoint_mb_mean"] / r16["best_checkpoint_mb_mean"],
            "peak_memory_reduction_vs_full_percent": 100
            * (1 - r16["peak_cuda_memory_mb_mean"] / full["peak_cuda_memory_mb_mean"]),
            "train_time_reduction_vs_full_percent": 100
            * (1 - r16["train_seconds_mean"] / full["train_seconds_mean"]),
        },
        "limitations": [
            f"Only {len(seeds)} seeds were run, so uncertainty estimates remain descriptive rather than strong inferential evidence.",
            "deterministic=false, so reruns with the same seed are not guaranteed to be bitwise identical.",
            "git_commit is null; the separately recorded SHA-256 snapshot verifies the matching local training files but does not preserve Git history.",
            "Runs were collected in two launcher waves with different GPU concurrency; wall-clock timing can be affected by GPU assignment and shared-server contention.",
            "The same test split was evaluated for all configurations; treat rank selection as validation-led and avoid post-hoc claims based only on the best test mean.",
        ],
    }
    with (output / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
