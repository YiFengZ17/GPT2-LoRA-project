"""Run the complete Frozen/LoRA/Full comparison as isolated subprocesses."""

import argparse
import csv
import gc
import json
import os
import queue
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Experiment:
    mode: str
    seed: int
    rank: Optional[int] = None

    @property
    def name(self) -> str:
        rank_part = f"-r{self.rank}" if self.rank is not None else ""
        return f"{self.mode}{rank_part}-seed{self.seed}"


@dataclass(frozen=True)
class RunOutcome:
    experiment: Experiment
    gpu: Optional[int]
    return_code: int
    result_path: Path
    log_path: Path


def parse_args(arguments: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Frozen, LoRA rank ablations, and Full fine-tuning for multiple "
            "seeds, then aggregate accuracy, F1, time, memory, and parameter count."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/main-study"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 2026])
    parser.add_argument("--ranks", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--precision", choices=("auto", "fp32", "fp16"), default="auto"
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        help="Physical GPU indices. Without this option only the first GPU is used.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        help="Maximum concurrent runs; defaults to one run per selected GPU.",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    args = parser.parse_args(arguments)

    if any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be non-negative")
    if any(rank <= 0 for rank in args.ranks):
        parser.error("ranks must be positive")
    if args.epochs <= 0 or args.batch_size <= 0 or args.max_length <= 0:
        parser.error("epochs, batch-size, and max-length must be positive")
    if args.gradient_accumulation_steps <= 0:
        parser.error("gradient-accumulation-steps must be positive")
    if args.num_workers < 0:
        parser.error("num-workers must be non-negative")
    if args.parallel is not None and args.parallel <= 0:
        parser.error("parallel must be positive")
    if args.device == "cpu" and args.gpus:
        parser.error("--gpus cannot be used with --device cpu")
    return args


def build_experiments(seeds: Iterable[int], ranks: Iterable[int]) -> List[Experiment]:
    experiments: List[Experiment] = []
    for seed in seeds:
        experiments.append(Experiment("frozen", seed))
        experiments.extend(Experiment("lora", seed, rank) for rank in ranks)
        experiments.append(Experiment("full", seed))
    return experiments


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def command_for(
    experiment: Experiment,
    run_dir: Path,
    args: argparse.Namespace,
) -> List[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_experiment.py"),
        "--mode",
        experiment.mode,
        "--seed",
        str(experiment.seed),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--max-length",
        str(args.max_length),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--precision",
        args.precision,
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--output-dir",
        str(run_dir),
    ]
    if experiment.rank is not None:
        command.extend(["--rank", str(experiment.rank)])
    if args.deterministic:
        command.append("--deterministic")
    for argument_name in (
        "max_train_samples",
        "max_validation_samples",
        "max_test_samples",
    ):
        value = getattr(args, argument_name)
        if value is not None:
            command.extend([f"--{argument_name.replace('_', '-')}", str(value)])

    latest_checkpoint = run_dir / "latest.pt"
    result_path = run_dir / "result.json"
    if latest_checkpoint.exists() and not result_path.exists():
        command.extend(["--resume-from", str(latest_checkpoint)])
    return command


def run_and_tee(
    command: Sequence[str],
    log_path: Path,
    experiment_name: str,
    gpu: Optional[int],
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    prefix = f"[{experiment_name}|GPU {gpu if gpu is not None else 'CPU'}] "
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(
            f"\nCUDA_VISIBLE_DEVICES={gpu if gpu is not None else ''} "
            f"$ {' '.join(command)}\n"
        )
        log_file.flush()
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(prefix + line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return process.wait()


def discover_gpus(args: argparse.Namespace) -> List[Optional[int]]:
    if args.device == "cpu":
        return [None]
    if args.gpus:
        if len(set(args.gpus)) != len(args.gpus):
            raise ValueError("--gpus contains duplicate indices")
        return list(args.gpus)

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "Could not discover GPUs; pass --gpus explicitly or use --device cpu"
        ) from error

    gpu_indices = [
        int(line.strip()) for line in result.stdout.splitlines() if line.strip()
    ]
    if not gpu_indices:
        raise RuntimeError("nvidia-smi did not report any GPUs")
    # Do not occupy every card on a shared server unless the user explicitly
    # lists the GPUs allocated to this experiment.
    return [gpu_indices[0]]


def prepare_shared_assets() -> None:
    """Download/cache shared assets once before concurrent workers start."""
    from transformers import GPT2LMHeadModel

    from data import create_tokenizer, load_sst5

    print("Preparing SST-5, GPT-2 tokenizer, and pretrained weights once...")
    load_sst5()
    create_tokenizer()
    pretrained = GPT2LMHeadModel.from_pretrained("gpt2")
    del pretrained
    gc.collect()
    print("Shared assets are ready; starting experiment workers.")


def execute_experiment(
    experiment: Experiment,
    run_dir: Path,
    args: argparse.Namespace,
    available_gpus: "queue.Queue[Optional[int]]",
) -> RunOutcome:
    gpu = available_gpus.get()
    log_path = run_dir / "train.log"
    result_path = run_dir / "result.json"
    try:
        command = command_for(experiment, run_dir, args)
        print(
            f"Starting {experiment.name} on "
            f"{'CPU' if gpu is None else f'GPU {gpu}'}",
            flush=True,
        )
        return_code = run_and_tee(
            command,
            log_path,
            experiment_name=experiment.name,
            gpu=gpu,
        )
        return RunOutcome(
            experiment=experiment,
            gpu=gpu,
            return_code=return_code,
            result_path=result_path,
            log_path=log_path,
        )
    finally:
        available_gpus.put(gpu)


def flatten_result(experiment: Experiment, result: Dict[str, Any]) -> Dict[str, Any]:
    metadata = result["metadata"]
    test = result["test"]
    validation = result["best_validation"]
    peak_bytes = result.get("peak_cuda_memory_bytes")
    return {
        "name": experiment.name,
        "status": result["status"],
        "mode": experiment.mode,
        "rank": experiment.rank,
        "seed": experiment.seed,
        "precision": metadata["precision"],
        "effective_batch_size": metadata["effective_batch_size"],
        "physical_gpu": metadata["environment"].get("cuda_visible_devices"),
        "trainable_parameters": metadata["trainable_parameters"],
        "total_parameters": metadata["total_parameters"],
        "trainable_fraction": metadata["trainable_fraction"],
        "validation_accuracy": validation["accuracy"],
        "validation_macro_f1": validation["macro_f1"],
        "test_accuracy": test["accuracy"],
        "test_macro_f1": test["macro_f1"],
        "train_seconds": result["train_seconds"],
        "wall_seconds": result["wall_seconds"],
        "peak_cuda_memory_mb": peak_bytes / (1024 ** 2) if peak_bytes else None,
        "best_checkpoint_mb": result["best_checkpoint_bytes"] / (1024 ** 2),
        "epochs_completed": result["epochs_completed"],
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, Optional[int]], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["mode"], row["rank"]), []).append(row)

    aggregated = []
    for (mode, rank), group in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1] or 0)
    ):
        row: Dict[str, Any] = {
            "mode": mode,
            "rank": rank,
            "runs": len(group),
            "trainable_parameters": group[0]["trainable_parameters"],
            "trainable_fraction": group[0]["trainable_fraction"],
        }
        for metric in (
            "validation_accuracy",
            "validation_macro_f1",
            "test_accuracy",
            "test_macro_f1",
            "train_seconds",
            "wall_seconds",
            "peak_cuda_memory_mb",
            "best_checkpoint_mb",
        ):
            values = [entry[metric] for entry in group if entry[metric] is not None]
            row[f"{metric}_mean"] = statistics.mean(values) if values else None
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregated.append(row)
    return aggregated


def main(arguments: Optional[Sequence[str]] = None) -> int:
    args = parse_args(arguments)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiments = build_experiments(args.seeds, args.ranks)
    gpu_slots = discover_gpus(args)
    parallelism = min(args.parallel or len(gpu_slots), len(gpu_slots))
    gpu_slots = gpu_slots[:parallelism]
    manifest = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "experiments": [experiment.name for experiment in experiments],
        "gpu_slots": gpu_slots,
        "parallelism": parallelism,
    }
    write_json(args.output_dir / "manifest.json", manifest)

    completed_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    pending: List[Tuple[Experiment, Path]] = []
    for index, experiment in enumerate(experiments, start=1):
        run_dir = args.output_dir / experiment.name
        result_path = run_dir / "result.json"
        command = command_for(experiment, run_dir, args)

        if result_path.exists():
            print(f"[{index}/{len(experiments)}] Reusing {experiment.name}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            completed_rows.append(flatten_result(experiment, result))
        elif args.dry_run:
            gpu = gpu_slots[(index - 1) % len(gpu_slots)]
            print(
                f"[{index}/{len(experiments)}] GPU {gpu}: " + " ".join(command)
            )
        else:
            pending.append((experiment, run_dir))

    if not args.dry_run and pending:
        prepare_shared_assets()
        available_gpus: "queue.Queue[Optional[int]]" = queue.Queue()
        for gpu in gpu_slots:
            available_gpus.put(gpu)

        with ThreadPoolExecutor(max_workers=parallelism) as executor:
            futures = {
                executor.submit(
                    execute_experiment,
                    experiment,
                    run_dir,
                    args,
                    available_gpus,
                ): experiment
                for experiment, run_dir in pending
            }
            for future in as_completed(futures):
                experiment = futures[future]
                try:
                    outcome = future.result()
                except Exception as error:
                    failure = {
                        "name": experiment.name,
                        "gpu": None,
                        "return_code": None,
                        "error": repr(error),
                    }
                    failures.append(failure)
                    write_json(args.output_dir / "failures.json", failures)
                    print(f"FAILED {experiment.name}: {error!r}", flush=True)
                    continue
                if outcome.return_code != 0 or not outcome.result_path.exists():
                    print(
                        f"FAILED {experiment.name} on GPU {outcome.gpu}; "
                        f"see {outcome.log_path}",
                        flush=True,
                    )
                    failure = {
                        "name": experiment.name,
                        "gpu": outcome.gpu,
                        "return_code": outcome.return_code,
                        "log": str(outcome.log_path),
                    }
                    failures.append(failure)
                    write_json(args.output_dir / "failures.json", failures)
                    continue

                print(
                    f"Completed {experiment.name} on GPU {outcome.gpu}",
                    flush=True,
                )
                result = json.loads(outcome.result_path.read_text(encoding="utf-8"))
                completed_rows.append(flatten_result(experiment, result))
                completed_rows.sort(key=lambda row: row["name"])
                write_csv(args.output_dir / "summary.csv", completed_rows)
                write_json(args.output_dir / "summary.json", completed_rows)
                aggregated_rows = aggregate(completed_rows)
                write_csv(args.output_dir / "aggregate.csv", aggregated_rows)
                write_json(args.output_dir / "aggregate.json", aggregated_rows)

    if completed_rows:
        completed_rows.sort(key=lambda row: row["name"])
        write_csv(args.output_dir / "summary.csv", completed_rows)
        write_json(args.output_dir / "summary.json", completed_rows)
        aggregated_rows = aggregate(completed_rows)
        write_csv(args.output_dir / "aggregate.csv", aggregated_rows)
        write_json(args.output_dir / "aggregate.json", aggregated_rows)

    # Retained for CLI compatibility. Parallel workers already in flight are
    # allowed to finish so checkpoints and logs are not corrupted.
    if args.fail_fast and failures:
        print("fail-fast requested; all already-started workers were finalized")

    manifest.update(
        {
            "status": "completed" if not failures else "completed_with_failures",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "completed_runs": len(completed_rows),
            "failed_runs": len(failures),
        }
    )
    write_json(args.output_dir / "manifest.json", manifest)
    print(
        f"\nSuite finished: {len(completed_rows)} completed, "
        f"{len(failures)} failed. Summary: {args.output_dir / 'aggregate.csv'}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
