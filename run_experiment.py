
import argparse
import json
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import DatasetDict
from torch import nn

from classifier import SentimentClassifier
from data import create_dataloaders, create_tokenizer, load_sst5
from lora import inject_lora_into_qv
from model import Transformer
from train import create_optimizer, evaluate, fit, load_checkpoint


DEFAULT_LEARNING_RATES = {
    "frozen": 1e-3,
    "lora": 2e-4,
    "full": 2e-5,
}

# argument value
def parse_args(arguments: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GPT-2 on SST-5 with frozen, LoRA, or full adaptation."
    )
    parser.add_argument(
        "--mode",
        choices=("frozen", "lora", "full"),
        required=True,
    )
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "fp16"),
        default="auto",
        help="auto selects FP16 on CUDA and FP32 on CPU.",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic PyTorch algorithms when available.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run one epoch on at most 100 examples per split.",
    )

    args = parser.parse_args(arguments)

    if args.gradient_accumulation_steps <= 0:
        parser.error("gradient-accumulation-steps must be positive")

    if args.alpha is None:
        args.alpha = 2 * args.rank
    if args.epochs is None:
        args.epochs = 1 if args.debug else 3
    if args.learning_rate is None:
        args.learning_rate = DEFAULT_LEARNING_RATES[args.mode]

    if args.debug:
        if args.max_train_samples is None:
            args.max_train_samples = 100
        if args.max_validation_samples is None:
            args.max_validation_samples = 100
        if args.max_test_samples is None:
            args.max_test_samples = 100

    if args.output_dir is None:
        run_name = f"lora-r{args.rank}" if args.mode == "lora" else args.mode
        if args.debug:
            run_name += "-debug"
        args.output_dir = Path("runs") / run_name

    return args

# seed
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_reproducibility(seed: int, deterministic: bool) -> None:
    set_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def git_commit() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None

# select device
def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(device_name)


def limit_dataset(
    dataset: DatasetDict,
    train_samples: Optional[int],
    validation_samples: Optional[int],
    test_samples: Optional[int],
) -> DatasetDict:
    limits = {
        "train": train_samples,
        "validation": validation_samples,
        "test": test_samples,
    }
    limited = DatasetDict()

    for split_name, split in dataset.items():
        limit = limits[split_name]
        if limit is not None:
            if limit <= 0:
                raise ValueError(
                    f"{split_name} sample limit must be positive, got {limit}"
                )
            split = split.select(range(min(limit, len(split))))
        limited[split_name] = split

    return limited

# 3 mode
def configure_model(
    transformer: Transformer,
    mode: str,
    rank: int,
    alpha: float,
    lora_dropout: float,
) -> SentimentClassifier:
    # Initialize the classifier before mode-specific LoRA modules so that the
    # same seed gives every adaptation method the same classifier head.
    model = SentimentClassifier(transformer)
    if mode == "frozen":
        for parameter in transformer.parameters():
            parameter.requires_grad = False
    elif mode == "lora":
        inject_lora_into_qv(
            model=transformer,
            rank=rank,
            alpha=alpha,
            dropout=lora_dropout,
        )
    elif mode == "full":
        for parameter in transformer.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError(f"unsupported training mode: {mode}")

    return model

# summarize the amount of arguments
def count_parameters(model: nn.Module) -> tuple:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def main(arguments: Optional[List[str]] = None) -> None:
    args = parse_args(arguments)
    configure_reproducibility(args.seed, args.deterministic)
    device = select_device(args.device)
    precision = args.precision
    if precision == "auto":
        precision = "fp16" if device.type == "cuda" else "fp32"
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("--precision fp16 requires --device cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    wall_start = time.perf_counter()

    run_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    write_json(
        args.output_dir / "config.json",
        {
            "status": "running",
            "started_at": started_at.isoformat(),
            "arguments": run_config,
            "git_commit": git_commit(),
        },
    )

    dataset = limit_dataset(
        dataset=load_sst5(),
        train_samples=args.max_train_samples,
        validation_samples=args.max_validation_samples,
        test_samples=args.max_test_samples,
    )
    
    tokenizer = create_tokenizer()
    dataloaders = create_dataloaders(
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )

    transformer = Transformer.from_pretrained()
    model = configure_model(
        transformer=transformer,
        mode=args.mode,
        rank=args.rank,
        alpha=args.alpha,
        lora_dropout=args.lora_dropout,
    )
    optimizer = create_optimizer(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_parameters, trainable_parameters = count_parameters(model)
    metadata = {
        "mode": args.mode,
        "rank": args.rank if args.mode == "lora" else None,
        "alpha": args.alpha if args.mode == "lora" else None,
        "lora_dropout": args.lora_dropout if args.mode == "lora" else None,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "effective_batch_size": (
            args.batch_size * args.gradient_accumulation_steps
        ),
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "precision": precision,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
        "dataset_sizes": {
            split_name: len(split) for split_name, split in dataset.items()
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "git_commit": git_commit(),
        },
    }

    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"Precision: {precision}")
    print(
        "Batch size: "
        f"{args.batch_size} x {args.gradient_accumulation_steps} accumulation "
        f"= {args.batch_size * args.gradient_accumulation_steps} effective"
    )
    print(f"Output directory: {args.output_dir}")
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    if args.mode == "full" and device.type == "cpu":
        print("Warning: full fine-tuning on CPU may be very slow and use swap.")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history = fit(
        model=model,
        train_dataloader=dataloaders["train"],
        validation_dataloader=dataloaders["validation"],
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        checkpoint_dir=args.output_dir,
        resume_from=args.resume_from,
        metadata=metadata,
        precision=precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    best_checkpoint = args.output_dir / "best.pt"
    load_checkpoint(
        path=best_checkpoint,
        model=model,
        device=device,
    )
    test_start = time.perf_counter()
    test_metrics = evaluate(
        model=model,
        dataloader=dataloaders["test"],
        device=device,
        use_amp=precision == "fp16",
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    test_seconds = time.perf_counter() - test_start

    finished_at = datetime.now(timezone.utc)
    best_validation = max(
        history,
        key=lambda result: result["validation"]["accuracy"],
    )["validation"]
    result = {
        "status": "completed",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "wall_seconds": time.perf_counter() - wall_start,
        "train_seconds": sum(
            epoch["timing"]["train_seconds"] for epoch in history
        ),
        "validation_seconds": sum(
            epoch["timing"]["validation_seconds"] for epoch in history
        ),
        "test_seconds": test_seconds,
        "metadata": metadata,
        "best_validation": best_validation,
        "test": asdict(test_metrics),
        "epochs_completed": len(history),
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None
        ),
        "best_checkpoint_bytes": best_checkpoint.stat().st_size,
    }
    write_json(args.output_dir / "result.json", result)
    write_json(
        args.output_dir / "config.json",
        {
            "status": "completed",
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "arguments": run_config,
            "git_commit": metadata["environment"]["git_commit"],
        },
    )
    print(
        f"Test | loss {test_metrics.loss:.4f}, "
        f"accuracy {test_metrics.accuracy:.4f}, "
        f"macro-F1 {test_metrics.macro_f1:.4f}"
    )


if __name__ == "__main__":
    main()
