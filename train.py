import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW, Optimizer


@dataclass(frozen=True)
class ClassificationMetrics:
    loss: float
    accuracy: float
    examples: int
    macro_f1: float
    per_class_accuracy: List[Optional[float]]
    confusion_matrix: List[List[int]]


def _classification_metrics(
    total_loss: float,
    total_examples: int,
    confusion_matrix: Tensor,
) -> ClassificationMetrics:
    matrix = confusion_matrix.to(torch.float64)
    true_counts = matrix.sum(dim=1)
    predicted_counts = matrix.sum(dim=0)
    true_positives = matrix.diag()

    per_class_accuracy = [
        (true_positives[index] / true_counts[index]).item()
        if true_counts[index] > 0
        else None
        for index in range(matrix.shape[0])
    ]
    f1_scores = []
    for index in range(matrix.shape[0]):
        denominator = true_counts[index] + predicted_counts[index]
        f1_scores.append(
            (2 * true_positives[index] / denominator).item()
            if denominator > 0
            else 0.0
        )

    return ClassificationMetrics(
        loss=total_loss / total_examples,
        accuracy=true_positives.sum().item() / total_examples,
        examples=total_examples,
        macro_f1=sum(f1_scores) / len(f1_scores),
        per_class_accuracy=per_class_accuracy,
        confusion_matrix=confusion_matrix.tolist(),
    )


def create_optimizer(
    model: nn.Module,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
) -> Optimizer:
    if learning_rate <= 0:
        raise ValueError(
            f"learning_rate must be positive, got {learning_rate}"
        )
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("model does not contain any trainable parameters")

    return AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable[Dict[str, Tensor]],
    optimizer: Optimizer,
    device: torch.device,
    use_amp: bool = False,
    scaler: Optional[Any] = None,
    gradient_accumulation_steps: int = 1,
) -> ClassificationMetrics:
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if use_amp and scaler is None:
        raise ValueError("FP16 training requires a gradient scaler")

    try:
        num_batches = len(dataloader)  # type: ignore[arg-type]
    except TypeError:
        num_batches = None
    if gradient_accumulation_steps > 1 and num_batches is None:
        raise ValueError("gradient accumulation requires a sized dataloader")

    model.train()
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_examples = 0
    confusion_matrix = None

    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels)

        assert num_batches is not None or gradient_accumulation_steps == 1
        if num_batches is None:
            accumulation_group_size = 1
            should_step = True
        else:
            group_start = (
                batch_index // gradient_accumulation_steps
            ) * gradient_accumulation_steps
            accumulation_group_size = min(
                gradient_accumulation_steps,
                num_batches - group_start,
            )
            should_step = (
                (batch_index + 1) % gradient_accumulation_steps == 0
                or batch_index + 1 == num_batches
            )

        backward_loss = loss / accumulation_group_size
        if use_amp:
            scaler.scale(backward_loss).backward()
        else:
            backward_loss.backward()

        if should_step:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = labels.shape[0]
        total_loss += loss.detach().to(torch.float64) * batch_size
        total_examples += batch_size
        predictions = logits.argmax(dim=-1)
        num_classes = logits.shape[-1]
        batch_confusion = torch.bincount(
            (labels * num_classes + predictions).detach(),
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)
        confusion_matrix = (
            batch_confusion
            if confusion_matrix is None
            else confusion_matrix + batch_confusion
        )

    if total_examples == 0:
        raise ValueError("dataloader did not produce any examples")

    assert confusion_matrix is not None
    return _classification_metrics(
        total_loss.item(), total_examples, confusion_matrix.cpu()
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: Iterable[Dict[str, Tensor]],
    device: torch.device,
    use_amp: bool = False,
) -> ClassificationMetrics:

    model.eval()
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_examples = 0
    confusion_matrix = None

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels)

        batch_size = labels.shape[0]
        total_loss += loss.detach().to(torch.float64) * batch_size
        total_examples += batch_size
        predictions = logits.argmax(dim=-1)
        num_classes = logits.shape[-1]
        batch_confusion = torch.bincount(
            (labels * num_classes + predictions).detach(),
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)
        confusion_matrix = (
            batch_confusion
            if confusion_matrix is None
            else confusion_matrix + batch_confusion
        )

    if total_examples == 0:
        raise ValueError("dataloader did not produce any examples")

    assert confusion_matrix is not None
    return _classification_metrics(
        total_loss.item(), total_examples, confusion_matrix.cpu()
    )


def get_trainable_state_dict(model: nn.Module) -> Dict[str, Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    best_validation_accuracy: float,
    history: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
    include_optimizer: bool = True,
    data_generator_state: Optional[Tensor] = None,
    scaler_state: Optional[Dict[str, Any]] = None,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "format_version": 2,
        "epoch": epoch,
        "trainable_state_dict": get_trainable_state_dict(model),
        "best_validation_accuracy": best_validation_accuracy,
        "history": history,
        "metadata": metadata or {},
        "rng_state": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "data_generator_state": data_generator_state,
        "scaler_state": scaler_state,
    }
    if include_optimizer:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(checkpoint_path)


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    device: torch.device = torch.device("cpu"),
    restore_rng_state: bool = False,
) -> Dict[str, Any]:
    checkpoint = torch.load(
        Path(path),
        map_location=device,
        weights_only=False,
    )

    saved_names = set(checkpoint["trainable_state_dict"].keys())
    trainable_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if saved_names != trainable_names:
        raise ValueError(
            "checkpoint trainable parameters do not match the model: "
            f"saved={sorted(saved_names)}, current={sorted(trainable_names)}"
        )

    _, unexpected_keys = model.load_state_dict(
        checkpoint["trainable_state_dict"],
        strict=False,
    )
    if unexpected_keys:
        raise ValueError(f"unexpected checkpoint parameters: {unexpected_keys}")

    if optimizer is not None:
        if "optimizer_state_dict" not in checkpoint:
            raise ValueError("checkpoint does not contain optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if restore_rng_state and "rng_state" in checkpoint:
        rng_state = checkpoint["rng_state"]
        random.setstate(rng_state["python"])
        torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and rng_state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng_state["cuda"])

    return checkpoint


def fit(
    model: nn.Module,
    train_dataloader: Iterable[Dict[str, Tensor]],
    validation_dataloader: Iterable[Dict[str, Tensor]],
    optimizer: Optimizer,
    device: torch.device,
    epochs: int,
    checkpoint_dir: Union[str, Path] = "checkpoints",
    resume_from: Optional[Union[str, Path]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    precision: str = "fp32",
    gradient_accumulation_steps: int = 1,
) -> List[Dict[str, Any]]:
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if precision not in {"fp32", "fp16"}:
        raise ValueError(f"unsupported precision: {precision}")
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("FP16 AMP requires a CUDA device")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")

    model.to(device)
    use_amp = precision == "fp16"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    checkpoint_dir = Path(checkpoint_dir)
    history: List[Dict[str, Any]] = []
    best_validation_accuracy = float("-inf")
    start_epoch = 1

    if resume_from is not None:
        checkpoint = load_checkpoint(
            path=resume_from,
            model=model,
            optimizer=optimizer,
            device=device,
            restore_rng_state=True,
        )
        history = list(checkpoint["history"])
        best_validation_accuracy = checkpoint["best_validation_accuracy"]
        start_epoch = checkpoint["epoch"] + 1
        if metadata is None:
            metadata = checkpoint["metadata"]
        if scaler is not None and checkpoint.get("scaler_state") is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])
        data_generator = getattr(train_dataloader, "generator", None)
        if data_generator is not None and checkpoint.get("data_generator_state") is not None:
            data_generator.set_state(checkpoint["data_generator_state"])

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        train_start = time.perf_counter()
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        train_seconds = time.perf_counter() - train_start
        validation_start = time.perf_counter()
        validation_metrics = evaluate(
            model=model,
            dataloader=validation_dataloader,
            device=device,
            use_amp=use_amp,
        )
        validation_seconds = time.perf_counter() - validation_start

        epoch_result = {
            "epoch": epoch,
            "train": asdict(train_metrics),
            "validation": asdict(validation_metrics),
            "timing": {
                "train_seconds": train_seconds,
                "validation_seconds": validation_seconds,
                "epoch_seconds": time.perf_counter() - epoch_start,
            },
        }
        history.append(epoch_result)

        improved = validation_metrics.accuracy > best_validation_accuracy
        if improved:
            best_validation_accuracy = validation_metrics.accuracy

        checkpoint_arguments = {
            "model": model,
            "optimizer": optimizer,
            "epoch": epoch,
            "best_validation_accuracy": best_validation_accuracy,
            "history": history,
            "metadata": metadata,
            "data_generator_state": (
                train_dataloader.generator.get_state()
                if getattr(train_dataloader, "generator", None) is not None
                else None
            ),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
        }
        save_checkpoint(
            path=checkpoint_dir / "latest.pt",
            **checkpoint_arguments,
        )
        if improved:
            save_checkpoint(
                path=checkpoint_dir / "best.pt",
                include_optimizer=False,
                **checkpoint_arguments,
            )

        history_path = checkpoint_dir / "history.json"
        temporary_history_path = history_path.with_suffix(".json.tmp")
        temporary_history_path.write_text(
            json.dumps(history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_history_path.replace(history_path)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss {train_metrics.loss:.4f}, "
            f"accuracy {train_metrics.accuracy:.4f} | "
            f"validation loss {validation_metrics.loss:.4f}, "
            f"accuracy {validation_metrics.accuracy:.4f}"
        )

    return history
