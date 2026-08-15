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
) -> ClassificationMetrics:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("dataloader did not produce any examples")

    return ClassificationMetrics(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
        examples=total_examples,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: Iterable[Dict[str, Tensor]],
    device: torch.device,
) -> ClassificationMetrics:

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("dataloader did not produce any examples")

    return ClassificationMetrics(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
        examples=total_examples,
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
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "format_version": 1,
        "epoch": epoch,
        "trainable_state_dict": get_trainable_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_validation_accuracy": best_validation_accuracy,
        "history": history,
        "metadata": metadata or {},
    }

    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(checkpoint_path)


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    device: torch.device = torch.device("cpu"),
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
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

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
) -> List[Dict[str, Any]]:
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")

    model.to(device)
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
        )
        history = list(checkpoint["history"])
        best_validation_accuracy = checkpoint["best_validation_accuracy"]
        start_epoch = checkpoint["epoch"] + 1
        if metadata is None:
            metadata = checkpoint["metadata"]

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
        )
        validation_metrics = evaluate(
            model=model,
            dataloader=validation_dataloader,
            device=device,
        )

        epoch_result = {
            "epoch": epoch,
            "train": asdict(train_metrics),
            "validation": asdict(validation_metrics),
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
        }
        save_checkpoint(
            path=checkpoint_dir / "latest.pt",
            **checkpoint_arguments,
        )
        if improved:
            save_checkpoint(
                path=checkpoint_dir / "best.pt",
                **checkpoint_arguments,
            )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss {train_metrics.loss:.4f}, "
            f"accuracy {train_metrics.accuracy:.4f} | "
            f"validation loss {validation_metrics.loss:.4f}, "
            f"accuracy {validation_metrics.accuracy:.4f}"
        )

    return history
