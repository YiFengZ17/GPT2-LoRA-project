import torch
from torch import nn

from train import (
    create_optimizer,
    evaluate,
    fit,
    load_checkpoint,
    train_one_epoch,
)


class TinyClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(10, 4)
        self.classifier = nn.Linear(4, 2)

        for parameter in self.embedding.parameters():
            parameter.requires_grad = False

    def forward(self, input_ids, attention_mask):
        hidden = self.embedding(input_ids[:, 0])
        return self.classifier(hidden)


class FixedClassifier(nn.Module):

    def forward(self, input_ids, attention_mask):
        positive_score = input_ids[:, 0].float()
        negative_score = 1.0 - positive_score
        return torch.stack([negative_score, positive_score], dim=-1)


def make_batch():
    return {
        "input_ids": torch.tensor([[0, 2], [1, 3], [0, 4], [1, 5]]),
        "attention_mask": torch.ones(4, 2, dtype=torch.long),
        "labels": torch.tensor([0, 1, 0, 1]),
    }


def test_optimizer_and_training_update_only_trainable_parameters():
    torch.manual_seed(0)

    model = TinyClassifier()
    optimizer = create_optimizer(
        model,
        learning_rate=1e-2,
        weight_decay=0.0,
    )

    embedding_before = model.embedding.weight.detach().clone()
    classifier_before = model.classifier.weight.detach().clone()

    metrics = train_one_epoch(
        model=model,
        dataloader=[make_batch()],
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert torch.equal(model.embedding.weight, embedding_before)
    assert not torch.equal(model.classifier.weight, classifier_before)
    assert metrics.examples == 4
    assert metrics.loss > 0
    assert 0.0 <= metrics.accuracy <= 1.0


def test_evaluate_reports_loss_and_accuracy_without_gradients():
    model = FixedClassifier()

    metrics = evaluate(
        model=model,
        dataloader=[make_batch()],
        device=torch.device("cpu"),
    )

    assert metrics.examples == 4
    assert metrics.loss > 0
    assert metrics.accuracy == 1.0
    assert model.training is False


def test_fit_saves_best_latest_and_resumes(tmp_path):
    torch.manual_seed(0)
    model = TinyClassifier()
    optimizer = create_optimizer(
        model,
        learning_rate=1e-2,
        weight_decay=0.0,
    )

    history = fit(
        model=model,
        train_dataloader=[make_batch()],
        validation_dataloader=[make_batch()],
        optimizer=optimizer,
        device=torch.device("cpu"),
        epochs=2,
        checkpoint_dir=tmp_path,
        metadata={"mode": "test"},
    )

    latest_path = tmp_path / "latest.pt"
    best_path = tmp_path / "best.pt"
    assert latest_path.exists()
    assert best_path.exists()
    assert len(history) == 2

    trained_classifier = model.classifier.weight.detach().clone()

    torch.manual_seed(0)
    resumed_model = TinyClassifier()
    resumed_optimizer = create_optimizer(
        resumed_model,
        learning_rate=1e-2,
        weight_decay=0.0,
    )
    checkpoint = load_checkpoint(
        path=latest_path,
        model=resumed_model,
        optimizer=resumed_optimizer,
    )

    assert checkpoint["epoch"] == 2
    assert checkpoint["metadata"] == {"mode": "test"}
    assert set(checkpoint["trainable_state_dict"]) == {
        "classifier.weight",
        "classifier.bias",
    }
    assert torch.equal(resumed_model.classifier.weight, trained_classifier)

    resumed_history = fit(
        model=resumed_model,
        train_dataloader=[make_batch()],
        validation_dataloader=[make_batch()],
        optimizer=resumed_optimizer,
        device=torch.device("cpu"),
        epochs=3,
        checkpoint_dir=tmp_path,
        resume_from=latest_path,
    )

    assert len(resumed_history) == 3
    assert resumed_history[-1]["epoch"] == 3
