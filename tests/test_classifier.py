from types import SimpleNamespace

import torch
from torch import nn

from classifier import SentimentClassifier
from model import ModelConfig, Transformer


class StubTransformer(nn.Module):

    def __init__(self):
        super().__init__()

        self.config = SimpleNamespace(d_model=3)

    def get_hidden_states(
        self,
        input_ids,
        attention_mask=None,
    ):
        return torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [9.0, 0.0, 0.0],
                ],
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 9.0, 0.0],
                    [0.0, 9.0, 0.0],
                ],
            ]
        )


def test_classifier_output_shape():
    config = ModelConfig(
        d_model=24,
        n_heads=3,
        n_layers=2,
        context_length=16,
        vocab_size=100,
    )
    transformer = Transformer(config)
    model = SentimentClassifier(transformer)

    input_ids = torch.randint(0, 100, (2, 5))
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
        ]
    )

    logits = model(input_ids, attention_mask)

    assert logits.shape == (2, 5)


def test_classifier_uses_last_real_token():
    model = SentimentClassifier(
        StubTransformer(),
        num_classes=3,
    )

    with torch.no_grad():
        model.classifier.weight.copy_(torch.eye(3))
        model.classifier.bias.zero_()

    input_ids = torch.zeros(2, 4, dtype=torch.long)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ]
    )

    logits = model(input_ids, attention_mask)
    expected = torch.tensor(
        [
            [3.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )

    assert torch.equal(logits, expected)
