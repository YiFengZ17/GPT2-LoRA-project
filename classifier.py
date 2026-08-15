import torch
from torch import Tensor, nn

from model import Transformer

class SentimentClassifier(nn.Module):

    def __init__(
        self,
        transformer: Transformer,
        # classify  5 emotion classes
        num_classes: int = 5,
    ):
        super().__init__()

        self.transformer = transformer
        self.classifier = nn.Linear(
            transformer.config.d_model,
            num_classes,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        hidden = self.transformer.get_hidden_states(
            input_ids,
            attention_mask=attention_mask,
        )

        lengths = attention_mask.sum(dim=1)

        if torch.any(lengths == 0):
            raise ValueError("Every input must contain at least one real token")

        last_token_indices = lengths.to(torch.long) - 1
        batch_indices = torch.arange(
            input_ids.shape[0],
            device=input_ids.device,
        )

        sentence_hidden = hidden[
            batch_indices,
            last_token_indices,
        ]

        logits = self.classifier(sentence_hidden)
        return logits