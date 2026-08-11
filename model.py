"""
A bare-bones GPT-2 style transformer.
"""

import math
from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

from weight_converter import state_dict_converter


@dataclass
class ModelConfig:
    d_model: int
    n_heads: int
    n_layers: int
    context_length: int
    vocab_size: int


class CausalAttention(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        assert config.d_model % config.n_heads == 0
        self.d_attention = int(config.d_model / config.n_heads)

        self.W_k = nn.Linear(config.d_model, self.d_attention * config.n_heads)
        self.W_q = nn.Linear(config.d_model, self.d_attention * config.n_heads)
        self.W_v = nn.Linear(config.d_model, self.d_attention * config.n_heads)

        self.W_o = nn.Linear(self.d_attention * config.n_heads, config.d_model)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length)).view(
                1, 1, config.context_length, config.context_length
            ),
            persistent=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        n_heads = x.shape[-1] // self.d_attention

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.reshape(B, T, n_heads, self.d_attention).transpose(1, 2)
        k = k.reshape(B, T, n_heads, self.d_attention).transpose(1, 2)
        v = v.reshape(B, T, n_heads, self.d_attention).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_attention)

        mask = self.causal_mask[:, :, :T, :T]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)

        head_output = weights @ v
        head_output = head_output.transpose(1, 2)
        head_output = head_output.reshape(B, T, D)

        output = self.W_o(head_output)
        return output


class GELU(nn.Module):
    """
    Implementation of the GELU activation function used by GPT-2.
    """

    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))  # fmt: skip


class MLP(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.d_model, 4 * config.d_model)
        self.fc2 = nn.Linear(4 * config.d_model, config.d_model)
        self.gelu = GELU()

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.fc1(x)
        hidden = self.gelu(hidden)
        output = self.fc2(hidden)
        return output


class DecoderBlock(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.mlp = MLP(config)
        self.attention = CausalAttention(config)
        self.pre_layer_norm = nn.LayerNorm(config.d_model)
        self.post_layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, x: Tensor) -> Tensor:
        x_ln = self.pre_layer_norm(x)
        x1 = self.attention(x_ln) + x

        x1_ln = self.post_layer_norm(x1)
        output = self.mlp(x1_ln) + x1
        return output


class Transformer(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.context_length, config.d_model)
        self.backbone = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.n_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # GPT-2 shares the token embedding and language-model output weights.
        self.lm_head.weight = self.embeddings.weight

        self._init_weights()

    def _init_weights(self):

        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)

    def get_hidden_states(self, x: Tensor) -> Tensor:
        B, T = x.shape
        assert T <= self.config.context_length

        x_token_embed = self.embeddings(x)

        positions = torch.arange(T, device=x.device)
        x_position_embed = self.position_embeddings(positions)

        hidden = x_position_embed + x_token_embed

        for block in self.backbone:
            hidden = block(hidden)

        hidden = self.final_layer_norm(hidden)
        return hidden

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.get_hidden_states(x)
        logits = self.lm_head(hidden)
        return logits

    @torch.no_grad()
    def generate(
        self,
        x: Tensor,
        num_new_tokens: int,
    ) -> Tensor:
        for _ in range(num_new_tokens):
            x_context = x[:, -self.config.context_length:]

            logits = self(x_context)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            x = torch.cat([x, next_token], dim=1)

        return x

    def get_loss_on_batch(self, input_ids: Tensor) -> Tensor:
        input_x = input_ids[:, :-1]
        labels = input_ids[:, 1:]

        logits = self(input_x)

        logits = logits.reshape(-1, self.config.vocab_size)
        labels = labels.reshape(-1)

        loss = F.cross_entropy(logits, labels)
        return loss

    @classmethod
    def from_pretrained(cls):
        """
        Load GPT-2 Small and convert its Hugging Face weights.
        """

        config = ModelConfig(
            d_model=768,
            n_heads=12,
            n_layers=12,
            context_length=1024,
            vocab_size=50257,
        )

        model = cls(config)

        model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
        converted_state_dict: Dict[str, Tensor] = state_dict_converter(
            model_hf.state_dict()
        )

        model.load_state_dict(converted_state_dict)
        return model
