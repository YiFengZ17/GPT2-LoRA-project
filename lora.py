import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F

class LoRALinear(nn.Module):
    def __init__(
            self,
            base_linear: nn.Linear,
            rank: int,
            alpha: float,
            dropout: float,
    ):
        super().__init__()

        if rank <= 0:
            raise ValueError(f"rank must be positive,got {rank}")
        
        self.base = base_linear

        # define A matrix
        self.lora_A = nn.Linear(
            in_features=base_linear.in_features,
            out_features=rank,
            bias=False,
        )

        # define B matrix
        self.lora_B = nn.Linear(
            in_features=rank,
            out_features=base_linear.out_features,
            bias=False,
        )

        # define scaling parameter
        self.scaling = alpha / rank

        # define dropout layer
        self.dropout = nn.Dropout(dropout)

        # Froze the base matrix
        for parameter in self.base.parameters():
            parameter.requires_grad = False

        # initialize A randomly, B to all-zero
        nn.init.kaiming_uniform_(
            self.lora_A.weight,
            a=math.sqrt(5),
        )
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x:Tensor)->Tensor:
        base_output = self.base(x)

        lora_output = self.lora_B(self.lora_A(self.dropout(x)))
        return base_output + self.scaling * lora_output


def inject_lora_into_qv(
        model:nn.Module,
        rank:int,
        alpha:float,
        dropout:float,
    )->nn.Module:
    # Froze GPT2 firstly
    for parameter in model.parameters():
        parameter.requires_grad = False
    # change Q,V attention in every decoder block
    for block in model.backbone:
        attention = block.attention
        attention.W_q = LoRALinear(
            base_linear=attention.W_q,
            rank = rank,
            alpha=alpha,
            dropout=dropout,
            )

        attention.W_v = LoRALinear(
            base_linear=attention.W_v,
            rank = rank,
            alpha=alpha,
            dropout=dropout,
            )
    return model