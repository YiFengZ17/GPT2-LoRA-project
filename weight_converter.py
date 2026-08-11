"""
Convert Hugging Face GPT-2 weights to the parameter names used by model.py.
"""

from typing import Dict

import torch
from torch import Tensor


def switch_backbone_key(key: str, value: Tensor) -> Dict[str, Tensor]:

    suffix_key_switcher = {
        "ln_1.weight": "pre_layer_norm.weight",
        "ln_1.bias": "pre_layer_norm.bias",
        "ln_2.weight": "post_layer_norm.weight",
        "ln_2.bias": "post_layer_norm.bias",
        "mlp.c_fc.weight": "mlp.fc1.weight",
        "mlp.c_fc.bias": "mlp.fc1.bias",
        "mlp.c_proj.weight": "mlp.fc2.weight",
        "mlp.c_proj.bias": "mlp.fc2.bias",
        "attn.c_proj.weight": "attention.W_o.weight",
        "attn.c_proj.bias": "attention.W_o.bias",
    }

    layer_num: str = key.split(".")[2]
    suffix: str = ".".join(key.split(".")[3:])

    if "c_attn.weight" in key:
        _, embedding_dim = value.shape
        W_q, W_k, W_v = value.split(embedding_dim, dim=0)

        return {
            f"backbone.{layer_num}.attention.W_q.weight": W_q,
            f"backbone.{layer_num}.attention.W_k.weight": W_k,
            f"backbone.{layer_num}.attention.W_v.weight": W_v,
        }

    elif "c_attn.bias" in key:
        embedding_dim_times_3, = value.shape
        assert embedding_dim_times_3 % 3 == 0
        embedding_dim = embedding_dim_times_3 // 3

        b_q, b_k, b_v = value.split(embedding_dim, dim=0)

        return {
            f"backbone.{layer_num}.attention.W_q.bias": b_q,
            f"backbone.{layer_num}.attention.W_k.bias": b_k,
            f"backbone.{layer_num}.attention.W_v.bias": b_v,
        }

    else:
        return {
            f"backbone.{layer_num}.{suffix_key_switcher[suffix]}": value
        }


@torch.no_grad()
def state_dict_converter(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """
    Convert a Hugging Face GPT-2 state dict to our Transformer state dict.
    """

    key_switcher = {
        "transformer.wte.weight": "embeddings.weight",
        "transformer.wpe.weight": "position_embeddings.weight",
        "transformer.ln_f.weight": "final_layer_norm.weight",
        "transformer.ln_f.bias": "final_layer_norm.bias",
        "lm_head.weight": "lm_head.weight",
    }

    new_state_dict = {}

    transposed = [
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    ]

    for key, value in state_dict.items():
        if any(key.endswith(weight_name) for weight_name in transposed):
            value = value.t()

        if ".h." in key:
            new_state_dict = new_state_dict | switch_backbone_key(
                key,
                value.clone().detach(),
            )
        else:
            new_state_dict[key_switcher[key]] = value.clone().detach()

    return new_state_dict

