from lora import LoRALinear, inject_lora_into_qv
from model import ModelConfig, Transformer
import torch
from torch import nn, Tensor
from torch.nn import functional as F

# initialization test
def test_initial_output_matches_base():
    torch.manual_seed(0)

    base = nn.Linear(4, 3)
    x = torch.randn(2, 5, 4)

    expected = base(x)

    lora = LoRALinear(
        base_linear=base,
        rank=2,
        alpha=4,
        dropout=0.0,
    )
    actual = lora(x)

    assert torch.allclose(actual, expected)

# test Frozen 
def test_only_lora_parameters_are_trainable():
    base = nn.Linear(4, 3)

    lora = LoRALinear(
        base_linear=base,
        rank=2,
        alpha=4,
        dropout=0.0,
    )


    assert all(
        not parameter.requires_grad
        for parameter in lora.base.parameters()
    )


    assert lora.lora_A.weight.requires_grad
    assert lora.lora_B.weight.requires_grad

    trainable_parameters = sum(
        parameter.numel()
        for parameter in lora.parameters()
        if parameter.requires_grad
    )

    expected_trainable_parameters = 2 * (4 + 3)

    assert trainable_parameters == expected_trainable_parameters


# gradient flow test
def test_gradient_flow_on_first_backward():
    torch.manual_seed(0)

    base = nn.Linear(4, 3)
    lora = LoRALinear(
        base_linear=base,
        rank=2,
        alpha=4,
        dropout=0.0,
    )

    x = torch.randn(2, 5, 4)

    output = lora(x)
    loss = output.sum()
    loss.backward()

    assert lora.base.weight.grad is None
    assert lora.base.bias.grad is None

  
    assert lora.lora_A.weight.grad is not None
    assert lora.lora_B.weight.grad is not None

   
    assert torch.count_nonzero(lora.lora_A.weight.grad) == 0

 
    assert torch.count_nonzero(lora.lora_B.weight.grad) > 0

# test injection
def test_lora_injection_preserves_initial_model_output():
    torch.manual_seed(0)

    config = ModelConfig(
        d_model=24,
        n_heads=3,
        n_layers=2,
        context_length=16,
        vocab_size=100,
    )
    model = Transformer(config)
    model.eval()

    input_ids = torch.randint(0, 100, (2, 8))

    with torch.no_grad():
        expected = model(input_ids)

    inject_lora_into_qv(
        model=model,
        rank=2,
        alpha=4,
        dropout=0.0,
    )

    with torch.no_grad():
        actual = model(input_ids)


    assert torch.allclose(actual, expected)

    lora_modules = [
        module
        for module in model.modules()
        if isinstance(module, LoRALinear)
    ]

    assert len(lora_modules) == 4

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    # Because rank = 2, 2 layers means 2 Q and 2 V, which are 4 LoRA linear
    expected_trainable_parameters = 4 * 2 * (24 + 24)

    assert trainable_parameters == expected_trainable_parameters