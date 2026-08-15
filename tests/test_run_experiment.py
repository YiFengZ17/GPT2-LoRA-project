import torch

from lora import LoRALinear
from model import ModelConfig, Transformer
from run_experiment import configure_model, parse_args, select_device


def make_transformer():
    return Transformer(
        ModelConfig(
            d_model=24,
            n_heads=3,
            n_layers=2,
            context_length=16,
            vocab_size=100,
        )
    )


def test_debug_arguments_choose_safe_defaults():
    args = parse_args(["--mode", "lora", "--debug"])

    assert args.rank == 8
    assert args.alpha == 16
    assert args.epochs == 1
    assert args.learning_rate == 2e-4
    assert args.max_train_samples == 100
    assert args.max_validation_samples == 100
    assert args.max_test_samples == 100
    assert str(args.output_dir) == "runs/lora-r8-debug"


def test_configure_model_supports_all_training_modes():
    frozen_model = configure_model(
        make_transformer(),
        mode="frozen",
        rank=2,
        alpha=4,
        lora_dropout=0.0,
    )
    assert all(
        not parameter.requires_grad
        for parameter in frozen_model.transformer.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in frozen_model.classifier.parameters()
    )

    lora_model = configure_model(
        make_transformer(),
        mode="lora",
        rank=2,
        alpha=4,
        lora_dropout=0.0,
    )
    lora_modules = [
        module
        for module in lora_model.modules()
        if isinstance(module, LoRALinear)
    ]
    assert len(lora_modules) == 4
    assert all(
        "lora_A" in name or "lora_B" in name or name.startswith("classifier.")
        for name, parameter in lora_model.named_parameters()
        if parameter.requires_grad
    )

    full_model = configure_model(
        make_transformer(),
        mode="full",
        rank=2,
        alpha=4,
        lora_dropout=0.0,
    )
    assert all(parameter.requires_grad for parameter in full_model.parameters())


def test_select_device_rejects_unavailable_cuda():
    if not torch.cuda.is_available():
        try:
            select_device("cuda")
        except RuntimeError:
            pass
        else:
            raise AssertionError("select_device should reject unavailable CUDA")
