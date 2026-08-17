"""
Utilities for loading and tokenizing the local SST-5 dataset.
"""

from pathlib import Path
from typing import Dict, List

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase


DATA_PATH = Path(__file__).resolve().parent / "data" / "sst5"
DATASET_ID = "SetFit/sst5"


def load_sst5() -> DatasetDict:
    """
    Load the locally saved SST-5 train, validation, and test splits.
    """

    if DATA_PATH.exists():
        return load_from_disk(str(DATA_PATH))

    # A fresh clone intentionally does not contain the dataset. Download it
    # once, then persist the exact snapshot used by subsequent runs.
    dataset = load_dataset(DATASET_ID)
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"expected DatasetDict from {DATASET_ID}, got {type(dataset)}")
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(DATA_PATH))
    return dataset


def create_tokenizer() -> PreTrainedTokenizerBase:
    """
    Load the GPT-2 tokenizer (downloading it on first use) and configure
    right-side padding.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2",
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return tokenizer


class SST5Collator:
    """
    Tokenize and dynamically pad a list of SST-5 examples into one batch.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
    ):
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: List[Dict]) -> Dict[str, Tensor]:
        if not examples:
            raise ValueError("examples must contain at least one sample")

        texts = [example["text"] for example in examples]
        labels = torch.tensor(
            [example["label"] for example in examples],
            dtype=torch.long,
        )

        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": labels,
        }


def create_dataloaders(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 8,
    max_length: int = 128,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool = False,
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test DataLoaders for SST-5.
    """

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}")

    required_splits = {"train", "validation", "test"}
    missing_splits = required_splits - set(dataset.keys())
    if missing_splits:
        raise ValueError(f"dataset is missing splits: {sorted(missing_splits)}")

    collator = SST5Collator(
        tokenizer=tokenizer,
        max_length=max_length,
    )
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    return {
        "train": DataLoader(
            dataset["train"],
            batch_size=batch_size,
            shuffle=True,
            generator=train_generator,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        ),
        "validation": DataLoader(
            dataset["validation"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        ),
        "test": DataLoader(
            dataset["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        ),
    }
