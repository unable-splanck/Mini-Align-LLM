from typing import Optional

import torch
from torch import nn
from transformers import AutoConfig


class ScalarValueHead(nn.Module):
    """A minimal scalar value head for PPO-style experiments."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.value_head(hidden_states).squeeze(-1)


def build_value_head(model_name_or_path: str) -> ScalarValueHead:
    config = AutoConfig.from_pretrained(model_name_or_path)
    hidden_size: Optional[int] = getattr(config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(config, "n_embd", None)
    if hidden_size is None:
        raise ValueError(f"Cannot infer hidden size for model: {model_name_or_path}")
    return ScalarValueHead(hidden_size)

