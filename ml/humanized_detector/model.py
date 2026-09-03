"""Small Transformer encoder trained from random weights."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int = 192
    heads: int = 6
    layers: int = 4
    max_tokens: int = 256


class TinyTransformerClassifier(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(config.max_tokens, config.hidden_size)
        layer = nn.TransformerEncoderLayer(config.hidden_size, config.heads, config.hidden_size * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        padding = input_ids.eq(0)
        hidden = self.encoder(self.token_embedding(input_ids) + self.position_embedding(positions), src_key_padding_mask=padding)
        return self.classifier(hidden[:, 0]).squeeze(-1)
