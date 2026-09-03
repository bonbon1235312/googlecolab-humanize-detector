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
    dropout: float = 0.15


class TinyTransformerClassifier(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(config.max_tokens, config.hidden_size)
        layer = nn.TransformerEncoderLayer(
            config.hidden_size,
            config.heads,
            config.hidden_size * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        padding = input_ids.eq(0)
        hidden = self.encoder(self.token_embedding(input_ids) + self.position_embedding(positions), src_key_padding_mask=padding)
        return hidden[:, 0]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(input_ids)).squeeze(-1)


class MultiWindowClassifier(nn.Module):
    """Apply one shared tiny encoder to beginning, middle, and end windows."""

    def __init__(self, config: ModelConfig, pooling: str = "mean") -> None:
        super().__init__()
        if pooling not in {"mean", "attention"}:
            raise ValueError("pooling must be 'mean' or 'attention'")
        self.pooling = pooling
        self.text_encoder = TinyTransformerClassifier(config)
        self.attention = nn.Linear(config.hidden_size, 1) if pooling == "attention" else None
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 3:
            raise ValueError("input_ids must have shape [batch, windows, tokens]")
        batch_size, window_count, token_count = input_ids.shape
        valid_windows = input_ids.ne(0).any(dim=-1)
        flat = input_ids.reshape(batch_size * window_count, token_count).clone()
        invalid = ~valid_windows.reshape(-1)
        if invalid.any():
            flat[invalid, 0] = 1
        embeddings = self.text_encoder.encode(flat).reshape(batch_size, window_count, -1)
        if self.pooling == "mean":
            weights = valid_windows.unsqueeze(-1).to(embeddings.dtype)
            pooled = (embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        else:
            assert self.attention is not None
            scores = self.attention(embeddings).squeeze(-1).masked_fill(~valid_windows, float("-inf"))
            pooled = (torch.softmax(scores, dim=1).unsqueeze(-1) * embeddings).sum(dim=1)
        return self.classifier(pooled).squeeze(-1)


class StructuralOnlyClassifier(nn.Module):
    """Small MLP baseline for cached structural features."""

    def __init__(self, feature_count: int, hidden_size: int = 48, dropout: float = 0.15) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(feature_count, hidden_size), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(-1)
