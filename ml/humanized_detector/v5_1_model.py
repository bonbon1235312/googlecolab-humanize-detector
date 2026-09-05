"""Bundled 12M-class encoder for the V5.1 humanizer candidate."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class V51ModelConfig:
    """Architecture settings for the V5.1 shared-window encoder."""

    vocab_size: int
    hidden_size: int = 384
    heads: int = 8
    layers: int = 6
    max_tokens: int = 256
    dropout: float = 0.15

    def __post_init__(self) -> None:
        if self.vocab_size <= 1 or self.hidden_size <= 0 or self.heads <= 0 or self.layers <= 0 or self.max_tokens <= 0:
            raise ValueError("V5.1 model dimensions must be positive")
        if self.hidden_size % self.heads:
            raise ValueError("hidden_size must divide evenly by heads")


class V51FusedClassifier(nn.Module):
    """Shared text encoder, cross-window attention, FiLM, and shallow head."""

    def __init__(self, config: V51ModelConfig, feature_count: int) -> None:
        super().__init__()
        if feature_count <= 0:
            raise ValueError("feature_count must be positive")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(config.max_tokens, config.hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers, enable_nested_tensor=False)
        self.cross_window_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.cross_window_norm = nn.LayerNorm(config.hidden_size)
        self.film = nn.Sequential(
            nn.Linear(feature_count, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size * 2),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 1),
        )

    def _encode_windows(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        if input_ids.ndim != 3:
            raise ValueError("input_ids must have shape [batch, windows, tokens]")
        batch_size, window_count, token_count = input_ids.shape
        if token_count > self.config.max_tokens:
            raise ValueError("input token count exceeds model max_tokens")
        valid_windows = input_ids.ne(0).any(dim=-1)
        safe_windows = input_ids.reshape(batch_size * window_count, token_count).clone()
        invalid = ~valid_windows.reshape(-1)
        if invalid.any():
            safe_windows[invalid, 0] = 1
        padding = safe_windows.eq(0)
        positions = torch.arange(token_count, device=input_ids.device).unsqueeze(0)
        encoded = self.encoder(
            self.token_embedding(safe_windows) + self.position_embedding(positions),
            src_key_padding_mask=padding,
        )
        valid_tokens = (~padding).unsqueeze(-1).to(encoded.dtype)
        windows = (encoded * valid_tokens).sum(dim=1) / valid_tokens.sum(dim=1).clamp_min(1)
        windows = windows.reshape(batch_size, window_count, -1)
        safe_valid_windows = valid_windows.clone()
        empty_examples = ~safe_valid_windows.any(dim=1)
        if empty_examples.any():
            safe_valid_windows[empty_examples, 0] = True
        return windows, safe_valid_windows

    def forward(self, input_ids: Tensor, features: Tensor) -> Tensor:
        windows, valid_windows = self._encode_windows(input_ids)
        contextual, _ = self.cross_window_attention(
            windows,
            windows,
            windows,
            key_padding_mask=~valid_windows,
            need_weights=False,
        )
        windows = self.cross_window_norm(windows + contextual)
        weights = valid_windows.unsqueeze(-1).to(windows.dtype)
        text_embedding = (windows * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        gamma, beta = self.film(features).chunk(2, dim=-1)
        fused = (1 + torch.tanh(gamma)) * text_embedding + beta
        return self.head(fused).squeeze(-1)
