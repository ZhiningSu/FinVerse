from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tokenizers import DualVQMarketTokenizer


class ForecastOutputMixin:
    def _loss_output(self, price_pred, price_target):
        loss = torch.tensor(0.0, device=price_pred.device)
        if price_target is not None and price_target.numel() > 0:
            loss = F.mse_loss(price_pred, price_target)
        return {"price_pred": price_pred, "loss": loss}


class LSTMForecaster(nn.Module, ForecastOutputMixin):
    def __init__(self, price_dim: int = 6, hidden_dim: int = 256, output_dim: int = 6, num_steps: int = 30):
        super().__init__()
        self.num_steps = num_steps
        self.output_dim = output_dim
        self.encoder = nn.LSTM(price_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_steps * output_dim),
        )

    def forward(self, price_seq, news_feat=None, macro_feat=None, edge_index=None, edge_weight=None, action=None, price_target=None):
        _, (hidden, _) = self.encoder(price_seq)
        h = hidden[-1]
        pred = self.head(h).view(price_seq.size(0), self.num_steps, self.output_dim)
        return self._loss_output(pred, price_target)


class GRUForecaster(nn.Module, ForecastOutputMixin):
    def __init__(self, price_dim: int = 6, hidden_dim: int = 256, output_dim: int = 6, num_steps: int = 30):
        super().__init__()
        self.num_steps = num_steps
        self.output_dim = output_dim
        self.encoder = nn.GRU(price_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_steps * output_dim),
        )

    def forward(self, price_seq, news_feat=None, macro_feat=None, edge_index=None, edge_weight=None, action=None, price_target=None):
        _, hidden = self.encoder(price_seq)
        h = hidden[-1]
        pred = self.head(h).view(price_seq.size(0), self.num_steps, self.output_dim)
        return self._loss_output(pred, price_target)


class DLinearForecaster(nn.Module, ForecastOutputMixin):
    def __init__(self, price_dim: int = 6, lookback: int = 30, output_dim: int = 6, num_steps: int = 30, hidden_dim: int = 256):
        super().__init__()
        self.output_dim = output_dim
        self.num_steps = num_steps
        self.linear = nn.Linear(lookback, num_steps)
        self.channel_proj = nn.Linear(price_dim, output_dim)

    def forward(self, price_seq, news_feat=None, macro_feat=None, edge_index=None, edge_weight=None, action=None, price_target=None):
        pred = self.linear(price_seq.transpose(1, 2)).transpose(1, 2)
        pred = self.channel_proj(pred)
        return self._loss_output(pred, price_target)


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 512):
        super().__init__()
        positions = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_len, hidden_dim)
        pe[:, 0::2] = torch.sin(positions * div)
        pe[:, 1::2] = torch.cos(positions * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerForecaster(nn.Module, ForecastOutputMixin):
    def __init__(self, price_dim: int = 6, hidden_dim: int = 256, output_dim: int = 6, num_steps: int = 30):
        super().__init__()
        self.output_dim = output_dim
        self.num_steps = num_steps
        self.input_proj = nn.Linear(price_dim, hidden_dim)
        self.pos = PositionalEncoding(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_steps * output_dim),
        )

    def forward(self, price_seq, news_feat=None, macro_feat=None, edge_index=None, edge_weight=None, action=None, price_target=None):
        x = self.pos(self.input_proj(price_seq))
        h = self.encoder(x).mean(dim=1)
        pred = self.head(h).view(price_seq.size(0), self.num_steps, self.output_dim)
        return self._loss_output(pred, price_target)


class PatchTSTForecaster(nn.Module, ForecastOutputMixin):
    def __init__(
        self,
        price_dim: int = 6,
        hidden_dim: int = 256,
        output_dim: int = 6,
        num_steps: int = 30,
        patch_len: int = 5,
        stride: int = 3,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_steps = num_steps
        self.patch_len = patch_len
        self.stride = stride
        self.patch_proj = nn.Linear(price_dim * patch_len, hidden_dim)
        self.pos = PositionalEncoding(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_steps * output_dim))

    def forward(self, price_seq, news_feat=None, macro_feat=None, edge_index=None, edge_weight=None, action=None, price_target=None):
        patches = price_seq.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 3, 2).reshape(price_seq.size(0), patches.size(1), -1)
        x = self.pos(self.patch_proj(patches))
        h = self.encoder(x).mean(dim=1)
        pred = self.head(h).view(price_seq.size(0), self.num_steps, self.output_dim)
        return self._loss_output(pred, price_target)


class ITransformerForecaster(nn.Module, ForecastOutputMixin):
    def __init__(self, price_dim: int = 6, lookback: int = 30, hidden_dim: int = 256, output_dim: int = 6, num_steps: int = 30):
        super().__init__()
        self.price_dim = price_dim
        self.output_dim = output_dim
        self.num_steps = num_steps
        self.value_proj = nn.Linear(lookback, hidden_dim)
        self.var_embed = nn.Parameter(torch.randn(1, price_dim, hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_steps))
        self.channel_proj = nn.Linear(price_dim, output_dim)

    def forward(self, price_seq, news_feat=None, macro_feat=None, edge_index=None, edge_weight=None, action=None, price_target=None):
        x = price_seq.transpose(1, 2)
        h = self.value_proj(x) + self.var_embed[:, : x.size(1)]
        h = self.encoder(h)
        pred = self.head(h).transpose(1, 2)
        pred = self.channel_proj(pred)
        return self._loss_output(pred, price_target)


class KronosMiniForecaster(nn.Module, ForecastOutputMixin):
    """
    Lightweight Kronos-style baseline.

    It discretizes recent K-line/market patterns with dual VQ codes, then uses a
    small token-level Transformer to predict the future price trajectory. This is
    not the official Kronos checkpoint; it is a reproducible mini baseline that
    keeps the same tokenization spirit under the current project data format.
    """

    def __init__(self, price_dim: int = 6, hidden_dim: int = 256, output_dim: int = 6, num_steps: int = 30):
        super().__init__()
        self.output_dim = output_dim
        self.num_steps = num_steps
        self.tokenizer = DualVQMarketTokenizer(
            price_dim=price_dim,
            hidden_dim=hidden_dim,
            token_dim=max(32, hidden_dim // 4),
            num_temporal_codes=128,
            num_cross_codes=128,
        )
        self.price_proj = nn.Linear(price_dim, hidden_dim)
        self.pos = PositionalEncoding(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.token_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_steps * output_dim))

    def forward(self, price_seq, news_feat=None, macro_feat=None, edge_index=None, edge_weight=None, action=None, price_target=None):
        tokenized = self.tokenizer(price_seq)
        x = self.pos(self.price_proj(price_seq))
        seq_h = self.encoder(x).mean(dim=1)
        token_h = torch.cat([tokenized["temporal_h"], tokenized["cross_h"]], dim=-1)
        h = seq_h + self.token_fusion(token_h)
        pred = self.head(h).view(price_seq.size(0), self.num_steps, self.output_dim)
        out = self._loss_output(pred, price_target)
        out.update(
            {
                "vq_loss": tokenized["vq_loss"],
                "temporal_token_ids": tokenized["temporal_token_ids"],
                "cross_token_ids": tokenized["cross_token_ids"],
            }
        )
        return out
