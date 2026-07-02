from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tokenizers import DualVQMarketTokenizer


def _align_feature_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    if x.size(-1) < target_dim:
        pad = torch.zeros(*x.shape[:-1], target_dim - x.size(-1), device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=-1)
    if x.size(-1) > target_dim:
        return x[..., :target_dim]
    return x


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, max(hidden_dim // 2, 1)),
            nn.Tanh(),
            nn.Linear(max(hidden_dim // 2, 1), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(weights * x, dim=1)


class ForecastOutputMixin:
    def _init_forecast_common(
        self,
        price_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_steps: int,
        news_dim: int = 384,
        macro_dim: int = 8,
        action_dim: int = 8,
    ) -> None:
        self.price_dim = price_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_steps = num_steps
        self.news_dim = news_dim
        self.macro_dim = macro_dim
        self.action_dim = action_dim
        self.input_norm = nn.LayerNorm(price_dim)

        self.news_encoder = nn.Sequential(
            nn.Linear(news_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.news_pool = AttentionPool(hidden_dim)
        self.macro_encoder = nn.Sequential(
            nn.Linear(macro_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.macro_pool = AttentionPool(hidden_dim)
        self.graph_encoder = nn.Sequential(
            nn.Linear(price_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.modality_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.decoder_cell = nn.GRUCell(hidden_dim + output_dim, hidden_dim)
        self.step_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.return_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 1)),
            nn.GELU(),
            nn.Linear(max(hidden_dim // 2, 1), output_dim),
        )
        self.regime_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, max(hidden_dim // 4, 1)),
            nn.GELU(),
            nn.Linear(max(hidden_dim // 4, 1), 4),
        )

    def _normalize_price(self, price_seq: torch.Tensor) -> torch.Tensor:
        price_seq = _align_feature_dim(price_seq, self.price_dim)
        mean = price_seq.mean(dim=1, keepdim=True)
        std = price_seq.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
        return self.input_norm((price_seq - mean) / std)

    def _encode_optional_sequence(
        self,
        seq: torch.Tensor | None,
        encoder: nn.Module,
        pool: nn.Module,
        feature_dim: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if seq is None:
            return torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        seq = seq.to(device=device, dtype=dtype)
        if seq.dim() == 2:
            seq = seq.unsqueeze(1)
        seq = _align_feature_dim(seq, feature_dim)
        return pool(encoder(seq))

    def _edges_for_batch(self, edge_index, edge_weight, batch_idx: int, num_nodes: int, device: torch.device):
        if edge_index is None or edge_index.numel() == 0:
            base_edges = torch.empty(2, 0, dtype=torch.long, device=device)
            weights = torch.empty(0, dtype=torch.float32, device=device)
        else:
            if edge_index.dim() == 3:
                base_edges = edge_index[batch_idx].to(device=device, dtype=torch.long)
            else:
                base_edges = edge_index.to(device=device, dtype=torch.long)
            if edge_weight is None or edge_weight.numel() == 0:
                weights = torch.ones(base_edges.size(-1), device=device)
            elif edge_weight.dim() == 2:
                weights = edge_weight[batch_idx].to(device=device, dtype=torch.float32)
            else:
                weights = edge_weight.to(device=device, dtype=torch.float32)
        base_edges = base_edges.clamp(0, num_nodes - 1)
        reverse_edges = base_edges.flip(0)
        self_edges = torch.arange(num_nodes, device=device, dtype=torch.long).repeat(2, 1)
        all_edges = torch.cat([base_edges, reverse_edges, self_edges], dim=1)
        all_weights = torch.cat(
            [weights, weights, torch.ones(num_nodes, device=device, dtype=torch.float32)],
            dim=0,
        )
        return all_edges, all_weights

    def _graph_context(self, price_seq: torch.Tensor, edge_index=None, edge_weight=None) -> torch.Tensor:
        latest = _align_feature_dim(price_seq[:, -1], self.price_dim)
        batch_size, num_nodes = latest.shape
        graph_mean = torch.zeros_like(latest)
        for b in range(batch_size):
            edges, weights = self._edges_for_batch(edge_index, edge_weight, b, num_nodes, latest.device)
            src, dst = edges[0], edges[1]
            weights = weights.to(device=latest.device, dtype=latest.dtype)
            numer = torch.zeros(num_nodes, device=latest.device, dtype=latest.dtype)
            denom = torch.zeros(num_nodes, device=latest.device, dtype=latest.dtype)
            numer.index_add_(0, dst, latest[b, src] * weights)
            denom.index_add_(0, dst, weights.abs().clamp_min(1e-6))
            graph_mean[b] = numer / denom.clamp_min(1e-6)
        graph_delta = latest - graph_mean
        return self.graph_encoder(torch.cat([latest, graph_mean, graph_delta], dim=-1))

    def _action_context(
        self,
        action: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if action is None:
            action = torch.zeros(batch_size, self.action_dim, device=device, dtype=dtype)
        else:
            action = action.to(device=device, dtype=dtype)
            if action.dim() == 3:
                action = action[:, 0]
            action = _align_feature_dim(action, self.action_dim)
        return self.action_encoder(action)

    def _compose_context(
        self,
        base_h: torch.Tensor,
        price_seq_norm: torch.Tensor,
        news_feat=None,
        macro_feat=None,
        edge_index=None,
        edge_weight=None,
        action=None,
    ) -> torch.Tensor:
        batch_size = base_h.size(0)
        device = base_h.device
        dtype = base_h.dtype
        news_h = self._encode_optional_sequence(
            news_feat, self.news_encoder, self.news_pool, self.news_dim, batch_size, device, dtype
        )
        macro_h = self._encode_optional_sequence(
            macro_feat, self.macro_encoder, self.macro_pool, self.macro_dim, batch_size, device, dtype
        )
        graph_h = self._graph_context(price_seq_norm, edge_index, edge_weight)
        action_h = self._action_context(action, batch_size, device, dtype)
        fused = self.modality_fusion(torch.cat([base_h, news_h, macro_h, graph_h, action_h], dim=-1))
        return self.context_norm(base_h + fused)

    def _decode_autoregressive(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = context
        prev_pred = torch.zeros(context.size(0), self.output_dim, device=context.device, dtype=context.dtype)
        preds = []
        states = []
        for _ in range(self.num_steps):
            state = self.decoder_cell(torch.cat([context, prev_pred], dim=-1), state)
            pred = self.step_head(state)
            prev_pred = pred
            preds.append(pred)
            states.append(state)
        return torch.stack(preds, dim=1), torch.stack(states, dim=1)

    def _loss_output(self, price_pred, states, price_target):
        return_pred = self.return_head(states[:, -1])
        regime_logits = self.regime_head(states)
        price_loss = price_pred.new_tensor(0.0)
        return_loss = price_pred.new_tensor(0.0)
        if price_target is not None and price_target.numel() > 0:
            target = price_target.to(device=price_pred.device, dtype=price_pred.dtype)
            if target.dim() == 2:
                target = target.unsqueeze(-1)
            target = _align_feature_dim(target, self.output_dim)
            steps = min(target.size(1), price_pred.size(1))
            target = target[:, :steps]
            pred = price_pred[:, :steps]
            price_loss = F.mse_loss(pred, target)
            return_loss = F.mse_loss(return_pred, target.mean(dim=1))
        loss = price_loss + 0.1 * return_loss
        return {
            "price_pred": price_pred,
            "return_pred": return_pred,
            "regime_logits": regime_logits,
            "loss": loss,
            "price_loss": price_loss,
            "return_loss": return_loss,
        }

    def _forecast_from_context(
        self,
        base_h: torch.Tensor,
        price_seq_norm: torch.Tensor,
        news_feat=None,
        macro_feat=None,
        edge_index=None,
        edge_weight=None,
        action=None,
        price_target=None,
    ):
        context = self._compose_context(base_h, price_seq_norm, news_feat, macro_feat, edge_index, edge_weight, action)
        price_pred, states = self._decode_autoregressive(context)
        return self._loss_output(price_pred, states, price_target)


class LSTMForecaster(nn.Module, ForecastOutputMixin):
    def __init__(self, price_dim: int = 6, hidden_dim: int = 256, output_dim: int = 6, num_steps: int = 30):
        super().__init__()
        self._init_forecast_common(price_dim, hidden_dim, output_dim, num_steps)
        self.encoder = nn.LSTM(price_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.1)

    def forward(self, price_seq, news_feat=None, macro_feat=None, edge_index=None, edge_weight=None, action=None, price_target=None):
        x = self._normalize_price(price_seq)
        _, (hidden, _) = self.encoder(x)
        return self._forecast_from_context(hidden[-1], x, news_feat, macro_feat, edge_index, edge_weight, action, price_target)


class GRUForecaster(nn.Module, ForecastOutputMixin):
    def __init__(self, price_dim: int = 6, hidden_dim: int = 256, output_dim: int = 6, num_steps: int = 30):
        super().__init__()
        self._init_forecast_common(price_dim, hidden_dim, output_dim, num_steps)
        self.encoder = nn.GRU(price_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.1)

    def forward(self, price_seq, news_feat=None, macro_feat=None, edge_index=None, edge_weight=None, action=None, price_target=None):
        x = self._normalize_price(price_seq)
        _, hidden = self.encoder(x)
        return self._forecast_from_context(hidden[-1], x, news_feat, macro_feat, edge_index, edge_weight, action, price_target)


class DLinearForecaster(nn.Module, ForecastOutputMixin):
    def __init__(self, price_dim: int = 6, lookback: int = 30, output_dim: int = 6, num_steps: int = 30, hidden_dim: int = 256):
        super().__init__()
        self._init_forecast_common(price_dim, hidden_dim, output_dim, num_steps)
        self.linear = nn.Linear(lookback, num_steps)
        self.channel_proj = nn.Linear(price_dim, output_dim)
        self.context_proj = nn.Sequential(
            nn.Linear(num_steps * output_dim + price_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, price_seq, news_feat=None, macro_feat=None, edge_index=None, edge_weight=None, action=None, price_target=None):
        x = self._normalize_price(price_seq)
        direct = self.linear(x.transpose(1, 2)).transpose(1, 2)
        direct = self.channel_proj(direct)
        base_h = self.context_proj(torch.cat([x[:, -1], direct.flatten(1)], dim=-1))
        return self._forecast_from_context(base_h, x, news_feat, macro_feat, edge_index, edge_weight, action, price_target)


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
        self._init_forecast_common(price_dim, hidden_dim, output_dim, num_steps)
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
        self.pool = AttentionPool(hidden_dim)

    def forward(self, price_seq, news_feat=None, macro_feat=None, edge_index=None, edge_weight=None, action=None, price_target=None):
        x = self._normalize_price(price_seq)
        h = self.pool(self.encoder(self.pos(self.input_proj(x))))
        return self._forecast_from_context(h, x, news_feat, macro_feat, edge_index, edge_weight, action, price_target)


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
        self._init_forecast_common(price_dim, hidden_dim, output_dim, num_steps)
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
        self.pool = AttentionPool(hidden_dim)

    def forward(self, price_seq, news_feat=None, macro_feat=None, edge_index=None, edge_weight=None, action=None, price_target=None):
        x = self._normalize_price(price_seq)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 3, 2).reshape(x.size(0), patches.size(1), -1)
        h = self.pool(self.encoder(self.pos(self.patch_proj(patches))))
        return self._forecast_from_context(h, x, news_feat, macro_feat, edge_index, edge_weight, action, price_target)


class ITransformerForecaster(nn.Module, ForecastOutputMixin):
    def __init__(self, price_dim: int = 6, lookback: int = 30, hidden_dim: int = 256, output_dim: int = 6, num_steps: int = 30):
        super().__init__()
        self._init_forecast_common(price_dim, hidden_dim, output_dim, num_steps)
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
        self.pool = AttentionPool(hidden_dim)

    def forward(self, price_seq, news_feat=None, macro_feat=None, edge_index=None, edge_weight=None, action=None, price_target=None):
        x = self._normalize_price(price_seq)
        h = self.value_proj(x.transpose(1, 2)) + self.var_embed[:, : x.size(2)]
        h = self.pool(self.encoder(h))
        return self._forecast_from_context(h, x, news_feat, macro_feat, edge_index, edge_weight, action, price_target)


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
        self._init_forecast_common(price_dim, hidden_dim, output_dim, num_steps)
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
        self.pool = AttentionPool(hidden_dim)
        self.token_fusion = nn.Sequential(
            nn.LayerNorm((hidden_dim // 2) * 3),
            nn.Linear((hidden_dim // 2) * 3, hidden_dim),
            nn.GELU(),
        )

    def forward(self, price_seq, news_feat=None, macro_feat=None, edge_index=None, edge_weight=None, action=None, price_target=None):
        x = self._normalize_price(price_seq)
        tokenized = self.tokenizer(x, edge_index=edge_index, edge_weight=edge_weight)
        seq_h = self.pool(self.encoder(self.pos(self.price_proj(x))))
        token_h = torch.cat([tokenized["temporal_h"], tokenized["cross_h"], tokenized["fused_h"]], dim=-1)
        h = seq_h + self.token_fusion(token_h)
        out = self._forecast_from_context(h, x, news_feat, macro_feat, edge_index, edge_weight, action, price_target)
        out.update(
            {
                "vq_loss": tokenized["vq_loss"],
                "temporal_token_ids": tokenized["temporal_token_ids"],
                "cross_token_ids": tokenized["cross_token_ids"],
                "temporal_perplexity": tokenized["temporal_perplexity"],
                "cross_perplexity": tokenized["cross_perplexity"],
            }
        )
        return out


class TimesFMStyleForecaster(nn.Module, ForecastOutputMixin):
    """
    Lightweight TimesFM-style baseline.

    This is not the official Google TimesFM pretrained checkpoint. It uses the
    same broad idea of patching a long context into time tokens and decoding a
    multi-horizon forecast, but is trained from scratch on the project data.
    """

    def __init__(
        self,
        price_dim: int = 6,
        hidden_dim: int = 256,
        output_dim: int = 6,
        num_steps: int = 30,
        patch_len: int = 6,
        stride: int = 3,
    ):
        super().__init__()
        self._init_forecast_common(price_dim, hidden_dim, output_dim, num_steps)
        self.patch_len = patch_len
        self.stride = stride
        self.patch_proj = nn.Linear(price_dim * patch_len, hidden_dim)
        self.context_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.pos = PositionalEncoding(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)
        self.pool = AttentionPool(hidden_dim)

    def forward(self, price_seq, news_feat=None, macro_feat=None, edge_index=None, edge_weight=None, action=None, price_target=None):
        x = self._normalize_price(price_seq)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 3, 2).reshape(x.size(0), patches.size(1), -1)
        tokens = self.pos(self.patch_proj(patches))
        encoded = self.encoder(tokens)
        h = self.pool(encoded)
        h = h * self.context_gate(h)
        return self._forecast_from_context(h, x, news_feat, macro_feat, edge_index, edge_weight, action, price_target)


class ChronosMiniForecaster(nn.Module, ForecastOutputMixin):
    """
    Lightweight Chronos-mini baseline.

    This is not the official Amazon Chronos pretrained model. It discretizes
    numerical time-series values into token bins, embeds the tokens, and trains a
    small Transformer from scratch to forecast continuous future returns.
    """

    def __init__(
        self,
        price_dim: int = 6,
        hidden_dim: int = 256,
        output_dim: int = 6,
        num_steps: int = 30,
        vocab_size: int = 256,
        clip_value: float = 5.0,
    ):
        super().__init__()
        self._init_forecast_common(price_dim, hidden_dim, output_dim, num_steps)
        self.vocab_size = vocab_size
        self.clip_value = clip_value
        self.value_embed = nn.Embedding(vocab_size, hidden_dim)
        self.channel_embed = nn.Parameter(torch.randn(1, 1, price_dim, hidden_dim) * 0.02)
        self.pos = PositionalEncoding(hidden_dim, max_len=1024)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)
        self.pool = AttentionPool(hidden_dim)

    def _tokenize(self, price_seq):
        clipped = price_seq.clamp(-self.clip_value, self.clip_value)
        scaled = (clipped + self.clip_value) / (2 * self.clip_value)
        return torch.clamp((scaled * (self.vocab_size - 1)).long(), 0, self.vocab_size - 1)

    def forward(self, price_seq, news_feat=None, macro_feat=None, edge_index=None, edge_weight=None, action=None, price_target=None):
        x = self._normalize_price(price_seq)
        token_ids = self._tokenize(x)
        emb = self.value_embed(token_ids)
        emb = emb + self.channel_embed[:, :, : x.size(2)]
        emb = emb.reshape(x.size(0), x.size(1) * x.size(2), -1)
        emb = self.pos(emb)
        h = self.pool(self.encoder(emb))
        return self._forecast_from_context(h, x, news_feat, macro_feat, edge_index, edge_weight, action, price_target)
