from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(x), dim=1)
        pooled = torch.sum(weights * x, dim=1)
        return pooled, weights.squeeze(-1)


class VectorQuantizer(nn.Module):
    """VQ-VAE quantizer with straight-through gradients."""

    def __init__(
        self,
        num_codes: int,
        code_dim: int,
        beta: float = 0.25,
        usage_decay: float = 0.99,
        reset_interval: int = 250,
        dead_code_threshold: float = 1e-4,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta
        self.usage_decay = usage_decay
        self.reset_interval = reset_interval
        self.dead_code_threshold = dead_code_threshold
        self.codebook = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)
        self.register_buffer("usage_ema", torch.zeros(num_codes))
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def _maybe_reset_dead_codes(self, flat_z: torch.Tensor) -> None:
        if not self.training or self.reset_interval <= 0 or flat_z.numel() == 0:
            return
        updates = int(self.num_updates.item())
        if updates == 0 or updates % self.reset_interval != 0:
            return
        dead_mask = self.usage_ema < self.dead_code_threshold
        if not torch.any(dead_mask):
            return
        dead_count = int(dead_mask.sum().item())
        sample_idx = torch.randint(0, flat_z.size(0), (dead_count,), device=flat_z.device)
        samples = flat_z.detach()[sample_idx]
        noise = torch.randn_like(samples) * 0.01
        self.codebook.weight.data[dead_mask] = samples + noise
        self.usage_ema[dead_mask] = self.usage_ema.mean().clamp_min(self.dead_code_threshold)

    @torch.no_grad()
    def _update_usage(self, encodings: torch.Tensor) -> None:
        if not self.training:
            return
        avg_probs = encodings.mean(dim=0).to(device=self.usage_ema.device, dtype=self.usage_ema.dtype)
        self.usage_ema.mul_(self.usage_decay).add_(avg_probs, alpha=1.0 - self.usage_decay)
        self.num_updates.add_(1)

    def forward(self, z: torch.Tensor) -> dict:
        original_shape = z.shape
        flat_z = z.reshape(-1, self.code_dim)
        self._maybe_reset_dead_codes(flat_z)

        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_z @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=1).unsqueeze(0)
        )
        token_ids = distances.argmin(dim=1)
        quantized = self.codebook(token_ids).view(original_shape)

        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(z, quantized.detach())
        loss = codebook_loss + self.beta * commitment_loss

        quantized_st = z + (quantized - z).detach()
        encodings = F.one_hot(token_ids, self.num_codes).float()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-8).log()).sum())
        self._update_usage(encodings)
        active_codes = (avg_probs > 0).sum().to(z.device)
        dead_codes = (self.usage_ema <= self.dead_code_threshold).sum().to(z.device)

        return {
            "quantized": quantized_st,
            "token_ids": token_ids.view(original_shape[:-1]),
            "loss": loss,
            "perplexity": perplexity,
            "active_codes": active_codes,
            "dead_codes": dead_codes,
        }


class DualVQMarketTokenizer(nn.Module):
    """
    Dual VQ tokenization for HMSC.

    The temporal tokenizer discretizes per-sample historical price patterns.
    The cross-sectional tokenizer discretizes same-day market/neighbor structure.
    """

    def __init__(
        self,
        price_dim: int,
        hidden_dim: int,
        token_dim: int = 64,
        num_temporal_codes: int = 256,
        num_cross_codes: int = 256,
        beta: float = 0.25,
        attn_heads: int = 4,
    ):
        super().__init__()
        self.price_dim = price_dim
        while token_dim % attn_heads != 0 and attn_heads > 1:
            attn_heads -= 1
        self.temporal_encoder = nn.GRU(price_dim, token_dim, batch_first=True)
        self.temporal_pool = AttentionPool(token_dim)
        self.temporal_proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
        )
        self.cross_encoder = nn.Sequential(
            nn.Linear(price_dim * 4, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, token_dim),
            nn.LayerNorm(token_dim),
        )

        self.temporal_vq = VectorQuantizer(num_temporal_codes, token_dim, beta=beta)
        self.cross_vq = VectorQuantizer(num_cross_codes, token_dim, beta=beta)

        self.temporal_out = nn.Linear(token_dim, hidden_dim // 2)
        self.cross_out = nn.Linear(token_dim, hidden_dim // 2)
        self.fused_out = nn.Linear(token_dim, hidden_dim // 2)
        self.temporal_cross_attn = nn.MultiheadAttention(token_dim, attn_heads, batch_first=True)
        self.cross_temporal_attn = nn.MultiheadAttention(token_dim, attn_heads, batch_first=True)
        self.temporal_gate = nn.Sequential(nn.Linear(token_dim * 2, token_dim), nn.Sigmoid())
        self.cross_gate = nn.Sequential(nn.Linear(token_dim * 2, token_dim), nn.Sigmoid())
        self.token_norm = nn.LayerNorm(token_dim)
        self.fusion_proj = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )

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
            [
                weights,
                weights,
                torch.ones(num_nodes, device=device, dtype=torch.float32),
            ],
            dim=0,
        )
        return all_edges, all_weights

    def _graph_cross_input(
        self,
        latest: torch.Tensor,
        edge_index=None,
        edge_weight=None,
        sector_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latest = _align_feature_dim(latest, self.price_dim)
        batch_size, num_nodes = latest.shape
        graph_context = torch.zeros_like(latest)
        for b in range(batch_size):
            edges, weights = self._edges_for_batch(edge_index, edge_weight, b, num_nodes, latest.device)
            src, dst = edges[0], edges[1]
            weights = weights.to(device=latest.device, dtype=latest.dtype)
            numer = torch.zeros(num_nodes, device=latest.device, dtype=latest.dtype)
            denom = torch.zeros(num_nodes, device=latest.device, dtype=latest.dtype)
            numer.index_add_(0, dst, latest[b, src] * weights)
            denom.index_add_(0, dst, weights.abs().clamp_min(1e-6))
            graph_context[b] = numer / denom.clamp_min(1e-6)

        if sector_feat is None:
            sector_context = torch.zeros_like(latest)
        else:
            sector_context = sector_feat.to(device=latest.device, dtype=latest.dtype)
            if sector_context.dim() == 3:
                sector_context = sector_context.mean(dim=-1)
            sector_context = _align_feature_dim(sector_context, self.price_dim)
        graph_delta = latest - graph_context
        return torch.cat([latest, graph_context, graph_delta, sector_context], dim=-1)

    def forward(
        self,
        price_seq: torch.Tensor,
        edge_index=None,
        edge_weight=None,
        sector_feat: torch.Tensor | None = None,
    ) -> dict:
        price_seq = _align_feature_dim(price_seq, self.price_dim)
        temporal_seq, _ = self.temporal_encoder(price_seq)
        temporal_pooled, temporal_attention = self.temporal_pool(temporal_seq)
        temporal_embed = self.temporal_proj(temporal_pooled)
        temporal = self.temporal_vq(temporal_embed)

        cross_input = self._graph_cross_input(price_seq[:, -1], edge_index, edge_weight, sector_feat)
        cross_embed = self.cross_encoder(cross_input)
        cross = self.cross_vq(cross_embed)

        temporal_token = temporal["quantized"]
        cross_token = cross["quantized"]
        temporal_ctx, _ = self.temporal_cross_attn(
            temporal_token.unsqueeze(1),
            cross_token.unsqueeze(1),
            cross_token.unsqueeze(1),
        )
        cross_ctx, _ = self.cross_temporal_attn(
            cross_token.unsqueeze(1),
            temporal_token.unsqueeze(1),
            temporal_token.unsqueeze(1),
        )
        temporal_ctx = temporal_ctx.squeeze(1)
        cross_ctx = cross_ctx.squeeze(1)
        temporal_gate = self.temporal_gate(torch.cat([temporal_token, temporal_ctx], dim=-1))
        cross_gate = self.cross_gate(torch.cat([cross_token, cross_ctx], dim=-1))
        temporal_fused = self.token_norm(temporal_token + temporal_gate * temporal_ctx)
        cross_fused = self.token_norm(cross_token + cross_gate * cross_ctx)
        fused_token = self.fusion_proj(torch.cat([temporal_fused, cross_fused], dim=-1))

        temporal_h = self.temporal_out(temporal_fused)
        cross_h = self.cross_out(cross_fused)
        fused_h = self.fused_out(fused_token)
        vq_loss = temporal["loss"] + cross["loss"]

        return {
            "temporal_h": temporal_h,
            "cross_h": cross_h,
            "fused_h": fused_h,
            "vq_loss": vq_loss,
            "temporal_token_ids": temporal["token_ids"],
            "cross_token_ids": cross["token_ids"],
            "temporal_perplexity": temporal["perplexity"],
            "cross_perplexity": cross["perplexity"],
            "temporal_attention": temporal_attention,
            "temporal_active_codes": temporal["active_codes"],
            "cross_active_codes": cross["active_codes"],
            "temporal_dead_codes": temporal["dead_codes"],
            "cross_dead_codes": cross["dead_codes"],
        }
