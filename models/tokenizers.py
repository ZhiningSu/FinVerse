from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """VQ-VAE quantizer with straight-through gradients."""

    def __init__(self, num_codes: int, code_dim: int, beta: float = 0.25):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta
        self.codebook = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z: torch.Tensor) -> dict:
        original_shape = z.shape
        flat_z = z.reshape(-1, self.code_dim)

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

        return {
            "quantized": quantized_st,
            "token_ids": token_ids.view(original_shape[:-1]),
            "loss": loss,
            "perplexity": perplexity,
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
    ):
        super().__init__()
        self.temporal_encoder = nn.GRU(price_dim, token_dim, batch_first=True)
        self.temporal_proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
        )
        self.cross_encoder = nn.Sequential(
            nn.Linear(price_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, token_dim),
            nn.LayerNorm(token_dim),
        )

        self.temporal_vq = VectorQuantizer(num_temporal_codes, token_dim, beta=beta)
        self.cross_vq = VectorQuantizer(num_cross_codes, token_dim, beta=beta)

        self.temporal_out = nn.Linear(token_dim, hidden_dim // 2)
        self.cross_out = nn.Linear(token_dim, hidden_dim // 2)

    def forward(self, price_seq: torch.Tensor) -> dict:
        temporal_seq, _ = self.temporal_encoder(price_seq)
        temporal_embed = self.temporal_proj(temporal_seq[:, -1])
        temporal = self.temporal_vq(temporal_embed)

        cross_input = price_seq[:, -1]
        cross_embed = self.cross_encoder(cross_input)
        cross = self.cross_vq(cross_embed)

        temporal_h = self.temporal_out(temporal["quantized"])
        cross_h = self.cross_out(cross["quantized"])
        vq_loss = temporal["loss"] + cross["loss"]

        return {
            "temporal_h": temporal_h,
            "cross_h": cross_h,
            "vq_loss": vq_loss,
            "temporal_token_ids": temporal["token_ids"],
            "cross_token_ids": cross["token_ids"],
            "temporal_perplexity": temporal["perplexity"],
            "cross_perplexity": cross["perplexity"],
        }
