from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tokenizers import DualVQMarketTokenizer


class MultiModalEncoder(nn.Module):

    def __init__(
        self,
        price_dim: int = 7,
        news_dim: int = 384,
        macro_dim: int = 8,
        graph_dim: int = 5,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        use_dual_vq: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.price_dim = price_dim
        self.use_dual_vq = use_dual_vq
        self.price_lstm = nn.LSTM(price_dim, hidden_dim, batch_first=True)
        self.price_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.news_encoder = nn.Sequential(
            nn.Linear(news_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        self.macro_encoder = nn.Sequential(
            nn.Linear(macro_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
        )

        self.graph_encoder = nn.Sequential(
            nn.Linear(price_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )
        self.graph_pool = nn.Linear(hidden_dim // 2, hidden_dim // 2)

        if use_dual_vq:
            self.dual_vq_tokenizer = DualVQMarketTokenizer(
                price_dim=price_dim,
                hidden_dim=hidden_dim,
                token_dim=hidden_dim // 4,
                num_temporal_codes=256,
                num_cross_codes=256,
            )
            fusion_dim = hidden_dim + hidden_dim // 2 + hidden_dim // 2 + hidden_dim // 2 + hidden_dim
        else:
            self.dual_vq_tokenizer = None
            fusion_dim = hidden_dim + hidden_dim // 2 + hidden_dim // 2 + hidden_dim // 2

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.to_latent = nn.Linear(hidden_dim, latent_dim * 2)

    def forward(self, price_seq, news_feat, macro_feat, edge_index, edge_weight):
        if news_feat.dim() == 2:
            news_feat = news_feat.unsqueeze(1)
        if macro_feat.dim() == 2:
            macro_feat = macro_feat.unsqueeze(1)
        if price_seq.size(2) < self.price_dim:
            price_seq = torch.cat(
                [
                    price_seq,
                    torch.zeros(
                        price_seq.size(0),
                        price_seq.size(1),
                        self.price_dim - price_seq.size(2),
                        device=price_seq.device,
                        dtype=price_seq.dtype,
                    ),
                ],
                dim=2,
            )
        elif price_seq.size(2) > self.price_dim:
            price_seq = price_seq[:, :, : self.price_dim]

        lstm_out, _ = self.price_lstm(price_seq)
        price_h = self.price_encoder(lstm_out)
        price_h = price_h.mean(dim=1)
        news_h = self.news_encoder(news_feat).mean(dim=1)
        macro_h = self.macro_encoder(macro_feat).mean(dim=1)
        n_nodes = self.price_dim
        if price_seq.size(2) < n_nodes:
            price_seq_padded = torch.cat([
                price_seq,
                torch.zeros(price_seq.size(0), price_seq.size(1), n_nodes - price_seq.size(2), device=price_seq.device)
            ], dim=2)
        else:
            price_seq_padded = price_seq[:, :, :n_nodes]
        node_encoded = self.graph_encoder(price_seq_padded)
        graph_h = node_encoded.mean(dim=1)

        aux = {
            "vq_loss": price_seq.new_tensor(0.0),
            "temporal_token_ids": None,
            "cross_token_ids": None,
            "temporal_perplexity": price_seq.new_tensor(0.0),
            "cross_perplexity": price_seq.new_tensor(0.0),
        }
        parts = [price_h, news_h, macro_h, graph_h]
        if self.dual_vq_tokenizer is not None:
            vq = self.dual_vq_tokenizer(price_seq)
            parts.extend([vq["temporal_h"], vq["cross_h"]])
            aux.update(vq)

        combined = torch.cat(parts, dim=-1)
        fused = self.fusion(combined)
        stats = self.to_latent(fused)
        mu, logvar = stats[..., :self.latent_dim], stats[..., self.latent_dim:]
        return mu, logvar, aux


class GraphAttentionEncoder(nn.Module):

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight):
        graph_feat = x.mean(dim=0, keepdim=True)
        D = graph_feat.shape[-1]
        mlp = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
        )
        return self.out_proj(mlp(graph_feat.squeeze(0)))


class TransitionModel(nn.Module):

    def __init__(self, latent_dim: int = 128, action_dim: int = 8, hidden_dim: int = 256):
        super().__init__()
        self.rssm_transition = nn.GRUCell(latent_dim + action_dim, latent_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

    def forward(self, prev_latent, action):
        if prev_latent.dim() == 2:
            prev_latent = prev_latent.unsqueeze(1)
        if action.dim() == 2:
            action = action.unsqueeze(1)
        inp = torch.cat([prev_latent, action], dim=-1)
        if inp.size(1) > 1:
            inp = inp.squeeze(1)
        else:
            inp = inp.squeeze(1)
        h = self.rssm_transition(inp, prev_latent.squeeze(1))
        prior_stats = self.prior_net(h)
        return h, prior_stats

    def get_prior(self, prev_latent):
        prior_stats = self.prior_net(prev_latent)
        return prior_stats


class ObservationDecoder(nn.Module):

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256, num_tickers: int = 80):
        super().__init__()
        self.price_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.return_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_tickers),
        )
        self.regime_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 4),
        )

    def forward(self, latent, horizon: int = 30):
        price_pred = self.price_head(latent)
        price_pred = price_pred.squeeze(-1)
        return_pred = self.return_head(latent)
        regime_logits = self.regime_head(latent)
        return price_pred, return_pred, regime_logits


class WorldModel(nn.Module):

    def __init__(
        self,
        price_dim: int = 7,
        news_dim: int = 384,
        macro_dim: int = 8,
        graph_dim: int = 5,
        action_dim: int = 8,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_tickers: int = 80,
        use_dual_vq: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_tickers = num_tickers

        self.encoder = MultiModalEncoder(
            price_dim,
            news_dim,
            macro_dim,
            graph_dim,
            hidden_dim,
            latent_dim,
            use_dual_vq=use_dual_vq,
        )
        self.transition = TransitionModel(latent_dim, action_dim, hidden_dim)
        self.decoder = ObservationDecoder(latent_dim, hidden_dim, num_tickers)

        self.news_projector = nn.Linear(768, news_dim)
        self.action_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def kl_divergence(self, q_mu, q_logvar, p_mu, p_logvar):
        p_var = p_logvar.exp()
        q_var = q_logvar.exp()
        kl = (q_var + (q_mu - p_mu) ** 2) / (p_var + 1e-8) + p_logvar - q_logvar - 1
        return 0.5 * kl.sum(dim=-1)

    def encode(self, price_seq, news_feat, macro_feat, edge_index, edge_weight, return_aux: bool = False):
        q_mu, q_logvar, aux = self.encoder(price_seq, news_feat, macro_feat, edge_index, edge_weight)
        if return_aux:
            return q_mu, q_logvar, aux
        return q_mu, q_logvar

    def imagine(self, prior_h, actions, horizon: int = 30):
        self.eval()
        imagined_states = []
        h = prior_h
        actions = actions.to(prior_h.device)
        if actions.dim() == 2:
            actions = actions.unsqueeze(1).expand(-1, horizon, -1)
        for t in range(horizon):
            a = actions[:, t]
            inp = torch.cat([h, a], dim=-1)
            h = self.transition.rssm_transition(inp, h)
            imagined_states.append(h)
        return torch.stack(imagined_states, dim=1)

    def forward(self, price_seq, news_feat, macro_feat, edge_index, edge_weight, action, price_target=None):
        q_mu, q_logvar, aux = self.encode(price_seq, news_feat, macro_feat, edge_index, edge_weight, return_aux=True)
        z = self.reparameterize(q_mu, q_logvar)
        prior_h, prior_stats = self.transition(prev_latent=z, action=action)
        p_mu = prior_stats[..., : self.latent_dim]
        p_logvar = prior_stats[..., self.latent_dim:]
        kl = self.kl_divergence(q_mu, q_logvar, p_mu, p_logvar)

        imagined = self.imagine(z, action.unsqueeze(1).expand(-1, 30, -1), horizon=30)
        price_pred, return_pred, regime_logits = self.decoder(imagined, horizon=30)

        loss = torch.tensor(0.0, device=z.device)
        if price_target is not None and price_target.numel() > 0:
            target_next = price_target[:, :, 0] if price_target.dim() == 3 else price_target
            recon_loss = F.mse_loss(price_pred, target_next)
            loss = loss + recon_loss

        return {
            "z": z,
            "prior_h": prior_h,
            "kl": kl,
            "price_pred": price_pred,
            "return_pred": return_pred,
            "regime_logits": regime_logits,
            "loss": loss,
            "vq_loss": aux["vq_loss"],
            "temporal_token_ids": aux["temporal_token_ids"],
            "cross_token_ids": aux["cross_token_ids"],
            "temporal_perplexity": aux["temporal_perplexity"],
            "cross_perplexity": aux["cross_perplexity"],
        }

    def reparameterize(self, mu, logvar):
        if self.training:
            std = (logvar * 0.5).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def init_hidden(self, batch_size: int, device: torch.device):
        return torch.zeros(batch_size, self.latent_dim, device=device)
