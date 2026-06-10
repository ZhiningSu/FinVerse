from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PriceOnlyGRU(nn.Module):

    def __init__(
        self,
        price_dim: int = 6,
        hidden_dim: int = 256,
        output_dim: int = 6,
        num_steps: int = 30,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.output_dim = output_dim

        self.encoder = nn.GRU(price_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.1)
        self.decoder = nn.GRUCell(hidden_dim, hidden_dim)

        self.predict = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, price_seq, news_feat=None, macro_feat=None, edge_index=None, edge_weight=None, action=None, price_target=None):
        _, hidden = self.encoder(price_seq)
        if isinstance(hidden, tuple):
            hidden = hidden[-1]
        if hidden.dim() == 3:
            hidden = hidden.mean(dim=0)

        preds = []
        h = hidden
        for _ in range(self.num_steps):
            h = self.decoder(h, h)
            pred = self.predict(h)
            preds.append(pred)

        price_pred = torch.stack(preds, dim=1)

        loss = torch.tensor(0.0, device=price_seq.device)
        if price_target is not None and price_target.numel() > 0:
            loss = F.mse_loss(price_pred, price_target)

        return {
            "price_pred": price_pred,
            "loss": loss,
            "hidden": h,
        }


class MultiModalNoRollout(nn.Module):

    def __init__(
        self,
        price_dim: int = 6,
        news_dim: int = 384,
        macro_dim: int = 8,
        graph_dim: int = 5,
        action_dim: int = 8,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_tickers: int = 66,
    ):
        super().__init__()
        self.latent_dim = latent_dim

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

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2 + hidden_dim // 2 + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.to_latent = nn.Linear(hidden_dim, latent_dim)

        self.transition = nn.GRUCell(latent_dim + action_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
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

    def encode(self, price_seq, news_feat, macro_feat, edge_index, edge_weight):
        lstm_out, _ = self.price_lstm(price_seq)
        price_h = self.price_encoder(lstm_out).mean(dim=1)
        news_h = self.news_encoder(news_feat).mean(dim=1)
        macro_h = self.macro_encoder(macro_feat).mean(dim=1)
        n_nodes = int(edge_index.max().item()) + 1
        if price_seq.size(2) < n_nodes:
            padded = torch.cat([
                price_seq,
                torch.zeros(price_seq.size(0), price_seq.size(1), n_nodes - price_seq.size(2), device=price_seq.device)
            ], dim=2)
        else:
            padded = price_seq[:, :, :n_nodes]
        graph_h = self.graph_encoder(padded).mean(dim=1)
        combined = torch.cat([price_h, news_h, macro_h, graph_h], dim=-1)
        fused = self.fusion(combined)
        return self.to_latent(fused)

    def forward(self, price_seq, news_feat, macro_feat, edge_index, edge_weight, action, price_target=None):
        z = self.encode(price_seq, news_feat, macro_feat, edge_index, edge_weight)

        imagined = []
        h = z
        for _ in range(30):
            inp = torch.cat([h, action], dim=-1)
            h = self.transition(inp, h)
            imagined.append(h)
        imagined = torch.stack(imagined, dim=1)

        price_pred = self.decoder(imagined).squeeze(-1)
        return_pred = self.return_head(imagined[:, -1])
        regime_logits = self.regime_head(imagined[:, -1])

        loss = torch.tensor(0.0, device=z.device)
        if price_target is not None and price_target.numel() > 0:
            target = price_target[:, :, 0] if price_target.dim() == 3 else price_target
            loss = F.mse_loss(price_pred, target)

        return {
            "price_pred": price_pred,
            "return_pred": return_pred,
            "regime_logits": regime_logits,
            "loss": loss,
            "z": z,
            "kl": torch.tensor(0.0, device=z.device),
        }


class NoGraphWorldModel(nn.Module):

    def __init__(
        self,
        price_dim: int = 6,
        news_dim: int = 384,
        macro_dim: int = 8,
        graph_dim: int = 5,
        action_dim: int = 8,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_tickers: int = 66,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_tickers = num_tickers

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

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2 + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.to_latent = nn.Linear(hidden_dim, latent_dim * 2)

        self.transition = nn.GRUCell(latent_dim + action_dim, latent_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

        self.decoder_price = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.decoder_return = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_tickers),
        )
        self.decoder_regime = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 4),
        )

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

    def reparameterize(self, mu, logvar):
        if self.training:
            std = (logvar * 0.5).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def encode(self, price_seq, news_feat, macro_feat, edge_index, edge_weight):
        lstm_out, _ = self.price_lstm(price_seq)
        price_h = self.price_encoder(lstm_out).mean(dim=1)
        news_h = self.news_encoder(news_feat).mean(dim=1)
        macro_h = self.macro_encoder(macro_feat).mean(dim=1)

        combined = torch.cat([price_h, news_h, macro_h], dim=-1)
        fused = self.fusion(combined)
        stats = self.to_latent(fused)
        mu, logvar = stats[..., :self.latent_dim], stats[..., self.latent_dim:]
        return mu, logvar

    def imagine(self, prior_h, actions, horizon: int = 30):
        imagined = []
        h = prior_h
        actions = actions.to(prior_h.device)
        if actions.dim() == 2:
            actions = actions.unsqueeze(1).expand(-1, horizon, -1)
        for t in range(horizon):
            a = actions[:, t]
            inp = torch.cat([h, a], dim=-1)
            h = self.transition(inp, h)
            imagined.append(h)
        return torch.stack(imagined, dim=1)

    def forward(self, price_seq, news_feat, macro_feat, edge_index, edge_weight, action, price_target=None):
        q_mu, q_logvar = self.encode(price_seq, news_feat, macro_feat, edge_index, edge_weight)
        z = self.reparameterize(q_mu, q_logvar)

        prior_h = z
        prior_stats = self.prior_net(prior_h)
        p_mu = prior_stats[..., :self.latent_dim]
        p_logvar = prior_stats[..., self.latent_dim:]
        kl = self.kl_divergence(q_mu, q_logvar, p_mu, p_logvar)

        imagined = self.imagine(z, action.unsqueeze(1).expand(-1, 30, -1), horizon=30)
        price_pred = self.decoder_price(imagined).squeeze(-1)
        return_pred = self.decoder_return(imagined[:, -1])
        regime_logits = self.decoder_regime(imagined[:, -1])

        loss = torch.tensor(0.0, device=z.device)
        if price_target is not None and price_target.numel() > 0:
            target = price_target[:, :, 0] if price_target.dim() == 3 else price_target
            loss = F.mse_loss(price_pred, target)

        return {
            "z": z,
            "prior_h": prior_h,
            "kl": kl,
            "price_pred": price_pred,
            "return_pred": return_pred,
            "regime_logits": regime_logits,
            "loss": loss,
        }
