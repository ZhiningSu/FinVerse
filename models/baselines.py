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


def _target_for_prediction(pred: torch.Tensor, target: torch.Tensor | None) -> torch.Tensor | None:
    if target is None or target.numel() == 0:
        return None
    target = target.to(device=pred.device, dtype=pred.dtype)
    steps = min(pred.size(1), target.size(1))
    pred_channels = pred.size(-1) if pred.dim() == 3 else None
    target = target[:, :steps]
    if pred.dim() == 2 and target.dim() == 3:
        target = target[:, :, 0]
    elif pred.dim() == 3 and target.dim() == 2:
        target = target.unsqueeze(-1).expand(-1, -1, pred_channels)
    elif pred.dim() == 3 and target.dim() == 3:
        target = _align_feature_dim(target, pred_channels)
    return target


def _mse_loss(pred: torch.Tensor, target: torch.Tensor | None) -> torch.Tensor:
    target = _target_for_prediction(pred, target)
    if target is None:
        return pred.new_tensor(0.0)
    steps = min(pred.size(1), target.size(1))
    return F.mse_loss(pred[:, :steps], target[:, :steps])


def _kl_divergence(q_mu: torch.Tensor, q_logvar: torch.Tensor, p_mu: torch.Tensor, p_logvar: torch.Tensor) -> torch.Tensor:
    p_var = p_logvar.exp()
    q_var = q_logvar.exp()
    kl = (q_var + (q_mu - p_mu) ** 2) / (p_var + 1e-8) + p_logvar - q_logvar - 1
    return 0.5 * kl.sum(dim=-1)


def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor, training: bool) -> torch.Tensor:
    if training:
        std = (logvar * 0.5).exp()
        return mu + torch.randn_like(std) * std
    return mu


def _prepare_action_sequence(
    action: torch.Tensor | None,
    batch_size: int,
    horizon: int,
    action_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if action is None:
        return torch.zeros(batch_size, horizon, action_dim, device=device, dtype=dtype)
    action = action.to(device=device, dtype=dtype)
    if action.dim() == 2:
        action = action.unsqueeze(1).expand(-1, horizon, -1)
    if action.size(1) < horizon:
        pad = action[:, -1:].expand(-1, horizon - action.size(1), -1)
        action = torch.cat([action, pad], dim=1)
    return _align_feature_dim(action[:, :horizon], action_dim)


def _edge_index_and_weight(edge_index, edge_weight, batch_idx: int, num_nodes: int, device: torch.device):
    if edge_index is None or edge_index.numel() == 0:
        base_edges = torch.empty(2, 0, dtype=torch.long, device=device)
        weights = torch.empty(0, dtype=torch.float32, device=device)
    else:
        base_edges = edge_index[batch_idx] if edge_index.dim() == 3 else edge_index
        base_edges = base_edges.to(device=device, dtype=torch.long).clamp(0, num_nodes - 1)
        if edge_weight is None or edge_weight.numel() == 0:
            weights = torch.ones(base_edges.size(-1), device=device)
        else:
            weights = edge_weight[batch_idx] if edge_weight.dim() == 2 else edge_weight
            weights = weights.to(device=device, dtype=torch.float32)
    reverse_edges = base_edges.flip(0)
    self_edges = torch.arange(num_nodes, device=device, dtype=torch.long).repeat(2, 1)
    edges = torch.cat([base_edges, reverse_edges, self_edges], dim=1)
    weights = torch.cat([weights, weights, torch.ones(num_nodes, device=device)], dim=0)
    return edges, weights


def _graph_neighbor_series(price_seq: torch.Tensor, edge_index, edge_weight, num_nodes: int) -> torch.Tensor:
    price_seq = _align_feature_dim(price_seq, num_nodes)
    batch_size, _, _ = price_seq.shape
    latest = price_seq[:, -1]
    neighbor_latest = torch.zeros_like(latest)
    for b in range(batch_size):
        edges, weights = _edge_index_and_weight(edge_index, edge_weight, b, num_nodes, price_seq.device)
        src, dst = edges[0], edges[1]
        weights = weights.to(device=price_seq.device, dtype=price_seq.dtype)
        numer = torch.zeros(num_nodes, device=price_seq.device, dtype=price_seq.dtype)
        denom = torch.zeros(num_nodes, device=price_seq.device, dtype=price_seq.dtype)
        numer.index_add_(0, dst, latest[b, src] * weights)
        denom.index_add_(0, dst, weights.abs().clamp_min(1e-6))
        neighbor_latest[b] = numer / denom.clamp_min(1e-6)
    graph_seq = price_seq.clone()
    graph_seq[:, -1] = 0.5 * (latest + neighbor_latest)
    return graph_seq


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
        loss = _mse_loss(price_pred, price_target)

        return {
            "price_pred": price_pred,
            "loss": loss,
            "hidden": h,
            "kl": price_seq.new_tensor(0.0),
            "vq_loss": price_seq.new_tensor(0.0),
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
        self.price_dim = price_dim
        self.news_dim = news_dim
        self.macro_dim = macro_dim
        self.action_dim = action_dim
        self.num_steps = 30

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
        price_seq = _align_feature_dim(price_seq, self.price_dim)
        news_feat = _align_feature_dim(news_feat, self.news_dim)
        macro_feat = _align_feature_dim(macro_feat, self.macro_dim)
        lstm_out, _ = self.price_lstm(price_seq)
        price_h = self.price_encoder(lstm_out).mean(dim=1)
        news_h = self.news_encoder(news_feat).mean(dim=1)
        macro_h = self.macro_encoder(macro_feat).mean(dim=1)
        graph_seq = _graph_neighbor_series(price_seq, edge_index, edge_weight, self.price_dim)
        graph_h = self.graph_encoder(graph_seq).mean(dim=1)
        combined = torch.cat([price_h, news_h, macro_h, graph_h], dim=-1)
        fused = self.fusion(combined)
        return self.to_latent(fused)

    def forward(self, price_seq, news_feat, macro_feat, edge_index, edge_weight, action, price_target=None):
        z = self.encode(price_seq, news_feat, macro_feat, edge_index, edge_weight)

        actions = _prepare_action_sequence(
            action,
            batch_size=z.size(0),
            horizon=self.num_steps,
            action_dim=self.action_dim,
            device=z.device,
            dtype=z.dtype,
        )
        imagined = []
        h = z
        for t in range(self.num_steps):
            inp = torch.cat([h, actions[:, t]], dim=-1)
            h = self.transition(inp, h)
            imagined.append(h)
        imagined = torch.stack(imagined, dim=1)

        price_pred = self.decoder(imagined).squeeze(-1)
        return_pred = self.return_head(imagined[:, -1])
        regime_logits = self.regime_head(imagined[:, -1])

        loss = _mse_loss(price_pred, price_target)

        return {
            "price_pred": price_pred,
            "return_pred": return_pred,
            "regime_logits": regime_logits,
            "loss": loss,
            "z": z,
            "kl": z.new_tensor(0.0),
            "vq_loss": z.new_tensor(0.0),
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
        self.price_dim = price_dim
        self.news_dim = news_dim
        self.macro_dim = macro_dim
        self.action_dim = action_dim

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
        return _kl_divergence(q_mu, q_logvar, p_mu, p_logvar)

    def reparameterize(self, mu, logvar):
        return _reparameterize(mu, logvar, self.training)

    def encode(self, price_seq, news_feat, macro_feat, edge_index, edge_weight):
        price_seq = _align_feature_dim(price_seq, self.price_dim)
        news_feat = _align_feature_dim(news_feat, self.news_dim)
        macro_feat = _align_feature_dim(macro_feat, self.macro_dim)
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
        actions = _prepare_action_sequence(
            actions,
            batch_size=prior_h.size(0),
            horizon=horizon,
            action_dim=self.action_dim,
            device=prior_h.device,
            dtype=prior_h.dtype,
        )
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

        imagined = self.imagine(z, action, horizon=30)
        price_pred = self.decoder_price(imagined).squeeze(-1)
        return_pred = self.decoder_return(imagined[:, -1])
        regime_logits = self.decoder_regime(imagined[:, -1])

        loss = _mse_loss(price_pred, price_target)

        return {
            "z": z,
            "prior_h": prior_h,
            "kl": kl,
            "price_pred": price_pred,
            "return_pred": return_pred,
            "regime_logits": regime_logits,
            "loss": loss,
            "vq_loss": z.new_tensor(0.0),
        }


class DreamerStyleRSSM(nn.Module):
    """
    Dreamer-style RSSM baseline trained from scratch.

    The baseline keeps a deterministic recurrent state and stochastic latent
    posterior/prior, but uses a compact price-only encoder so that it serves as a
    clean world-model baseline rather than another HMSC variant.
    """

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
        num_steps: int = 30,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.price_dim = price_dim
        self.action_dim = action_dim
        self.price_encoder = nn.GRU(price_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.1)
        self.posterior = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )
        self.rssm = nn.GRUCell(latent_dim + action_dim, hidden_dim)
        self.prior = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.return_head = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_tickers),
        )
        self.regime_head = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 4),
        )

    def reparameterize(self, mu, logvar):
        return _reparameterize(mu, logvar, self.training)

    def kl_divergence(self, q_mu, q_logvar, p_mu, p_logvar):
        return _kl_divergence(q_mu, q_logvar, p_mu, p_logvar)

    def forward(self, price_seq, news_feat=None, macro_feat=None, edge_index=None, edge_weight=None, action=None, price_target=None):
        price_seq = _align_feature_dim(price_seq, self.price_dim)
        _, h_seq = self.price_encoder(price_seq)
        h_det = h_seq[-1]
        q_stats = self.posterior(h_det)
        q_mu, q_logvar = q_stats[:, : self.latent_dim], q_stats[:, self.latent_dim :]
        z = self.reparameterize(q_mu, q_logvar)
        actions = _prepare_action_sequence(
            action,
            batch_size=price_seq.size(0),
            horizon=self.num_steps,
            action_dim=self.action_dim,
            device=price_seq.device,
            dtype=price_seq.dtype,
        )
        prior_stats = self.prior(h_det)
        p_mu, p_logvar = prior_stats[:, : self.latent_dim], prior_stats[:, self.latent_dim :]
        kl = self.kl_divergence(q_mu, q_logvar, p_mu, p_logvar)

        preds = []
        states = []
        h = h_det
        z_t = z
        for t in range(self.num_steps):
            h = self.rssm(torch.cat([z_t, actions[:, t]], dim=-1), h)
            prior_stats = self.prior(h)
            p_mu = prior_stats[:, : self.latent_dim]
            p_logvar = prior_stats[:, self.latent_dim :]
            z_t = self.reparameterize(p_mu, p_logvar)
            state = torch.cat([h, z_t], dim=-1)
            states.append(state)
            preds.append(self.decoder(state).squeeze(-1))
        state_path = torch.stack(states, dim=1)
        price_pred = torch.stack(preds, dim=1)
        return_pred = self.return_head(state_path[:, -1])
        regime_logits = self.regime_head(state_path[:, -1])

        loss = _mse_loss(price_pred, price_target)
        return {
            "z": z,
            "prior_h": h_det,
            "kl": kl,
            "price_pred": price_pred,
            "return_pred": return_pred,
            "regime_logits": regime_logits,
            "loss": loss,
            "vq_loss": price_seq.new_tensor(0.0),
        }
