from __future__ import annotations

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
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(weights * x, dim=1)


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
        self.price_input = nn.Linear(price_dim, hidden_dim)
        self.price_pos = nn.Parameter(torch.zeros(1, 256, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.price_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.price_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.price_pool = AttentionPool(hidden_dim)

        self.news_encoder = nn.Sequential(
            nn.Linear(news_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        self.news_pool = AttentionPool(hidden_dim // 2)

        self.macro_encoder = nn.Sequential(
            nn.Linear(macro_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
        )
        self.macro_pool = AttentionPool(hidden_dim // 2)

        self.graph_encoder = GraphAttentionEncoder(
            num_nodes=price_dim,
            hidden_dim=hidden_dim // 2,
            num_heads=4,
            num_layers=2,
        )

        if use_dual_vq:
            self.dual_vq_tokenizer = DualVQMarketTokenizer(
                price_dim=price_dim,
                hidden_dim=hidden_dim,
                token_dim=hidden_dim // 4,
                num_temporal_codes=256,
                num_cross_codes=256,
            )
            self.vq_token_proj = nn.Linear(hidden_dim // 2, hidden_dim)
            self.vq_cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
            self.vq_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
            self.vq_residual_norm = nn.LayerNorm(hidden_dim)
            self.vq_summary = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
            )
            fusion_dim = hidden_dim + hidden_dim // 2 + hidden_dim // 2 + hidden_dim // 2 + hidden_dim // 2
        else:
            self.dual_vq_tokenizer = None
            self.vq_token_proj = None
            self.vq_cross_attn = None
            self.vq_gate = None
            self.vq_residual_norm = None
            self.vq_summary = None
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
        price_seq = _align_feature_dim(price_seq, self.price_dim)

        price_h_seq = self.price_input(price_seq)
        price_h_seq = price_h_seq + self.price_pos[:, : price_h_seq.size(1)]
        price_h_seq = self.price_transformer(price_h_seq)
        price_h_seq = self.price_encoder(price_h_seq)
        price_h = self.price_pool(price_h_seq)
        news_h = self.news_pool(self.news_encoder(news_feat))
        macro_h = self.macro_pool(self.macro_encoder(macro_feat))
        graph_h = self.graph_encoder(price_seq, edge_index, edge_weight)

        aux = {
            "vq_loss": price_seq.new_tensor(0.0),
            "temporal_token_ids": None,
            "cross_token_ids": None,
            "temporal_perplexity": price_seq.new_tensor(0.0),
            "cross_perplexity": price_seq.new_tensor(0.0),
            "temporal_active_codes": price_seq.new_tensor(0.0),
            "cross_active_codes": price_seq.new_tensor(0.0),
            "temporal_dead_codes": price_seq.new_tensor(0.0),
            "cross_dead_codes": price_seq.new_tensor(0.0),
        }
        parts = [price_h, news_h, macro_h, graph_h]
        if self.dual_vq_tokenizer is not None:
            vq = self.dual_vq_tokenizer(price_seq, edge_index=edge_index, edge_weight=edge_weight)
            vq_representations = [vq["temporal_h"], vq["cross_h"]]
            if "fused_h" in vq:
                vq_representations.append(vq["fused_h"])
            vq_tokens = torch.stack(
                [self.vq_token_proj(item) for item in vq_representations],
                dim=1,
            )
            vq_attn, _ = self.vq_cross_attn(price_h.unsqueeze(1), vq_tokens, vq_tokens)
            vq_attn = vq_attn.squeeze(1)
            gate = self.vq_gate(torch.cat([price_h, vq_attn], dim=-1))
            price_h = self.vq_residual_norm(price_h + gate * vq_attn)
            vq_context = self.vq_summary(torch.cat([vq_tokens.mean(dim=1), vq_attn], dim=-1))
            parts = [price_h, news_h, macro_h, graph_h, vq_context]
            aux.update(vq)

        combined = torch.cat(parts, dim=-1)
        fused = self.fusion(combined)
        stats = self.to_latent(fused)
        mu, logvar = stats[..., :self.latent_dim], stats[..., self.latent_dim:]
        return mu, logvar, aux


class GraphAttentionEncoder(nn.Module):

    def __init__(self, num_nodes: int, hidden_dim: int, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        while hidden_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.node_temporal = nn.GRU(1, hidden_dim, batch_first=True)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.edge_bias = nn.Linear(1, num_heads)
        self.edge_value = nn.Linear(1, hidden_dim)
        self.attn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.pool_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def _edges_for_batch(self, edge_index, edge_weight, batch_idx: int, device: torch.device):
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

        base_edges = base_edges.clamp(0, self.num_nodes - 1)
        reverse_edges = base_edges.flip(0)
        self_edges = torch.arange(self.num_nodes, device=device, dtype=torch.long).repeat(2, 1)
        all_edges = torch.cat([base_edges, reverse_edges, self_edges], dim=1)
        all_weights = torch.cat(
            [
                weights,
                weights,
                torch.ones(self.num_nodes, device=device, dtype=torch.float32),
            ],
            dim=0,
        )
        return all_edges, all_weights

    def _message_pass(self, node_h: torch.Tensor, edge_index, edge_weight) -> torch.Tensor:
        batch_size, num_nodes, _ = node_h.shape
        updated = []
        for b in range(batch_size):
            h = node_h[b]
            edges, weights = self._edges_for_batch(edge_index, edge_weight, b, h.device)
            src, dst = edges[0], edges[1]
            q = self.query(h).view(num_nodes, self.num_heads, self.head_dim)
            k = self.key(h).view(num_nodes, self.num_heads, self.head_dim)
            v = self.value(h).view(num_nodes, self.num_heads, self.head_dim)
            edge_signal = weights.to(dtype=h.dtype).unsqueeze(-1)
            edge_bias = self.edge_bias(edge_signal)
            edge_value = self.edge_value(edge_signal).view(-1, self.num_heads, self.head_dim)
            messages = torch.zeros_like(h)
            for node in range(num_nodes):
                mask = dst == node
                if not torch.any(mask):
                    messages[node] = h[node]
                    continue
                src_idx = src[mask]
                score = (q[node].unsqueeze(0) * k[src_idx]).sum(dim=-1)
                score = score / (self.head_dim ** 0.5) + edge_bias[mask]
                alpha = torch.softmax(score, dim=0)
                values = v[src_idx] + edge_value[mask]
                messages[node] = torch.sum(alpha.unsqueeze(-1) * values, dim=0).reshape(self.hidden_dim)
            updated.append(messages)
        return torch.stack(updated, dim=0)

    def forward(self, price_seq, edge_index, edge_weight):
        price_seq = _align_feature_dim(price_seq, self.num_nodes)
        batch_size, seq_len, num_nodes = price_seq.shape
        node_series = price_seq.transpose(1, 2).reshape(batch_size * num_nodes, seq_len, 1)
        _, h = self.node_temporal(node_series)
        node_h = self.node_norm(h[-1].reshape(batch_size, num_nodes, self.hidden_dim))
        for attn_norm, ff in zip(self.attn_norms, self.layers):
            messages = self._message_pass(node_h, edge_index, edge_weight)
            node_h = attn_norm(node_h + messages)
            node_h = node_h + ff(node_h)
        weights = torch.softmax(self.pool_score(node_h), dim=1)
        graph_h = torch.sum(weights * node_h, dim=1)
        return self.out_proj(graph_h)


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
        if action.dim() == 2:
            action = action.unsqueeze(1)
        h = prev_latent[:, -1] if prev_latent.dim() == 3 else prev_latent
        for t in range(action.size(1)):
            inp = torch.cat([h, action[:, t]], dim=-1)
            h = self.rssm_transition(inp, h)
        prior_stats = self.prior_net(h)
        return h, prior_stats

    def get_prior(self, prev_latent):
        prior_stats = self.prior_net(prev_latent)
        return prior_stats


class ObservationDecoder(nn.Module):

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256, num_tickers: int = 80):
        super().__init__()
        self.latent_dim = latent_dim
        self.rollout_cell = nn.GRUCell(latent_dim + 1, latent_dim)
        self.state_norm = nn.LayerNorm(latent_dim)
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
        if latent.dim() == 2:
            latent = latent.unsqueeze(1).expand(-1, horizon, -1)
        steps = min(horizon, latent.size(1))
        state = latent[:, 0]
        prev_price = latent.new_zeros(latent.size(0), 1)
        prices = []
        states = []
        for t in range(steps):
            context = latent[:, t]
            state = self.rollout_cell(torch.cat([context, prev_price], dim=-1), state)
            state = self.state_norm(state + context)
            price_t = self.price_head(state)
            prev_price = price_t
            prices.append(price_t.squeeze(-1))
            states.append(state)
        if steps < horizon:
            for _ in range(horizon - steps):
                state = self.rollout_cell(torch.cat([state, prev_price], dim=-1), state)
                state = self.state_norm(state)
                price_t = self.price_head(state)
                prev_price = price_t
                prices.append(price_t.squeeze(-1))
                states.append(state)
        state_path = torch.stack(states, dim=1)
        price_pred = torch.stack(prices, dim=1)
        return_pred = self.return_head(state_path[:, -1])
        regime_logits = self.regime_head(state_path)
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
        self.action_dim = action_dim

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

    def _compose_action(self, latent: torch.Tensor, action: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        learned_action = torch.tanh(self.action_net(latent))
        if action is None:
            return learned_action, learned_action
        action = action.to(latent.device, dtype=latent.dtype)
        if action.dim() == 3:
            action = action[:, 0]
        if action.size(-1) != learned_action.size(-1):
            action = _align_feature_dim(action.unsqueeze(1), learned_action.size(-1)).squeeze(1)
        policy_action = torch.tanh(action + learned_action)
        return policy_action, learned_action

    def imagine(self, prior_h, actions, horizon: int = 30):
        imagined_states = []
        h = prior_h
        actions = actions.to(device=prior_h.device, dtype=prior_h.dtype)
        if actions.dim() == 2:
            actions = actions.unsqueeze(1).expand(-1, horizon, -1)
        if actions.size(-1) != self.action_dim:
            actions = _align_feature_dim(actions, self.action_dim)
        if actions.size(1) < horizon:
            pad = actions[:, -1:].expand(-1, horizon - actions.size(1), -1)
            actions = torch.cat([actions, pad], dim=1)
        for t in range(horizon):
            a = actions[:, t]
            inp = torch.cat([h, a], dim=-1)
            h = self.transition.rssm_transition(inp, h)
            imagined_states.append(h)
        return torch.stack(imagined_states, dim=1)

    def _action_policy_loss(self, policy_action: torch.Tensor, price_target: torch.Tensor | None) -> torch.Tensor:
        if price_target is None or price_target.numel() == 0:
            return policy_action.new_tensor(0.0)
        if price_target.dim() == 3:
            target_returns = price_target[:, : min(5, price_target.size(1)), :].mean(dim=1)
        elif price_target.dim() == 2:
            target_returns = price_target[:, : min(5, price_target.size(1))].mean(dim=1, keepdim=True)
        else:
            return policy_action.new_tensor(0.0)
        target_returns = torch.nan_to_num(target_returns, nan=0.0, posinf=0.0, neginf=0.0).clamp(-0.2, 0.2)
        target_returns = _align_feature_dim(target_returns.unsqueeze(1), policy_action.size(-1)).squeeze(1)
        return -(torch.tanh(policy_action) * target_returns.detach()).mean()

    def forward(self, price_seq, news_feat, macro_feat, edge_index, edge_weight, action, price_target=None):
        q_mu, q_logvar, aux = self.encode(price_seq, news_feat, macro_feat, edge_index, edge_weight, return_aux=True)
        z = self.reparameterize(q_mu, q_logvar)
        policy_action, learned_action = self._compose_action(z, action)
        prior_h, prior_stats = self.transition(prev_latent=z, action=policy_action)
        p_mu = prior_stats[..., : self.latent_dim]
        p_logvar = prior_stats[..., self.latent_dim:]
        kl = self.kl_divergence(q_mu, q_logvar, p_mu, p_logvar)

        imagined = self.imagine(z, policy_action.unsqueeze(1).expand(-1, 30, -1), horizon=30)
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
            "action_pred": policy_action,
            "learned_action": learned_action,
            "action_reg_loss": learned_action.pow(2).mean(),
            "action_policy_loss": self._action_policy_loss(policy_action, price_target),
            "temporal_token_ids": aux["temporal_token_ids"],
            "cross_token_ids": aux["cross_token_ids"],
            "temporal_perplexity": aux["temporal_perplexity"],
            "cross_perplexity": aux["cross_perplexity"],
            "temporal_active_codes": aux["temporal_active_codes"],
            "cross_active_codes": aux["cross_active_codes"],
            "temporal_dead_codes": aux["temporal_dead_codes"],
            "cross_dead_codes": aux["cross_dead_codes"],
        }

    def reparameterize(self, mu, logvar):
        if self.training:
            std = (logvar * 0.5).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def init_hidden(self, batch_size: int, device: torch.device):
        return torch.zeros(batch_size, self.latent_dim, device=device)
