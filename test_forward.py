from __future__ import annotations

import torch

torch.set_num_threads(4)

from models.world_model import WorldModel

model = WorldModel(
    price_dim=7,
    news_dim=384,
    macro_dim=8,
    graph_dim=5,
    action_dim=8,
    latent_dim=128,
    hidden_dim=256,
    num_tickers=80,
)

print(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

B, T, F_price = 4, 20, 7
price_seq = torch.randn(B, T, F_price)
news_feat = torch.randn(B, 384)
macro_feat = torch.randn(B, 8)
edge_index = torch.randint(0, B, (2, B * 3))
edge_weight = torch.rand(B * 3)
action = torch.randn(B, 8)
price_target = torch.randn(B, 10, 80 * 5)

output = model(price_seq, news_feat, macro_feat, edge_index, edge_weight, action, price_target)

print(f"\nz shape: {output['z'].shape}")
print(f"prior_h shape: {output['prior_h'].shape}")
print(f"kl shape: {output['kl'].shape}")
print(f"price_pred shape: {output['price_pred'].shape}")
print(f"return_pred shape: {output['return_pred'].shape}")
print(f"regime_logits shape: {output['regime_logits'].shape}")
print(f"\n✅ Model forward pass successful!")