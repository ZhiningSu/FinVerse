from __future__ import annotations

import argparse
import json
import logging
import os
import random
import ssl
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from datasets.finworld_dataset import FinWorldDataset, collate_fn
from models.baselines import DreamerStyleRSSM, PriceOnlyGRU, MultiModalNoRollout, NoGraphWorldModel
from models.forecasting_baselines import (
    ChronosMiniForecaster,
    DLinearForecaster,
    GRUForecaster,
    ITransformerForecaster,
    KronosMiniForecaster,
    LSTMForecaster,
    PatchTSTForecaster,
    TimesFMStyleForecaster,
    TransformerForecaster,
)
from models.world_model import WorldModel
from trainers.trainer import Trainer, WorldModelLoss, BaselineLoss

ssl._create_default_https_context = ssl._create_unverified_context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


DEFAULT_CONFIG = {
    "data_root": "data/processed/real_90",
    "output_dir": "outputs",
    "latent_dim": 128,
    "hidden_dim": 256,
    "kl_weight": 0.001,
    "recon_weight": 1.0,
    "regime_weight": 0.05,
    "vq_weight": 0.001,
    "action_weight": 0.001,
    "policy_weight": 0.001,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "num_epochs": 30,
    "gradient_clip": 1.0,
    "log_interval": 10,
    "val_interval": 1,
    "save_interval": 5,
    "num_workers": 0,
    "max_train_episodes": None,
    "max_val_episodes": None,
    "target_mode": "return",
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def build_parser():
    parser = argparse.ArgumentParser(description="Train FinWorld Model")
    parser.add_argument("--data-root", default=DEFAULT_CONFIG["data_root"])
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--latent-dim", type=int, default=DEFAULT_CONFIG["latent_dim"])
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_CONFIG["hidden_dim"])
    parser.add_argument("--kl-weight", type=float, default=DEFAULT_CONFIG["kl_weight"])
    parser.add_argument("--recon-weight", type=float, default=DEFAULT_CONFIG["recon_weight"])
    parser.add_argument("--regime-weight", type=float, default=DEFAULT_CONFIG["regime_weight"])
    parser.add_argument("--vq-weight", type=float, default=DEFAULT_CONFIG["vq_weight"])
    parser.add_argument("--action-weight", type=float, default=DEFAULT_CONFIG["action_weight"])
    parser.add_argument("--policy-weight", type=float, default=DEFAULT_CONFIG["policy_weight"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_CONFIG["num_epochs"])
    parser.add_argument("--gradient-clip", type=float, default=DEFAULT_CONFIG["gradient_clip"])
    parser.add_argument("--log-interval", type=int, default=DEFAULT_CONFIG["log_interval"])
    parser.add_argument("--val-interval", type=int, default=DEFAULT_CONFIG["val_interval"])
    parser.add_argument("--save-interval", type=int, default=DEFAULT_CONFIG["save_interval"])
    parser.add_argument("--num-workers", type=int, default=DEFAULT_CONFIG["num_workers"])
    parser.add_argument("--max-train-episodes", type=int, default=None)
    parser.add_argument("--max-val-episodes", type=int, default=None)
    parser.add_argument("--target-mode", choices=["return", "price"], default=DEFAULT_CONFIG["target_mode"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--device", default=DEFAULT_CONFIG["device"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-save", action="store_true", help="Run training without writing checkpoints.")
    parser.add_argument("--price-stats", type=str, default=None)
    parser.add_argument("--macro-stats", type=str, default=None)
    parser.add_argument("--model", type=str, default="full",
        choices=[
            "full", "price_only", "multi_noroll", "no_graph",
            "lstm", "gru", "dlinear", "transformer", "patchtst", "itransformer",
            "kronos_mini", "chronos_mini", "timesfm", "vanilla_rssm",
            "dreamer_rssm", "finverse",
        ],
        help="Model family to train.")
    return parser


def load_stats(path: str | Path | None):
    if path is None:
        return {}
    with open(path) as f:
        return json.load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(model_name: str, args, device):
    extra = {
        "price_dim": 6,
        "news_dim": 384,
        "macro_dim": 8,
        "graph_dim": 5,
        "action_dim": 8,
        "num_tickers": int(getattr(args, "num_tickers", 90)),
    }
    world_extra = {
        **extra,
        "num_sectors": int(getattr(args, "num_sectors", 32)),
    }
    if model_name in {"full", "finverse"}:
        model = WorldModel(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim, **world_extra).to(device)
        criterion = WorldModelLoss(
            kl_weight=args.kl_weight,
            recon_weight=args.recon_weight,
            regime_weight=args.regime_weight,
            vq_weight=args.vq_weight,
            action_weight=args.action_weight,
            policy_weight=args.policy_weight,
        )
    elif model_name == "vanilla_rssm":
        model = WorldModel(
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            use_dual_vq=False,
            **world_extra,
        ).to(device)
        criterion = WorldModelLoss(
            kl_weight=args.kl_weight,
            recon_weight=args.recon_weight,
            regime_weight=args.regime_weight,
            vq_weight=0.0,
            action_weight=args.action_weight,
            policy_weight=args.policy_weight,
        )
    elif model_name == "dreamer_rssm":
        model = DreamerStyleRSSM(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim, **extra).to(device)
        criterion = WorldModelLoss(
            kl_weight=args.kl_weight,
            recon_weight=args.recon_weight,
            regime_weight=args.regime_weight,
            vq_weight=0.0,
            action_weight=args.action_weight,
            policy_weight=args.policy_weight,
        )
    elif model_name == "price_only":
        model = PriceOnlyGRU(price_dim=6, hidden_dim=args.hidden_dim, output_dim=6, num_steps=30).to(device)
        criterion = BaselineLoss()
    elif model_name == "multi_noroll":
        model = MultiModalNoRollout(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim, **extra).to(device)
        criterion = BaselineLoss()
    elif model_name == "no_graph":
        model = NoGraphWorldModel(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim, **extra).to(device)
        criterion = WorldModelLoss(
            kl_weight=args.kl_weight,
            recon_weight=args.recon_weight,
            regime_weight=args.regime_weight,
            vq_weight=args.vq_weight,
            action_weight=args.action_weight,
            policy_weight=args.policy_weight,
        )
    elif model_name == "lstm":
        model = LSTMForecaster(price_dim=6, hidden_dim=args.hidden_dim, output_dim=6, num_steps=30).to(device)
        criterion = BaselineLoss()
    elif model_name == "gru":
        model = GRUForecaster(price_dim=6, hidden_dim=args.hidden_dim, output_dim=6, num_steps=30).to(device)
        criterion = BaselineLoss()
    elif model_name == "dlinear":
        model = DLinearForecaster(price_dim=6, hidden_dim=args.hidden_dim, output_dim=6, num_steps=30).to(device)
        criterion = BaselineLoss()
    elif model_name == "transformer":
        model = TransformerForecaster(price_dim=6, hidden_dim=args.hidden_dim, output_dim=6, num_steps=30).to(device)
        criterion = BaselineLoss()
    elif model_name == "patchtst":
        model = PatchTSTForecaster(price_dim=6, hidden_dim=args.hidden_dim, output_dim=6, num_steps=30).to(device)
        criterion = BaselineLoss()
    elif model_name == "itransformer":
        model = ITransformerForecaster(price_dim=6, hidden_dim=args.hidden_dim, output_dim=6, num_steps=30).to(device)
        criterion = BaselineLoss()
    elif model_name == "kronos_mini":
        model = KronosMiniForecaster(price_dim=6, hidden_dim=args.hidden_dim, output_dim=6, num_steps=30).to(device)
        criterion = BaselineLoss()
    elif model_name == "chronos_mini":
        model = ChronosMiniForecaster(price_dim=6, hidden_dim=args.hidden_dim, output_dim=6, num_steps=30).to(device)
        criterion = BaselineLoss()
    elif model_name == "timesfm":
        model = TimesFMStyleForecaster(price_dim=6, hidden_dim=args.hidden_dim, output_dim=6, num_steps=30).to(device)
        criterion = BaselineLoss()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model, criterion


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading datasets...")
    train_dataset = FinWorldDataset(
        root=args.data_root,
        split="train",
        price_stats=load_stats(args.price_stats),
        macro_stats=load_stats(args.macro_stats),
        max_episodes=args.max_train_episodes,
        target_mode=args.target_mode,
    )
    val_dataset = FinWorldDataset(
        root=args.data_root,
        split="validation",
        price_stats=load_stats(args.price_stats),
        macro_stats=load_stats(args.macro_stats),
        max_episodes=args.max_val_episodes,
        target_mode=args.target_mode,
    )

    LOGGER.info("Train episodes: %d | Val episodes: %d", len(train_dataset), len(val_dataset))
    args.num_tickers = int(train_dataset.price_buffer.shape[1])
    LOGGER.info("Detected ticker universe size: %d", args.num_tickers)
    args.num_sectors = max(len(getattr(train_dataset, "sector_vocab", [])), 1)
    LOGGER.info("Detected sector vocabulary size: %d", args.num_sectors)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn) if val_dataset else None

    LOGGER.info("Building model (type=%s)...", args.model)
    model, criterion = build_model(args.model, args, device)

    LOGGER.info("Model params: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        output_dir=output_dir,
        gradient_clip=args.gradient_clip,
        log_interval=args.log_interval,
        val_interval=args.val_interval,
    )

    start_epoch = 0
    if args.resume:
        LOGGER.info("Resuming from %s", args.resume)
        start_epoch = trainer.load_checkpoint(args.resume) + 1

    LOGGER.info("Starting training for %d epochs...", args.num_epochs)
    for epoch in range(start_epoch, args.num_epochs):
        train_loss = trainer.train_epoch(epoch)
        scheduler.step()

        val_loss = 0.0
        if epoch % args.val_interval == 0 and val_loader:
            val_loss = trainer.validate(epoch)

        selection_loss = trainer.last_val_metrics.get("recon", val_loss) if val_loader else val_loss
        is_best = selection_loss < trainer.best_val_loss
        if is_best:
            trainer.best_val_loss = selection_loss

        if not args.no_save and (epoch % args.save_interval == 0 or is_best):
            trainer.save_checkpoint(epoch, is_best=is_best)

        LOGGER.info(
            "Epoch %d | Train Loss: %.4f | Val Loss: %.4f | Val Recon: %.4f | Best Recon: %.4f",
            epoch,
            train_loss,
            val_loss,
            selection_loss,
            trainer.best_val_loss,
        )

    if args.no_save:
        LOGGER.info("Training complete. Checkpoint saving was disabled (--no-save).")
    else:
        LOGGER.info("Training complete. Checkpoints saved to %s", output_dir)


if __name__ == "__main__":
    main()
