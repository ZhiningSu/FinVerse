from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


class WorldModelLoss(nn.Module):
    def __init__(
        self,
        kl_weight: float = 0.1,
        recon_weight: float = 1.0,
        regime_weight: float = 0.05,
        vq_weight: float = 0.05,
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.recon_weight = recon_weight
        self.regime_weight = regime_weight
        self.vq_weight = vq_weight

    def forward(self, model_output, price_target=None, regime_target=None):
        loss = model_output["loss"]
        kl = model_output["kl"].mean()
        vq_loss = model_output.get("vq_loss", torch.tensor(0.0, device=loss.device))
        total = loss + self.kl_weight * kl + self.vq_weight * vq_loss
        return total, {
            "kl": kl.item(),
            "recon": loss.item(),
            "vq": vq_loss.item(),
            "temporal_perplexity": self._as_float(model_output.get("temporal_perplexity", 0.0)),
            "cross_perplexity": self._as_float(model_output.get("cross_perplexity", 0.0)),
        }

    @staticmethod
    def _as_float(value):
        if isinstance(value, torch.Tensor):
            return value.detach().item()
        return float(value)


class BaselineLoss(nn.Module):
    def forward(self, model_output, price_target=None, regime_target=None):
        loss = model_output["loss"]
        return loss, {"kl": 0.0, "recon": loss.item()}


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        output_dir: str | Path,
        gradient_clip: float = 1.0,
        log_interval: int = 10,
        val_interval: int = 1,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gradient_clip = gradient_clip
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "kl": [],
            "recon": [],
            "vq": [],
            "temporal_perplexity": [],
            "cross_perplexity": [],
        }

    def train_epoch(self, epoch: int):
        self.model.train()
        epoch_losses = []
        epoch_kls = []
        epoch_recons = []
        epoch_vqs = []
        epoch_temporal_ppl = []
        epoch_cross_ppl = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} Train")
        for batch in pbar:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            price_seq = batch["price_seq"]
            news_feat = batch["news_feat"]
            macro_feat = batch["macro_feat"]
            edge_index = batch["edge_index"]
            edge_weight = batch["edge_weight"]
            price_target = batch["price_target"]
            action = batch["action"]

            output = self.model(price_seq, news_feat, macro_feat, edge_index, edge_weight, action, price_target)
            total_loss, metrics = self.criterion(output, price_target)

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()

            epoch_losses.append(total_loss.item())
            epoch_kls.append(metrics["kl"])
            epoch_recons.append(metrics["recon"])
            epoch_vqs.append(metrics.get("vq", 0.0))
            epoch_temporal_ppl.append(metrics.get("temporal_perplexity", 0.0))
            epoch_cross_ppl.append(metrics.get("cross_perplexity", 0.0))

            self.global_step += 1
            if self.global_step % self.log_interval == 0:
                pbar.set_postfix(
                    loss=f"{total_loss.item():.4f}",
                    kl=f"{metrics['kl']:.4f}",
                    vq=f"{metrics.get('vq', 0.0):.4f}",
                )

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        self.history["train_loss"].append(avg_loss)
        self.history["kl"].append(sum(epoch_kls) / len(epoch_kls))
        self.history["recon"].append(sum(epoch_recons) / len(epoch_recons))
        self.history["vq"].append(sum(epoch_vqs) / len(epoch_vqs))
        self.history["temporal_perplexity"].append(sum(epoch_temporal_ppl) / len(epoch_temporal_ppl))
        self.history["cross_perplexity"].append(sum(epoch_cross_ppl) / len(epoch_cross_ppl))

        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int):
        if self.val_loader is None:
            return 0.0
        self.model.eval()
        val_losses = []
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} Val")
        for batch in pbar:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            price_seq = batch["price_seq"]
            news_feat = batch["news_feat"]
            macro_feat = batch["macro_feat"]
            edge_index = batch["edge_index"]
            edge_weight = batch["edge_weight"]
            price_target = batch["price_target"]
            action = batch["action"]
            output = self.model(price_seq, news_feat, macro_feat, edge_index, edge_weight, action, price_target)
            total_loss, _ = self.criterion(output, price_target)
            val_losses.append(total_loss.item())

        avg_val = sum(val_losses) / len(val_losses)
        self.history["val_loss"].append(avg_val)
        return avg_val

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "history": self.history,
            "global_step": self.global_step,
        }
        torch.save(ckpt, self.output_dir / "last_checkpoint.pt")
        if is_best:
            torch.save(ckpt, self.output_dir / "best_checkpoint.pt")

    def load_checkpoint(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.history = ckpt.get("history", {"train_loss": [], "val_loss": [], "kl": [], "recon": []})
        self.history.setdefault("vq", [])
        self.history.setdefault("temporal_perplexity", [])
        self.history.setdefault("cross_perplexity", [])
        self.global_step = ckpt.get("global_step", 0)
        return ckpt["epoch"]
