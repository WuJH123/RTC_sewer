"""Trainer: PyTorch training loop with AMP, gradient clipping, checkpointing.

Features:
  - AMP (mixed precision) for 8GB VRAM
  - Gradient clipping (max_norm=1.0)
  - AdamW optimizer
  - Event-equal sampling weights
  - Checkpoint save/load
  - Training history logging
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast

logger = logging.getLogger(__name__)


class TwinDynamicsTrainer:
    """Training loop for TwinGraphDynamics model.

    Handles mixed precision, gradient accumulation, checkpointing,
    and training history logging.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_grad_norm: float = 1.0,
        device: str = "cuda",
        amp: bool = True,
        grad_accum_steps: int = 1,
        output_dir: str | Path = "outputs/v42_training",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.max_grad_norm = float(max_grad_norm)
        self.amp = amp and self.device.type == "cuda"
        self.grad_accum_steps = int(grad_accum_steps)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scaler = GradScaler(enabled=self.amp)

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.train_history: list[dict[str, float]] = []

    def train_epoch(
        self,
        dataloader: Any,
        epoch: int | None = None,
    ) -> dict[str, float]:
        """Train for one epoch.

        dataloader should yield batches with keys matching the model's
        forward() and loss function's expected inputs.
        """
        if epoch is not None:
            self.epoch = epoch
        self.model.train()
        epoch_losses: dict[str, float] = {}
        n_batches = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = self._to_device(batch)

            # Mixed precision forward
            with autocast(enabled=self.amp):
                pred = self.model(
                    state_history=batch["state_history"],
                    rainfall=batch["rainfall"],
                    action_candidate=batch["action_candidate"],
                    action_reference=batch["action_reference"],
                    edge_index=batch["edge_index"],
                    node_static=batch["node_static"],
                    action_node_map=batch["action_node_map"],
                )

                # Compute loss
                if self.loss_fn is not None:
                    loss_dict = self.loss_fn(pred, batch.get("target", {}))
                    if isinstance(loss_dict, dict):
                        loss = sum(loss_dict.values())
                    else:
                        loss = loss_dict
                else:
                    loss = pred.get("delta", torch.zeros(())).mean()

            # Scale loss for gradient accumulation
            loss = loss / self.grad_accum_steps

            # Backward
            self.scaler.scale(loss).backward()

            # Gradient accumulation step
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Accumulate losses
            loss_val = loss.detach().item() * self.grad_accum_steps
            for k, v in (loss_dict.items() if isinstance(loss_dict, dict) else [("loss", loss)]):
                val = v.detach().item() if torch.is_tensor(v) else v
                epoch_losses[k] = epoch_losses.get(k, 0.0) + val
            n_batches += 1

        # Average losses
        epoch_losses = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
        epoch_losses["lr"] = self.optimizer.param_groups[0]["lr"]
        epoch_losses["epoch_time"] = time.time() - t0
        self.train_history.append(epoch_losses)
        self.epoch += 1

        logger.info(f"Epoch {self.epoch}: {epoch_losses}")
        return epoch_losses

    def _to_device(self, batch: dict) -> dict:
        """Recursively move batch tensors to device."""
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device)
            elif isinstance(v, dict):
                out[k] = self._to_device(v)
            else:
                out[k] = v
        return out

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        """Save model, optimizer, and training state."""
        if path is None:
            path = self.output_dir / f"checkpoint_epoch{self.epoch}.pt"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "train_history": self.train_history,
        }, path)
        logger.info(f"Checkpoint saved: {path}")
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model, optimizer, and training state."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)
        self.train_history = ckpt.get("train_history", [])
        logger.info(f"Checkpoint loaded: {path} (epoch={self.epoch})")

    def save_history(self, path: str | Path | None = None) -> Path:
        """Save training history as JSON."""
        if path is None:
            path = self.output_dir / "train_history.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.train_history, f, indent=2)
        return path
