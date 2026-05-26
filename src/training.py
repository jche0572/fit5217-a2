"""Training helpers for Task 1 PyTorch sequence-to-sequence models."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn
from tqdm.auto import tqdm

from src.preprocessing import PAD_ID


def _step_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    criterion: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Compute cross-entropy loss for one padded mini-batch."""
    src = batch["src"].to(device)
    src_lengths = batch["src_lengths"].to(device)
    tgt = batch["tgt"].to(device)

    logits = model(src, tgt, src_lengths)
    gold = tgt[:, 1:]
    return criterion(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip: float = 1.0,
    show_progress: bool = True,
) -> float:
    """Train ``model`` for one epoch and return average token loss."""
    model.train()
    total_loss = 0.0
    total_batches = 0
    iterator = tqdm(dataloader, desc="train", leave=False, disable=not show_progress)

    for batch in iterator:
        optimizer.zero_grad(set_to_none=True)
        loss = _step_loss(model, batch, criterion, device)
        loss.backward()
        # RNN 训练容易梯度爆炸, baseline 统一裁剪到 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1
        iterator.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    show_progress: bool = True,
) -> float:
    """Evaluate ``model`` for one epoch and return average token loss."""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    iterator = tqdm(dataloader, desc="eval", leave=False, disable=not show_progress)

    for batch in iterator:
        loss = _step_loss(model, batch, criterion, device)
        total_loss += loss.item()
        total_batches += 1
        iterator.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_batches, 1)


class EarlyStopping:
    """Simple dev-loss early stopping with best-checkpoint tracking."""

    def __init__(self, patience: int = 3, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.bad_epochs = 0

    def step(self, dev_loss: float) -> bool:
        """Return True when training should stop."""
        if dev_loss < self.best_loss - self.min_delta:
            self.best_loss = dev_loss
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def make_criterion() -> nn.Module:
    """Create the baseline cross-entropy loss that ignores ``<PAD>``."""
    return nn.CrossEntropyLoss(ignore_index=PAD_ID)


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    train_losses: List[float],
    dev_losses: List[float],
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model, optimizer, config, and loss history to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "train_losses": train_losses,
        "dev_losses": dev_losses,
        "config": config or {},
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(state, path)


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Load a checkpoint into ``model`` and optionally ``optimizer``."""
    try:
        state = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # 兼容较老 PyTorch, 当时 torch.load 还没有 weights_only 参数
        state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state
