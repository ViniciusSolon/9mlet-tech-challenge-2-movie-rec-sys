"""Training loop helpers for PyTorch recommenders."""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from data.splits import temporal_train_test_split


def frame_to_loader(
    frame: pd.DataFrame,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Convert a ratings frame into a TensorDataset loader."""
    features = torch.tensor(
        frame[["user_idx", "movie_idx"]].values,
        dtype=torch.long,
    )
    targets = torch.tensor(frame["rating"].values, dtype=torch.float32)
    return DataLoader(
        TensorDataset(features, targets),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def build_loaders(
    frame: pd.DataFrame,
    val_split: float,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    """Split data temporally (fallback random) into train/validation loaders."""
    train_df, val_df = temporal_train_test_split(
        frame,
        test_ratio=val_split,
        seed=seed,
    )
    return (
        frame_to_loader(train_df, batch_size, shuffle=True),
        frame_to_loader(val_df, batch_size, shuffle=False),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
) -> float:
    """Run one training or validation epoch and return mean loss."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    with torch.set_grad_enabled(is_train):
        for batch_x, batch_y in loader:
            total_loss += _step(
                model, batch_x, batch_y, criterion, optimizer, device, is_train
            )
    return total_loss / len(loader)


def _step(
    model: nn.Module,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    is_train: bool,
) -> float:
    """Execute a single optim/eval step and return the batch loss value."""
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    loss = criterion(model(batch_x).squeeze(), batch_y)
    if is_train and optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return float(loss.item())


def update_checkpoint(
    model: nn.Module,
    val_loss: float,
    best_val_loss: float,
    patience_count: int,
    patience_limit: int,
    model_path: str,
) -> tuple[float, int, bool]:
    """Update best checkpoint and early-stopping counters.

    Returns:
        Tuple of (best_val_loss, patience_count, should_stop).
    """
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), model_path)
        return val_loss, 0, False
    patience_count += 1
    return best_val_loss, patience_count, patience_count >= patience_limit


def build_optimizer(
    model: nn.Module,
    lr: float,
) -> tuple[optim.Optimizer, Any]:
    """Create Adam optimizer and ReduceLROnPlateau scheduler."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=2,
        factor=0.5,
    )
    return optimizer, scheduler
