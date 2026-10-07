"""
src/model.py
Multi-Head Neural Architecture for joint quality & degradation prediction.

Receives dataset embedding from DatasetEncoder and outputs:
  - 6 quality dimension scores  ∈ [0, 1]
  - 2 downstream degradation scores (light, heavy)  ∈ [0, 1]

All outputs are trained jointly with a weighted MSE loss.
"""

from __future__ import annotations
from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.encoder import DatasetEncoder, N_COL_STATS, N_ROW_POOL, extract_dataset_features
from src.corruption import CorruptedDataset, CORRUPTION_DIMS
from src.utils import get_logger, GLOBAL_SEED

logger = get_logger("model")

# ──────────────────────────────────────────────────────────────────
# Output names
# ──────────────────────────────────────────────────────────────────
QUALITY_HEADS   = CORRUPTION_DIMS                          # 6 heads
DEGRAD_HEADS    = ["light_drop", "heavy_drop"]             # 2 heads
ALL_HEADS       = QUALITY_HEADS + DEGRAD_HEADS             # 8 total
N_OUTPUTS       = len(ALL_HEADS)


# ──────────────────────────────────────────────────────────────────
# Dataset wrapper for DataLoader
# ──────────────────────────────────────────────────────────────────
class TabularQualityDataset(Dataset):
    """
    PyTorch Dataset wrapping a list of CorruptedDataset objects.
    Converts raw arrays → (col_stats, row_pool, targets) tensors.
    """

    MAX_COLS = 200  # fixed column dimension for padding

    def __init__(self, variants: list):
        self.data: list[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        for v in variants:
            col_stats, row_pool = extract_dataset_features(v.X, max_cols_for_encoding=self.MAX_COLS)

            # Build target vector [8]: 6 severities + 2 degradation scores
            targets = np.concatenate([
                v.severities,                                   # [6]
                np.array([v.light_drop, v.heavy_drop], dtype=np.float32),  # [2]
            ])

            self.data.append((col_stats, row_pool, targets.astype(np.float32)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        col_stats, row_pool, targets = self.data[idx]
        return (
            torch.from_numpy(col_stats),    # [MAX_COLS, N_COL_STATS]
            torch.from_numpy(row_pool),     # [N_ROW_POOL]
            torch.from_numpy(targets),      # [N_OUTPUTS]
        )


# ──────────────────────────────────────────────────────────────────
# Neural model
# ──────────────────────────────────────────────────────────────────
class PredictionHeads(nn.Module):
    """Independent linear + sigmoid heads for each output."""

    def __init__(self, embed_dim: int, n_outputs: int):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, 1),
                nn.Sigmoid(),
            )
            for _ in range(n_outputs)
        ])

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """emb: [B, embed_dim]  →  [B, n_outputs]"""
        return torch.cat([h(emb) for h in self.heads], dim=-1)


class DataQualityNet(nn.Module):
    """
    Full multi-head neural network.

    Architecture
    ──────────────────────────────────────────────────────────────
    (col_stats, row_pool)
        → DatasetEncoder            → embedding [embed_dim]
        → shared trunk MLP          → trunk [trunk_dim]
        → PredictionHeads (×8)      → predictions [8]  ∈ [0,1]
    """

    def __init__(
        self,
        embed_dim:  int   = 128,
        trunk_dim:  int   = 256,
        col_dim:    int   = 64,
        n_heads:    int   = 4,
        n_seeds:    int   = 4,
        dropout:    float = 0.15,
        max_cols:   int   = 200,
    ):
        super().__init__()
        self.encoder = DatasetEncoder(
            col_stats_dim = N_COL_STATS,
            col_dim       = col_dim,
            n_heads       = n_heads,
            n_seeds       = n_seeds,
            embed_dim     = embed_dim,
            dropout       = dropout,
        )
        self.trunk = nn.Sequential(
            nn.Linear(embed_dim, trunk_dim),
            nn.LayerNorm(trunk_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(trunk_dim, trunk_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = PredictionHeads(trunk_dim, N_OUTPUTS)

    def forward(
        self,
        col_stats: torch.Tensor,  # [B, n_cols, N_COL_STATS]
        row_pool:  torch.Tensor,  # [B, N_ROW_POOL]
    ) -> torch.Tensor:
        """Returns predictions [B, N_OUTPUTS] ∈ [0, 1]."""
        emb  = self.encoder(col_stats, row_pool)   # [B, embed_dim]
        feat = self.trunk(emb)                     # [B, trunk_dim]
        return self.heads(feat)                    # [B, N_OUTPUTS]

    def predict_numpy(
        self,
        col_stats: np.ndarray,
        row_pool:  np.ndarray,
        device:    str = "cpu",
    ) -> np.ndarray:
        """
        Convenience method for single-dataset inference.
        Returns np.ndarray [N_OUTPUTS].
        """
        self.eval()
        with torch.no_grad():
            cs = torch.from_numpy(col_stats).unsqueeze(0).to(device)  # [1, n_cols, 16]
            rp = torch.from_numpy(row_pool).unsqueeze(0).to(device)   # [1, 8]
            out = self.forward(cs, rp)
        return out.squeeze(0).cpu().numpy()


# ──────────────────────────────────────────────────────────────────
# Loss function
# ──────────────────────────────────────────────────────────────────
def quality_loss(
    preds:   torch.Tensor,  # [B, N_OUTPUTS]
    targets: torch.Tensor,  # [B, N_OUTPUTS]
    quality_weight: float = 1.0,
    degrad_weight:  float = 2.0,  # downstream scores weighted higher
) -> torch.Tensor:
    """
    Weighted MSE across all heads.
    Degradation heads get higher weight since they are the key application target.
    """
    quality_loss = F.mse_loss(preds[:, :6], targets[:, :6])
    degrad_loss  = F.mse_loss(preds[:, 6:], targets[:, 6:])
    return quality_weight * quality_loss + degrad_weight * degrad_loss


# ──────────────────────────────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────────────────────────────
def train_epoch(
    model:     DataQualityNet,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    str = "cpu",
) -> float:
    model.train()
    total_loss = 0.0
    for col_stats, row_pool, targets in loader:
        col_stats = col_stats.to(device)
        row_pool  = row_pool.to(device)
        targets   = targets.to(device)

        optimizer.zero_grad()
        preds = model(col_stats, row_pool)
        loss  = quality_loss(preds, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


@torch.no_grad()
def eval_epoch(
    model:   DataQualityNet,
    loader:  DataLoader,
    device:  str = "cpu",
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Returns (mean_loss, all_preds [N, 8], all_targets [N, 8])."""
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for col_stats, row_pool, targets in loader:
        col_stats = col_stats.to(device)
        row_pool  = row_pool.to(device)
        targets   = targets.to(device)

        preds = model(col_stats, row_pool)
        loss  = quality_loss(preds, targets)
        total_loss += loss.item()
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_preds   = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    return total_loss / max(1, len(loader)), all_preds, all_targets


def fit_model(
    model:          DataQualityNet,
    train_variants: list,
    val_variants:   list,
    n_epochs:       int   = 50,
    batch_size:     int   = 32,
    lr:             float = 3e-4,
    device:         str   = "cpu",
    patience:       int   = 10,
) -> Dict:
    """
    Train the model with early stopping.
    Returns dict with training history and best val loss.
    """
    train_ds = TabularQualityDataset(train_variants)
    val_ds   = TabularQualityDataset(val_variants)

    g = torch.Generator()
    g.manual_seed(GLOBAL_SEED)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr/20)

    best_val_loss = float("inf")
    best_state    = None
    history       = {"train_loss": [], "val_loss": []}
    patience_cnt  = 0

    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, _, _ = eval_epoch(model, val_loader, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1

        if epoch % 10 == 0:
            logger.debug(f"  Epoch {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f}")

        if patience_cnt >= patience:
            logger.debug(f"  Early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_val_loss"] = best_val_loss
    return history
