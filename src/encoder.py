"""
src/encoder.py
Permutation-Invariant Dataset Encoder.

Produces a fixed-size dataset-level embedding from tabular data
of arbitrary shape (n_rows × n_cols) by:

  1. Per-column statistics extraction (column-wise meta-features)
  2. Row-wise feature aggregation (row pooling)
  3. DeepSets encoder over column representations → set embedding
  4. Optional Set Transformer attention (for richer interactions)
  5. Final projection to fixed-size embedding
"""

from __future__ import annotations
import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import get_logger, GLOBAL_SEED

logger = get_logger("encoder")

# ──────────────────────────────────────────────────────────────────
# Column-wise meta-feature extraction (CPU, numpy)
# ──────────────────────────────────────────────────────────────────
N_COL_STATS = 16  # number of per-column statistics


def extract_column_stats(X: np.ndarray, max_rows: int = 5000) -> np.ndarray:
    """
    Extract N_COL_STATS statistics per column from a numpy array.
    Handles NaN values gracefully.

    Parameters
    ----------
    X : np.ndarray  shape [n_rows, n_cols]
    max_rows : int  subsample for speed

    Returns
    -------
    col_stats : np.ndarray  shape [n_cols, N_COL_STATS]
    """
    if X.shape[0] > max_rows:
        idx = np.random.RandomState(GLOBAL_SEED).choice(X.shape[0], max_rows, replace=False)
        X = X[idx]

    n_cols = X.shape[1]
    stats = np.zeros((n_cols, N_COL_STATS), dtype=np.float32)

    for c in range(n_cols):
        col = X[:, c].astype(np.float64)
        valid = col[~np.isnan(col)]

        if len(valid) == 0:
            stats[c] = 0.0
            stats[c, 10] = 1.0  # 100% missing
            continue

        n_total = len(col)
        n_valid = len(valid)

        # 0: mean
        stats[c, 0]  = float(np.mean(valid))
        # 1: std
        stats[c, 1]  = float(np.std(valid) + 1e-8)
        # 2: min
        stats[c, 2]  = float(np.min(valid))
        # 3: max
        stats[c, 3]  = float(np.max(valid))
        # 4: median
        stats[c, 4]  = float(np.median(valid))
        # 5: IQR
        q75, q25 = np.percentile(valid, [75, 25])
        stats[c, 5]  = float(q75 - q25)
        # 6: skewness
        if stats[c, 1] > 1e-8:
            stats[c, 6] = float(np.mean(((valid - stats[c, 0]) / stats[c, 1]) ** 3))
        # 7: kurtosis
        if stats[c, 1] > 1e-8:
            stats[c, 7] = float(np.mean(((valid - stats[c, 0]) / stats[c, 1]) ** 4) - 3)
        # 8: fraction unique
        stats[c, 8]  = float(np.unique(valid).shape[0] / n_valid)
        # 9: fraction zeros
        stats[c, 9]  = float(np.sum(np.abs(valid) < 1e-8) / n_valid)
        # 10: fraction missing
        stats[c, 10] = float((n_total - n_valid) / n_total)
        # 11: 5th percentile
        stats[c, 11] = float(np.percentile(valid, 5))
        # 12: 95th percentile
        stats[c, 12] = float(np.percentile(valid, 95))
        # 13: outlier fraction (>3σ)
        mu, sigma = stats[c, 0], stats[c, 1]
        stats[c, 13] = float(np.sum(np.abs(valid - mu) > 3 * sigma) / n_valid)
        # 14: autocorrelation (lag-1) if enough data
        if n_valid > 10:
            try:
                ac = float(np.corrcoef(valid[:-1], valid[1:])[0, 1])
                stats[c, 14] = 0.0 if np.isnan(ac) else ac
            except Exception:
                stats[c, 14] = 0.0
        # 15: log(std + 1)  – scale-invariant spread
        stats[c, 15] = float(math.log1p(stats[c, 1]))

    # Clip to reasonable range to prevent exploding inputs
    stats = np.clip(stats, -50.0, 50.0)
    # Replace any remaining NaN/inf
    stats = np.nan_to_num(stats, nan=0.0, posinf=50.0, neginf=-50.0)
    return stats.astype(np.float32)


def extract_row_stats(X: np.ndarray, max_rows: int = 5000) -> np.ndarray:
    """
    Compute row-level aggregated statistics (pool over rows to get a
    fixed-size global summary independent of n_rows).

    Returns
    -------
    row_pool : np.ndarray  shape [8]  — global pool over row-level stats
    """
    if X.shape[0] > max_rows:
        idx = np.random.RandomState(GLOBAL_SEED).choice(X.shape[0], max_rows, replace=False)
        X = X[idx]

    row_means  = np.nanmean(X, axis=1)  # [n_rows]
    row_stds   = np.nanstd(X, axis=1)
    row_miss   = np.mean(np.isnan(X), axis=1)
    row_maxabs = np.nanmax(np.abs(X), axis=1)

    pool = np.array([
        float(np.mean(row_means)),   float(np.std(row_means)),
        float(np.mean(row_stds)),    float(np.std(row_stds)),
        float(np.mean(row_miss)),    float(np.std(row_miss)),
        float(np.mean(row_maxabs)),  float(np.std(row_maxabs)),
    ], dtype=np.float32)

    return np.clip(np.nan_to_num(pool, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)


N_ROW_POOL = 8   # dimensionality of row_pool


# ──────────────────────────────────────────────────────────────────
# Neural encoder modules (PyTorch)
# ──────────────────────────────────────────────────────────────────
class ColumnMLP(nn.Module):
    """Per-column MLP: maps each column's statistics to a latent vector."""

    def __init__(self, in_dim: int = N_COL_STATS, hidden_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, n_cols, N_COL_STATS]
        return self.net(x)  # [batch, n_cols, out_dim]


class MultiHeadAttentionPool(nn.Module):
    """
    Set Transformer-style pooling:
    Inducing-point multihead attention for O(n) permutation-invariant
    column aggregation → fixed k seed vectors.
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_seeds: int = 4):
        super().__init__()
        self.n_seeds = n_seeds
        self.d_model = d_model
        self.seeds   = nn.Parameter(torch.randn(1, n_seeds, d_model))
        self.attn    = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
        self.norm    = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [batch, n_cols, d_model]
        returns: [batch, n_seeds * d_model]
        """
        B = x.size(0)
        seeds = self.seeds.expand(B, -1, -1)   # [B, n_seeds, d_model]
        out, _ = self.attn(seeds, x, x)        # [B, n_seeds, d_model]
        out = self.norm(out + seeds)
        return out.reshape(B, -1)              # [B, n_seeds * d_model]


class DeepSetsEncoder(nn.Module):
    """
    DeepSets aggregation: φ is applied per-element, then sum-pooled.
    Permutation invariant by construction.
    """

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.rho = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [batch, n_elements, d_model]
        returns: [batch, d_model]
        """
        phi_out = self.phi(x)       # [B, n, d_model]
        pooled  = phi_out.sum(dim=1)  # [B, d_model]  ← permutation invariant
        return self.rho(pooled)     # [B, d_model]


class DatasetEncoder(nn.Module):
    """
    Full permutation-invariant dataset encoder.

    Architecture
    ──────────────
    column stats [n_cols, 16]
        → ColumnMLP → [n_cols, col_dim]
        → DeepSets sum-pool → [col_dim]
        ↕ (concatenate)
    Set Transformer pool → [n_seeds * col_dim]
        → concat with DeepSets output and row_pool [8]
        → projection MLP → embedding [embed_dim]

    The encoder is permutation-invariant in both columns and rows.
    """

    def __init__(
        self,
        col_stats_dim: int  = N_COL_STATS,
        col_dim:       int  = 64,
        n_heads:       int  = 4,
        n_seeds:       int  = 4,
        embed_dim:     int  = 128,
        dropout:       float = 0.1,
    ):
        super().__init__()
        self.col_mlp      = ColumnMLP(col_stats_dim, col_dim * 2, col_dim)
        self.deepsets     = DeepSetsEncoder(col_dim)
        self.attn_pool    = MultiHeadAttentionPool(col_dim, n_heads, n_seeds)

        # Input to final projection: deepsets [col_dim] + attnpool [n_seeds*col_dim] + row_pool [8]
        proj_in_dim = col_dim + n_seeds * col_dim + N_ROW_POOL
        self.proj = nn.Sequential(
            nn.Linear(proj_in_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(
        self,
        col_stats: torch.Tensor,   # [batch, n_cols, N_COL_STATS]
        row_pool:  torch.Tensor,   # [batch, N_ROW_POOL]
    ) -> torch.Tensor:
        """Returns: embedding [batch, embed_dim]"""
        col_repr = self.col_mlp(col_stats)          # [B, n_cols, col_dim]
        ds_repr  = self.deepsets(col_repr)           # [B, col_dim]
        attn_repr = self.attn_pool(col_repr)         # [B, n_seeds*col_dim]
        combined  = torch.cat([ds_repr, attn_repr, row_pool], dim=-1)
        return self.proj(combined)                   # [B, embed_dim]


# ──────────────────────────────────────────────────────────────────
# Dataset feature extraction (combines CPU stats + GPU encoder)
# ──────────────────────────────────────────────────────────────────
def extract_dataset_features(
    X: np.ndarray,
    encoder: Optional[DatasetEncoder] = None,
    device: str = "cpu",
    max_cols_for_encoding: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract (col_stats, row_pool) for a single dataset.
    Used for encoding before feeding into the neural model.

    Parameters
    ----------
    X : np.ndarray  [n_rows, n_cols]
    encoder : DatasetEncoder or None
    device : str
    max_cols_for_encoding : int  pad/truncate to this dimension

    Returns
    -------
    col_stats : np.ndarray  [max_cols_for_encoding, N_COL_STATS]
    row_pool  : np.ndarray  [N_ROW_POOL]
    """
    col_stats_raw = extract_column_stats(X)  # [n_cols, 16]
    row_pool      = extract_row_stats(X)     # [8]

    # Pad or truncate column dimension to fixed size
    n_cols = col_stats_raw.shape[0]
    if n_cols < max_cols_for_encoding:
        pad = np.zeros((max_cols_for_encoding - n_cols, N_COL_STATS), dtype=np.float32)
        col_stats = np.vstack([col_stats_raw, pad])
    else:
        col_stats = col_stats_raw[:max_cols_for_encoding]

    return col_stats, row_pool
