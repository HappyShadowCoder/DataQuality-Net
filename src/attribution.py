"""
src/attribution.py
Column and Row-Level Attribution + Precision@k Evaluation.

Methods:
  1. Occlusion-based column attribution — zero out one column's stats
     and measure prediction change (model-agnostic, exact).
  2. Gradient-based row attribution — gradient of output w.r.t. row
     stats representation (approximation since rows are pooled).
  3. Precision@k evaluation against ground-truth corruption masks.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.encoder import (
    extract_column_stats, extract_row_stats,
    N_COL_STATS, N_ROW_POOL,
)
from src.model import TabularQualityDataset
from src.model import DataQualityNet, N_OUTPUTS, QUALITY_HEADS
from src.corruption import CorruptionMasks
from src.utils import get_logger

logger = get_logger("attribution")


# ──────────────────────────────────────────────────────────────────
# Column-level occlusion attribution
# ──────────────────────────────────────────────────────────────────
def column_occlusion_scores(
    model:     DataQualityNet,
    X:         np.ndarray,         # [n_rows, n_cols]  raw feature array
    device:    str = "cpu",
    max_cols:  int = 200,
) -> np.ndarray:
    """
    Compute attribution score per original column using occlusion.

    Strategy:
      For each column c, zero out col_stats[c] (all 16 stats → 0).
      Measure the mean absolute change across all 8 output heads.
      Columns with large change have high attribution.

    Returns
    -------
    attr_scores : np.ndarray [n_cols]  — higher = more attributed
    """
    from src.encoder import extract_dataset_features

    col_stats_full, row_pool = extract_dataset_features(X, max_cols_for_encoding=max_cols)
    n_cols_orig = min(X.shape[1], max_cols)

    # Baseline prediction
    model.eval()
    with torch.no_grad():
        cs_t  = torch.from_numpy(col_stats_full).unsqueeze(0).to(device)
        rp_t  = torch.from_numpy(row_pool).unsqueeze(0).to(device)
        base_pred = model(cs_t, rp_t).cpu().numpy().squeeze(0)   # [N_OUTPUTS]

    attr_scores = np.zeros(n_cols_orig, dtype=np.float32)

    for c in range(n_cols_orig):
        cs_occluded = col_stats_full.copy()
        cs_occluded[c] = 0.0   # zero out the column stats

        with torch.no_grad():
            cs_t = torch.from_numpy(cs_occluded).unsqueeze(0).to(device)
            occ_pred = model(cs_t, rp_t).cpu().numpy().squeeze(0)

        attr_scores[c] = float(np.abs(occ_pred - base_pred).mean())

    return attr_scores


# ──────────────────────────────────────────────────────────────────
# Row-level gradient attribution
# ──────────────────────────────────────────────────────────────────
def row_gradient_scores(
    model:    DataQualityNet,
    X:        np.ndarray,       # [n_rows, n_cols]
    device:   str = "cpu",
    max_rows: int = 5000,
    max_cols: int = 200,
    target_head: int = 0,       # which output head to attribute toward
) -> np.ndarray:
    """
    Approximate row-level attribution via gradient of col_stats w.r.t.
    each column's contribution (since row pooling aggregates all rows,
    true per-row gradient is approximated by leave-one-out on a subsample).

    For computational tractability:
      - Subsample max_rows rows
      - Compute gradient of sum(output[target_head]) w.r.t. col_stats
      - Use column variance explained per row as proxy for row influence

    Returns
    -------
    row_scores : np.ndarray [n_rows]  — approximate attribution per row
    """
    from src.encoder import extract_column_stats, extract_row_stats

    n_rows = X.shape[0]
    if n_rows > max_rows:
        idx = np.random.RandomState(42).choice(n_rows, max_rows, replace=False)
        X_sub = X[idx]
        row_idx = idx
    else:
        X_sub = X
        row_idx = np.arange(n_rows)

    # Use gradient of col_stats tensor w.r.t. output
    col_stats_raw = extract_column_stats(X_sub, max_rows=max_rows)
    row_pool_val  = extract_row_stats(X_sub, max_rows=max_rows)

    n_cols = min(col_stats_raw.shape[0], max_cols)
    col_stats_full = np.zeros((max_cols, N_COL_STATS), dtype=np.float32)
    col_stats_full[:n_cols] = col_stats_raw[:n_cols]

    cs_t = torch.from_numpy(col_stats_full).unsqueeze(0).float().to(device)
    rp_t = torch.from_numpy(row_pool_val).unsqueeze(0).float().to(device)
    cs_t.requires_grad_(True)

    model.eval()
    output = model(cs_t, rp_t)[:, target_head].sum()
    output.backward()

    col_grads = cs_t.grad.squeeze(0).detach().cpu().numpy()  # [max_cols, 16]
    col_importance = np.abs(col_grads).sum(axis=1)[:n_cols]  # [n_cols]

    # Row-level score: weighted sum of column importances by row's distance
    # from column mean (outlier rows deviate more from column mean)
    col_means = np.nanmean(X_sub, axis=0)
    col_stds  = np.nanstd(X_sub, axis=0) + 1e-8
    # z-scores of each row
    z = np.abs((X_sub[:, :n_cols] - col_means[:n_cols]) / col_stds[:n_cols])
    z = np.nan_to_num(z, nan=0.0)

    # Row attribution = weighted average of z-scores by column importance
    w = col_importance / (col_importance.sum() + 1e-8)
    row_scores_sub = (z * w).sum(axis=1)  # [n_rows_sub]

    # Map back to full row array
    row_scores = np.zeros(n_rows, dtype=np.float32)
    row_scores[row_idx] = row_scores_sub.astype(np.float32)

    return row_scores


# ──────────────────────────────────────────────────────────────────
# Precision@k evaluation
# ──────────────────────────────────────────────────────────────────
def precision_at_k(
    scores:   np.ndarray,  # [N] attribution scores (higher = more suspicious)
    mask:     np.ndarray,  # [N] bool ground-truth mask (True = corrupted)
    k_fracs:  List[float] = [0.05, 0.10, 0.20],
) -> Dict[str, float]:
    """
    Compute Precision@k: fraction of ground-truth corrupted items among
    the top-k% highest-scored items.

    Parameters
    ----------
    scores : [N]  — attribution scores
    mask   : [N]  — ground-truth boolean mask
    k_fracs : list of fractions to evaluate at

    Returns
    -------
    dict: {"P@5%": ..., "P@10%": ..., "P@20%": ...}
    """
    n = len(scores)
    if n == 0 or mask.sum() == 0:
        return {f"P@{int(k*100)}%": 0.0 for k in k_fracs}

    order  = np.argsort(-scores)  # descending
    sorted_mask = mask[order]

    result = {}
    for k_frac in k_fracs:
        k   = max(1, int(n * k_frac))
        prec = float(sorted_mask[:k].mean())
        result[f"P@{int(k_frac*100)}%"] = prec

    return result


# ──────────────────────────────────────────────────────────────────
# Attribution evaluator — runs over a list of corrupted variants
# ──────────────────────────────────────────────────────────────────
class AttributionEvaluator:

    def __init__(
        self,
        model:  DataQualityNet,
        device: str = "cpu",
    ):
        self.model  = model
        self.device = device

    def evaluate(
        self,
        variants:  list,   # list of CorruptedDataset
        k_fracs:   List[float] = [0.05, 0.10, 0.20],
        max_variants: int = 100,
    ) -> Dict:
        """
        For each variant:
          - Compute column occlusion scores
          - Evaluate Precision@k against redundant_cols mask
          - Compute row scores and evaluate against outlier_rows mask

        Returns aggregated Precision@k statistics.
        """
        col_pk_records = []
        row_pk_records = []

        for i, v in enumerate(variants[:max_variants]):
            if i % 20 == 0:
                logger.debug(f"  Attribution eval {i}/{min(len(variants), max_variants)} …")

            try:
                # --- Column attribution ---
                col_attr = column_occlusion_scores(
                    self.model, v.X, self.device
                )
                redundant_mask = v.masks.redundant_cols[:len(col_attr)]
                if len(redundant_mask) == len(col_attr) and redundant_mask.sum() > 0:
                    pk = precision_at_k(col_attr, redundant_mask.astype(bool), k_fracs)
                    col_pk_records.append(pk)

                # --- Row attribution (targeting outlier head = index 0) ---
                row_attr = row_gradient_scores(
                    self.model, v.X, self.device, target_head=0
                )
                outlier_mask = v.masks.outlier_rows[:len(row_attr)]
                if len(outlier_mask) == len(row_attr) and outlier_mask.sum() > 0:
                    pk = precision_at_k(row_attr, outlier_mask.astype(bool), k_fracs)
                    row_pk_records.append(pk)

            except Exception as e:
                logger.debug(f"  Attribution failed for variant {i}: {e}")
                continue

        def _agg(records):
            if not records:
                return {}
            keys = records[0].keys()
            return {
                k: {
                    "mean": float(np.mean([r[k] for r in records])),
                    "std":  float(np.std([r[k] for r in records])),
                }
                for k in keys
            }

        return {
            "column_attribution": {
                "metric": "Precision@k (redundant/leaky features)",
                "n_evaluated": len(col_pk_records),
                "results": _agg(col_pk_records),
            },
            "row_attribution": {
                "metric": "Precision@k (outlier rows)",
                "n_evaluated": len(row_pk_records),
                "results": _agg(row_pk_records),
            },
        }
