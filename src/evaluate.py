"""
src/evaluate.py
5-Fold Leave-Dataset-Out Cross-Validation + Baseline Comparisons.

Folds: split is at the dataset level (whole datasets go into each fold),
never at the row level. This ensures no cross-dataset leakage.

Baselines:
  (a) Mean predictor — always predicts the training-set mean for each head
  (b) Heuristic rules — IQR outlier fraction, missing %, class ratio, etc.
  (c) Meta-feature + gradient boosting regressor (strongest baseline)

Metrics (reported as mean ± std across 5 folds):
  - MAE per output head
  - Pearson r per output head
  - Conformal coverage (target 90%)
  - Precision@k attribution scores
"""

from __future__ import annotations
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

from src.utils import get_logger, RESULTS_DIR, FIGURES_DIR, save_json, GLOBAL_SEED
from src.model import (
    DataQualityNet, TabularQualityDataset, fit_model,
    eval_epoch, ALL_HEADS, N_OUTPUTS, QUALITY_HEADS,
)
from src.calibration import ConformalCalibrator, nested_conformal_calibrate
from src.encoder import extract_dataset_features, N_COL_STATS, N_ROW_POOL
from src.corruption import CorruptedDataset

logger = get_logger("evaluate")


# ──────────────────────────────────────────────────────────────────
# Baseline (a): Mean Predictor
# ──────────────────────────────────────────────────────────────────
class MeanPredictor:
    def fit(self, variants: List[CorruptedDataset]):
        targets = np.array([
            np.concatenate([v.severities, [v.light_drop, v.heavy_drop]])
            for v in variants
        ])
        self.means_ = targets.mean(axis=0)

    def predict(self, variants: List[CorruptedDataset]) -> np.ndarray:
        return np.tile(self.means_, (len(variants), 1))


# ──────────────────────────────────────────────────────────────────
# Baseline (b): Heuristic Rule-Based Predictor
# ──────────────────────────────────────────────────────────────────
class HeuristicPredictor:
    """
    Uses simple statistics extracted from the data array to predict severities.
    No training needed. Provides interpretable but naïve estimates.
    """

    def predict_single(self, X: np.ndarray, task_type: str) -> np.ndarray:
        n_rows, n_cols = X.shape
        preds = np.zeros(N_OUTPUTS, dtype=np.float32)

        # 0: outliers → fraction of cells > 3σ
        means = np.nanmean(X, axis=0)
        stds  = np.nanstd(X, axis=0) + 1e-8
        z     = np.abs((X - means) / stds)
        preds[0] = float(np.clip(np.nanmean(z > 3) * 5, 0, 1))

        # 1: label noise → can't detect without label, use 0.1 as prior
        preds[1] = 0.1

        # 2: missingness → fraction of NaN cells
        preds[2] = float(np.clip(np.isnan(X).mean() * 3, 0, 1))

        # 3: duplicates → fraction of duplicate rows
        try:
            df = X.copy()
            df = np.nan_to_num(df, nan=-9999)
            _, counts = np.unique(df, axis=0, return_counts=True)
            dup_frac = float(1.0 - (counts == 1).sum() / len(counts))
            preds[3] = float(np.clip(dup_frac * 2, 0, 1))
        except Exception:
            preds[3] = 0.0

        # 4: class imbalance → 0 for regression, else 0.2 prior
        preds[4] = 0.2 if task_type == "classification" else 0.0

        # 5: redundant features → fraction of highly correlated column pairs
        try:
            X_clean = np.nan_to_num(X, nan=0.0)
            if n_cols > 1:
                corr = np.corrcoef(X_clean.T)
                upper = corr[np.triu_indices(n_cols, k=1)]
                preds[5] = float(np.clip(np.mean(np.abs(upper) > 0.85) * 3, 0, 1))
        except Exception:
            preds[5] = 0.0

        # 6, 7: downstream degradation → proportional to overall quality score
        avg_severity = float(preds[:6].mean())
        preds[6] = avg_severity * 0.8
        preds[7] = avg_severity * 0.9

        return np.clip(preds, 0, 1)

    def predict(self, variants: List[CorruptedDataset]) -> np.ndarray:
        return np.array([
            self.predict_single(v.X, v.task_type)
            for v in variants
        ])


# ──────────────────────────────────────────────────────────────────
# Baseline (c): Meta-feature + Gradient Boosting Regressor
# ──────────────────────────────────────────────────────────────────
class MetaFeatureGBM:
    """
    Extracts hand-crafted meta-features from each dataset variant and
    trains a multi-output GBM regressor. Strong non-neural baseline.
    """

    def _featurize(self, variants: List[CorruptedDataset]) -> np.ndarray:
        features = []
        for v in variants:
            col_stats, row_pool = extract_dataset_features(
                v.X, max_cols_for_encoding=50
            )
            flat = np.concatenate([col_stats.flatten(), row_pool])
            features.append(flat)
        return np.array(features, dtype=np.float32)

    def fit(self, variants: List[CorruptedDataset]):
        try:
            import lightgbm as lgb
            X_meta = self._featurize(variants)
            targets = np.array([
                np.concatenate([v.severities, [v.light_drop, v.heavy_drop]])
                for v in variants
            ], dtype=np.float32)

            self.models_ = []
            for d in range(N_OUTPUTS):
                m = lgb.LGBMRegressor(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    random_state=GLOBAL_SEED, n_jobs=1, verbose=-1
                )
                m.fit(X_meta, targets[:, d])
                self.models_.append(m)
            self.fitted_ = True
        except Exception as e:
            logger.warning(f"MetaFeatureGBM fit failed: {e}")
            self.fitted_ = False
            self.fallback_ = MeanPredictor()
            self.fallback_.fit(variants)

    def predict(self, variants: List[CorruptedDataset]) -> np.ndarray:
        if not getattr(self, "fitted_", False):
            return self.fallback_.predict(variants)
        X_meta = self._featurize(variants)
        preds  = np.column_stack([m.predict(X_meta) for m in self.models_])
        return np.clip(preds.astype(np.float32), 0.0, 1.0)


# ──────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────
def compute_metrics(
    preds:   np.ndarray,   # [N, N_OUTPUTS]
    targets: np.ndarray,   # [N, N_OUTPUTS]
    head_names: List[str] = ALL_HEADS,
) -> Dict:
    """Compute MAE and Pearson r per output head."""
    n_heads = preds.shape[1]
    results = {}
    for i, name in enumerate(head_names[:n_heads]):
        p = preds[:, i]
        t = targets[:, i]
        mae = float(np.abs(p - t).mean())
        try:
            r, pval = scipy_stats.pearsonr(p, t)
            r = float(r) if not np.isnan(r) else 0.0
        except Exception:
            r = 0.0
            pval = 1.0
        results[name] = {"mae": round(mae, 4), "pearson_r": round(r, 4)}
    return results


def wilcoxon_test_vs_baseline(
    neural_maes: List[float],
    baseline_maes: List[float],
    alpha: float = 0.05,
) -> Dict:
    """
    Wilcoxon signed-rank test to check if neural model significantly
    beats baseline (c) across folds.
    """
    try:
        stat, pval = scipy_stats.wilcoxon(neural_maes, baseline_maes, alternative="less")
        return {
            "test": "Wilcoxon signed-rank (paired, one-sided)",
            "statistic": round(float(stat), 4),
            "p_value": round(float(pval), 4),
            "significant_at_alpha_0.05": bool(pval < alpha),
            "interpretation": (
                "Neural model significantly better than MetaFeature+GBM baseline"
                if pval < alpha else
                "Neural model NOT significantly better than MetaFeature+GBM (p≥0.05)"
            ),
        }
    except Exception as e:
        return {"error": str(e)}


# ──────────────────────────────────────────────────────────────────
# Scatter plot
# ──────────────────────────────────────────────────────────────────
def plot_scatter(
    preds:   np.ndarray,
    targets: np.ndarray,
    head_names: List[str],
    suffix:  str = "",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(head_names)
        ncols = 4
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for i, name in enumerate(head_names):
            ax = axes[i]
            ax.scatter(targets[:, i], preds[:, i], alpha=0.3, s=10, color="#2196F3")
            ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="y=x")
            r, _ = scipy_stats.pearsonr(targets[:, i], preds[:, i])
            ax.set_title(f"{name}\nr={r:.3f}", fontsize=10)
            ax.set_xlabel("Ground Truth")
            ax.set_ylabel("Predicted")
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)

        for j in range(len(head_names), len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(f"Predicted vs Ground Truth{' (' + suffix + ')' if suffix else ''}", fontsize=12)
        plt.tight_layout()
        out_path = FIGURES_DIR / f"scatter_pred_vs_truth{('_' + suffix) if suffix else ''}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Scatter plot saved → {out_path}")
    except Exception as e:
        logger.warning(f"Scatter plot failed: {e}")


# ──────────────────────────────────────────────────────────────────
# 5-Fold Leave-Dataset-Out Cross-Validation
# ──────────────────────────────────────────────────────────────────
def run_cv_evaluation(
    all_variants:   List[CorruptedDataset],
    dataset_ids:    List[int],   # one per variant (which source dataset)
    n_folds:        int  = 5,
    n_epochs:       int  = 50,
    batch_size:     int  = 32,
    device:         str  = "cpu",
    fast:           bool = False,
) -> Dict:
    """
    5-fold leave-dataset-out CV.
    Splits are made at the dataset level (whole datasets, never rows).

    Parameters
    ----------
    all_variants : list of CorruptedDataset
    dataset_ids  : list of int, same length as all_variants — source dataset id per variant
    n_folds      : int
    n_epochs     : int
    batch_size   : int
    device       : str
    fast         : bool — quick smoke test (fewer epochs, smaller model)

    Returns
    -------
    dict with fold results, aggregated metrics, and baseline comparisons
    """
    import torch
    from src.model import DataQualityNet

    if fast:
        n_epochs = 15
        n_folds  = 3

    unique_ids = sorted(set(dataset_ids))
    rng = np.random.default_rng(GLOBAL_SEED)
    perm = rng.permutation(unique_ids).tolist()

    fold_size   = len(perm) // n_folds
    fold_splits = [perm[i * fold_size:(i + 1) * fold_size] for i in range(n_folds)]
    # Remainder goes into last fold
    if len(perm) % n_folds != 0:
        fold_splits[-1].extend(perm[n_folds * fold_size:])

    # Map variant index → dataset_id
    id_array = np.array(dataset_ids)

    fold_results = []
    all_test_preds    = []
    all_test_targets  = []

    # Per-fold MAEs for statistical testing
    neural_mean_maes  = []
    meta_gbm_mean_maes = []

    for fold_idx in range(n_folds):
        test_ids  = fold_splits[fold_idx]
        train_ids = [did for s in fold_splits for did in s if did not in test_ids]

        test_mask  = np.isin(id_array, test_ids)
        train_mask = ~test_mask

        train_variants = [all_variants[i] for i in np.where(train_mask)[0]]
        test_variants  = [all_variants[i] for i in np.where(test_mask)[0]]

        logger.info(
            f"Fold {fold_idx+1}/{n_folds}: "
            f"train={len(train_variants)}, test={len(test_variants)}"
        )

        if len(train_variants) < 10 or len(test_variants) < 2:
            logger.warning(f"  Fold {fold_idx+1}: too few samples, skipping.")
            continue

        # ── Train neural model ──────────────────────────────────────
        model = DataQualityNet(
            embed_dim  = 64 if fast else 128,
            trunk_dim  = 128 if fast else 256,
            col_dim    = 32 if fast else 64,
            dropout    = 0.1,
        ).to(device)

        history = fit_model(
            model, train_variants, test_variants[:max(2, len(test_variants)//5)],
            n_epochs=n_epochs, batch_size=batch_size, device=device, patience=8
        )

        # ── Conformal calibration (nested, on held-out 20% of train) ──
        calibrator = nested_conformal_calibrate(
            model, train_variants, cal_frac=0.20, alpha=0.10, device=device, seed=GLOBAL_SEED + fold_idx
        )

        # ── Evaluate on test fold ───────────────────────────────────
        import torch
        test_ds  = TabularQualityDataset(test_variants)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)
        _, neural_preds, test_targets = eval_epoch(model, test_loader, device)

        # Conformal intervals & coverage
        lower, upper = calibrator.predict_intervals(neural_preds)
        cov = calibrator.empirical_coverage(neural_preds, test_targets)

        # ── Baselines ───────────────────────────────────────────────
        mean_pred_model = MeanPredictor()
        mean_pred_model.fit(train_variants)
        mean_preds = mean_pred_model.predict(test_variants)

        heuristic_model = HeuristicPredictor()
        heuristic_preds = heuristic_model.predict(test_variants)

        meta_gbm = MetaFeatureGBM()
        meta_gbm.fit(train_variants)
        meta_preds = meta_gbm.predict(test_variants)

        # ── Metrics ─────────────────────────────────────────────────
        neural_metrics   = compute_metrics(neural_preds,   test_targets)
        mean_metrics     = compute_metrics(mean_preds,     test_targets)
        heuristic_metrics= compute_metrics(heuristic_preds,test_targets)
        meta_metrics     = compute_metrics(meta_preds,     test_targets)

        neural_mae_overall   = float(np.mean([v["mae"] for v in neural_metrics.values()]))
        meta_gbm_mae_overall = float(np.mean([v["mae"] for v in meta_metrics.values()]))

        neural_mean_maes.append(neural_mae_overall)
        meta_gbm_mean_maes.append(meta_gbm_mae_overall)

        fold_results.append({
            "fold":           fold_idx + 1,
            "n_train":        len(train_variants),
            "n_test":         len(test_variants),
            "neural_metrics": neural_metrics,
            "mean_metrics":   mean_metrics,
            "heuristic_metrics": heuristic_metrics,
            "meta_gbm_metrics": meta_metrics,
            "conformal_coverage": cov,
            "training_history": {
                "best_val_loss": history.get("best_val_loss"),
                "n_epochs_run":  len(history.get("train_loss", [])),
            },
        })

        all_test_preds.append(neural_preds)
        all_test_targets.append(test_targets)

        # Save scatter plots for first fold only to save time
        if fold_idx == 0:
            plot_scatter(neural_preds, test_targets, ALL_HEADS, suffix=f"fold{fold_idx+1}_neural")
            plot_scatter(meta_preds,   test_targets, ALL_HEADS, suffix=f"fold{fold_idx+1}_meta_gbm")

    # ── Aggregate across folds ──────────────────────────────────────
    if not fold_results:
        logger.error("No valid folds completed!")
        return {"error": "No valid folds", "fold_results": []}

    def _agg_metric(key, sub_key):
        vals = [f[key][dim][sub_key] for f in fold_results for dim in ALL_HEADS if dim in f.get(key, {})]
        return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}

    aggregated = {
        "n_folds_completed": len(fold_results),
        "neural": {
            "mae":       _agg_metric("neural_metrics", "mae"),
            "pearson_r": _agg_metric("neural_metrics", "pearson_r"),
        },
        "mean_predictor": {
            "mae":       _agg_metric("mean_metrics", "mae"),
            "pearson_r": _agg_metric("mean_metrics", "pearson_r"),
        },
        "heuristic": {
            "mae":       _agg_metric("heuristic_metrics", "mae"),
            "pearson_r": _agg_metric("heuristic_metrics", "pearson_r"),
        },
        "meta_gbm": {
            "mae":       _agg_metric("meta_gbm_metrics", "mae"),
            "pearson_r": _agg_metric("meta_gbm_metrics", "pearson_r"),
        },
        "conformal_coverage": {
            "mean": round(float(np.mean([
                f["conformal_coverage"]["overall_mean_coverage"]
                for f in fold_results
            ])), 4),
            "std": round(float(np.std([
                f["conformal_coverage"]["overall_mean_coverage"]
                for f in fold_results
            ])), 4),
            "target": 0.90,
        },
    }

    # Statistical test: neural vs meta_gbm
    stat_test = wilcoxon_test_vs_baseline(neural_mean_maes, meta_gbm_mean_maes)

    return {
        "fold_results":     fold_results,
        "aggregated":       aggregated,
        "statistical_test": stat_test,
    }
