"""
src/composite.py
Composite Usability Scorer — outputs a unified 0–100 score.

Task-conditioned weighting: classification tasks care more about
label noise and class imbalance; regression cares more about outliers
and missingness.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.model import ALL_HEADS, QUALITY_HEADS, DEGRAD_HEADS
from src.encoder import extract_dataset_features
from src.utils import get_logger

logger = get_logger("composite")

# ──────────────────────────────────────────────────────────────────
# Per-task weights for the 6 quality dimensions
# Weights sum to 1.0 per task type.
# ──────────────────────────────────────────────────────────────────
WEIGHTS = {
    "classification": {
        "outliers":           0.10,
        "label_noise":        0.25,  # highest: directly corrupts the learning signal
        "missingness":        0.20,
        "duplicates":         0.10,
        "class_imbalance":    0.25,  # high: drastically affects classifier metrics
        "redundant_features": 0.10,
    },
    "regression": {
        "outliers":           0.25,  # highest: regression is not robust to extremes
        "label_noise":        0.20,
        "missingness":        0.25,  # high: imputation can significantly bias regression
        "duplicates":         0.10,
        "class_imbalance":    0.00,  # N/A for regression
        "redundant_features": 0.20,
    },
}

# Downstream degradation penalty weight (applied on top of quality score)
DEGRAD_PENALTY_WEIGHT = 0.30   # 30% of final score penalised by predicted degradation


# ──────────────────────────────────────────────────────────────────
# Scorer
# ──────────────────────────────────────────────────────────────────
class CompositeUsabilityScorer:
    """
    Converts the 8-dimensional model output into a single 0–100 score
    and individual quality dimension ratings.
    """

    def score(
        self,
        predictions: np.ndarray,        # [N_OUTPUTS=8] from model, values in [0,1]
        task_type:   str = "classification",
        return_breakdown: bool = True,
    ) -> Dict:
        """
        Compute composite usability score.

        Parameters
        ----------
        predictions : np.ndarray [8]
            Model output: first 6 are corruption severity scores (0=clean, 1=severe),
            last 2 are downstream degradation scores [light_drop, heavy_drop].
        task_type : str  "classification" | "regression"
        return_breakdown : bool

        Returns
        -------
        dict with keys:
          usability_score   : float [0, 100]  — higher is better
          quality_subscores : dict {dim: score [0,100]}
          degradation_scores: dict {light_drop, heavy_drop}
          task_type         : str
          grade             : str  A/B/C/D/F
        """
        if task_type not in WEIGHTS:
            task_type = "classification"

        w = WEIGHTS[task_type]

        # Severity predictions [0,1] → invert to get quality (0=bad, 1=clean)
        # Then scale to [0,100]
        quality_raw = np.clip(predictions[:6], 0.0, 1.0)
        quality_score_raw = 1.0 - quality_raw  # [0,1]: higher = better quality

        # Weighted quality score
        weights = np.array([w[dim] for dim in QUALITY_HEADS], dtype=np.float32)
        # Normalise (handles class_imbalance=0 for regression)
        weights = weights / (weights.sum() + 1e-8)

        weighted_quality = float(np.dot(weights, quality_score_raw))  # [0, 1]

        # Downstream degradation penalty (average of light & heavy)
        light_drop = float(np.clip(predictions[6], 0.0, 1.0))
        heavy_drop = float(np.clip(predictions[7], 0.0, 1.0))
        mean_degrad = (light_drop + heavy_drop) / 2.0

        # Final score: quality base × (1 - degradation_penalty)
        penalty     = mean_degrad * DEGRAD_PENALTY_WEIGHT
        final_score = weighted_quality * (1.0 - penalty)
        usability   = float(np.clip(final_score * 100, 0.0, 100.0))

        # Individual quality sub-scores
        quality_subscores = {
            dim: float(round((1.0 - float(quality_raw[i])) * 100, 1))
            for i, dim in enumerate(QUALITY_HEADS)
        }

        grade = _grade(usability)

        result = {
            "usability_score":    round(usability, 1),
            "grade":              grade,
            "task_type":          task_type,
            "degradation_scores": {
                "light_model_drop": round(light_drop * 100, 1),
                "heavy_model_drop": round(heavy_drop * 100, 1),
            },
        }
        if return_breakdown:
            result["quality_subscores"] = quality_subscores
            result["weights_used"]      = {k: round(float(v), 3) for k, v in zip(QUALITY_HEADS, weights)}

        return result

    def remediation_tips(self, predictions: np.ndarray, task_type: str = "classification") -> list:
        """
        Return ranked list of actionable remediation tips based on predicted severities.
        """
        quality_raw = np.clip(predictions[:6], 0.0, 1.0)
        ranked = sorted(enumerate(quality_raw), key=lambda x: -x[1])

        tips_map = {
            "outliers":           "Apply IQR-based or isolation forest outlier removal before training.",
            "label_noise":        "Use confident-learning (e.g., Cleanlab) to identify and re-label noisy samples.",
            "missingness":        "Use multiple imputation (e.g., IterativeImputer) or MICE; report missing fractions per column.",
            "duplicates":         "Run deduplication with hash-based exact matching and fuzzy near-duplicate detection.",
            "class_imbalance":    "Apply SMOTE, class-weighted losses, or stratified sampling to address imbalance.",
            "redundant_features": "Use mutual information ranking or VIF analysis to prune collinear/leaky features.",
        }

        tips = []
        for idx, severity in ranked:
            dim = QUALITY_HEADS[idx]
            if severity > 0.15:
                tips.append({
                    "dimension": dim,
                    "severity":  round(float(severity), 3),
                    "tip":       tips_map[dim],
                    "priority":  "HIGH" if severity > 0.5 else "MEDIUM",
                })
        return tips


def _grade(score: float) -> str:
    if score >= 85:  return "A"
    if score >= 70:  return "B"
    if score >= 55:  return "C"
    if score >= 40:  return "D"
    return "F"


def assess_dataframe(
    df: pd.DataFrame,
    model,
    task_type: str = "classification",
    device: str = "cpu",
) -> Tuple[Dict, np.ndarray, List[Dict]]:
    """
    Robust end-to-end tabular dataset assessment:
    1. Preprocesses both numeric and categorical columns.
    2. Measures empirical data quality metrics (missingness, duplicates, outliers, redundancy, imbalance).
    3. Normalizes features to match model training distribution (zero-mean, unit-variance).
    4. Runs neural forward pass for single-pass degradation estimation.
    5. Fuses empirical evidence with neural degradation to produce a reliable, differentiated usability score.
    """
    # 1. Empirical checks on raw data
    raw_missing = float(df.isnull().values.mean())
    raw_dups = float(df.duplicated().mean())

    # 2. Encode categorical columns
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == "object" or df_clean[col].dtype.name == "category":
            df_clean[col] = LabelEncoder().fit_transform(df_clean[col].astype(str))
    
    X = df_clean.values.astype(np.float32)

    # 3. Empirical outliers & redundancy
    X_num = np.nan_to_num(X, nan=0.0)
    stds = np.std(X_num, axis=0) + 1e-8
    means = np.mean(X_num, axis=0)
    z = np.abs((X_num - means) / stds)
    raw_outliers = float(np.mean(z > 3))

    if X_num.shape[1] > 1:
        corr = np.corrcoef(X_num.T)
        corr = np.nan_to_num(corr)
        upper = np.abs(corr[np.triu_indices(X_num.shape[1], k=1)])
        raw_red = float(np.mean(upper > 0.80))
    else:
        raw_red = 0.0

    # 4. Class imbalance from target (last column)
    target = df.iloc[:, -1]
    val_counts = target.value_counts(normalize=True).values
    raw_imb = float(val_counts[0] - val_counts[-1]) if len(val_counts) > 1 and task_type == "classification" else 0.0

    # 5. Standardize to match training distribution so inputs don't saturate at clip boundary
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    col_stats, row_pool = extract_dataset_features(X_scaled)
    preds = model.predict_numpy(col_stats, row_pool, device)

    # 6. Fused predictions: blend model predictions with directly observed table properties
    fused_preds = preds.copy()
    fused_preds[0] = float(np.clip(0.4 * preds[0] + 0.6 * min(1.0, raw_outliers * 10), 0.0, 1.0))
    fused_preds[2] = float(np.clip(0.3 * preds[2] + 0.7 * min(1.0, raw_missing * 3), 0.0, 1.0))
    fused_preds[3] = float(np.clip(0.3 * preds[3] + 0.7 * min(1.0, raw_dups * 3), 0.0, 1.0))
    fused_preds[4] = float(np.clip(0.4 * preds[4] + 0.6 * raw_imb, 0.0, 1.0)) if task_type == "classification" else 0.0
    fused_preds[5] = float(np.clip(0.4 * preds[5] + 0.6 * min(1.0, raw_red * 3), 0.0, 1.0))

    # ── DEEP BEYOND-PANDAS DIAGNOSTICS ──
    deep_diag = {}

    # 1. Target Leakage / Spurious Proxy Features
    if X_num.shape[1] > 1:
        target_vec = X_num[:, -1]
        leakage_detected = []
        for i in range(X_num.shape[1] - 1):
            col_std = np.std(X_num[:, i])
            target_std = np.std(target_vec)
            if col_std > 1e-8 and target_std > 1e-8:
                r_val = abs(float(np.corrcoef(X_num[:, i], target_vec)[0, 1]))
                if r_val > 0.90:
                    leakage_detected.append({"feature": str(df.columns[i]), "correlation": round(r_val, 3)})
        deep_diag["target_leakage"] = leakage_detected

    # 2. Multivariate Outliers (Joint distribution Mahalanobis covariance distance)
    try:
        sample_size = min(len(X_scaled), 1000)
        sample_X = X_scaled[:sample_size]
        cov = np.cov(sample_X.T) + 1e-4 * np.eye(sample_X.shape[1])
        inv_cov = np.linalg.pinv(cov)
        mahal_dist = np.sum((sample_X @ inv_cov) * sample_X, axis=1)
        threshold = np.percentile(mahal_dist, 98)
        multi_outlier_idx = np.where(mahal_dist > threshold)[0].tolist()
        deep_diag["multivariate_outliers"] = {
            "count": len(multi_outlier_idx),
            "sample_indices": multi_outlier_idx[:10],
            "severity": round(float(np.mean(mahal_dist > threshold)), 3)
        }
    except Exception:
        deep_diag["multivariate_outliers"] = {"count": 0, "sample_indices": [], "severity": 0.0}

    # 3. Near-Duplicate Fuzzy Leakage Rows (Vectorized Cosine Similarity > 0.99)
    sample_n = min(len(X_scaled), 1000)
    try:
        if sample_n > 5:
            sample_sub = X_scaled[:sample_n]
            norms = np.linalg.norm(sample_sub, axis=1, keepdims=True) + 1e-8
            sub_norm = sample_sub / norms
            sim_mat = sub_norm @ sub_norm.T
            np.fill_diagonal(sim_mat, 0.0)
            near_dup_count = int(np.sum(sim_mat > 0.99) // 2)
            deep_diag["fuzzy_near_duplicates"] = {
                "count": near_dup_count,
                "note": "Rows differing only by minor numerical jitter (bypasses exact .duplicated())"
            }
        else:
            deep_diag["fuzzy_near_duplicates"] = {"count": 0, "note": "Insufficient rows"}
    except Exception:
        deep_diag["fuzzy_near_duplicates"] = {"count": 0, "note": "N/A"}

    # 4. Label Inconsistency / Contradiction Index (Vectorized Neighborhood Check)
    if task_type == "classification" and len(df) > 10:
        try:
            sample_sub = X_scaled[:sample_n]
            sq_norms = np.sum(sample_sub**2, axis=1, keepdims=True)
            dist_mat = sq_norms + sq_norms.T - 2.0 * (sample_sub @ sample_sub.T)
            np.fill_diagonal(dist_mat, np.inf)
            k = min(5, sample_n - 1)
            top_k_idx = np.argpartition(dist_mat, kth=k, axis=1)[:, :k]
            target_labels = np.array(target.iloc[:sample_n])
            neighbor_labels = target_labels[top_k_idx]
            mismatch_count = np.sum(neighbor_labels != target_labels[:, None], axis=1)
            inconsistent_samples = int(np.sum(mismatch_count >= (k // 2 + 1)))
            deep_diag["label_inconsistency"] = {
                "count": inconsistent_samples,
                "percentage": round(inconsistent_samples / sample_n * 100, 2),
                "note": "Samples whose feature neighbors belong to opposing classes (Cleanlab error candidate)"
            }
        except Exception:
            deep_diag["label_inconsistency"] = {"count": 0, "percentage": 0.0, "note": "N/A"}

    scorer = CompositeUsabilityScorer()
    report = scorer.score(fused_preds, task_type=task_type, return_breakdown=True)
    report["deep_diagnostics"] = deep_diag
    tips = scorer.remediation_tips(fused_preds, task_type=task_type)

    return report, fused_preds, tips
