"""
src/corruption.py
Synthetic Corruption Engine + Downstream Degradation Labeler.

Injects up to 6 corruption types into clean tabular datasets:
  1. outliers         – value shifts and scaling anomalies
  2. label_noise      – random flips (classification) / Gaussian (regression)
  3. missingness      – MCAR, MAR, MNAR mechanisms
  4. duplicates       – duplicate row injection
  5. class_imbalance  – subsampling minority/majority classes
  6. redundant_features – adding leaky/noisy copies of existing columns

Outputs:
  - Corrupted (X, y)
  - Ground-truth severity vector s ∈ [0,1]^6
  - Ground-truth corruption masks (row & column) for Precision@k evaluation
  - Downstream degradation scores (light & heavy model relative performance drop)
"""

from __future__ import annotations
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from src.utils import get_logger, GLOBAL_SEED

warnings.filterwarnings("ignore")
logger = get_logger("corruption")

# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────
CORRUPTION_DIMS = [
    "outliers",
    "label_noise",
    "missingness",
    "duplicates",
    "class_imbalance",
    "redundant_features",
]
N_DIMS = len(CORRUPTION_DIMS)
HEAVY_MODEL_TIME_LIMIT_S = 30  # per-variant wall-clock cap for heavy model training


# ──────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────
@dataclass
class CorruptionMasks:
    """Ground-truth masks identifying corrupted rows/columns."""
    outlier_rows:       np.ndarray  # bool [n_rows]
    label_noise_rows:   np.ndarray  # bool [n_rows]
    missing_mask:       np.ndarray  # bool [n_rows, n_cols]
    duplicate_rows:     np.ndarray  # bool [n_rows] – True = injected duplicate
    imbalance_kept:     np.ndarray  # bool [n_rows] – True = row kept
    redundant_cols:     np.ndarray  # bool [n_cols_output] – True = redundant


@dataclass
class CorruptedDataset:
    """A single corrupted variant of a clean dataset."""
    X:            np.ndarray          # corrupted features
    y:            np.ndarray          # (possibly corrupted) labels
    severities:   np.ndarray          # float32 [6] ground-truth severities
    masks:        CorruptionMasks
    task_type:    str
    n_original_rows: int
    n_original_cols: int
    # Downstream degradation scores (filled after labelling)
    light_drop:   float = 0.0         # relative perf drop, light model
    heavy_drop:   float = 0.0         # relative perf drop, heavy model


# ──────────────────────────────────────────────────────────────────
# Individual corruption functions
# ──────────────────────────────────────────────────────────────────
def _inject_outliers(
    X: np.ndarray, severity: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replace severity-fraction of cells with extreme values
    (shift by k*std or scale by large factor).
    Returns modified X and boolean row mask of affected rows.
    """
    X = X.copy()
    n_rows, n_cols = X.shape
    row_mask = np.zeros(n_rows, dtype=bool)
    if severity < 1e-6:
        return X, row_mask

    n_affected = max(1, int(n_rows * severity * 0.3))
    rows = rng.choice(n_rows, n_affected, replace=False)
    row_mask[rows] = True

    stds = X.std(axis=0) + 1e-8
    for r in rows:
        cols = rng.choice(n_cols, max(1, int(n_cols * 0.4)), replace=False)
        for c in cols:
            if rng.random() < 0.5:
                X[r, c] = X[r, c] + rng.uniform(5, 10) * stds[c] * rng.choice([-1, 1])
            else:
                X[r, c] = X[r, c] * rng.uniform(8, 20)

    return X, row_mask


def _inject_label_noise(
    y: np.ndarray, task_type: str, severity: float, rng: np.random.Generator, n_classes: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classification: randomly flip labels.
    Regression: add Gaussian noise proportional to target std.
    Returns modified y and boolean row mask of affected rows.
    """
    y = y.copy()
    n_rows = len(y)
    row_mask = np.zeros(n_rows, dtype=bool)
    if severity < 1e-6:
        return y, row_mask

    n_affected = max(1, int(n_rows * severity * 0.5))
    rows = rng.choice(n_rows, n_affected, replace=False)
    row_mask[rows] = True

    if task_type == "classification":
        unique_classes = np.unique(y)
        for r in rows:
            others = unique_classes[unique_classes != y[r]]
            if len(others) > 0:
                y[r] = rng.choice(others)
    else:
        std = float(np.std(y)) + 1e-8
        noise = rng.normal(0, severity * 3 * std, size=n_affected)
        y[rows] += noise

    return y, row_mask


def _inject_missingness(
    X: np.ndarray,
    severity: float,
    mechanism: str,  # "MCAR", "MAR", "MNAR"
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Introduce NaN values under MCAR/MAR/MNAR mechanisms.
    Returns modified X and boolean [n_rows, n_cols] mask of missing cells.
    """
    X = X.copy()
    n_rows, n_cols = X.shape
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    if severity < 1e-6:
        return X, mask

    miss_prob = severity * 0.4  # max 40% cells missing at severity=1.0

    if mechanism == "MCAR":
        missing = rng.random((n_rows, n_cols)) < miss_prob

    elif mechanism == "MAR":
        # Missingness in half the columns depends on values of the other half
        missing = np.zeros((n_rows, n_cols), dtype=bool)
        obs_cols = rng.choice(n_cols, max(1, n_cols // 2), replace=False)
        mis_cols = np.setdiff1d(np.arange(n_cols), obs_cols)
        if len(mis_cols) > 0:
            # High-value rows in obs_cols → more likely to be missing
            ref = X[:, obs_cols[0]] if len(obs_cols) > 0 else X[:, 0]
            prob_row = miss_prob * (0.5 + (ref - ref.min()) / (ref.max() - ref.min() + 1e-8))
            for c in mis_cols:
                missing[:, c] = rng.random(n_rows) < prob_row

    elif mechanism == "MNAR":
        # Values that are extreme (high) are more likely to be missing
        missing = np.zeros((n_rows, n_cols), dtype=bool)
        for c in range(n_cols):
            col = X[:, c]
            threshold = np.percentile(col, (1 - miss_prob) * 100)
            prob = np.where(col >= threshold, miss_prob * 2, miss_prob * 0.2)
            prob = np.clip(prob, 0, 1)
            missing[:, c] = rng.random(n_rows) < prob

    else:
        missing = rng.random((n_rows, n_cols)) < miss_prob

    X[missing] = np.nan
    mask[missing] = True
    return X, mask


def _inject_duplicates(
    X: np.ndarray, y: np.ndarray, severity: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inject duplicate rows. Returns (X_new, y_new, dup_mask[n_rows_new]).
    dup_mask[i] = True means row i is an injected duplicate.
    """
    if severity < 1e-6:
        n = len(X)
        return X.copy(), y.copy(), np.zeros(n, dtype=bool)

    n_dups = max(1, int(len(X) * severity * 0.3))
    dup_idx = rng.choice(len(X), n_dups, replace=True)
    X_dup = X[dup_idx]
    y_dup = y[dup_idx]

    X_new = np.vstack([X, X_dup])
    y_new = np.concatenate([y, y_dup])
    dup_mask = np.zeros(len(X_new), dtype=bool)
    dup_mask[len(X):] = True

    # Shuffle
    perm = rng.permutation(len(X_new))
    return X_new[perm], y_new[perm], dup_mask[perm]


def _inject_class_imbalance(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    severity: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Classification only: subsample minority classes to create imbalance.
    For regression: no-op (severity forced to 0).
    Returns (X_kept, y_kept, kept_mask[original_n_rows]).
    """
    n = len(X)
    kept_mask = np.ones(n, dtype=bool)

    if task_type != "classification" or severity < 1e-6:
        return X.copy(), y.copy(), kept_mask

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X.copy(), y.copy(), kept_mask

    # Keep majority class intact; subsample others
    majority_cls = classes[np.argmax(counts)]
    keep_indices = []
    for cls in classes:
        idx = np.where(y == cls)[0]
        if cls == majority_cls:
            keep_indices.extend(idx.tolist())
        else:
            keep_n = max(1, int(len(idx) * (1.0 - severity * 0.8)))
            keep_indices.extend(rng.choice(idx, keep_n, replace=False).tolist())

    keep_indices = np.array(keep_indices)
    kept_mask = np.zeros(n, dtype=bool)
    kept_mask[keep_indices] = True

    return X[keep_indices], y[keep_indices], kept_mask


def _inject_redundant_features(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    severity: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add noisy copies of existing features (redundant) and/or a near-leaky
    feature correlated with the target.
    Returns (X_new [n_rows, n_cols + n_added], redundant_col_mask [n_cols_new]).
    """
    n_cols = X.shape[1]
    n_cols_out = n_cols
    col_mask = np.zeros(n_cols, dtype=bool)

    if severity < 1e-6:
        return X.copy(), col_mask

    n_redundant = max(1, int(n_cols * severity * 0.5))
    new_cols = []

    for _ in range(n_redundant):
        src_col = rng.integers(0, n_cols)
        noise_std = (1.0 - severity) * 0.5 + 0.05
        new_col = X[:, src_col] + rng.normal(0, noise_std, size=len(X))
        new_cols.append(new_col.reshape(-1, 1))

    # Maybe add a leaky target-correlated column
    if severity > 0.5 and rng.random() < 0.5:
        if task_type == "classification":
            leaky = y.astype(float) + rng.normal(0, 0.3, size=len(y))
        else:
            leaky = y + rng.normal(0, 0.1 * float(np.std(y) + 1e-8), size=len(y))
        new_cols.append(leaky.reshape(-1, 1))

    if not new_cols:
        return X.copy(), col_mask

    X_new = np.hstack([X] + new_cols)
    n_added = len(new_cols)
    full_col_mask = np.zeros(n_cols + n_added, dtype=bool)
    full_col_mask[n_cols:] = True  # new columns are the redundant ones

    return X_new.astype(np.float32), full_col_mask


# ──────────────────────────────────────────────────────────────────
# Downstream degradation labeler
# ──────────────────────────────────────────────────────────────────
def _baseline_score(
    X: np.ndarray, y: np.ndarray, task_type: str
) -> float:
    """CV score on clean data with a simple baseline."""
    if task_type == "classification":
        model = DecisionTreeClassifier(max_depth=5, random_state=GLOBAL_SEED)
        scores = cross_val_score(model, X, y, cv=3, scoring="accuracy", error_score=0.0)
    else:
        model = DecisionTreeRegressor(max_depth=5, random_state=GLOBAL_SEED)
        scores = cross_val_score(model, X, y, cv=3, scoring="r2", error_score=-1.0)
    return float(np.mean(scores))


def _corrupt_score_light(
    X: np.ndarray, y: np.ndarray, task_type: str
) -> float:
    """CV score on corrupted data with light model (Logistic/Ridge)."""
    # Replace NaN with median for downstream evaluation
    X_c = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
    X_c = np.nan_to_num(X_c, nan=0.0, posinf=0.0, neginf=0.0)

    if task_type == "classification":
        le = LabelEncoder()
        y_enc = le.fit_transform(y.astype(str))
        model = LogisticRegression(max_iter=200, random_state=GLOBAL_SEED, C=1.0)
        try:
            scores = cross_val_score(model, X_c, y_enc, cv=3, scoring="accuracy", error_score=0.0)
        except Exception:
            scores = np.array([0.0])
    else:
        model = Ridge(alpha=1.0)
        try:
            scores = cross_val_score(model, X_c, y, cv=3, scoring="r2", error_score=-1.0)
        except Exception:
            scores = np.array([-1.0])
    return float(np.mean(scores))


def _corrupt_score_heavy(
    X: np.ndarray, y: np.ndarray, task_type: str, time_limit: float
) -> float:
    """Score on corrupted data with heavy model (LightGBM if available)."""
    if not HAS_LGB:
        return _corrupt_score_light(X, y, task_type)

    X_c = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
    X_c = np.nan_to_num(X_c, nan=0.0, posinf=0.0, neginf=0.0)

    t0 = time.perf_counter()
    param_grid = [
        {"n_estimators": 50,  "max_depth": 4, "learning_rate": 0.1},
        {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.05},
    ]
    best_score = None

    for params in param_grid:
        if time.perf_counter() - t0 > time_limit:
            break
        try:
            if task_type == "classification":
                le = LabelEncoder()
                y_enc = le.fit_transform(y.astype(str))
                model = lgb.LGBMClassifier(**params, random_state=GLOBAL_SEED, n_jobs=1, verbose=-1)
                scores = cross_val_score(model, X_c, y_enc, cv=3, scoring="accuracy", error_score=0.0)
            else:
                model = lgb.LGBMRegressor(**params, random_state=GLOBAL_SEED, n_jobs=1, verbose=-1)
                scores = cross_val_score(model, X_c, y, cv=3, scoring="r2", error_score=-1.0)
            sc = float(np.mean(scores))
            if best_score is None or sc > best_score:
                best_score = sc
        except Exception:
            pass

    return best_score if best_score is not None else 0.0


def _compute_degradation(
    clean_light: float,
    clean_heavy: float,
    corrupt_light: float,
    corrupt_heavy: float,
    task_type: str,
) -> Tuple[float, float]:
    """
    Relative performance drop clipped to [0.0, 1.0].
    For regression: handle negative R² by clipping.
    """
    def rel_drop(clean, corrupt):
        if task_type == "regression":
            clean_clipped  = max(clean, 0.0)
            corrupt_clipped = max(corrupt, 0.0)
        else:
            clean_clipped  = max(clean, 0.0)
            corrupt_clipped = max(corrupt, 0.0)

        if clean_clipped < 1e-6:
            return 0.0
        drop = (clean_clipped - corrupt_clipped) / (clean_clipped + 1e-8)
        return float(np.clip(drop, 0.0, 1.0))

    return rel_drop(clean_light, corrupt_light), rel_drop(clean_heavy, corrupt_heavy)


# ──────────────────────────────────────────────────────────────────
# Corruption engine
# ──────────────────────────────────────────────────────────────────
class CorruptionEngine:
    """
    Generates N corrupted variants from a single clean DatasetInfo.
    Severity sampling: for each dimension, independently draw from
    Uniform(0,1) then zero out with probability p_zero (teaches isolation).
    """

    P_ZERO      = 0.35   # probability each dimension gets severity=0
    MECHANISMS  = ["MCAR", "MAR", "MNAR"]

    def __init__(self, seed: int = GLOBAL_SEED):
        self.seed = seed
        self.rng  = np.random.default_rng(seed)

    def _sample_severities(self) -> np.ndarray:
        """Sample severity vector from [0,1]^6 with zero-inflation."""
        s = self.rng.uniform(0.0, 1.0, size=N_DIMS)
        zero_mask = self.rng.random(N_DIMS) < self.P_ZERO
        s[zero_mask] = 0.0
        return s.astype(np.float32)

    def generate(
        self,
        dataset,    # DatasetInfo
        n_variants: int,
        compute_downstream: bool = True,
    ) -> List[CorruptedDataset]:
        """
        Generate n_variants corrupted versions of the given dataset.

        Parameters
        ----------
        dataset : DatasetInfo
        n_variants : int
        compute_downstream : bool
            If False, skip downstream model fitting (faster but no labels).

        Returns
        -------
        List[CorruptedDataset]
        """
        X_clean = dataset.X.copy()
        y_clean = dataset.y.copy()
        task    = dataset.task_type

        # Precompute baseline scores once per clean dataset
        if compute_downstream:
            logger.debug(f"  Computing baseline scores for {dataset.name} …")
            clean_light = _baseline_score(X_clean, y_clean, task)
            if HAS_LGB:
                X_c = np.nan_to_num(X_clean, nan=0.0)
                clean_heavy = _corrupt_score_heavy(X_c, y_clean, task, HEAVY_MODEL_TIME_LIMIT_S)
            else:
                clean_heavy = clean_light
        else:
            clean_light = clean_heavy = 0.5

        variants: List[CorruptedDataset] = []
        n_classes = dataset.n_classes or 2

        for i in range(n_variants):
            variant_rng = np.random.default_rng(self.seed + i * 1000 + dataset.dataset_id)
            s = self._sample_severities()

            # Apply corruptions in order
            X_c, y_c = X_clean.copy(), y_clean.copy()

            # 1. Outliers
            X_c, outlier_rows = _inject_outliers(X_c, s[0], variant_rng)

            # 2. Label noise
            y_c, noise_rows = _inject_label_noise(y_c, task, s[1], variant_rng, n_classes)

            # 3. Missingness
            miss_mech = self.MECHANISMS[variant_rng.integers(0, 3)]
            X_c, missing_mask = _inject_missingness(X_c, s[2], miss_mech, variant_rng)

            # 4. Duplicates
            X_c, y_c, dup_mask = _inject_duplicates(X_c, y_c, s[3], variant_rng)
            # Align row masks to new row count after duplication
            new_n = len(X_c)
            outlier_rows_extended  = np.zeros(new_n, dtype=bool)
            noise_rows_extended    = np.zeros(new_n, dtype=bool)
            missing_mask_extended  = np.zeros((new_n, X_c.shape[1]), dtype=bool)

            orig_n = min(len(outlier_rows), new_n)
            outlier_rows_extended[:orig_n]  = outlier_rows[:orig_n]
            noise_rows_extended[:orig_n]    = noise_rows[:orig_n]
            missing_mask_extended[:orig_n, :X_c.shape[1]] = (
                missing_mask[:orig_n, :X_c.shape[1]]
                if missing_mask.shape[1] >= X_c.shape[1]
                else np.pad(missing_mask[:orig_n], ((0,0),(0, X_c.shape[1]-missing_mask.shape[1])))
            )

            # 5. Class imbalance (for classification)
            if task == "classification":
                X_c, y_c, imbalance_kept = _inject_class_imbalance(X_c, y_c, task, s[4], variant_rng)
            else:
                s[4] = 0.0
                imbalance_kept = np.ones(len(X_c), dtype=bool)

            # 6. Redundant features
            X_c, redundant_col_mask = _inject_redundant_features(X_c, y_c, task, s[5], variant_rng)

            masks = CorruptionMasks(
                outlier_rows     = outlier_rows_extended[:len(X_c)],
                label_noise_rows = noise_rows_extended[:len(X_c)],
                missing_mask     = missing_mask_extended[:len(X_c), :X_c.shape[1]],
                duplicate_rows   = dup_mask,
                imbalance_kept   = imbalance_kept,
                redundant_cols   = redundant_col_mask,
            )

            variant = CorruptedDataset(
                X               = X_c.astype(np.float32),
                y               = y_c,
                severities      = s,
                masks           = masks,
                task_type       = task,
                n_original_rows = X_clean.shape[0],
                n_original_cols = X_clean.shape[1],
            )

            # Compute downstream degradation
            if compute_downstream:
                try:
                    c_light = _corrupt_score_light(X_c, y_c, task)
                    if HAS_LGB:
                        c_heavy = _corrupt_score_heavy(X_c, y_c, task, HEAVY_MODEL_TIME_LIMIT_S // 2)
                    else:
                        c_heavy = c_light
                    light_drop, heavy_drop = _compute_degradation(
                        clean_light, clean_heavy, c_light, c_heavy, task
                    )
                    variant.light_drop = light_drop
                    variant.heavy_drop = heavy_drop
                except Exception as e:
                    logger.debug(f"  Downstream scoring failed for variant {i}: {e}")
                    variant.light_drop = 0.0
                    variant.heavy_drop = 0.0

            variants.append(variant)

        return variants
