"""
src/data_loader.py
Download and preprocess 25–40 tabular datasets from OpenML.
Enforces row cap of 20,000, handles missing values, encodes categoricals.
"""

from __future__ import annotations
import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import openml
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.utils import DATA_DIR, get_logger, retry, GLOBAL_SEED

warnings.filterwarnings("ignore")
logger = get_logger("data_loader")

# ──────────────────────────────────────────────────────────────────
# Dataset catalogue
# ──────────────────────────────────────────────────────────────────
CLASSIFICATION_IDS = [
    31,    # credit-g (German Credit) — known label noise & missingness
    37,    # diabetes
    40,    # sonar
    44,    # spambase
    54,    # vehicle
    59,    # ionosphere
    188,   # eucalyptus
    307,   # vowel
    458,   # analcatdata_authorship
    469,   # analcatdata_dmft
    1049,  # pc4
    1050,  # pc3
    1063,  # kc2
    1067,  # kc1
    1068,  # pc1
    1461,  # bank-marketing — severe class imbalance
    1480,  # ilpd
    1494,  # qsar-biodeg
    1510,  # wdbc
    40687, # Australian
    42,    # soybean
    23,    # cmc
    1497,  # wall-robot-navigation
    4534,  # PhishingWebsites
]

REGRESSION_IDS = [
    197,   # 529_pollen
    216,   # elevators
    317,   # boston (small, well-known)
    507,   # space_ga
    560,   # bodyfat
    564,   # kin8nm
    586,   # fri_c3_1000_25
    589,   # fri_c0_1000_10
]

# Datasets with documented real-world quality issues (sanity checks)
QUALITY_ISSUE_IDS = {
    31:   "known label noise and missing value patterns (UCI documentation)",
    1461: "severe class imbalance ~88% negative class (Moro et al. 2014)",
}

MAX_ROWS = 20_000
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────
# Dataset info container
# ──────────────────────────────────────────────────────────────────
class DatasetInfo:
    """Holds a cleaned dataset ready for corruption injection."""

    def __init__(
        self,
        dataset_id: int,
        name: str,
        task_type: str,  # "classification" | "regression"
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        n_classes: Optional[int],
        quality_note: Optional[str] = None,
    ):
        self.dataset_id   = dataset_id
        self.name         = name
        self.task_type    = task_type
        self.X            = X
        self.y            = y
        self.feature_names = feature_names
        self.n_classes    = n_classes
        self.quality_note = quality_note
        self.n_rows       = X.shape[0]
        self.n_cols       = X.shape[1]

    def __repr__(self) -> str:
        return (
            f"DatasetInfo(id={self.dataset_id}, name={self.name!r}, "
            f"task={self.task_type}, shape=({self.n_rows}×{self.n_cols}), "
            f"n_classes={self.n_classes})"
        )


# ──────────────────────────────────────────────────────────────────
# Download helpers
# ──────────────────────────────────────────────────────────────────
@retry(max_attempts=3, delay=10.0)
def _fetch_openml(dataset_id: int) -> openml.datasets.OpenMLDataset:
    return openml.datasets.get_dataset(
        dataset_id,
        download_data=True,
        download_qualities=False,
        download_features_meta_data=False,
    )


def _preprocess(
    X_raw: pd.DataFrame,
    y_raw: pd.Series,
    task_type: str,
    rng: np.random.Generator,
    max_rows: int = MAX_ROWS,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    1. Cap rows (random subsample).
    2. Drop constant columns.
    3. Encode categoricals (label encoding → numeric).
    4. Impute missing with median/mode.
    5. Standard-scale all features (zero-mean, unit-variance).
    6. Encode target for classification.
    Returns: X (float64), y (float64 or int), feature_names.
    """
    df = X_raw.copy()

    # ── Row cap ──────────────────────────────────────────────────
    if len(df) > max_rows:
        idx = rng.choice(len(df), max_rows, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
        y_raw = y_raw.iloc[idx].reset_index(drop=True)

    feature_names = list(df.columns)

    # ── Drop constant columns ────────────────────────────────────
    non_const = [c for c in df.columns if df[c].nunique(dropna=False) > 1]
    df = df[non_const]
    feature_names = non_const

    # ── Encode categorical features ──────────────────────────────
    for col in df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # ── Impute missing ────────────────────────────────────────────
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(df.astype(float))

    # ── Scale ────────────────────────────────────────────────────
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # ── Target ───────────────────────────────────────────────────
    if task_type == "classification":
        le_y = LabelEncoder()
        y = le_y.fit_transform(y_raw.astype(str)).astype(np.int64)
    else:
        y = pd.to_numeric(y_raw, errors="coerce").fillna(y_raw.median()).to_numpy(dtype=np.float64)

    return X.astype(np.float32), y, feature_names


def _load_dataset(dataset_id: int, task_type: str, rng: np.random.Generator) -> Optional[DatasetInfo]:
    cache_path = CACHE_DIR / f"{dataset_id}_{task_type}.pkl"
    if cache_path.exists():
        logger.info(f"[{dataset_id}] Loading from cache")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    try:
        logger.info(f"[{dataset_id}] Fetching from OpenML …")
        ds = _fetch_openml(dataset_id)
        X_raw, y_raw, _, _ = ds.get_data(dataset_format="dataframe", target=ds.default_target_attribute)

        if y_raw is None:
            logger.warning(f"[{dataset_id}] No target — skipping")
            return None

        X, y, feat_names = _preprocess(X_raw, y_raw, task_type, rng)

        if X.shape[0] < 50 or X.shape[1] < 2:
            logger.warning(f"[{dataset_id}] Too small after preprocessing — skipping")
            return None

        n_classes = int(np.unique(y).shape[0]) if task_type == "classification" else None
        info = DatasetInfo(
            dataset_id    = dataset_id,
            name          = ds.name,
            task_type     = task_type,
            X             = X,
            y             = y,
            feature_names = feat_names,
            n_classes     = n_classes,
            quality_note  = QUALITY_ISSUE_IDS.get(dataset_id),
        )

        with open(cache_path, "wb") as f:
            pickle.dump(info, f)
        logger.info(f"[{dataset_id}] ✓ {info}")
        return info

    except Exception as e:
        logger.warning(f"[{dataset_id}] FAILED: {e}")
        return None


# ──────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────
def load_all_datasets(
    fast: bool = False,
    seed: int = GLOBAL_SEED,
) -> List[DatasetInfo]:
    """
    Download & preprocess all datasets.

    Parameters
    ----------
    fast : bool
        If True, use a small subset for quick smoke testing.
    seed : int
        Global random seed.

    Returns
    -------
    List[DatasetInfo]
    """
    rng = np.random.default_rng(seed)
    openml.config.apikey = ""  # public API, no key needed

    if fast:
        # Quick smoke test: 6 classification + 3 regression
        clf_ids = [31, 37, 44, 1461, 1510, 1494]
        reg_ids = [317, 564, 589]
    else:
        clf_ids = sorted(CLASSIFICATION_IDS)
        reg_ids = sorted(REGRESSION_IDS)

    datasets: List[DatasetInfo] = []

    logger.info(f"Loading {len(clf_ids)} classification + {len(reg_ids)} regression datasets …")

    for did in clf_ids:
        ds = _load_dataset(did, "classification", rng)
        if ds is not None:
            datasets.append(ds)

    for did in reg_ids:
        ds = _load_dataset(did, "regression", rng)
        if ds is not None:
            datasets.append(ds)

    logger.info(f"Successfully loaded {len(datasets)} datasets.")

    # Log quality-issue datasets
    for ds in datasets:
        if ds.quality_note:
            logger.info(f"  [QUALITY-ISSUE] {ds.name} (id={ds.dataset_id}): {ds.quality_note}")

    return datasets
