"""
src/utils.py
Shared utilities: seed management, logging, path helpers, timing.
"""

import os
import json
import time
import random
import logging
import functools
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Dict, Optional

import numpy as np
import psutil

# ──────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results"
FIGURES_DIR  = PROJECT_ROOT / "figures"
LOGS_DIR     = PROJECT_ROOT / "logs"

for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────
# Global seed
# ──────────────────────────────────────────────────────────────────
GLOBAL_SEED = 42


def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ──────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────
def get_logger(name: str = "dataquality", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        fh = logging.FileHandler(LOGS_DIR / "run.log")
        fh.setFormatter(fmt)
        logger.addHandler(sh)
        logger.addHandler(fh)
        logger.setLevel(level)
    return logger


logger = get_logger()


# ──────────────────────────────────────────────────────────────────
# Timing & Memory
# ──────────────────────────────────────────────────────────────────
@contextmanager
def timer(label: str = ""):
    """Context manager that measures wall-clock time."""
    t0 = time.perf_counter()
    mem0 = psutil.Process().memory_info().rss / 1024 ** 2
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        mem1 = psutil.Process().memory_info().rss / 1024 ** 2
        logger.debug(f"[{label}] {elapsed:.3f}s | ΔRSS={mem1-mem0:+.1f}MB")


def peak_memory_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 ** 2


# ──────────────────────────────────────────────────────────────────
# JSON persistence
# ──────────────────────────────────────────────────────────────────
def save_json(obj: Any, path: Path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=_json_serial)


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _json_serial(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


# ──────────────────────────────────────────────────────────────────
# Config persistence
# ──────────────────────────────────────────────────────────────────
def save_config(config: Dict, filename: str = "experiment_config.json") -> None:
    save_json(config, LOGS_DIR / filename)
    logger.info(f"Config saved → {LOGS_DIR / filename}")


# ──────────────────────────────────────────────────────────────────
# Retry decorator
# ──────────────────────────────────────────────────────────────────
def retry(max_attempts: int = 3, delay: float = 5.0, exceptions=(Exception,)):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    logger.warning(f"[retry {attempt}/{max_attempts}] {fn.__name__}: {e}")
                    if attempt == max_attempts:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator
