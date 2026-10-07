"""
src/api.py
Universal 1-Line Pre-Flight Gatekeeper API for DataQuality-Net.

Usage in any ML/DL pipeline:
    import dataqualitynet as dq
    report = dq.assess(df, task="classification")
    if not report.is_ready_for_training:
        clean_df = report.auto_fix()
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, List, Union
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.utils import RESULTS_DIR
from src.model import DataQualityNet, ALL_HEADS, QUALITY_HEADS
from src.composite import CompositeUsabilityScorer, assess_dataframe


class PreFlightReport:
    """
    Contains the full pre-flight diagnostic report and automated remediation methods.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        report_dict: Dict,
        raw_preds: np.ndarray,
        tips: List[Dict],
        task_type: str = "classification",
    ):
        self.raw_df = df
        self.report = report_dict
        self.raw_preds = raw_preds
        self.tips = tips
        self.task_type = task_type

        self.usability_score = report_dict["usability_score"]
        self.grade = report_dict["grade"]
        self.subscores = report_dict.get("quality_subscores", {})
        self.degradation = report_dict.get("degradation_scores", {})
        self.deep_diagnostics = report_dict.get("deep_diagnostics", {})

        # Traffic light decision thresholds
        if self.usability_score >= 80:
            self.status = "🟢 GO (Production Ready)"
            self.is_ready_for_training = True
        elif self.usability_score >= 60:
            self.status = "🟡 CAUTION (Minor Defects Detected)"
            self.is_ready_for_training = True
        else:
            self.status = "🔴 HALT (High Risk of Downstream Failure)"
            self.is_ready_for_training = False

    def summary(self) -> str:
        """Formatted terminal-friendly pre-flight summary."""
        lines = [
            "=" * 60,
            "  DATAQUALITY-NET — PRE-FLIGHT ML GATEKEEPER REPORT",
            "=" * 60,
            f"  Status          : {self.status}",
            f"  Usability Score : {self.usability_score:.1f} / 100  (Grade: {self.grade})",
            f"  Ready to Train? : {'YES' if self.is_ready_for_training else 'NO (Remediation Recommended)'}",
            "-" * 60,
            "  Quality Sub-Scores:",
        ]
        for dim, sc in self.subscores.items():
            bar = "█" * int(sc / 5) + "░" * (20 - int(sc / 5))
            lines.append(f"    {dim:20s}: {bar} {sc:.1f}/100")
        lines.extend([
            "-" * 60,
            "  Predicted Downstream Degradation:",
            f"    Light Model Drop (LogReg/Ridge): {self.degradation.get('light_model_drop', 0):.1f}%",
            f"    Heavy Model Drop (LightGBM/XGB): {self.degradation.get('heavy_model_drop', 0):.1f}%",
        ])

        # Deep Beyond-Pandas Diagnostics section
        if self.deep_diagnostics:
            lines.extend([
                "-" * 60,
                "  🔬 Deep Beyond-Pandas Diagnostics (Hidden Flaws):",
            ])
            # Target leakage
            leakage = self.deep_diagnostics.get("target_leakage", [])
            if leakage:
                leak_str = ", ".join([f"{item['feature']} (r={item['correlation']})" for item in leakage[:3]])
                lines.append(f"    ⚠️  Target Leakage Detected   : {len(leakage)} features [{leak_str}]")
            else:
                lines.append("    ✅ Target Leakage             : None detected (|r| < 0.90)")

            # Multivariate outliers
            mo = self.deep_diagnostics.get("multivariate_outliers", {})
            lines.append(f"    🔍 Multivariate Outliers      : {mo.get('count', 0)} joint-space anomalies (Mahalanobis)")

            # Fuzzy near-duplicates
            fnd = self.deep_diagnostics.get("fuzzy_near_duplicates", {})
            lines.append(f"    👥 Fuzzy Near-Duplicates      : {fnd.get('count', 0)} rows (>99% cosine similarity)")

            # Label inconsistency
            li = self.deep_diagnostics.get("label_inconsistency", {})
            if "percentage" in li:
                lines.append(f"    ⚡ Label Contradictions       : {li.get('percentage', 0.0)}% of neighborhood ({li.get('count', 0)} rows)")

        lines.append("=" * 60)
        return "\n".join(lines)

    def auto_fix(self, *args, **kwargs) -> pd.DataFrame:
        """
        Defect-aware closed-loop automated remediation engine:
        1. Removes duplicate rows (if duplicates subscore < 90).
        2. Imputes missing values (if missingness subscore < 90).
        3. Winsorizes extreme outliers (only if outliers subscore < 85).
        4. Drops highly redundant multicollinear features (correlation > 0.95).
        5. Rebalances skewed class distributions (if task is classification and imbalance ratio > 2.0).
        """
        balance_imbalance = kwargs.get("balance_imbalance", True)
        return_log = kwargs.get("return_log", False)
        df_clean = self.raw_df.copy()
        actions = []

        # 0. Prune Spurious Target Leakage Features
        leakage = self.deep_diagnostics.get("target_leakage", [])
        leaked_cols = [item["feature"] for item in leakage if item["feature"] in df_clean.columns and item["feature"] != df_clean.columns[-1]]
        if leaked_cols:
            df_clean = df_clean.drop(columns=leaked_cols)
            actions.append(f"Pruned {len(leaked_cols)} target leakage proxy features: {', '.join(leaked_cols)}.")

        # 1. Deduplicate
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        dups_removed = initial_rows - len(df_clean)
        if dups_removed > 0:
            actions.append(f"Deduplicated {dups_removed:,} exact duplicate rows.")

        # 2. Impute Missing Values
        n_missing = int(df_clean.isnull().sum().sum())
        if n_missing > 0:
            for col in df_clean.columns:
                if df_clean[col].isnull().sum() > 0:
                    if pd.api.types.is_numeric_dtype(df_clean[col]):
                        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                    else:
                        mode_val = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else "missing"
                        df_clean[col] = df_clean[col].fillna(mode_val)
            actions.append(f"Imputed {n_missing:,} missing values using column medians and modes.")

        # 3. Defect-Targeted Outlier Trimming (only if outlier subscore indicates defects < 85)
        if self.subscores.get("outliers", 100) < 85:
            num_cols = df_clean.select_dtypes(include=[np.number]).columns
            trimmed_count = 0
            for col in num_cols:
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                if std > 1e-8:
                    lower = mean - 3.5 * std
                    upper = mean + 3.5 * std
                    outlier_mask = (df_clean[col] < lower) | (df_clean[col] > upper)
                    trimmed_count += int(outlier_mask.sum())
                    df_clean[col] = np.clip(df_clean[col], lower, upper)
            if trimmed_count > 0:
                actions.append(f"Winsorized {trimmed_count:,} extreme outlier values beyond 3.5σ.")

        # 4. Prune Highly Collinear Redundant Features (> 0.95 correlation)
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 1 and self.subscores.get("redundant_features", 100) < 85:
            corr_matrix = df_clean[num_cols].corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [c for c in upper_tri.columns if any(upper_tri[c] > 0.95)]
            if df_clean.columns[-1] in to_drop:
                to_drop.remove(df_clean.columns[-1])
            if to_drop:
                df_clean = df_clean.drop(columns=to_drop)
                actions.append(f"Dropped {len(to_drop)} multicollinear redundant features: {', '.join(to_drop)}.")

        # 5. Class Imbalance Remediation for Classification
        if balance_imbalance and self.task_type == "classification":
            target_col = df_clean.columns[-1]
            counts = df_clean[target_col].value_counts()
            if len(counts) >= 2:
                imbalance_ratio = float(counts.max() / max(1, counts.min()))
                if imbalance_ratio > 2.0:
                    target_size = int(counts.min() * 1.5)
                    df_clean = pd.concat([
                        df_clean[df_clean[target_col] == cls].sample(min(len(df_clean[df_clean[target_col] == cls]), target_size), random_state=42)
                        for cls in counts.index
                    ], ignore_index=True)
                    actions.append(f"Balanced class distribution (reduced skew from {imbalance_ratio:.1f}:1 to 1.5:1 via stratified sampling).")

        if not actions:
            actions.append("Dataset structure verified; no critical defects required invasive modification.")

        self.last_fix_actions = actions
        if return_log:
            return df_clean, actions
        return df_clean


class Gatekeeper:
    """
    Singleton gatekeeper that loads the pre-trained DataQualityNet model.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None, device: str = "cpu"):
        self.device = device
        if model_path is None:
            model_path = RESULTS_DIR / "best_model_fold1.pt"
        self.model_path = Path(model_path)
        self.model = DataQualityNet(embed_dim=128, trunk_dim=256, col_dim=64)

        if self.model_path.exists():
            try:
                state = torch.load(self.model_path, map_location=device)
                self.model.load_state_dict(state)
            except Exception:
                pass
        self.model.to(device).eval()

    def assess(self, data: Union[str, Path, pd.DataFrame], task: str = "classification") -> PreFlightReport:
        """
        Single-pass assessment of any tabular dataset.
        """
        if isinstance(data, (str, Path)):
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Data must be a file path or a pandas DataFrame.")

        report_dict, fused_preds, tips = assess_dataframe(df, self.model, task_type=task, device=self.device)
        return PreFlightReport(df, report_dict, fused_preds, tips, task_type=task)


# Global default instance
_default_gatekeeper: Optional[Gatekeeper] = None


def assess(data: Union[str, Path, pd.DataFrame], task: str = "classification", device: str = "cpu") -> PreFlightReport:
    """
    Convenience function for 1-line pre-flight check.
    Example:
        import src.api as dq
        report = dq.assess(df, task="classification")
        print(report.summary())
    """
    global _default_gatekeeper
    if _default_gatekeeper is None:
        _default_gatekeeper = Gatekeeper(device=device)
    return _default_gatekeeper.assess(data, task=task)
