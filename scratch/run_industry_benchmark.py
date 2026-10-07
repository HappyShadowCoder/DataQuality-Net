"""
scratch/run_industry_benchmark.py
Empirical Benchmark: DataQuality-Net vs. Cleanlab vs. Rule-Based Profiling
Across 5 Diverse Real-World Tabular Datasets.
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, cross_val_score
import lightgbm as lgb

import src.api as dq
from cleanlab.filter import find_label_issues

datasets = [
    ("Breast Cancer WDBC", "data/test/breast_cancer_wdbc.csv"),
    ("Indian Liver Patient", "data/test/indian_liver_patient.csv"),
    ("German Credit", "data/test/credit_g.csv"),
    ("Wall Robot Navigation", "data/test/wall_robot_navigation.csv"),
    ("Bank Marketing", "data/test/bank_marketing.csv"),
]

def run_benchmark():
    results = []

    for name, path in datasets:
        print(f"\n==========================================")
        print(f"BENCHMARKING: {name} ({path})")
        print(f"==========================================")
        df = pd.read_csv(path)
        n_rows, n_cols = df.shape

        # ── 1. DataQuality-Net (Single-Pass Neural) ──
        t0 = time.perf_counter()
        report = dq.assess(df, task="classification")
        t1 = time.perf_counter()
        dqn_latency_ms = (t1 - t0) * 1000.0

        dqn_score = report.usability_score
        dqn_grade = report.grade
        dqn_subscores = report.subscores
        dqn_degrad = report.degradation
        deep_diag = getattr(report, "deep_diagnostics", {})

        # Auto-fix execution test
        t_fix0 = time.perf_counter()
        clean_df, fix_actions = report.auto_fix(return_log=True)
        re_report = dq.assess(clean_df, task="classification")
        t_fix1 = time.perf_counter()
        dqn_autofix_ms = (t_fix1 - t_fix0) * 1000.0
        dqn_remediated_score = re_report.usability_score

        # ── Preprocessing for Baselines ──
        df_clean_base = df.copy()
        for col in df_clean_base.columns:
            if df_clean_base[col].dtype == "object" or df_clean_base[col].dtype.name == "category":
                df_clean_base[col] = LabelEncoder().fit_transform(df_clean_base[col].astype(str))
            else:
                df_clean_base[col] = df_clean_base[col].fillna(df_clean_base[col].median())

        X_base = df_clean_base.iloc[:, :-1].values
        y_base = LabelEncoder().fit_transform(df_clean_base.iloc[:, -1].values)

        # ── 2. Cleanlab (Confident Learning via Logistic Regression CV) ──
        sample_size = min(len(y_base), 5000)
        X_cl_sample = X_base[:sample_size]
        y_cl_sample = y_base[:sample_size]

        t_cl0 = time.perf_counter()
        scaler = StandardScaler()
        X_cl_s = scaler.fit_transform(X_cl_sample)
        clf = LogisticRegression(max_iter=300, random_state=42, n_jobs=1)
        try:
            pred_probs = cross_val_predict(clf, X_cl_s, y_cl_sample, cv=3, method="predict_proba", n_jobs=1)
            cl_label_issues = find_label_issues(labels=y_cl_sample, pred_probs=pred_probs, n_jobs=1)
            cl_issue_count = int(np.sum(cl_label_issues))
        except Exception as e:
            cl_issue_count = 0
            print("Cleanlab error:", e)
        t_cl1 = time.perf_counter()
        cl_latency_ms = (t_cl1 - t_cl0) * 1000.0

        # ── 3. Rule-Based Statistical Profiling (Great Expectations style) ──
        t_rule0 = time.perf_counter()
        missing_cnt = int(df.isnull().sum().sum())
        dup_cnt = int(df.duplicated().sum())
        # Z-score 3-sigma outliers
        z_scores = np.abs((X_base - np.mean(X_base, axis=0)) / (np.std(X_base, axis=0) + 1e-8))
        z_outliers = int(np.sum(z_scores > 3))
        # Pearson Collinearity > 0.90
        corr = np.corrcoef(X_base.T)
        corr = np.nan_to_num(corr)
        upper = np.abs(corr[np.triu_indices(X_base.shape[1], k=1)])
        collinear_pairs = int(np.sum(upper > 0.90))
        t_rule1 = time.perf_counter()
        rule_latency_ms = (t_rule1 - t_rule0) * 1000.0

        # ── 4. Actual Downstream Performance Impact (LightGBM on Raw vs Remediated) ──
        try:
            lgb_raw = lgb.LGBMClassifier(random_state=42, verbose=-1, n_estimators=50, n_jobs=1)
            cv_raw = float(np.mean(cross_val_score(lgb_raw, X_base[:2000], y_base[:2000], cv=3, n_jobs=1)))
            # Prepare remediated for LightGBM
            df_rem_base = clean_df.copy()
            for col in df_rem_base.columns:
                if df_rem_base[col].dtype == "object" or df_rem_base[col].dtype.name == "category":
                    df_rem_base[col] = LabelEncoder().fit_transform(df_rem_base[col].astype(str))
                else:
                    df_rem_base[col] = df_rem_base[col].fillna(df_rem_base[col].median())
            X_rem = df_rem_base.iloc[:, :-1].values
            y_rem = LabelEncoder().fit_transform(df_rem_base.iloc[:, -1].values)
            lgb_clean = lgb.LGBMClassifier(random_state=42, verbose=-1, n_estimators=50, n_jobs=1)
            cv_clean = float(np.mean(cross_val_score(lgb_clean, X_rem[:2000], y_rem[:2000], cv=3, n_jobs=1)))
        except Exception as e:
            cv_raw, cv_clean = 0.0, 0.0
            print("LGBM error:", e)

        dataset_summary = {
            "dataset_name": name,
            "path": path,
            "rows": n_rows,
            "cols": n_cols,
            "dqn": {
                "latency_ms": round(dqn_latency_ms, 2),
                "usability_score": round(dqn_score, 1),
                "grade": dqn_grade,
                "remediated_score": round(dqn_remediated_score, 1),
                "score_lift": round(dqn_remediated_score - dqn_score, 1),
                "autofix_latency_ms": round(dqn_autofix_ms, 2),
                "autofix_actions": fix_actions,
                "subscores": {k: round(v, 1) for k, v in dqn_subscores.items()},
                "degradation_estimates": {k: round(v, 1) for k, v in dqn_degrad.items()},
                "deep_diagnostics": {
                    "target_leakage_features": len(deep_diag.get("target_leakage", [])),
                    "mahalanobis_outliers": deep_diag.get("multivariate_outliers", {}).get("count", 0),
                    "fuzzy_near_duplicates": deep_diag.get("fuzzy_near_duplicates", {}).get("count", 0),
                    "label_inconsistency_pct": deep_diag.get("label_inconsistency", {}).get("percentage", 0.0),
                    "label_inconsistency_count": deep_diag.get("label_inconsistency", {}).get("count", 0),
                }
            },
            "cleanlab": {
                "latency_ms": round(cl_latency_ms, 2),
                "label_issues_found": cl_issue_count,
                "sample_evaluated": sample_size
            },
            "rules_profiling": {
                "latency_ms": round(rule_latency_ms, 2),
                "missing_cells": missing_cnt,
                "exact_duplicates": dup_cnt,
                "zscore_outliers": z_outliers,
                "collinear_pairs_gt_90": collinear_pairs
            },
            "empirical_lgbm_cv": {
                "raw_acc": round(float(cv_raw), 4),
                "clean_acc": round(float(cv_clean), 4),
                "accuracy_gain_pct": round(float((cv_clean - cv_raw) * 100), 2)
            }
        }

        results.append(dataset_summary)
        print(f"DQN Latency: {dqn_latency_ms:.2f} ms | Cleanlab Latency: {cl_latency_ms:.2f} ms | Rules Latency: {rule_latency_ms:.2f} ms")
        print(f"DQN Score: {dqn_score:.1f} -> {dqn_remediated_score:.1f} (Lift: +{dqn_remediated_score - dqn_score:.1f})")
        print(f"Cleanlab Issues: {cl_issue_count} | DQN Inconsistencies: {deep_diag.get('label_inconsistency', {}).get('count', 0)}")
        print(f"LightGBM Accuracy: {cv_raw:.4f} -> {cv_clean:.4f} (Delta: {cv_clean - cv_raw:+.4f})")

    out_path = Path("results/industry_benchmark_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nBenchmark completed successfully! Saved to results/industry_benchmark_results.json")

if __name__ == "__main__":
    run_benchmark()
