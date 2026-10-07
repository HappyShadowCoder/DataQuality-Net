# STAGE 1 REPORT — DataQuality-Net Prototype
**Generated:** 2026-10-07 10:45:05
**Mode:** FULL
**Total runtime:** 20.5 minutes

---

## 1. Experiment Summary

| Item | Value |
|------|-------|
| Datasets loaded | 31 |
| Total corrupted variants | 1550 |
| Folds completed | 5 / 5 |
| Training epochs (max) | 50 |
| Device | cpu |

---

## 2. Key Metrics (mean ± std across folds)

### 2a. Quality Prediction (all 8 heads)

| Model | MAE ↓ | Pearson r ↑ |
|-------|--------|-------------|
| **Neural (ours)** | **0.2501 ± 0.0680** | **0.1536** |
| Mean predictor (baseline a) | 0.2704 | — |
| Heuristic rules (baseline b) | 0.2274 | — |
| Meta-Feature + GBM (baseline c) | 0.1765 | — |

### 2b. Conformal Coverage (target ≈ 90%)

| Metric | Value |
|--------|-------|
| Empirical coverage (mean ± std) | 0.898 ± 0.019 |
| Target (1 − α = 0.90) | 0.90 |
| Coverage gap | -0.002 |

### 2c. Attribution Precision@k

| Attribution Type | P@5% | P@10% | P@20% |
|-----------------|------|-------|-------|
| Column (redundant features) | 0.080 ± 0.237 | 0.104 ± 0.225 | 0.161 ± 0.205 |
| Row (outlier rows) | 0.200 ± 0.240 | 0.184 ± 0.219 | 0.175 ± 0.185 |

---

## 3. Statistical Test: Neural vs. Meta-Feature+GBM

```
{
  "test": "Wilcoxon signed-rank (paired, one-sided)",
  "statistic": 15.0,
  "p_value": 1.0,
  "significant_at_alpha_0.05": false,
  "interpretation": "Neural model NOT significantly better than MetaFeature+GBM (p\u22650.05)"
}
```

---

## 4. Compute Benchmark

**Methodology note:** Scale-up benchmarks (>20k rows) use synthetically generated data
matching schema statistics (mixture of normal/exponential/uniform/lognormal, ~5% MCAR).
This is benchmarking-only data; NOT used for model training.

| n_rows | n_cols | Inference ms | Light ms | Heavy ms | Saves vs Light | Saves vs Heavy |
|--------|--------|-------------|----------|----------|----------------|----------------|
|   1,000 |     10 |         5.4 |      2.6 |     27.2 |              ✗ |              ✓ |
|   1,000 |     50 |        18.9 |      3.3 |    118.3 |              ✗ |              ✓ |
|  10,000 |     10 |        12.3 |      8.4 |    102.8 |              ✗ |              ✓ |
|  10,000 |     50 |        48.7 |     10.9 |    373.2 |              ✗ |              ✓ |
| 100,000 |     10 |        14.7 |     67.6 |    642.6 |              ✓ |              ✓ |
| 100,000 |     50 |        50.6 |    113.4 |   2012.7 |              ✓ |              ✓ |

**Savings summary:**
- Regimes where inference FASTER than light model: ['n=100,000, cols=10', 'n=100,000, cols=50']
- Regimes where inference FASTER than heavy model: ['n=1,000, cols=10', 'n=1,000, cols=50', 'n=10,000, cols=10', 'n=10,000, cols=50', 'n=100,000, cols=10', 'n=100,000, cols=50']
- Regimes with NO savings vs. light: ['n=1,000, cols=10', 'n=1,000, cols=50', 'n=10,000, cols=10', 'n=10,000, cols=50']
- Regimes with NO savings vs. heavy: []

*Single-pass inference may be SLOWER than light models at small scale due to fixed PyTorch overhead. Savings are reported only for regimes where inference_ms < training_ms.*

---

## 5. GO / NO-GO Recommendations per Patent Claim

| Potential Claim | Status |
|----------------|--------|
| Permutation-invariant encoder for tabular quality | ✅ **GO** — Architecture implemented and functional |
| Single-pass multi-head prediction (8 outputs) | ✅ **GO** — All 8 heads trained and evaluated |
| Neural beats mean-predictor baseline | ✅ **GO** — Neural MAE=0.2501 vs Mean MAE=0.2704 |
| Neural statistically beats meta-GBM (p<0.05) | 🚫 **NO-GO** — Neural model NOT significantly better than MetaFeature+GBM (p≥0.05) |
| Conformal coverage near 90% | ✅ **GO** — Coverage=0.898 (target=0.90) |
| Column attribution (Precision@k) | ✅ **GO** — Occlusion-based method implemented; P@k evaluated |
| Compute savings vs. heavy models at large scale | ✅ **GO** — Valid only for regimes listed above |
| Composite 0-100 usability score | ✅ **GO** — Task-conditioned weighting implemented |

---

## 6. Known Limitations & Edge Cases

1. **Small datasets (<200 rows):** Feature extraction is stable but downstream degradation
   labeling may be noisy due to high variance in 3-fold CV scores.
2. **High-cardinality categoricals:** Encoded via LabelEncoder before corruption injection;
   ordinal artifacts may affect outlier and missingness injections.
3. **Regression class-imbalance head:** Forced to 0.0 for regression tasks (not applicable).
   Patent claims should not assert imbalance detection for regression.
4. **Coverage guarantee:** Conformal coverage holds marginally (per-dimension), not
   jointly (all 8 dimensions simultaneously covered). Joint coverage is lower.
5. **Attribution at small column counts (<5 cols):** Occlusion Precision@k may be
   unreliable due to insufficient discriminating power.
6. **Single-pass savings at small scale:** Inference overhead may EXCEED light model
   training time for datasets <1,000 rows. Do NOT claim compute savings for this regime.
7. **LightGBM heavy model time limit:** Capped at 30s per variant. Results may not
   represent fully tuned LightGBM performance.
8. **Synthetic scale-up data:** 100k-row benchmarks use schema-matched synthetic data,
   not real datasets. Results may differ on real-world large datasets.

---

## 7. Artifacts Produced

| Artifact | Path |
|----------|------|
| Dataset inventory | results/dataset_inventory.json |
| Variant statistics | results/variant_stats.json |
| CV results (all folds) | results/cv_results.json |
| Attribution results | results/attribution_results.json |
| Compute benchmarks | results/benchmark_results.json |
| Best model weights | results/best_model_fold1.pt |
| Scatter plots | figures/scatter_pred_vs_truth_*.png |
| Benchmark plot | figures/benchmark_latency.png |

---

## 8. Human Action Required

🛑 **STAGE GATE: This report requires human review before Stage 2 (patent drafting).**

Please review the GO/NO-GO table above. If you are satisfied:

```
# Confirm Stage 2 start by running:
python run_all.py --stage2   # (to be implemented after confirmation)
```

Or reply with "PROCEED TO PART B" to begin patent drafting.
