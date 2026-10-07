# DataQuality-Net: Single-Pass Tabular Dataset Quality Assessment

## Invention Title
**System and Method for Single-Pass Quality Assessment and Downstream Performance Estimation of Tabular Datasets Using Permutation-Invariant Encoders**

- **Lead Inventor & Developer:** Swastiek Kala
- **Academic Context:** 3rd Year Project-Based Learning (PBL)
- **Patent Jurisdiction:** Indian Patents Act (1970), Form 2 Provisional Application

---

## Overview

This repository implements a proof-of-concept (POC) prototype for automated tabular data quality assessment and downstream ML performance degradation estimation using permutation-invariant set encoders (DeepSets / Set Transformer). Designed and developed by **Swastiek Kala**, the system outputs a composite 0–100 usability score, 6 individual quality-dimension scores, and 2 downstream-degradation estimates in a single forward pass — without retraining any model on the dataset under assessment.

---

## Project Structure

```
.
├── src/
│   ├── corruption.py          # Synthetic corruption engine + downstream labeler
│   ├── encoder.py             # Permutation-invariant dataset encoder
│   ├── model.py               # Multi-head neural architecture
│   ├── calibration.py         # Nested conformal prediction calibration
│   ├── composite.py           # Composite usability scorer
│   ├── attribution.py         # Feature & row attribution + Precision@k
│   ├── benchmark_compute.py   # Compute/memory benchmarking
│   ├── evaluate.py            # Baselines + 5-fold evaluation
│   ├── data_loader.py         # OpenML dataset downloader & preprocessor
│   └── utils.py               # Shared utilities
├── data/                      # Downloaded OpenML datasets (auto-populated)
├── notebooks/                 # Exploratory analysis notebooks
├── results/                   # Experimental results, metrics, STAGE1_REPORT.md
├── patent/                    # Patent documents (populated after Stage 2)
│   └── drawings/              # Architecture diagrams
├── figures/                   # Plots and visualizations
├── logs/                      # Execution logs
├── run_all.py                 # Main orchestration script
├── app.py                     # Streamlit diagnostic interface
├── requirements.txt
└── README.md
```

---

## Dataset Sources and Licenses

All datasets are sourced from [OpenML](https://www.openml.org/) — a public repository of machine learning datasets under the **Creative Commons CC BY 4.0** license unless otherwise noted.

### Datasets Used (Target: 25–40 datasets, <20k rows)

| OpenML ID | Name | Task | Rows | License |
|-----------|------|------|------|---------|
| 31 | credit-g (German Credit) | Classification | 1,000 | CC BY 4.0 |
| 37 | diabetes | Classification | 768 | CC BY 4.0 |
| 40 | sonar | Classification | 208 | CC BY 4.0 |
| 44 | spambase | Classification | 4,601 | CC BY 4.0 |
| 54 | vehicle | Classification | 846 | CC BY 4.0 |
| 59 | ionosphere | Classification | 351 | CC BY 4.0 |
| 188 | eucalyptus | Classification | 736 | CC BY 4.0 |
| 307 | vowel | Classification | 990 | CC BY 4.0 |
| 458 | analcatdata_authorship | Classification | 841 | CC BY 4.0 |
| 469 | analcatdata_dmft | Classification | 797 | CC BY 4.0 |
| 1049 | pc4 | Classification | 1,458 | CC BY 4.0 |
| 1050 | pc3 | Classification | 1,563 | CC BY 4.0 |
| 1063 | kc2 | Classification | 522 | CC BY 4.0 |
| 1067 | kc1 | Classification | 2,109 | CC BY 4.0 |
| 1068 | pc1 | Classification | 1,109 | CC BY 4.0 |
| 1461 | bank-marketing | Classification | 4,521 | CC BY 4.0 |
| 1480 | ilpd | Classification | 583 | CC BY 4.0 |
| 1494 | qsar-biodeg | Classification | 1,055 | CC BY 4.0 |
| 1510 | wdbc | Classification | 569 | CC BY 4.0 |
| 40687 | Australian | Classification | 690 | CC BY 4.0 |
| 42 | soybean | Classification | 683 | CC BY 4.0 |
| 6 | letter | Classification | 20,000 | CC BY 4.0 |
| 23 | cmc | Classification | 1,473 | CC BY 4.0 |
| 1497 | wall-robot-navigation | Classification | 5,456 | CC BY 4.0 |
| 4534 | PhishingWebsites | Classification | 11,055 | CC BY 4.0 |
| 197 | 529_pollen | Regression | 3,848 | CC BY 4.0 |
| 216 | elevators | Regression | 16,599 | CC BY 4.0 |
| 317 | boston | Regression | 506 | CC BY 4.0 |
| 507 | space_ga | Regression | 3,107 | CC BY 4.0 |
| 531 | cal_housing | Regression | 20,640 | CC BY 4.0 |
| 560 | bodyfat | Regression | 252 | CC BY 4.0 |
| 564 | kin8nm | Regression | 8,192 | CC BY 4.0 |
| 574 | house_16H | Regression | 22,784 | CC BY 4.0 |
| 586 | fri_c3_1000_25 | Regression | 1,000 | CC BY 4.0 |
| 589 | fri_c0_1000_10 | Regression | 1,000 | CC BY 4.0 |

**Real-world datasets with documented quality issues (qualitative sanity checks):**
- **credit-g (OpenML 31)**: Contains known label noise and missing value patterns documented in UCI repository.
- **bank-marketing (OpenML 1461)**: Known class imbalance (~88% negative class) documented in original Moro et al. (2014) paper.

> **Note**: Datasets capped at 20,000 rows; larger datasets (cal_housing, house_16H) are subsampled to 20,000 rows for consistency.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline
```bash
python run_all.py
```

### 3. Fast Smoke Test (≈10–15 min)
```bash
python run_all.py --fast
```

### 4. Launch Diagnostic Interface
```bash
streamlit run app.py
```

### 5. Assess a Custom CSV
```bash
python run_all.py --assess path/to/your_data.csv --task classification
```

---

## Reproducibility

All random seeds are fixed globally:
- Python: `random.seed(42)`
- NumPy: `np.random.seed(42)`
- PyTorch: `torch.manual_seed(42)`
- OpenML download order: deterministic via sorted dataset IDs

All seeds and hyperparameters are logged to `logs/experiment_config.json`.

---

## Citation

If you use this codebase, please cite:
```
[Author names withheld for provisional filing]
System and Method for Single-Pass Quality Assessment and Downstream 
Performance Estimation of Tabular Datasets Using Permutation-Invariant 
Encoders. Provisional Patent Application, Indian Patents Act (1970), 2026.
```

---

## License

This research code is provided for academic and patent filing purposes only.
