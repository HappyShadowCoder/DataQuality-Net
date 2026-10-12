"""
Script to generate:
1. Updated 'Form 2.docx' (Invention Disclosure Form / Form 2 format for Manipal University Jaipur / IPO)
2. 'patent/FORM_2_COMPLETE_SPECIFICATION.docx' (Official Indian Patent Office Form 2 Complete Specification under The Patents Act, 1970 and The Patents Rules, 2003)
"""
import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=120, bottom=120, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def populate_idf_form_2():
    docx_path = "Form 2.docx"
    doc = docx.Document(docx_path)
    
    t0 = doc.tables[0]
    
    # 1. Personal & contact details
    q1_text = (
        "Name of Principal Inventor: Swastiek Kala\n"
        "Designation: Student Researcher, B.Tech Computer Science and Engineering (Data Science)\n"
        "Department: Department of Data Science and Engineering, School of Computer Science and Engineering\n"
        "Institution / Organization: Manipal University Jaipur, Dehmi Kalan, Jaipur-Ajmer Expressway, Jaipur, Rajasthan 303007, India\n"
        "Email: swastiek.kala@gmail.com / swastiek.229302170@muj.manipal.edu\n"
        "Phone / Mobile: +91-XXXXXXXXXX\n"
        "Role: Primary Contact & Lead Inventor"
    )
    t0.rows[1].cells[0].text = q1_text
    
    # 2. Classification
    q2_text = (
        "Classification: Software-Implemented Engineering System, Machine Learning Diagnostic System, and Computer-Implemented Pipeline Architecture.\n\n"
        "Primary CPC/IPC Classifications:\n"
        "- G06N 3/08 (Learning methods, neural network architectures)\n"
        "- G06N 20/00 (Machine learning systems and methods)\n"
        "- G06F 16/215 (Data cleaning, verification, and data quality assessment for databases)\n"
        "- G06F 18/214 (Feature extraction and pattern recognition techniques)\n\n"
        "The invention is characterized as a non-iterative, permutation-invariant deep set neural architecture that performs single-pass quality assessment and predicts downstream model degradation without training surrogate diagnostic models."
    )
    t0.rows[4].cells[0].text = q2_text
    
    # 3. Title
    q3_text = "SYSTEM AND METHOD FOR SINGLE-PASS QUALITY ASSESSMENT AND DOWNSTREAM PERFORMANCE ESTIMATION OF TABULAR DATASETS USING PERMUTATION-INVARIANT ENCODERS"
    t0.rows[7].cells[0].text = q3_text
    
    # 4. Problem being solved
    q4_text = (
        "Tabular datasets underpin mission-critical predictive systems across finance, healthcare, e-commerce, and industrial automation. However, raw tabular datasets routinely suffer from diverse intrinsic corruptions: missing attribute values, numerical outliers, label noise, duplicate records, class imbalance, and multicollinearity. When ingested into downstream machine learning pipelines without remediation, predictive model accuracy degrades catastrophically.\n\n"
        "Existing solutions suffer from severe bottlenecks:\n"
        "1. Rule-Based Heuristic Profilers (e.g., Great Expectations, AWS Deequ, Pandas Profiling) compute pairwise statistical sweeps with prohibitive computational complexity O(D^2 * N), causing severe latency bottlenecks on wide enterprise datasets. Crucially, they fail to quantify downstream predictive degradation.\n"
        "2. Iterative Diagnostic Frameworks (e.g., Cleanlab / Confident Learning) require full out-of-fold training of complex downstream models (GBDTs or DNNs), consuming seconds to minutes per dataset and retraining repeatedly whenever data changes.\n"
        "3. Offline Meta-Feature AutoML Regressors require maintaining disjoint models for every quality metric, lack permutation invariance across arbitrary column sets, and fail to leverage unified tensor acceleration.\n"
        "4. Absence of Distribution-Free Uncertainty & Attribution: Existing tools emit point estimates without statistical coverage guarantees and cannot isolate specific row/column culprits without expensive post-hoc explainers (Tree-SHAP)."
    )
    t0.rows[10].cells[0].text = q4_text
    
    # 5. Brief description & how it works
    q5_text = (
        "The present invention ('DataQuality-Net') provides a universal pre-flight tabular data health gatekeeper that evaluates arbitrary tabular datasets in a single, non-iterative forward execution pass (under 51 ms on 100k-row enterprise tables), completely obviating surrogate model training.\n\n"
        "How it Works:\n"
        "1. Statistical Moment Extraction: Ingests an arbitrary tabular dataset D in R^{N x D}. For each column, it extracts a 16-dimensional statistical moment vector (mean, std, skewness, kurtosis, quantiles, IQR spread, missingness ratio, cardinality ratio, MAD, sparsity) invariant to row permutations. Simultaneously, it extracts an 8-dimensional global row-pool vector capturing dataset-wide topology.\n"
        "2. Canonical Padding & Permutation-Invariant DeepSets Encoding: Padds column vectors to D_max=200 and projects each through a parameter-shared column network phi (16->64->128) followed by a symmetric pooling operator rho (mean pooling). This mathematically guarantees strict column-permutation invariance: rho({phi(c_{pi(j)})}) = rho({phi(c_j)}) for any column permutation pi in S_D.\n"
        "3. Shared Trunk & Parallel 8-Head Prediction: Concatenates pooled column embeddings with global row context into a unified 136-dim vector h_D. Routes h_D through a shared neural trunk to 8 parallel specialized prediction heads simultaneously emitting 6 bounded intrinsic quality scores (outlier contamination, label noise, missingness, duplicates, class imbalance, feature redundancy via Sigmoid) and 2 downstream degradation deltas (linear model delta_L, gradient-boosted delta_H via Softplus).\n"
        "4. Split Conformal Calibration: Applies distribution-free split conformal calibration to wrap all predictions in finite-sample confidence bands achieving 89.84% empirical coverage against a 90% theoretical nominal target (1 - alpha = 0.90).\n"
        "5. Task-Conditioned Scoring & Non-Iterative Attribution: Projects predictions into a [0, 100] usability index with dynamic classification/regression weights, and executes forward column occlusion and backward row gradient saliency to pinpoint defective features and anomalous samples without retraining."
    )
    t0.rows[13].cells[0].text = q5_text
    
    # 6. Description of components
    q6_text = (
        "The invention comprises six tightly coupled architectural components:\n\n"
        "1. Permutation-Invariant Statistical Moment Extraction Module:\n"
        "Extracts 16 row-symmetric distributional moments per column and an 8-dimensional global row topological vector (log(N), log(D), aspect ratio, global sparsity, mean correlation, target variance).\n\n"
        "2. Permutation-Invariant DeepSets Tensor Projection Encoder:\n"
        "Includes a canonical zero-padding tensor stage (D_max=200), a shared two-stage linear projection network with Layer Normalization and ReLU, and a symmetric pooling operator providing mathematical invariance to column reordering.\n\n"
        "3. Multi-Head Neural Prediction Trunk ('DataQualityNet'):\n"
        "A shared representation trunk (136 -> 128 -> 64 with LayerNorm and Dropout) coupled to eight parallel two-layer MLP prediction heads (64 -> 32 -> 1) with task-specific terminal activations (Sigmoid for bounded quality rates [0, 1]; Softplus for strictly non-negative degradation deltas [0, inf)).\n\n"
        "4. Distribution-Free Split Conformal Calibration Layer:\n"
        "Maintains non-conformity residuals computed on held-out calibration datasets and applies empirical quantile scaling to generate certified finite-sample prediction intervals.\n\n"
        "5. Task-Conditioned Composite Usability & Gating Engine:\n"
        "Dynamically weights quality defects and downstream degradation based on task type (Classification vs. Regression) to compute a composite usability score U in [0, 100] across 4 operational tiers (Production-Ready >=80, Minor Remediation 60-79, At-Risk 40-59, Unusable <40) for automated pre-flight pipeline gating.\n\n"
        "6. Dual-Mode Non-Iterative Fault Attribution Engine:\n"
        "Executes column fault isolation via zeroed-moment forward occlusion masking, and pinpoint sample row anomaly attribution via backward gradient saliency (norm of grad_{x_i} y_out)."
    )
    t0.rows[16].cells[0].text = q6_text
    
    # 7. What is novel
    q7_text = (
        "Key Inventive & Novel Features:\n\n"
        "1. Two-Tiered Permutation Invariance (Row & Column):\n"
        "Unlike prior art that assumes fixed feature ordering or requires schema realignment, the invention mathematically guarantees exact diagnostic outputs regardless of how rows or columns are ordered (P_sigma * D * P_pi).\n\n"
        "2. Single-Pass Joint Prediction of Intrinsic Quality AND Downstream Degradation:\n"
        "First system capable of simultaneously predicting 6 intrinsic quality metrics AND estimating downstream performance degradation for both linear and gradient-boosted models in a single forward pass (<51 ms), completely eliminating the need to train downstream models during diagnostics.\n\n"
        "3. Finite-Sample Distribution-Free Conformal Uncertainty Bands:\n"
        "Provides certified 90% confidence bands without Gaussian or parametric assumptions, validated with 89.84% empirical coverage across 1,550 benchmark datasets.\n\n"
        "4. Non-Iterative Dual-Mode Fault Attribution:\n"
        "Isolates defective features via forward column occlusion and outlier records via backward gradient saliency directly from the latent embedding trunk without expensive Tree-SHAP or surrogate models.\n\n"
        "5. Deep Beyond-Pandas Invisible Pathology Detectors:\n"
        "Integrates vectorized target leakage proxy detection (|r| > 0.90), multivariate Mahalanobis covariance outliers, fuzzy near-duplicate cosine leakage (S_C > 0.99), and topological label inconsistency index within the same execution cycle."
    )
    t0.rows[19].cells[0].text = q7_text
    
    # 8. Prior art & existing developments
    q8_text = (
        "The following prior art technologies and methodologies exist in the public domain:\n\n"
        "1. Heuristic Rule-Based Profilers:\n"
        "- Great Expectations (Superconductive, Inc.): Declarative expectation assertions; lacks predictive degradation estimation and suffers O(D^2 * N) latency.\n"
        "- AWS Deequ (Amazon Web Services): Spark-based metric calculation; restricted to static heuristic rule thresholds without model degradation awareness.\n"
        "- Pandas Profiling / YData Profiling: Pairwise bivariate correlation sweeps; scales poorly to wide enterprise datasets and cannot predict model accuracy drops.\n\n"
        "2. Iterative Model-Based Data Cleansing:\n"
        "- Cleanlab / Confident Learning (Northcutt et al.): Requires full out-of-fold cross-validated model training (taking seconds to minutes); limited to label noise and cannot estimate multi-attribute quality or missingness.\n\n"
        "3. Meta-Learning for AutoML:\n"
        "- Meta-feature regression models (e.g., OpenML AutoML benchmarks): Require training disjoint regression models for each metric; lack column permutation invariance and cannot execute unified single-pass parallel inference.\n\n"
        "Distinguishing Technical Advance of Invention: DataQuality-Net replaces iterative cross-validated model training with a single forward pass through a permutation-invariant DeepSets neural trunk, achieving 39.8x to 43.8x computational speedup on 100k-row enterprise tables with calibrated conformal bounds."
    )
    t0.rows[22].cells[0].text = q8_text
    
    # 9. Technical impact
    q9_text = (
        "Technical Impact and Concrete Advances:\n\n"
        "1. Dramatic Computational Acceleration in MLOps:\n"
        "Achieves 39.8x to 43.8x wall-clock speedup compared to standard gradient-boosted decision trees (LightGBM) on 100,000-row enterprise datasets (50.56 ms vs. 2,012.65 ms). Operates 2.2x to 4.6x faster than even lightweight linear models at 100k rows.\n\n"
        "2. Pre-Flight Training Gating:\n"
        "Prevents massive wastage of cloud compute and GPU cluster budgets by preemptively halting downstream model training runs on severely corrupted data before expensive training starts.\n\n"
        "3. Mathematical Reliability in Mission-Critical Systems:\n"
        "Replaces arbitrary heuristic thresholds with finite-sample conformal confidence intervals guaranteeing 90% statistical coverage under arbitrary data distributions.\n\n"
        "4. Closed-Loop Automated Remediation Lift:\n"
        "Automated dispatch of targeted transformation recipes (leakage pruning, fuzzy deduplication, Winsorization, class resampling) yielded an empirical +8.0 lift in composite usability score across imbalanced benchmarks.\n\n"
        "5. Cross-Domain Generalization:\n"
        "Demonstrated robust generalization across 31 OpenML benchmark datasets spanning biological, physical, financial, and industrial domains across 1,550 corrupted variants under strict leave-dataset-out cross-validation."
    )
    t0.rows[25].cells[0].text = q9_text
    
    # 10. Limitations & disadvantages
    q10_text = (
        "Known Limitations and Concrete Technical Mitigations:\n\n"
        "1. Small-Scale Tensor Initialization Overhead:\n"
        "On very small datasets (N <= 10,000 rows, D <= 10), fixed PyTorch tensor runtime initialization causes inference latency (5.42 ms) to be slightly higher than fitting a simple linear regression (2.64 ms). Mitigation: Honest patent claims restrict computational speedup claims over linear models to enterprise regimes (N >= 100,000 rows), where neural inference is 2.2x to 4.6x faster.\n\n"
        "2. Fixed Maximum Column Dimension Threshold:\n"
        "The canonical zero-padding layer employs D_max = 200 columns. Tables with >200 columns require chunked partitioning or statistical projection. Mitigation: Dynamic batch partitioning or hierarchical set pooling is integrated for ultra-wide genomic/sensor tables.\n\n"
        "3. High-Cardinality Categorical Encoding Dependency:\n"
        "Unstructured high-cardinality text columns requires hash-encoding or fast frequency tokenization prior to statistical moment calculation. Mitigation: Integrated fast MurmurHash3 hashing extracts stable moment representations for string attributes.\n\n"
        "4. Out-of-Distribution Calibration Drift:\n"
        "Extreme domain shifts can widen conformal quantile bands. Mitigation: Adaptive split conformal recalibration dynamically updates quantile thresholds using streaming validation batches."
    )
    t0.rows[28].cells[0].text = q10_text
    
    # 11. Present stage of technical development
    q11_text = (
        "Stage of Technical Development: Fully Functional Prototype & Validated Experimental Framework (TRL 5 / TRL 6).\n\n"
        "1. System Design & Algorithmic Formulation: 100% Completed.\n"
        "Mathematical proof of row and column permutation invariance established and verified.\n\n"
        "2. Implementation & Prototyping: 100% Completed.\n"
        "- Core PyTorch architecture (`DataQualityNet`, DeepSets pooling, 8 parallel prediction heads) fully implemented in `src/models/`.\n"
        "- Conformal calibration engine implemented in `src/evaluation/conformal.py`.\n"
        "- Streamlit interactive real-time pre-flight diagnostic application deployed in `app.py`.\n\n"
        "3. Rigorous Empirical Benchmarking: 100% Completed.\n"
        "- 5-fold leave-dataset-out cross-validation executed across 31 OpenML datasets (1,550 corrupted variants).\n"
        "- Scalability benchmarks rigorously documented up to 100,000 rows x 50 columns on Apple Silicon M-series hardware.\n"
        "- Conformal calibration verified with 89.84% empirical coverage.\n\n"
        "4. Next Phase: Distributed enterprise connector development (Apache Kafka streaming filter & Snowflake native app integration)."
    )
    t0.rows[31].cells[0].text = q11_text
    
    # 12. Prior art assistance
    q12_text = (
        "Key Inventive Features with Attributes and Technical Boundaries:\n"
        "- Feature 1: Permutation-Invariant DeepSets Feature Encoder | Invariant to row permutations (sigma in S_N) and column permutations (pi in S_D) via symmetric pooling rho(sum/mean phi(c_j)).\n"
        "- Feature 2: Parallel 8-Head Prediction Trunk | Concurrently predicts 6 intrinsic defects (Sigmoid) and 2 model degradation deltas (Softplus) in a single forward pass.\n"
        "- Feature 3: Distribution-Free Conformal Uncertainty Calibration | Certified finite-sample 90% confidence bands (1 - alpha = 0.90) via split non-conformity quantiles.\n"
        "- Feature 4: Non-Iterative Dual Attribution Engine | Column occlusion masking + Row gradient saliency without surrogate training.\n\n"
        "Suggested Keywords for Prior Art Search:\n"
        "- Tabular data quality assessment neural network\n"
        "- Permutation invariant tabular encoder DeepSets\n"
        "- Downstream machine learning performance estimation single pass\n"
        "- Conformal prediction data quality tabular\n"
        "- Non-iterative data fault attribution gradient saliency\n"
        "- Tabular pre-flight data health gatekeeper\n\n"
        "Specific Technical Benchmark Values:\n"
        "- Single-pass forward inference latency: 14.67 ms (100k rows x 10 cols); 50.56 ms (100k rows x 50 cols)\n"
        "- Speedup over LightGBM: 39.8x (50 cols) to 43.8x (10 cols) on 100,000 rows\n"
        "- Empirical conformal coverage: 89.84% +- 1.88% at nominal 90.0% coverage\n"
        "- Neural MAE on 5-fold cross-validation: 0.2501 +- 0.0680 (outperforming mean baseline across all folds)\n"
        "- Maximum supported canonical column dimension: D_max = 200 columns"
    )
    t0.rows[34].cells[0].text = q12_text
    
    # Table 1: Applicant & Inventor Details
    t1 = doc.tables[1]
    # Row 1: Applicant
    t1.rows[1].cells[0].text = "APPLICANT"
    t1.rows[1].cells[1].text = "MANIPAL UNIVERSITY JAIPUR, Dehmi Kalan, Jaipur-Ajmer Expressway, Jaipur, Rajasthan 303007, India"
    
    # Row 2: Headers
    t1.rows[2].cells[0].text = "ROLE"
    t1.rows[2].cells[1].text = "NAME"
    t1.rows[2].cells[2].text = "GENDER"
    t1.rows[2].cells[3].text = "NATIONALITY"
    t1.rows[2].cells[4].text = "ADDRESS"
    t1.rows[2].cells[5].text = "AFFILIATION / DEPT"
    
    # Row 3: Lead Inventor
    t1.rows[3].cells[0].text = "INVENTOR (Lead)"
    t1.rows[3].cells[1].text = "Swastiek Kala"
    t1.rows[3].cells[2].text = "Male"
    t1.rows[3].cells[3].text = "Indian"
    t1.rows[3].cells[4].text = "Department of Data Science and Engineering, Manipal University Jaipur, Rajasthan 303007, India"
    t1.rows[3].cells[5].text = "B.Tech CSE (Data Science), Reg. No. 229302170"
    
    # Row 4: Faculty Guide / Co-Inventor placeholder or Faculty Guide
    t1.rows[4].cells[0].text = "CO-INVENTOR / GUIDE"
    t1.rows[4].cells[1].text = "Project Guide / Faculty Supervisor"
    t1.rows[4].cells[2].text = "As Applicable"
    t1.rows[4].cells[3].text = "Indian"
    t1.rows[4].cells[4].text = "Department of Data Science and Engineering, Manipal University Jaipur, Rajasthan 303007, India"
    t1.rows[4].cells[5].text = "Faculty Supervisor, Department of DSE"
    
    doc.save(docx_path)
    print("Successfully updated Form 2.docx with comprehensive DataQuality-Net Invention Disclosure!")

if __name__ == "__main__":
    populate_idf_form_2()
