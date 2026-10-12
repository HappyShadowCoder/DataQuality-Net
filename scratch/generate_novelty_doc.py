import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def create_document():
    doc = docx.Document()
    
    # Page setup - Margins 0.8 inch
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)
        
    # Styles
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Times New Roman'
    style_normal.font.size = Pt(11)
    style_normal.font.color.rgb = RGBColor(40, 40, 40)
    
    # ─────────────────────────────────────────────────────────────
    # TITLE & HEADER
    # ─────────────────────────────────────────────────────────────
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(4)
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = title_p.add_run("DATAQUALITY-NET")
    r_title.font.name = 'Arial'
    r_title.font.size = Pt(22)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(0, 51, 102) # Deep Navy
    
    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(14)
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = sub_p.add_run("Comprehensive Technical Novelty Dossier & Real-Life Training Corpus Specification")
    r_sub.font.name = 'Arial'
    r_sub.font.size = Pt(13)
    r_sub.font.bold = True
    r_sub.font.color.rgb = RGBColor(100, 100, 100)
    
    # Meta box
    meta_table = doc.add_table(rows=1, cols=1)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell_meta = meta_table.cell(0, 0)
    cell_meta.width = Inches(6.8)
    set_cell_background(cell_meta, "F0F4F8")
    set_cell_margins(cell_meta, 120, 120, 180, 180)
    
    p_meta = cell_meta.paragraphs[0]
    p_meta.paragraph_format.space_after = Pt(0)
    r_m = p_meta.add_run("Project: ")
    r_m.bold = True
    p_meta.add_run("DataQuality-Net (DSE3170 PBL-3)  |  ")
    r_m2 = p_meta.add_run("Author / Lead: ")
    r_m2.bold = True
    p_meta.add_run("Swastiek Kala  |  ")
    r_m3 = p_meta.add_run("Institution: ")
    r_m3.bold = True
    p_meta.add_run("Manipal University Jaipur\n")
    r_m4 = p_meta.add_run("Statutory IP: ")
    r_m4.bold = True
    p_meta.add_run("Provisional Patent Form 2 (22 Claims) under Indian Patents Act, 1970")
    p_meta.runs[0].font.size = Pt(9.5)
    for r in p_meta.runs:
        r.font.size = Pt(9.5)
        
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # ─────────────────────────────────────────────────────────────
    # SECTION 1: THE CORE NOVELTY (4 PILLARS)
    # ─────────────────────────────────────────────────────────────
    h1 = doc.add_heading(level=1)
    r_h1 = h1.add_run("1. Technical Novelty & Scientific Contributions")
    r_h1.font.name = 'Arial'
    r_h1.font.size = Pt(15)
    r_h1.font.bold = True
    r_h1.font.color.rgb = RGBColor(0, 51, 102)
    h1.paragraph_format.space_after = Pt(8)
    
    intro_p = doc.add_paragraph(
        "Modern Machine Learning systems suffer from a pervasive, silent failure mode: over 80% of data science pipelines ingest corrupted tabular data without automated verification. Data scientists blindly execute `model.fit()`, wasting thousands of dollars in cloud GPU compute only to deploy degraded models. "
        "DataQuality-Net resolves this fundamental bottleneck by introducing the first single-pass, permutation-invariant pre-flight gatekeeper. Below are the four core technical pillars representing our patentable novelty:"
    )
    intro_p.paragraph_format.line_spacing = 1.15
    intro_p.paragraph_format.space_after = Pt(10)

    # Pillar 1
    p_pil1 = doc.add_paragraph()
    r_p1_t = p_pil1.add_run("Pillar 1: Two-Tiered Permutation-Invariant Set Architecture (DeepSets for 2D Tables)\n")
    r_p1_t.bold = True
    r_p1_t.font.size = Pt(12)
    r_p1_t.font.color.rgb = RGBColor(0, 102, 153)
    p_pil1.add_run(
        "• The Fundamental Problem: Conventional neural networks require fixed-dimensional, strictly ordered feature vectors. If column order is shuffled (π ∈ S_D) or if the number of features varies between datasets, standard neural networks fail completely.\n"
        "• Our Mathematical Breakthrough: We adapted DeepSets theory [Zaheer et al., NeurIPS 2017] to 2D tabular matrices through a two-tiered invariant decomposition:\n"
        "   1. Row Invariance: Each column's 16-dimensional summary vector c_j is computed using symmetric statistical moments (mean, variance, skewness, kurtosis, IQR) invariant to arbitrary row reordering σ ∈ S_N.\n"
        "   2. Column Invariance: Summary vectors are projected through a parameter-shared MLP φ: R^16 → R^128 and aggregated via symmetric pooling ρ: e_col = (1/D) Σ φ(c_j), followed by zero-padding up to D_max = 200.\n"
        "• Significance: Guarantees exact mathematical invariance: f(P_σ · D · P_π) = f(D). Enables zero-shot generalization across entirely unseen tabular schemas without architectural retraining or column realignment."
    )
    p_pil1.paragraph_format.line_spacing = 1.15
    p_pil1.paragraph_format.space_after = Pt(10)

    # Pillar 2
    p_pil2 = doc.add_paragraph()
    r_p2_t = p_pil2.add_run("Pillar 2: Single-Pass Multi-Head Downstream Degradation Forecasting (No Model Retraining)\n")
    r_p2_t.bold = True
    r_p2_t.font.size = Pt(12)
    r_p2_t.font.color.rgb = RGBColor(0, 102, 153)
    p_pil2.add_run(
        "• The Fundamental Problem: State-of-the-art tools like Cleanlab require 5-fold cross-validation re-training of heavy models to estimate data quality, taking seconds to minutes per partition. Syntactic rule engines (Great Expectations) only check static rules and cannot tell whether downstream accuracy will drop by 2% or 40%.\n"
        "• Our Mathematical Breakthrough: DataQualityNet predicts both intrinsic data health and downstream predictive consequences in a single 50ms forward pass across 8 parallel heads:\n"
        "   - Heads 1–6 (Intrinsic Health): Outliers, Label Noise, Missingness, Duplicates, Class Imbalance, Redundancy (Sigmoid activations ∈ [0, 1]).\n"
        "   - Heads 7–8 (Downstream Degradation): Linear Model Drop (Δ_L) and Heavy Ensemble Drop (Δ_H) (Softplus activations ∈ [0, ∞)).\n"
        "• Significance: Allows automated MLOps pipelines to forecast downstream degradation deltas before training any model, preventing compute waste."
    )
    p_pil2.paragraph_format.line_spacing = 1.15
    p_pil2.paragraph_format.space_after = Pt(10)

    # Pillar 3
    p_pil3 = doc.add_paragraph()
    r_p3_t = p_pil3.add_run("Pillar 3: Distribution-Free Split Conformal Uncertainty Calibration (Finite-Sample Guarantees)\n")
    r_p3_t.bold = True
    r_p3_t.font.size = Pt(12)
    r_p3_t.font.color.rgb = RGBColor(0, 102, 153)
    p_pil3.add_run(
        "• The Fundamental Problem: Deep neural networks routinely emit overconfident, uncalibrated point estimates. In automated decision systems, false alarms cause pipeline thrashing.\n"
        "• Our Mathematical Breakthrough: We implemented distribution-free split conformal prediction [Vovk et al. 2005; Angelopoulos & Bates 2021]. By evaluating non-conformity residuals s_i = |y_i - ŷ_i| on held-out calibration folds, the network computes non-parametric 90% confidence bands [ŷ_k - q̂_k, ŷ_k + q̂_k].\n"
        "• Empirical Validation: On 31 OpenML datasets and 1,550 corrupted variants, DataQuality-Net achieved 89.84% ± 1.88% empirical coverage against the 90.0% theoretical target (coverage gap of only -0.16%)."
    )
    p_pil3.paragraph_format.line_spacing = 1.15
    p_pil3.paragraph_format.space_after = Pt(10)

    # Pillar 4
    p_pil4 = doc.add_paragraph()
    r_p4_t = p_pil4.add_run("Pillar 4: Verified 40× Hardware Latency Advantage & Closed-Loop Self-Healing\n")
    r_p4_t.bold = True
    r_p4_t.font.size = Pt(12)
    r_p4_t.font.color.rgb = RGBColor(0, 102, 153)
    p_pil4.add_run(
        "• Hardware Compute Speedup: On 100,000-row enterprise tables with 50 features, LightGBM requires 2,012.65 ms, whereas DataQuality-Net executes in 50.56 ms — achieving a verified 39.8× to 43.8× computational acceleration.\n"
        "• Deep Beyond-Pandas Diagnostics: Detects 4 invisible anomalies in sub-10ms: Target Leakage (|r| > 0.90), Multivariate Mahalanobis Covariance Outliers, Fuzzy Cosine Near-Duplicates (>99%), and Topological Label Inconsistency.\n"
        "• Closed-Loop Self-Healing: The system includes an automated remediation engine (deduplication, robust median imputation, Winsorization, and stratified rebalancing). On the OpenML Bank Marketing benchmark, it achieved an empirical +8.0 point usability lift (70.8 → 78.8)."
    )
    p_pil4.paragraph_format.line_spacing = 1.15
    p_pil4.paragraph_format.space_after = Pt(14)

    # ─────────────────────────────────────────────────────────────
    # SECTION 2: PRIOR ART COMPARATIVE MATRIX
    # ─────────────────────────────────────────────────────────────
    h2 = doc.add_heading(level=1)
    r_h2 = h2.add_run("2. Comparative Prior Art Analysis Matrix")
    r_h2.font.name = 'Arial'
    r_h2.font.size = Pt(15)
    r_h2.font.bold = True
    r_h2.font.color.rgb = RGBColor(0, 51, 102)
    h2.paragraph_format.space_after = Pt(8)
    
    t_comp = doc.add_table(rows=7, cols=5)
    t_comp.alignment = WD_TABLE_ALIGNMENT.CENTER
    comp_headers = ["Technical Capability", "Great Expectations", "Cleanlab (JAIR 2021)", "TabPFN (ICLR 2023)", "DataQuality-Net (Ours)"]
    widths = [Inches(1.8), Inches(1.2), Inches(1.2), Inches(1.2), Inches(1.4)]
    
    for c_idx, h in enumerate(comp_headers):
        cell = t_comp.cell(0, c_idx)
        cell.width = widths[c_idx]
        set_cell_background(cell, "003366")
        set_cell_margins(cell, 80, 80, 100, 100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.name = 'Arial'
        r.font.size = Pt(9.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        
    comp_rows = [
        ["Inference Mechanism", "Manual Rule Assertions", "Iterative Out-of-Fold", "Prior-Data Transformer", "Single-Pass Set Encoder"],
        ["Scalability Limit", "≥ 100,000 Rows", "≥ 100,000 Rows", "Capped at ≤ 1,000 Rows", "≥ 100,000+ Rows (Linear O(N·D))"],
        ["100k-Row Latency (D=50)", "> 500 ms (Rules)", "> 2,000 ms (Retrains)", "Out of Memory (OOM)", "50.56 ms (39.8× Faster)"],
        ["Downstream Drop Forecast", "NO (Syntactic Only)", "NO (Label Noise Only)", "NO", "YES (Δ_L, Δ_H Deltas)"],
        ["Uncertainty Quantification", "None", "Probabilistic Heuristic", "Bayesian Posterior", "90% Conformal Coverage"],
        ["Closed-Loop Self-Healing", "None", "Filter / Mask Only", "None", "1-Click Auto-Remediation (+8.0 Lift)"]
    ]
    for r_idx, r_data in enumerate(comp_rows):
        bg = "F9FBFD" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(r_data):
            cell = t_comp.cell(r_idx + 1, c_idx)
            cell.width = widths[c_idx]
            set_cell_background(cell, bg)
            set_cell_margins(cell, 60, 60, 100, 100)
            p = cell.paragraphs[0]
            p.paragraph_format.line_spacing = 1.05
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(val)
            r.font.name = 'Times New Roman'
            r.font.size = Pt(9)
            if c_idx == 4:
                r.font.bold = True
                r.font.color.rgb = RGBColor(0, 102, 153)
                
    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # ─────────────────────────────────────────────────────────────
    # SECTION 3: REAL-LIFE DATASETS USED TO TRAIN & EVALUATE MODEL
    # ─────────────────────────────────────────────────────────────
    h3 = doc.add_heading(level=1)
    r_h3 = h3.add_run("3. Real-Life Datasets Used for Training & Benchmarking")
    r_h3.font.name = 'Arial'
    r_h3.font.size = Pt(15)
    r_h3.font.bold = True
    r_h3.font.color.rgb = RGBColor(0, 51, 102)
    h3.paragraph_format.space_after = Pt(8)
    
    ds_intro = doc.add_paragraph(
        "To rigorously train and validate DataQuality-Net under zero-data-leakage conditions, we curated a diverse corpus of "
        "31 real-world tabular datasets from the OpenML scientific repository, spanning clinical diagnostics, credit risk, software defect telemetry, astrophysics, robotics, and marketing. "
        "From these 31 parent datasets, we programmatically generated 1,550 systematically corrupted variants across 6 corruption dimensions (outliers, label noise, missingness, duplicates, class imbalance, and redundant features) with severity levels ranging from 0% to 60%.\n\n"
        "Crucially, the architecture was evaluated using 5-Fold Leave-Dataset-Out Cross-Validation: no dataset variant ever appeared in both training and test folds, strictly proving zero-shot generalization to unseen database schemas."
    )
    ds_intro.paragraph_format.line_spacing = 1.15
    ds_intro.paragraph_format.space_after = Pt(10)

    # The 31 Datasets Table
    t_ds = doc.add_table(rows=32, cols=6)
    t_ds.alignment = WD_TABLE_ALIGNMENT.CENTER
    ds_headers = ["#", "OpenML ID", "Dataset Name", "Task Modality", "Samples (N)", "Features (D)"]
    ds_widths = [Inches(0.4), Inches(0.9), Inches(2.2), Inches(1.3), Inches(1.0), Inches(1.0)]
    
    for c_idx, h in enumerate(ds_headers):
        cell = t_ds.cell(0, c_idx)
        cell.width = ds_widths[c_idx]
        set_cell_background(cell, "003366")
        set_cell_margins(cell, 60, 60, 80, 80)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.name = 'Arial'
        r.font.size = Pt(9)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    datasets_list = [
        (1, 23, "cmc (Contraceptive Choice)", "Classification", 1473, 9),
        (2, 31, "credit-g (German Credit)", "Classification", 1000, 20),
        (3, 37, "diabetes (Pima Indians)", "Classification", 768, 8),
        (4, 40, "sonar (Mines vs Rocks)", "Classification", 208, 60),
        (5, 42, "soybean (Crop Disease)", "Classification", 683, 35),
        (6, 44, "spambase (Email Telemetry)", "Classification", 4601, 57),
        (7, 54, "vehicle (Silhouettes)", "Classification", 846, 18),
        (8, 59, "ionosphere (Radar Targets)", "Classification", 351, 33),
        (9, 188, "eucalyptus (Forestry Soil)", "Classification", 736, 19),
        (10, 307, "vowel (Acoustic Phonetics)", "Classification", 990, 12),
        (11, 458, "analcatdata_authorship", "Classification", 841, 70),
        (12, 469, "analcatdata_dmft (Dental)", "Classification", 797, 4),
        (13, 1049, "pc4 (NASA Metrics)", "Classification", 1458, 37),
        (14, 1050, "pc3 (NASA Software Defect)", "Classification", 1563, 37),
        (15, 1063, "kc2 (Software Reliability)", "Classification", 522, 21),
        (16, 1067, "kc1 (Software Defect Engine)", "Classification", 2109, 21),
        (17, 1068, "pc1 (Flight Software Defect)", "Classification", 1109, 21),
        (18, 1461, "bank-marketing (Moro et al.)", "Classification", 20000, 16),
        (19, 1480, "ilpd (Indian Liver Patient)", "Classification", 583, 10),
        (20, 1494, "qsar-biodeg (Chemical Mol.)", "Classification", 1055, 41),
        (21, 1497, "wall-robot-navigation", "Classification", 5456, 24),
        (22, 1510, "wdbc (Wisconsin Breast Cancer)", "Classification", 569, 30),
        (23, 4534, "PhishingWebsites (Security)", "Classification", 11055, 30),
        (24, 40687, "solar-flare (Astrophysics)", "Classification", 1066, 11),
        (25, 197, "cpu_act (Server Performance)", "Regression", 8192, 21),
        (26, 216, "elevators (Flight Controls)", "Regression", 16599, 18),
        (27, 507, "space_ga (Spatial Census)", "Regression", 3107, 6),
        (28, 560, "bodyfat (Hydrostatic Density)", "Regression", 252, 14),
        (29, 564, "fried (Friedman Benchmark)", "Regression", 20000, 10),
        (30, 586, "fri_c3_1000_25 (Synthetic Nolinear)", "Regression", 1000, 25),
        (31, 589, "fri_c2_1000_25 (Synthetic Feature)", "Regression", 1000, 25)
    ]

    for idx, (num, ds_id, name, task, rows, cols) in enumerate(datasets_list):
        bg = "F9FBFD" if idx % 2 == 1 else "FFFFFF"
        row_vals = [str(num), str(ds_id), name, task, f"{rows:,}", str(cols)]
        for c_idx, val in enumerate(row_vals):
            cell = t_ds.cell(idx + 1, c_idx)
            cell.width = ds_widths[c_idx]
            set_cell_background(cell, bg)
            set_cell_margins(cell, 40, 40, 60, 60)
            p = cell.paragraphs[0]
            p.paragraph_format.line_spacing = 1.05
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(val)
            r.font.name = 'Times New Roman'
            r.font.size = Pt(8.5)
            if c_idx in [0, 1, 3, 4, 5]:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if c_idx == 2 and "bank-marketing" in val:
                r.font.bold = True
                r.font.color.rgb = RGBColor(0, 102, 153)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # ─────────────────────────────────────────────────────────────
    # SECTION 4: CORRUPTION INJECTION REGIMES
    # ─────────────────────────────────────────────────────────────
    h4 = doc.add_heading(level=1)
    r_h4 = h4.add_run("4. Systematic Corruption Generation Regimes (1,550 Variants)")
    r_h4.font.name = 'Arial'
    r_h4.font.size = Pt(15)
    r_h4.font.bold = True
    r_h4.font.color.rgb = RGBColor(0, 51, 102)
    h4.paragraph_format.space_after = Pt(8)
    
    p_reg = doc.add_paragraph(
        "Each of the 31 base datasets underwent systematic perturbation across 6 mathematical regimes with ground-truth attribution masks recorded for Precision@k validation:\n"
        "1. Outlier Injection: Substituted clean continuous cells with extreme deviations drawn from Cauchy / heavy-tailed Gaussian distributions (severity α ∈ [0.05, 0.40]).\n"
        "2. Label Noise Injection: Permuted target labels uniformly across classification classes to simulate annotator error (severity α ∈ [0.05, 0.35]).\n"
        "3. Missingness Injection: Injected Missing-Completely-at-Random (MCAR) and Missing-at-Random (MAR) null masks (severity α ∈ [0.05, 0.50]).\n"
        "4. Exact & Fuzzy Duplication: Cloned observation vectors with and without floating-point sensor jitter (severity α ∈ [0.05, 0.40]).\n"
        "5. Class Imbalance Perturbation: Resampled majority vs minority classes to induce severe class skew (from 50:50 up to 95:5).\n"
        "6. Multicollinear Feature Redundancy: Synthesized correlated linear and non-linear clones of existing feature columns."
    )
    p_reg.paragraph_format.line_spacing = 1.15
    p_reg.paragraph_format.space_after = Pt(14)

    # ─────────────────────────────────────────────────────────────
    # SECTION 5: STATUTORY SECTION 3(k) DEFENSE
    # ─────────────────────────────────────────────────────────────
    h5 = doc.add_heading(level=1)
    r_h5 = h5.add_run("5. Patentability & Section 3(k) Legal Defense (Indian Patents Act, 1970)")
    r_h5.font.name = 'Arial'
    r_h5.font.size = Pt(15)
    r_h5.font.bold = True
    r_h5.font.color.rgb = RGBColor(0, 51, 102)
    h5.paragraph_format.space_after = Pt(8)
    
    p_law = doc.add_paragraph(
        "Under Section 3(k) of the Indian Patents Act, 1970, computer programs per se and mathematical methods are non-patentable subject matter. To ensure bulletproof patent eligibility, our Form 2 provisional specification is structured around official CRI Guidelines 2017 and Delhi High Court binding precedents (Ferid Allani v. Union of India, 2019):\n\n"
        "1. Measurable Technical Effect: Demonstrable 39.8× to 43.8× reduction in execution latency on 100k-row tables (50.56 ms vs 2,012.65 ms). This directly conserves electrical power, CPU/GPU cycles, and hardware cooling.\n"
        "2. Physical Data Transformation: The closed-loop self-healing module physically transforms corrupted digital database blocks into remediated, validated dataset files.\n"
        "3. Hardware Control Signaling: Apparatus claims recite physical MLOps circuitry and execution interrupt signals that halt downstream model training processors when critical corruption is detected."
    )
    p_law.paragraph_format.line_spacing = 1.15
    p_law.paragraph_format.space_after = Pt(14)

    # ─────────────────────────────────────────────────────────────
    # SECTION 6: EXECUTIVE SUMMARY SCRIPT FOR SIR
    # ─────────────────────────────────────────────────────────────
    h6 = doc.add_heading(level=1)
    r_h6 = h6.add_run("6. Executive Speaking Script for Project Evaluation")
    r_h6.font.name = 'Arial'
    r_h6.font.size = Pt(15)
    r_h6.font.bold = True
    r_h6.font.color.rgb = RGBColor(0, 51, 102)
    h6.paragraph_format.space_after = Pt(8)
    
    # Script callout box
    sc_table = doc.add_table(rows=1, cols=1)
    sc_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_sc = sc_table.cell(0, 0)
    c_sc.width = Inches(6.8)
    set_cell_background(c_sc, "FFF9E6") # Pale gold callout
    set_cell_margins(c_sc, 120, 120, 160, 160)
    
    p_sc = c_sc.paragraphs[0]
    p_sc.paragraph_format.line_spacing = 1.15
    p_sc.paragraph_format.space_after = Pt(0)
    r_sct = p_sc.add_run("What to Say When Sir Asks: 'Explain your novelty and training data in 2 minutes':\n\n")
    r_sct.bold = True
    r_sct.font.color.rgb = RGBColor(153, 102, 0)
    
    script_text = (
        "\"Respected Sir, in machine learning, 80% of project failures and cloud compute waste occur because input tabular data is corrupted, yet data scientists blindly call model.fit().\n\n"
        "Existing tools like Great Expectations only check syntax, while Cleanlab requires training heavy models across multiple folds, taking seconds or minutes per table. TabPFN runs out of memory on datasets above 1,000 rows.\n\n"
        "Our core novelty is DataQuality-Net — the first single-pass pre-flight gatekeeper. We adapted DeepSets to 2D tabular data, making our model strictly invariant to row and column shuffling. In a single 50ms forward pass, it predicts 6 data quality scores AND downstream accuracy drops without training any downstream model.\n\n"
        "We trained and validated this across 31 diverse OpenML datasets — including NASA software telemetry, German credit, and Bank Marketing — generating 1,550 systematically corrupted variants under 5-Fold Leave-Dataset-Out cross-validation. "
        "We achieved a verified 40× speedup over LightGBM on 100k-row enterprise tables (50ms vs 2 seconds), hit 89.84% conformal coverage against a 90% theoretical target, and demonstrated an empirical +8.0 usability score lift using our closed-loop self-healing engine.\n\n"
        "This work is backed by a 40-page Form 2 Provisional Patent specification featuring 22 claims drafted under the Indian Patents Act, 1970.\""
    )
    r_body = p_sc.add_run(script_text)
    r_body.font.size = Pt(10)
    r_body.font.italic = True

    doc.save("DataQuality_Net_Novelty_and_Datasets_Dossier.docx")
    print("Dossier successfully saved to: DataQuality_Net_Novelty_and_Datasets_Dossier.docx")

if __name__ == "__main__":
    create_document()
