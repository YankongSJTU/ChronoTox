# ChronoTheraAtlas

**A Circadian Rhythm Atlas-Based Intelligent Platform for Multi-Tissue Chronotoxicity Exploration and Drug Timing Hypothesis Generation**

🔗 Live server: https://bmap.sjtu.edu.cn/ChronoTheraAtlas

---

## 📋 Repository Structure

```
ChronoTheraAtlas/
│
├── 📁 analysis/                        # Core computational pipeline
│   ├── metacycle_ensemble.py            # ★ MetaCycle ensemble (JTK+ARSER+Lomb-Scargle)
│   ├── metacycle_ensemble_v2.py         # ★ MetaCycle ensemble v2 (with CLI)
│   ├── metacycle_legacy.py              # Legacy MetaCycle implementation
│   ├── cosinor_fitting.py               # Cosinor model fitting module
│   ├── expression_visualisation.py      # Expression density heatmaps & curves
│   ├── fdr_correction.py               # Per-tissue BH-FDR correction
│   ├── run_fdr_analysis.py             # Full FDR analysis pipeline
│   ├── reanalysis_fdr.py                # FDR-corrected rhythmicity tables
│   ├── reanalysis_full.py               # Full reanalysis workflow
│   ├── reanalysis_final.py              # Final reanalysis with bootstrap CIs
│   ├── step2_rerun_stats.py            # Pathway enrichment statistics
│   └── generate_supplementary_tables.py # Supplementary Table S1-S6
│
├── 📁 data/                            # All data files
│   ├── data.csv                        # 63-tissue × 12-timepoint expression matrix
│   ├── rhythmic_expr/                  # 63 per-tissue FPKM time series (67 files)
│   ├── rhythmic_genes/                  # 63 per-tissue rhythmic gene tables
│   └── *.csv                           # MetaCycle/cosinor fit results
│
├── 📁 webapp/                          # Flask webserver (local deployment)
│   ├── app.py                          # Main server (504 lines)
│   ├── templates/index.html            # HTML template
│   └── static/                         # Static assets
│
├── 📁 geo_raw/                          # GEO original data
│   └── GSE98965_baboon_tissue_expression_FPKM.csv.gz

---

## 🔬 Core Analysis Methods

### 1. MetaCycle Ensemble Analysis

MetaCycle integrates three complementary rhythmicity detection algorithms:

| Algorithm | Method | Strengths |
|-----------|--------|-----------|
| **JTK_CYCLE** | Non-parametric, Bonferroni-corrected | Robust to outliers, handles uneven sampling |
| **ARSER** | Autoregressive spectral estimation | High sensitivity, accounts for serial correlation |
| **Lomb-Scargle** | Least-squares spectral analysis | Works with missing data, handles irregular time series |

**Implementation:** `analysis/metacycle_ensemble.py` and `analysis/metacycle_ensemble_v2.py`

```bash
python analysis/metacycle_ensemble.py --data data.csv --method metacycle
```

**Consensus output** (used throughout the paper):
- Consensus R² (model fit quality)
- Consensus p-value (meta2d_p)
- Amplitude (meta2d_AMP)
- Acrophase (meta2d_phase)
- Relative amplitude (meta2d_rAMP)

### 2. Dynamic Time Warping (DTW) for Phase Synchronization

DTW aligns target-gene expression profiles against pathway gene sets to quantify:
- **Phase concordance**: How synchronized are targets and pathway members?
- **Phase lag**: What's the optimal alignment shift?
- **Chronotoxicity index**: When do target peaks coincide with vulnerability windows?

```python
from analysis.metacycle_ensemble import dtw_alignment
dtw_alignment(target_profile, pathway_profile, radius=1)
```

### 3. Rhythm Score (RS)

```
RS = 0.7 × R² + 0.3 × min(A/M, 1)
```
Where:
- R² = coefficient of determination from cosinor fit
- A = oscillation amplitude
- M = mean expression (mesor)

Higher RS = more robust, high-amplitude rhythms.

### 4. Circadian Congruence Score (CCS)

```
CCS = Σ_t [RS_target(t) × enrichment(t) × (1 − |Δt_phase| / 12)]
```
Where:
- RS_target(t) = Rhythm Score of drug target in tissue t
- enrichment(t) = proportion of target pathway genes rhythmic in tissue t
- Δt_phase = acrophase difference between target peak and pathway peak (hours)
- 12 = maximum possible phase lag (hours)

### 5. FDR Correction

Per-tissue Benjamini-Hochberg false discovery rate correction (q<0.05):
- 12,612 genes nominal significant → 12,607 genes FDR-significant (99.96% retention)
- 9 genes ubiquitously rhythmic (≥30 tissues)
- 5,650 genes multi-tissue rhythmic (5-29 tissues)
- 6,953 genes tissue-specific (<5 tissues)

**Implementation:** `analysis/fdr_correction.py`

---

## 🤖 LLM Prompt Template

Located in: `reanalysis_output/Supplementary_Table_S4_LLM_Config.txt`

**Model:** DeepSeek-R1 (8B, locally deployed, 4-bit quantized)
**Inference:** vLLM/Ollama | Temperature: 0.7 | Top-p: 0.9 | Max tokens: 2048

### System Prompt
```text
You are a chronopharmacology research assistant. Your role is to synthesize
quantitative circadian rhythmicity data into interpretable research hypotheses
for clinical investigation. You will receive structured data including:
(a) drug target rhythmic profiles across 63 tissues,
(b) pathway enrichment statistics,
(c) clinical context variables.

IMPORTANT: All outputs are research hypotheses derived from computational
analysis of non-human primate transcriptomic data. They must not be presented
as validated clinical dosing recommendations. Always include the disclaimer:
'These outputs are research hypotheses for clinical investigation only and
must not be used to alter treatment without professional clinical review.'

Provide structured reports with the following sections:
1. Target Rhythmicity Summary (peak times, amplitudes, tissues)
2. Pathway Consistency Assessment (enrichment, coordination)
3. Chronotherapeutic Hypothesis (proposed timing window with rationale)
4. Limitations (species gap, transcript-to-protein gap, uncertainty intervals)
5. Clinical Investigation Recommendations (comparison with known schedules)
```

### User Prompt Template
```text
Analyze the chronotherapeutic potential of {drug_name} ({drugbank_id}).

Target rhythmicity data:
{target_data_json}

Pathway enrichment data:
{pathway_data_json}

Clinical context:
{clinical_context_json}

Generate a structured research hypothesis report for clinical investigation.
```

### Data Package Structure
- `target_data`: JSON array of `{gene, tissue, acrophase_actual, amplitude, rhythm_score, fdr_qvalue, period}`
- `pathway_data`: JSON array of `{pathway_id, pathway_name, rhythmic_ratio, top_rhythmic_genes}`
- `clinical_context`: JSON object with `{comorbidities, age, sex, organ_function}`

---

## 🧬 Data Lineage

```
GEO GSE98965 (baboon multi-tissue diurnal transcriptome)
    ↓ baboon-to-human ortholog mapping (biomaRt)
data/rhythmic_expr/{TISSUE}_processed.csv  (63 tissues × 12 ZT points, FPKM)
    ↓ gene-level aggregation + noise filtering (mean FPKM < 0.1)
    ↓ MetaCycle ensemble (JTK + ARSER + Lomb-Scargle)
analysis/metacycle_ensemble.py
    ↓ per-tissue BH-FDR correction (q<0.05)
reanalysis_output/{GENE}_fdr_tissues.csv
    ↓ drug-target association (DrugBank v5.1.13)
analysis/reanalysis_final.py
    ↓ pathway enrichment (KEGG)
reanalysis_output/pathway_enrichment_fdr.csv
    ↓ DTW phase synchronization
reanalysis_output/[DTW alignment results]
    ↓ LLM semantic synthesis (DeepSeek-R1)
附件/Midodrine_AI_report.pdf
```

---

## 🚀 Quick Start

### Requirements
- Python 3.10+
- pandas, numpy, matplotlib, seaborn, scipy
- Flask (for webapp)


### Run Web Server
```bash
python webapp/app.py
# Open http://localhost:5000
```

### Reproduce Analysis
```bash
# Step 1: MetaCycle ensemble analysis
python analysis/metacycle_ensemble.py --data data.csv --method metacycle

# Step 2: Per-tissue FDR correction
python analysis/fdr_correction.py

# Step 3: Full reanalysis with bootstrap CIs
python analysis/reanalysis_final.py

# Step 4: Generate supplementary tables
python analysis/generate_supplementary_tables.py
```

---

## 📊 Key Results

### 63-Tissue Circadian Atlas
- 12,612 unique genes exhibit significant circadian oscillations
- 1,057 rhythmic genes per tissue (average)
- 9 genes ubiquitously rhythmic across ≥30 tissues
- Median period: 23.5 hours

### Midodrine Case Study (ADRA1B)
- **9 FDR-significant tissues** (q<0.05)
- Strongest rhythm: Aorta (AOR), peak at ~05:56, R²=0.781
- **Recommended dosing window:** 06:00 (dawn transition)
- Convergence window: 04:00-08:00

### Validation (Supplementary Table S5)
- Predicted timing aligns with established chronotherapeutic schedules for:
  - Midodrine (dawn dosing)
  - Prednisone (morning dosing)
  - Aspirin (morning dosing)

---

## 📜 Citation

If you use this work, please cite:

```bibtex
@article{chronotheraAtlas2025,
  title={ChronoTheraAtlas: A Circadian Rhythm Atlas-Based Intelligent Platform
         for Multi-Tissue Chronotoxicity Exploration and Drug Timing
         Hypothesis Generation},
  author={Kong, Y. and Lu, H.},
  journal={Manuscript under revision},
  year={2025},
  url={https://bmap.sjtu.edu.cn/ChronoTheraAtlas}
}
```

---

## ⚠️ Disclaimer

All outputs from ChronoTheraAtlas are **research hypotheses** for clinical
investigation only. They are based on computational analysis of non-human
primate transcriptomic data and **must not be used to alter treatment**
without professional clinical review.

---

## 📄 License

MIT License - See [LICENSE](LICENSE)

---

## 🤝 Contributing

Issues and pull requests welcome. For major changes, please open an issue first
to discuss what you'd like to change.

---

## 📫 Contact

- **Project Lead:** Yan Kong 
