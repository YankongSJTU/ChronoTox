"""
Comprehensive re-analysis of ChronoTheraAtlas data addressing reviewer concerns.

Key analyses:
1. FDR correction on MetaCycle results (BH and BY)
2. Bootstrap confidence intervals for acrophase and amplitude
3. Formal definition of Rhythm Score and Circadian Congruence Score
4. Cross-validation with GTEx/ literature for key targets
5. Validation against known chronotherapeutic drugs
6. Robustness: sensitivity to alternative rhythmicity thresholds
7. Gene breadth and ubiquitous analysis with FDR-corrected data
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import json
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = '/export/home/kongyan/project/rhyweb'
GENE_DIR = f'{BASE_DIR}/data/rhythmic_genes'
EXPR_DIR = f'{BASE_DIR}/data/rhythmic_expr'
DATA_CSV = f'{BASE_DIR}/data.csv'

ZT_HOURS = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
ACTUAL_HOURS = (ZT_HOURS + 6) % 24

# ============================================================================
# 1. FDR CORRECTION ON METACYCLE RESULTS
# ============================================================================
print("=" * 70)
print("1. FDR CORRECTION ANALYSIS")
print("=" * 70)

# Load all rhythmic gene results
tissue_data = {}
for f in sorted(os.listdir(GENE_DIR)):
    if f.startswith('rhythmic_genes_') and f.endswith('_p0.05.csv'):
        tissue = f.replace('rhythmic_genes_', '').replace('_p0.05.csv', '')
        df = pd.read_csv(os.path.join(GENE_DIR, f))
        tissue_data[tissue] = df

tissues = sorted(tissue_data.keys())
print(f"Loaded {len(tissues)} tissues")

# For each tissue, apply FDR correction to meta2d_pvalue
# Collect all p-values across all tissues for global FDR
all_pvals = []
tissue_gene_pvals = {}  # tissue -> list of (gene, pval)

for tissue, df in tissue_data.items():
    pvals = df['meta2d_pvalue'].values
    genes = df['CycID'].values
    tissue_gene_pvals[tissue] = list(zip(genes, pvals))
    all_pvals.extend(pvals)

all_pvals = np.array(all_pvals)
print(f"Total gene-tissue pairs with p<0.05 (nominal): {len(all_pvals)}")
print(f"Unique genes across all tissues: {len(set().union(*[set(g for g,_ in tg) for tg in tissue_gene_pvals.values()]))}")

# Apply BH FDR correction globally (across all gene-tissue pairs)
from scipy.stats import false_discovery_control
# Manual BH FDR
sorted_idx = np.argsort(all_pvals)
sorted_pvals = all_pvals[sorted_idx]
n_tests = len(sorted_pvals)
bh_thresholds = sorted_pvals * n_tests / np.arange(1, n_tests + 1)
bh_critical = np.minimum.accumulate(bh_thresholds[::-1])[::-1]
bh_significant = sorted_pvals <= bh_critical
# Find the largest k where p(k) <= threshold
if np.any(bh_significant):
    bh_k = np.max(np.where(bh_significant)[0]) + 1
    bh_qvals = np.zeros(n_tests)
    bh_qvals[sorted_idx[:bh_k]] = sorted_pvals[:bh_k] * n_tests / np.arange(1, bh_k + 1)
    bh_qvals[sorted_idx[bh_k:]] = 1.0
else:
    bh_k = 0
    bh_qvals = np.ones(n_tests)

# Map back to tissue-gene pairs
global_bh_q = {}
pos = 0
for tissue in tissues:
    n_genes = len(tissue_gene_pvals[tissue])
    for i, (gene, pval) in enumerate(tissue_gene_pvals[tissue]):
        global_bh_q[(tissue, gene)] = bh_qvals[pos + i]
    pos += n_genes

# Compute per-tissue FDR counts
print(f"\n--- Per-Tissue FDR Results (global BH, alpha=0.05) ---")
print(f"{'Tissue':<10} {'p<0.05':>8} {'FDR<0.05':>10} {'FDR%':>8} {'FDR<0.1':>10} {'FDR<0.2':>10}")
print("-" * 50)

fdr_summary = {}
for tissue in tissues:
    df = tissue_data[tissue]
    genes = df['CycID'].values
    pvals = df['meta2d_pvalue'].values
    qvals = np.array([global_bh_q.get((tissue, g), 1.0) for g in genes])

    n_p05 = np.sum(pvals < 0.05)
    n_fdr05 = np.sum(qvals < 0.05)
    n_fdr10 = np.sum(qvals < 0.10)
    n_fdr20 = np.sum(qvals < 0.20)

    fdr_summary[tissue] = {
        'total_genes': len(df),
        'nominal_p05': n_p05,
        'fdr05': n_fdr05,
        'fdr10': n_fdr10,
        'fdr20': n_fdr20,
        'pct_fdr05': n_fdr05 / len(df) * 100 if len(df) > 0 else 0,
    }

    print(f"{tissue:<10} {n_p05:>8} {n_fdr05:>10} {n_fdr05/len(df)*100:>7.1f}% {n_fdr10:>10} {n_fdr20:>10}")

# Summary statistics
total_nominal = sum(fdr_summary[t]['nominal_p05'] for t in tissues)
total_fdr05 = sum(fdr_summary[t]['fdr05'] for t in tissues)
total_fdr10 = sum(fdr_summary[t]['fdr10'] for t in tissues)
total_fdr20 = sum(fdr_summary[t]['fdr20'] for t in tissues)

print(f"\n--- Global Summary ---")
print(f"Total nominal p<0.05 gene-tissue pairs: {total_nominal}")
print(f"After BH FDR (q<0.05): {total_fdr05} ({total_fdr05/total_nominal*100:.1f}% of nominal)")
print(f"After BH FDR (q<0.10): {total_fdr10} ({total_fdr10/total_nominal*100:.1f}% of nominal)")
print(f"After BH FDR (q<0.20): {total_fdr20} ({total_fdr20/total_nominal*100:.1f}% of nominal)")

# Unique genes passing FDR at different thresholds
unique_nominal = set()
unique_fdr05 = set()
unique_fdr10 = set()
unique_fdr20 = set()

for tissue in tissues:
    df = tissue_data[tissue]
    genes = df['CycID'].values
    pvals = df['meta2d_pvalue'].values
    qvals = np.array([global_bh_q.get((tissue, g), 1.0) for g in genes])

    unique_nominal.update(genes[pvals < 0.05])
    unique_fdr05.update(genes[qvals < 0.05])
    unique_fdr10.update(genes[qvals < 0.10])
    unique_fdr20.update(genes[qvals < 0.20])

print(f"\n--- Unique Genes (across all tissues) ---")
print(f"Nominal p<0.05: {len(unique_nominal)} unique genes")
print(f"FDR q<0.05: {len(unique_fdr05)} unique genes ({len(unique_fdr05)/len(unique_nominal)*100:.1f}% of nominal)")
print(f"FDR q<0.10: {len(unique_fdr10)} unique genes ({len(unique_fdr10)/len(unique_nominal)*100:.1f}% of nominal)")
print(f"FDR q<0.20: {len(unique_fdr20)} unique genes ({len(unique_fdr20)/len(unique_nominal)*100:.1f}% of nominal)")

# Gene breadth analysis with FDR-corrected data
print(f"\n--- Gene Breadth Analysis (FDR-corrected) ---")

# Breadth = number of tissues where gene is rhythmic (FDR<0.05)
breadth_fdr05 = {}
breadth_fdr10 = {}
breadth_fdr20 = {}
breadth_nominal = {}

for tissue in tissues:
    df = tissue_data[tissue]
    genes = df['CycID'].values
    pvals = df['meta2d_pvalue'].values
    qvals = np.array([global_bh_q.get((tissue, g), 1.0) for g in genes])

    for g, p, q in zip(genes, pvals, qvals):
        if p < 0.05:
            breadth_nominal[g] = breadth_nominal.get(g, 0) + 1
        if q < 0.05:
            breadth_fdr05[g] = breadth_fdr05.get(g, 0) + 1
        if q < 0.10:
            breadth_fdr10[g] = breadth_fdr10.get(g, 0) + 1
        if q < 0.20:
            breadth_fdr20[g] = breadth_fdr20.get(g, 0) + 1

# Categorize genes
def categorize_breadth(breadth_dict, total_tissues=63):
    ubiquitous = sum(1 for b in breadth_dict.values() if b >= 30)
    multi = sum(1 for b in breadth_dict.values() if 5 <= b < 30)
    specific = sum(1 for b in breadth_dict.values() if b < 5)
    return ubiquitous, multi, specific

ub_nom, mu_nom, sp_nom = categorize_breadth(breadth_nominal)
ub_05, mu_05, sp_05 = categorize_breadth(breadth_fdr05)
ub_10, mu_10, sp_10 = categorize_breadth(breadth_fdr10)
ub_20, mu_20, sp_20 = categorize_breadth(breadth_fdr20)

print(f"Breadth category | Nominal p<0.05 | FDR q<0.05 | FDR q<0.10 | FDR q<0.20")
print(f"Ubiquitous (≥30) | {ub_nom:>6} ({ub_nom/len(breadth_nominal)*100:.2f}%) | {ub_05:>6} ({ub_05/max(len(breadth_fdr05),1)*100:.2f}%) | {ub_10:>6} | {ub_20:>6}")
print(f"Multi-tissue(5-29)| {mu_nom:>6} ({mu_nom/len(breadth_nominal)*100:.2f}%) | {mu_05:>6} ({mu_05/max(len(breadth_fdr05),1)*100:.2f}%) | {mu_10:>6} | {mu_20:>6}")
print(f"Tissue-specific(<5)| {sp_nom:>6} ({sp_nom/len(breadth_nominal)*100:.2f}%) | {sp_05:>6} ({sp_05/max(len(breadth_fdr05),1)*100:.2f}%) | {sp_10:>6} | {sp_20:>6}")
print(f"Total unique genes | {len(breadth_nominal):>6} | {len(breadth_fdr05):>6} | {len(breadth_fdr10):>6} | {len(breadth_fdr20):>6}")

# Save FDR-corrected results
fdr_results = []
for tissue in tissues:
    df = tissue_data[tissue]
    genes = df['CycID'].values
    for _, row in df.iterrows():
        gene = row['CycID']
        qval = global_bh_q.get((tissue, gene), 1.0)
        fdr_results.append({
            'Tissue': tissue,
            'Gene': gene,
            'meta2d_pvalue': row['meta2d_pvalue'],
            'meta2d_BH_Q_global': qval,
            'meta2d_period': row['meta2d_period'],
            'meta2d_phase': row['meta2d_phase'],
            'meta2d_AMP': row['meta2d_AMP'],
            'meta2d_Base': row['meta2d_Base'],
            'meta2d_rAMP': row['meta2d_rAMP'],
            'nominal_sig': row['meta2d_pvalue'] < 0.05,
            'fdr05_sig': qval < 0.05,
            'fdr10_sig': qval < 0.10,
            'fdr20_sig': qval < 0.20,
        })

fdr_df = pd.DataFrame(fdr_results)
fdr_df.to_csv(f'{BASE_DIR}/metacycle_fdr_corrected_results.csv', index=False)
print(f"\nFDR-corrected results saved to metacycle_fdr_corrected_results.csv")
print(f"Nominal p<0.05 gene-tissue pairs: {total_nominal}")
print(f"FDR q<0.05 gene-tissue pairs: {total_fdr05} (retention rate: {total_fdr05/total_nominal*100:.1f}%)")
print(f"Unique genes nominal: {len(unique_nominal)}, FDR q<0.05: {len(unique_fdr05)}")
print(f"Retention of unique genes: {len(unique_fdr05)/len(unique_nominal)*100:.1f}%")
