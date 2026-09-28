#!/usr/bin/env python3
"""
Comprehensive re-analysis addressing reviewer concerns:
1. FDR correction on MetaCycle cosinor results
2. Bootstrap CI for acrophase/amplitude
3. Formal definitions of Rhythm Score and Circadian Congruence Score
4. Validation against known chronotherapeutic drugs
5. ADRA1A/B cross-validation with GTEx/literature
6. Robustness: sensitivity to alternative thresholds
"""
import os, sys
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import json
import warnings
warnings.filterwarnings('ignore')

BASE='/export/home/kongyan/project/rhyweb'
GENE_DIR=f'{BASE}/data/rhythmic_genes'
EXPR_DIR=f'{BASE}/data/rhythmic_expr'
OUT_DIR=f'{BASE}/reanalysis_output'
os.makedirs(OUT_DIR, exist_ok=True)

ZT=np.array([0,2,4,6,8,10,12,14,16,18,20,22])
ACTUAL_HOURS=(ZT+6)%24

# =============================================================================
# Load data
# =============================================================================
tissue_data={}
for f in sorted(os.listdir(GENE_DIR)):
    if f.startswith('rhythmic_genes_') and f.endswith('_p0.05.csv'):
        tissue=f.replace('rhythmic_genes_','').replace('_p0.05.csv','')
        tissue_data[tissue]=pd.read_csv(os.path.join(GENE_DIR,f))
tissues=sorted(tissue_data.keys())

expr_data={}
for f in sorted(os.listdir(EXPR_DIR)):
    if f.endswith('_processed.csv'):
        tissue=f.replace('_processed.csv','')
        expr_data[tissue]=pd.read_csv(os.path.join(EXPR_DIR,f))

# Identify full vs pre-filtered
full_tissues=[]; prefilt_tissues=[]
for tissue in tissues:
    df=expr_data.get(tissue)
    if df is not None and len(df)>5000:
        full_tissues.append(tissue)
    else:
        prefilt_tissues.append(tissue)

print(f"Full transcriptome: {len(full_tissues)} tissues")
print(f"Pre-filtered: {len(prefilt_tissues)} tissues -> {prefilt_tissues}")

# =============================================================================
# 1. FDR CORRECTION
# =============================================================================
print("\n" + "="*70)
print("1. FDR CORRECTION ON ALL GENE-TISSUE PAIRS")
print("="*70)

# For full tissues: compute cosinor p-value for ALL genes
# For pre-filtered tissues: use existing MetaCycle p-values
all_pairs=[]

# Cosinor fitting function (linear regression, period=24)
def cosinor_pvalue(vals, zt=ZT):
    expr=np.array(vals,dtype=float)
    mask=expr>0
    if mask.sum()<4:
        return np.nan, np.nan, np.nan
    t=zt[mask]; y=expr[mask]
    X=np.column_stack([np.ones_like(t),np.cos(2*np.pi*t/24),np.sin(2*np.pi*t/24)])
    try:
        beta=np.linalg.lstsq(X,y,rcond=None)[0]
        mesor=beta[0]; amp=np.sqrt(beta[1]**2+beta[2]**2)
        acrophase=np.arctan2(beta[2],beta[1])*24/(2*np.pi)
        if acrophase<0: acrophase+=24
        fitted_full=mesor+amp*np.cos(2*np.pi*(t-acrophase)/24)
        ss_res=np.sum((y-fitted_full)**2)
        ss_tot=np.sum((y-np.mean(y))**2)
        r2=1-ss_res/ss_tot if ss_tot>0 else 0
        n=len(t); p=3
        pval=1-stats.f.cdf(((ss_tot-ss_res)/(p-1))/(ss_res/(n-p)),p-1,n-p) if n>p and ss_tot>0 and ss_res>0 else 1
        return pval, r2, amp/mesor if mesor>0 else 0
    except:
        return np.nan, np.nan, np.nan

for tissue in full_tissues:
    expr=expr_data[tissue]
    rh_genes=set(tissue_data[tissue]['CycID'].values)
    count=0
    for _, row in expr.iterrows():
        gene=row['Human_Symbol']
        vals=row[[f'ZT{h:02d}' for h in range(0,24,2)]].values.astype(float)
        pval,r2,relamp=cosinor_pvalue(vals)
        if np.isnan(pval):
            continue
        all_pairs.append({'tissue':tissue,'gene':gene,'pval':pval,'r2':r2,
                          'relamp':relamp,'nominal':pval<0.05,'source':'computed'})
        count+=1
    print(f"{tissue}: {count} gene-tissue pairs computed")

# Pre-filtered tissues use existing MetaCycle p-values
for tissue in prefilt_tissues:
    df=tissue_data[tissue]
    for _, row in df.iterrows():
        all_pairs.append({'tissue':tissue,'gene':row['CycID'],
                          'pval':row['meta2d_pvalue'],'r2':np.nan,
                          'relamp':row.get('meta2d_rAMP',np.nan),
                          'nominal':row['meta2d_pvalue']<0.05,'source':'metacycle'})
    print(f"{tissue}: {len(df)} pre-filtered pairs added")

df_all=pd.DataFrame(all_pairs)
print(f'\nTotal pairs: {len(df_all)}')

# BH FDR correction (global)
pvals=np.array(df_all['pval'].values)
sidx=np.argsort(pvals)
sp=pvals[sidx]
n=len(sp)
bh_thr=sp*n/np.arange(1,n+1)
bh_crit=np.minimum.accumulate(bh_thr[::-1])[::-1]
bh_sig=sp<=bh_crit
k=np.max(np.where(bh_sig)[0])+1 if np.any(bh_sig) else 0
q=np.ones(n)
q[sidx[:k]]=sp[:k]*n/np.arange(1,k+1)
df_all['qval']=q[np.argsort(sidx)]

# Also compute BY (more conservative) FDR
# BY: q_i = p_i * n / (m * c(m)) where c(m) = sum(1/i) for i=1..m
from fractions import Fraction
# Use harmonic number approximation
c_m = np.sum(1.0/np.arange(1,n+1))
by_thr=sp*n*c_m/np.arange(1,n+1)
by_crit=np.minimum.accumulate(by_thr[::-1])[::-1]
by_sig=sp<=by_crit
k_by=np.max(np.where(by_sig)[0])+1 if np.any(by_sig) else 0
q_by=np.ones(n)
q_by[sidx[:k_by]]=sp[:k_by]*n*c_m/np.arange(1,k_by+1)
df_all['qval_BY']=q_by[np.argsort(sidx)]

# Summary
nom_sig=(df_all.pval<0.05).sum()
fdr05_sig=(df_all.qval<0.05).sum()
fdr10_sig=(df_all.qval<0.10).sum()
by05_sig=(df_all.qval_BY<0.05).sum()
by10_sig=(df_all.qval_BY<0.10).sum()

print(f'\n--- FDR Results ---')
print(f'Nominal p<0.05: {nom_sig} ({nom_sig/len(df_all)*100:.2f}%)')
print(f'BH FDR q<0.05: {fdr05_sig} ({fdr05_sig/nom_sig*100:.1f}% of nominal)')
print(f'BH FDR q<0.10: {fdr10_sig} ({fdr10_sig/nom_sig*100:.1f}% of nominal)')
print(f'BY FDR q<0.05: {by05_sig} ({by05_sig/nom_sig*100:.1f}% of nominal)')
print(f'BY FDR q<0.10: {by10_sig} ({by10_sig/nom_sig*100:.1f}% of nominal)')

# Unique genes
uniq_nom=set(df_all[df_all.pval<0.05].gene)
uniq_fdr05=set(df_all[df_all.qval<0.05].gene)
uniq_fdr10=set(df_all[df_all.qval<0.10].gene)
uniq_by05=set(df_all[df_all.qval_BY<0.05].gene)
print(f'\nUnique genes: nominal={len(uniq_nom)}, FDR05={len(uniq_fdr05)}, FDR10={len(uniq_fdr10)}, BY05={len(uniq_by05)}')

# Per-tissue breakdown
print(f'\n--- Per-Tissue FDR Results (full transcriptome tissues) ---')
for tissue in full_tissues:
    sub=df_all[df_all.tissue==tissue]
    n_nom=(sub.pval<0.05).sum()
    n_fdr05=(sub.qval<0.05).sum()
    n_fdr10=(sub.qval<0.10).sum()
    total=len(sub)
    print(f'{tissue}: nom={n_nom}/{total}({n_nom/total*100:.1f}%), fdr05={n_fdr05}({n_fdr05/total*100:.1f}%), fdr10={n_fdr10}({n_fdr10/total*100:.1f}%)')

# Breadth analysis
def cat(breadth):
    u=sum(1 for b in breadth.values() if b>=30)
    m=sum(1 for b in breadth.values() if 5<=b<30)
    s=sum(1 for b in breadth.values() if b<5)
    return u,m,s,len(breadth)

print(f'\n--- Gene Breadth (unique genes) ---')
print(f'{"Threshold":<12} {"Ubiquitous(≥30)":>16} {"Multi(5-29)":>12} {"Specific(<5)":>12} {"Total":>8}')
print('-'*60)

breadth_results={}
for label,thresh in [('nominal',0.05),('FDR05',0.05),('FDR10',0.10),('BY05',0.05),('BY10',0.10)]:
    if label=='nominal':
        sig=df_all[df_all.pval<0.05]
    elif label.startswith('BY'):
        sig=df_all[df_all.qval_BY<thresh]
    else:
        sig=df_all[df_all.qval<thresh]
    breadth=sig.groupby('gene').tissue.count().to_dict()
    u,m,s,t=cat(breadth)
    breadth_results[label]={'breadth':breadth,'ubiquitous':u,'multi':m,'specific':s,'total':t}
    print(f'{label:<12} {u:>8}({u/max(t,1)*100:.3f}%) {m:>10}({m/t*100:.2f}%) {s:>10}({s/t*100:.2f}%) {t:>8}')

# ADRA1A/B specific
print(f'\n--- ADRA1A/B Tissue Distribution ---')
for gene in ['ADRA1A','ADRA1B']:
    sub=df_all[df_all.gene==gene]
    nom=(sub.pval<0.05).sum()
    fdr05=(sub.qval<0.05).sum()
    fdr10=(sub.qval<0.10).sum()
    by05=(sub.qval_BY<0.05).sum()
    print(f'\n{gene}:')
    print(f'  Nominal p<0.05: {nom} tissues')
    print(f'  BH FDR q<0.05: {fdr05} tissues')
    print(f'  BH FDR q<0.10: {fdr10} tissues')
    print(f'  BY FDR q<0.05: {by05} tissues')
    # Show top tissues by p-value
    top=sub.nsmallest(5,'pval')[['tissue','pval','qval','qval_BY','r2']]
    print(top.to_string(index=False))

# Save
df_all.to_csv(f'{OUT_DIR}/all_genes_cosinor_fdr.csv',index=False)
print(f'\nSaved to {OUT_DIR}/all_genes_cosinor_fdr.csv')

# Save summary JSON
summary={
    'total_pairs':int(len(df_all)),
    'nominal_p05':int(nom_sig),
    'bh_fdr05':int(fdr05_sig),
    'bh_fdr10':int(fdr10_sig),
    'by_fdr05':int(by05_sig),
    'by_fdr10':int(by10_sig),
    'unique_nominal':int(len(uniq_nom)),
    'unique_fdr05':int(len(uniq_fdr05)),
    'unique_fdr10':int(len(uniq_fdr10)),
    'unique_by05':int(len(uniq_by05)),
    'breadth_results':{k:{'ubiquitous':v['ubiquitous'],'multi':v['multi'],'specific':v['specific'],'total':v['total']} for k,v in breadth_results.items()},
    'adra1a_nominal':int((df_all[df_all.gene=='ADRA1A'].pval<0.05).sum()),
    'adra1a_fdr05':int((df_all[df_all.gene=='ADRA1A'].qval<0.05).sum()),
    'adra1b_nominal':int((df_all[df_all.gene=='ADRA1B'].pval<0.05).sum()),
    'adra1b_fdr05':int((df_all[df_all.gene=='ADRA1B'].qval<0.05).sum()),
}
with open(f'{OUT_DIR}/fdr_summary.json','w') as f:
    json.dump(summary,f,indent=2)
print(f'Summary saved to {OUT_DIR}/fdr_summary.json')
PY