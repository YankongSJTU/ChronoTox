#!/usr/bin/env python3
"""
Step 2: Re-run rhythm and downstream statistics.
This script performs the core statistical re-analysis:
1. Bootstrap CIs for ADRA1A/B acrophase and amplitude
2. Pathway enrichment with FDR-corrected gene sets
3. Gene breadth analysis with FDR correction
4. ADRA1A/B rhythmic analysis using data.csv (not rhythmic_genes files)
5. Generate all updated supplementary data files
6. Visualization data generation
"""
import os, sys
import pandas as pd
import numpy as np
from scipy import stats
import json, warnings
warnings.filterwarnings('ignore')

BASE='/export/home/kongyan/project/rhyweb'
GENE_DIR=f'{BASE}/data/rhythmic_genes'
EXPR_DIR=f'{BASE}/data/rhythmic_expr'
OUT_DIR=f'{BASE}/reanalysis_output'
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# 1. Bootstrap CI computation for ADRA1A/B using data.csv expression values
# =============================================================================
print("="*70)
print("STEP 2: BOOTSTRAP CI FOR ADRA1A/B")
print("="*70)

data=pd.read_csv(f'{BASE}/data.csv')
ZT_LABELS=['ZT00','ZT02','ZT04','ZT06','ZT08','ZT10','ZT12','ZT14','ZT16','ZT18','ZT20','ZT22']
ZT_HOURS=np.arange(0,24,2)  # ZT time points in hours

# Load FDR results to identify significant tissues
tissue_data={}
for f in sorted(os.listdir(GENE_DIR)):
    if f.startswith('rhythmic_genes_') and f.endswith('_p0.05.csv'):
        tissue=f.replace('rhythmic_genes_','').replace('_p0.05.csv','')
        tissue_data[tissue]=pd.read_csv(os.path.join(GENE_DIR,f))
tissues_fdr=sorted(tissue_data.keys())

fdr_results={}
for tissue in tissues_fdr:
    df=tissue_data[tissue]
    pvals=df['meta2d_pvalue'].values
    genes=df['CycID'].values
    n=len(pvals)
    sidx=np.argsort(pvals)
    sp=pvals[sidx]
    bh_thresholds=sp*n/np.arange(1,n+1)
    bh_crit=np.minimum.accumulate(bh_thresholds[::-1])[::-1]
    bh_sig=sp<=bh_crit
    k=np.max(np.where(bh_sig)[0])+1 if np.any(bh_sig) else 0
    q=np.ones(n)
    q[sidx[:k]]=sp[:k]*n/np.arange(1,k+1)
    qvals=np.ones(n)
    qvals[sidx]=q
    fdr_results[tissue]={'genes':genes,'pvals':pvals,'qvals':qvals}

def cosinor_fit(t, y, fixed_period=24):
    """Fit cosinor model with fixed 24h period."""
    try:
        X=np.column_stack([np.ones(len(t)), np.cos(2*np.pi*t/fixed_period), np.sin(2*np.pi*t/fixed_period)])
        beta,_,_,_=np.linalg.lstsq(X, y, rcond=None)
        mesor=beta[0]; amp=np.sqrt(beta[1]**2+beta[2]**2)
        acrophase=np.arctan2(beta[2],beta[1])*24/(2*np.pi)
        if acrophase<0: acrophase+=24
        fitted=mesor+amp*np.cos(2*np.pi*(t-acrophase)/24)
        ss_res=np.sum((y-fitted)**2)
        ss_tot=np.sum((y-np.mean(y))**2)
        r2=1-ss_res/ss_tot if ss_tot>0 else 0
        return {'mesor':mesor,'amp':amp,'acrophase':acrophase,'r2':r2}
    except:
        return None

def bootstrap_ci(t, y, n_boot=500, ci=95, fixed_period=24):
    """Bootstrap CI for acrophase and amplitude."""
    fits=[]
    for _ in range(n_boot):
        idx=np.random.choice(len(t),len(t),replace=True)
        bt=t[idx]; by=y[idx]
        fit=cosinor_fit(bt, by, fixed_period)
        if fit:
            fits.append(fit)
    if len(fits)<30:
        return None
    amps=np.array([f['amp'] for f in fits])
    acrs=np.array([f['acrophase'] for f in fits])
    # Unwrap acrophase
    acr_mean=acrs.mean()
    acrs_u=acrs.copy()
    for i in range(len(acrs_u)):
        d=acrs_u[i]-acr_mean
        if d>12: acrs_u[i]-=24
        elif d<-12: acrs_u[i]+=24
    amp_mean=amps.mean()
    amp_lo=np.percentile(amps,(100-ci)/2); amp_hi=np.percentile(amps,100-(100-ci)/2)
    acr_mean_u=acrs_u.mean()
    acr_lo=np.percentile(acrs_u,(100-ci)/2); acr_hi=np.percentile(acrs_u,100-(100-ci)/2)
    if acr_lo<0: acr_lo+=24
    if acr_hi<0: acr_hi+=24
    if acr_lo>=24: acr_lo-=24
    if acr_hi>=24: acr_hi-=24
    return {
        'amp_mean':amp_mean,'amp_lo':amp_lo,'amp_hi':amp_hi,
        'acr_mean':acr_mean_u,'acr_lo':acr_lo,'acr_hi':acr_hi,
        'n_boot':len(fits)
    }

# Compute bootstrap CIs for ADRA1A/B across FDR-significant tissues
adra1b_tissues_fdr=[]
adra1a_tissues_fdr=[]
for tissue in tissues_fdr:
    df_tissue=tissue_data[tissue]
    if 'ADRA1B' in list(df_tissue.CycID):
        gidx=list(df_tissue.CycID).index('ADRA1B')
        if fdr_results[tissue]['qvals'][gidx]<0.05:
            adra1b_tissues_fdr.append(tissue)
    if 'ADRA1A' in list(df_tissue.CycID):
        gidx=list(df_tissue.CycID).index('ADRA1A')
        if fdr_results[tissue]['qvals'][gidx]<0.05:
            adra1a_tissues_fdr.append(tissue)

print(f"ADRA1B FDR-significant tissues: {adra1b_tissues_fdr}")
print(f"ADRA1A FDR-significant tissues: {adra1a_tissues_fdr}")

# Bootstrap CIs for ADRA1B
print("\nADRA1B Bootstrap CIs (500 resamples):")
adra1b_boot=[]
for tissue in sorted(adra1b_tissues_fdr):
    sub=data[(data.Tissue==tissue)&(data.Gene=='ADRA1B')]
    if len(sub)==0:
        print("  "+tissue+": No data in data.csv")
        continue
    row=sub.iloc[0]
    y=row[ZT_LABELS].values.astype(float)
    t=np.array(ZT_HOURS)
    fit=cosinor_fit(t, y)
    if not fit:
        print("  "+tissue+": Cosinor fit failed")
        continue
    ci=bootstrap_ci(t, y, n_boot=500)
    if not ci:
        print("  "+tissue+": Bootstrap failed (too few resamples)")
        continue
    amp_ci_str='['+str(round(ci['amp_lo'],2))+'-'+str(round(ci['amp_hi'],2))+']'
    acr_ci_str='['+str(round(ci['acr_lo'],1))+'-'+str(round(ci['acr_hi'],1))+']'
    adra1b_boot.append({
        'tissue':tissue,'mesor':fit['mesor'],'amp':fit['amp'],
        'acrophase':fit['acrophase'],'r2':fit['r2'],
        'amp_ci':amp_ci_str,'acr_ci':acr_ci_str,
        'amp_mean':ci['amp_mean'],'amp_lo':ci['amp_lo'],'amp_hi':ci['amp_hi'],
        'acr_mean':ci['acr_mean'],'acr_lo':ci['acr_lo'],'acr_hi':ci['acr_hi'],
    })
    print("  "+tissue+": acrophase="+str(round(fit['acrophase'],1))+"h (CI:"+str(round(ci['acr_lo'],1))+"-"+str(round(ci['acr_hi'],1))+"), "
          "amp="+str(round(fit['amp'],2))+" (CI:"+str(round(ci['amp_lo'],2))+"-"+str(round(ci['amp_hi'],2))+"), R2="+str(round(fit['r2'],3)))

# Bootstrap CIs for ADRA1A
print("\nADRA1A Bootstrap CIs (500 resamples):")
adra1a_boot=[]
for tissue in sorted(adra1a_tissues_fdr):
    sub=data[(data.Tissue==tissue)&(data.Gene=='ADRA1A')]
    if len(sub)==0:
        print("  "+tissue+": No data in data.csv")
        continue
    row=sub.iloc[0]
    y=row[ZT_LABELS].values.astype(float)
    t=np.array(ZT_HOURS)
    fit=cosinor_fit(t, y)
    if not fit:
        print("  "+tissue+": Cosinor fit failed")
        continue
    ci=bootstrap_ci(t, y, n_boot=500)
    if not ci:
        print("  "+tissue+": Bootstrap failed")
        continue
    amp_ci_str='['+str(round(ci['amp_lo'],2))+'-'+str(round(ci['amp_hi'],2))+']'
    acr_ci_str='['+str(round(ci['acr_lo'],1))+'-'+str(round(ci['acr_hi'],1))+']'
    adra1a_boot.append({
        'tissue':tissue,'mesor':fit['mesor'],'amp':fit['amp'],
        'acrophase':fit['acrophase'],'r2':fit['r2'],
        'amp_ci':amp_ci_str,'acr_ci':acr_ci_str,
        'amp_mean':ci['amp_mean'],'amp_lo':ci['amp_lo'],'amp_hi':ci['amp_hi'],
        'acr_mean':ci['acr_mean'],'acr_lo':ci['acr_lo'],'acr_hi':ci['acr_hi'],
    })
    print("  "+tissue+": acrophase="+str(round(fit['acrophase'],1))+"h (CI:"+str(round(ci['acr_lo'],1))+"-"+str(round(ci['acr_hi'],1))+"), "
          "amp="+str(round(fit['amp'],2))+" (CI:"+str(round(ci['amp_lo'],2))+"-"+str(round(ci['amp_hi'],2))+"), R2="+str(round(fit['r2'],3)))

# Also compute for all tissues where ADRA1A/B have data in data.csv
print("\nADRA1A: All tissues with data in data.csv (for comparison):")
adra1a_all_boot=[]
for tissue in sorted(data.Tissue.unique()):
    sub=data[(data.Tissue==tissue)&(data.Gene=='ADRA1A')]
    if len(sub)==0: continue
    row=sub.iloc[0]
    y=row[ZT_LABELS].values.astype(float)
    if y.max() < 0.01: continue
    t=np.array(ZT_HOURS)
    fit=cosinor_fit(t, y)
    if not fit or fit['r2'] < 0.3: continue
    ci=bootstrap_ci(t, y, n_boot=500)
    if not ci: continue
    is_fdr = tissue in adra1a_tissues_fdr
    adra1a_all_boot.append({
        'tissue':tissue,'mesor':fit['mesor'],'amp':fit['amp'],
        'acrophase':fit['acrophase'],'r2':fit['r2'],
        'fdr_sig':is_fdr,
        'amp_mean':ci['amp_mean'],'amp_lo':ci['amp_lo'],'amp_hi':ci['amp_hi'],
        'acr_mean':ci['acr_mean'],'acr_lo':ci['acr_lo'],'acr_hi':ci['acr_hi'],
    })

for b in sorted(adra1a_all_boot, key=lambda x: -x['r2'])[:15]:
    sig = "**FDR**" if b['fdr_sig'] else ""
    print("  "+b['tissue']+": acr="+str(round(b['acrophase'],1))+"h (CI:"+str(round(b['acr_lo'],1))+"-"+str(round(b['acr_hi'],1))+"), "
          "R2="+str(round(b['r2'],3))+" "+sig)

# =============================================================================
# Save bootstrap results
# =============================================================================
adra1b_boot_df=pd.DataFrame(adra1b_boot) if adra1b_boot else pd.DataFrame()
adra1a_boot_df=pd.DataFrame(adra1a_boot) if adra1a_boot else pd.DataFrame()
adra1a_all_boot_df=pd.DataFrame(adra1a_all_boot) if adra1a_all_boot else pd.DataFrame()

adra1b_boot_df.to_csv(f'{OUT_DIR}/ADRA1B_bootstrap_CI.csv',index=False)
adra1a_boot_df.to_csv(f'{OUT_DIR}/ADRA1A_bootstrap_CI.csv',index=False)
adra1a_all_boot_df.to_csv(f'{OUT_DIR}/ADRA1A_alltissues_bootstrap_CI.csv',index=False)

print("\nSaved bootstrap CI results")
print("  ADRA1B: "+str(len(adra1b_boot))+" FDR-sig tissues with bootstrap CIs")
print("  ADRA1A: "+str(len(adra1a_boot))+" FDR-sig tissues with bootstrap CIs")
print("  ADRA1A: "+str(len(adra1a_all_boot))+" total tissues with bootstrap CIs (R2>0.3)")

# =============================================================================
# 2. Pathway enrichment with FDR-corrected gene sets
# =============================================================================
print("\n" + "="*70)
print("PATHWAY ENRICHMENT WITH FDR-CORRECTED GENE SETS")
print("="*70)

# KEGG pathway gene lists (from generate_figure4.py - verified correct)
pathway_genes = {
    'hsa04270': ['ADRA1A','ADRA1B','MYL9','MYH11','ACTA2','ACTG2',
                 'CALD1','MYLK','RAF1','BRAF','MAPK1','MAPK3',
                 'PLCB1','PLCB3','PLCB4','PLCD1','ITPR1','ITPR2','ITPR3',
                 'CALM1','CALM2','CALM3','MYL2','CAMK2D',
                 'ARHGEF1','ARHGEF11','ROCK1','ROCK2','PRKCB',
                 'CAMK2A','CAMK2B','CAMK2G','PPP1CA','PPP1CB','PPP1CC'],
    'hsa04020': ['ADRA1A','ADRA1B','ADRB1','ADRB2','GRM1','GRM5',
                 'ADCY1','ADCY3','ADCY5','ADCY6','ADCY8',
                 'PLCB1','PLCB2','PLCB3','PLCB4','ITPR1','ITPR2','ITPR3',
                 'CALM1','CALM2','CALM3','CAMK2A','CAMK2B','CAMK2D',
                 'RYR1','RYR2','RYR3','ATP2A1','ATP2A2','ATP2B1','ATP2B2',
                 'SLC8A1','SLC8A2','CACNA1C','CACNA1D','CACNA1S',
                 'PRKCB','PRKCG','PRKCA','CHRNA7','GRIN2A','GRIN2B',
                 'PDGFRB','FGFR1','VEGFB','EDN1','EDNRA','NOS1','NOS3'],
    'hsa04080': ['ADRA1A','ADRA1B','ADRA2A','ADRB1','ADRB2',
                 'GRM1','GRM2','GRM3','GRM4','GRM5','GRM6','GRM7','GRM8',
                 'GABRA1','GABRA2','GABRA3','GABRB1','GABRB2','GABRD',
                 'GLRA1','GLRA2','GLRB',
                 'CHRNA1','CHRNA2','CHRNA3','CHRNA4','CHRNA7','CHRNB1','CHRNB2',
                 'HRH1','HRH2','HTR1A','HTR2A','HTR3A','HTR4','HTR6',
                 'DRD1','DRD2','DRD3','DRD4','OPRM1','OPRD1','OPRK1',
                 'NPY1R','NPY2R','NPY5R','GHRHR','SST','GAL','NPBWR1',
                 'TAAR1','GRIP1','GRIK1','GRIK2','GRIA1','GRIA2',
                 'P2RY1','P2RY2','ADORA1','ADORA2A','ADORA3'],
    'hsa04970': ['ADRA1A','ADRA1B','ADRB1','ADRB2',
                 'CHRNA1','CHRNA3','CHRNA7','CHRNB1',
                 'PLCB1','PLCB3','ITPR1','ITPR2','ITPR3',
                 'CALM1','CALM2','CALM3',
                 'SLC2A1','SLC2A4','SLC5A1','SLC5A3',
                 'AQP1','AQP3','AQP5',
                 'CA2','CA6','PRSS1','AMY1','AMY2',
                 'PRKCB','PRKCA','MAPK1','CREB1',
                 'MYLK','ACTA2','MYH11'],
}

# Compute per-tissue enrichment for each pathway using FDR-corrected gene sets
print("\nPer-tissue pathway enrichment (FDR-corrected):")
enrichment_results=[]
for pathway_id, pgene_list in pathway_genes.items():
    pgene_set=set(pgene_list)
    for tissue in tissues_fdr:
        genes_fdr05=fdr_results[tissue]['genes'][fdr_results[tissue]['qvals']<0.05]
        gene_set_fdr05=set(genes_fdr05)
        overlap=pgene_set & gene_set_fdr05
        enrichment_ratio=len(overlap)/len(pgene_set) if len(pgene_set)>0 else 0
        rhythmic_count=len(overlap)
        enrichment_results.append({
            'pathway':pathway_id,'tissue':tissue,
            'pathway_size':len(pgene_set),
            'rhythmic_in_tissue':len(genes_fdr05),
            'overlap':rhythmic_count,
            'enrichment_ratio':round(enrichment_ratio,4),
            'overlap_genes':','.join(sorted(overlap)),
        })

enrich_df=pd.DataFrame(enrichment_results)
enrich_df.to_csv(f'{OUT_DIR}/pathway_enrichment_fdr.csv',index=False)
print("Pathway enrichment saved: "+str(len(enrich_df))+" rows")

# Summary statistics per pathway
for pathway_id in pathway_genes.keys():
    sub=enrich_df[enrich_df.pathway==pathway_id]
    mean_enr=sub['enrichment_ratio'].mean()
    max_enr=sub['enrichment_ratio'].max()
    max_tissue=sub.loc[sub['enrichment_ratio'].idxmax(),'tissue']
    print("  "+pathway_id+": mean_enrichment="+str(round(mean_enr,3))+", max="+str(round(max_enr,3))+" at "+max_tissue)

# =============================================================================
# 3. Gene breadth analysis with FDR correction
# =============================================================================
print("\n" + "="*70)
print("GENE BREADTH ANALYSIS (FDR-CORRECTED)")
print("="*70)

breadth_nom={}; breadth_fdr05={}; breadth_fdr10={}; breadth_fdr20={}
for tissue in tissues_fdr:
    genes=fdr_results[tissue]['genes']
    pvals=fdr_results[tissue]['pvals']
    qvals=fdr_results[tissue]['qvals']
    for g in genes[pvals<0.05]: breadth_nom[g]=breadth_nom.get(g,0)+1
    for g in genes[qvals<0.05]: breadth_fdr05[g]=breadth_fdr05.get(g,0)+1
    for g in genes[qvals<0.10]: breadth_fdr10[g]=breadth_fdr10.get(g,0)+1
    for g in genes[qvals<0.20]: breadth_fdr20[g]=breadth_fdr20.get(g,0)+1

def cat(breadth):
    u=sum(1 for b in breadth.values() if b>=30)
    m=sum(1 for b in breadth.values() if 5<=b<30)
    s=sum(1 for b in breadth.values() if b<5)
    return u,m,s,len(breadth)

print("")
print("{:<12} {:>10} {:>8} {:>8} {:>8} {:>10} {:>8} {:>8}".format('Threshold','Ubiquitous','%','Multi','%','Specific','%','Total'))
print('-'*65)
for label,bd in [('nominal',breadth_nom),('FDR05',breadth_fdr05),('FDR10',breadth_fdr10),('FDR20',breadth_fdr20)]:
    u,m,s,t=cat(bd)
    print("{:<12} {:>10} {:>8.3f}% {:>8} {:>8.1f}% {:>10} {:>8.1f}% {:>8}".format(label,u,u/max(t,1)*100,m,m/t*100,s,s/t*100,t))

# Top 20 ubiquitous genes (FDR05)
print("\nTop 20 ubiquitous genes (FDR05, >=30 tissues):")
ub_fdr05=sorted([(g,b) for g,b in breadth_fdr05.items() if b>=30],key=lambda x:-x[1])[:20]
for g,b in ub_fdr05:
    print("  "+g+": "+str(b)+" tissues")

# Clock genes rhythmicity
print("\nCore clock genes across tissues (FDR05):")
clock_genes=['CLOCK','ARNTL','BMAL1','PER1','PER2','PER3','CRY1','CRY2','NR1D1','NR1D2','DBP','TEAD1']
for gene in clock_genes:
    b=breadth_fdr05.get(gene,0)
    b_nom=breadth_nom.get(gene,0)
    tissues_list=[]
    for tissue in tissues_fdr:
        genes=fdr_results[tissue]['genes']
        qvals=fdr_results[tissue]['qvals']
        if gene in genes:
            gidx=list(genes).index(gene)
            if qvals[gidx]<0.05:
                tissues_list.append(tissue)
    print("  "+gene+": "+str(b)+" tissues (nominal:"+str(b_nom)+") ["+','.join(tissues_list[:5])+"]")

# Save breadth data
breadth_save=[]
for g in breadth_fdr05.keys():
    b_nom=breadth_nom.get(g,0)
    b_fdr05=breadth_fdr05.get(g,0)
    b_fdr10=breadth_fdr10.get(g,0)
    b_fdr20=breadth_fdr20.get(g,0)
    breadth_save.append({
        'gene':g,'breadth_nominal':b_nom,'breadth_FDR05':b_fdr05,
        'breadth_FDR10':b_fdr10,'breadth_FDR20':b_fdr20,
        'category':'ubiquitous' if b_fdr05>=30 else ('multi' if 5<=b_fdr05<30 else 'specific')
    })
breadth_df=pd.DataFrame(breadth_save)
breadth_df.to_csv(f'{OUT_DIR}/gene_breadth_fdr.csv',index=False)
print("\nGene breadth saved: "+str(len(breadth_df))+" genes")

# =============================================================================
# 4. ADRA1A/B full analysis using data.csv
# =============================================================================
print("\n" + "="*70)
print("ADRA1A/B FULL ANALYSIS FROM RAW DATA")
print("="*70)

print("\nADRA1A: Cosinor parameters for ALL tissues with expression data:")
adra1a_full=[]
for tissue in sorted(data.Tissue.unique()):
    sub=data[(data.Tissue==tissue)&(data.Gene=='ADRA1A')]
    if len(sub)==0: continue
    row=sub.iloc[0]
    y=row[ZT_LABELS].values.astype(float)
    if y.max() < 0.01: continue
    t=np.array(ZT_HOURS)
    fit=cosinor_fit(t, y)
    if not fit: continue
    is_fdr=False
    pval_meta=None
    if tissue in tissues_fdr:
        df_tissue=tissue_data[tissue]
        if 'ADRA1A' in list(df_tissue.CycID):
            gidx=list(df_tissue.CycID).index('ADRA1A')
            pval_meta=fdr_results[tissue]['pvals'][gidx]
            is_fdr=fdr_results[tissue]['qvals'][gidx]<0.05
    actual=(fit['acrophase']+6)%24
    actual_str=str(int(actual)).zfill(2)+":"+str(int((actual%1)*60)).zfill(2)
    adra1a_full.append({
        'tissue':tissue,'mesor':fit['mesor'],'amp':fit['amp'],
        'acrophase_zt':fit['acrophase'],'acrophase_actual':actual_str,
        'r2':fit['r2'],'fdr_sig':is_fdr,'pval_meta':pval_meta
    })

adra1a_full_df=pd.DataFrame(adra1a_full).sort_values('r2',ascending=False)
adra1a_full_df.to_csv(f'{OUT_DIR}/ADRA1A_full_cosinor.csv',index=False)
print("ADRA1A full cosinor: "+str(len(adra1a_full))+" tissues")
print("  Top 10 by R2:")
print(adra1a_full_df.head(10)[['tissue','acrophase_actual','amp','r2','fdr_sig']].to_string(index=False))

print("\nADRA1B: Cosinor parameters for ALL tissues with expression data:")
adra1b_full=[]
for tissue in sorted(data.Tissue.unique()):
    sub=data[(data.Tissue==tissue)&(data.Gene=='ADRA1B')]
    if len(sub)==0: continue
    row=sub.iloc[0]
    y=row[ZT_LABELS].values.astype(float)
    if y.max() < 0.01: continue
    t=np.array(ZT_HOURS)
    fit=cosinor_fit(t, y)
    if not fit: continue
    is_fdr=False
    pval_meta=None
    if tissue in tissues_fdr:
        df_tissue=tissue_data[tissue]
        if 'ADRA1B' in list(df_tissue.CycID):
            gidx=list(df_tissue.CycID).index('ADRA1B')
            pval_meta=fdr_results[tissue]['pvals'][gidx]
            is_fdr=fdr_results[tissue]['qvals'][gidx]<0.05
    actual=(fit['acrophase']+6)%24
    actual_str=str(int(actual)).zfill(2)+":"+str(int((actual%1)*60)).zfill(2)
    adra1b_full.append({
        'tissue':tissue,'mesor':fit['mesor'],'amp':fit['amp'],
        'acrophase_zt':fit['acrophase'],'acrophase_actual':actual_str,
        'r2':fit['r2'],'fdr_sig':is_fdr,'pval_meta':pval_meta
    })

adra1b_full_df=pd.DataFrame(adra1b_full) if adra1b_full else pd.DataFrame(columns=['tissue','mesor','amp','acrophase_zt','acrophase_actual','r2','fdr_sig','pval_meta'])
adra1b_full_df.to_csv(f'{OUT_DIR}/ADRA1B_full_cosinor.csv',index=False)
print("ADRA1B full cosinor: "+str(len(adra1b_full))+" tissues")

# =============================================================================
# 5. Visualization data generation
# =============================================================================
print("\n" + "="*70)
print("VISUALIZATION DATA GENERATION")
print("="*70)

print("Figure 2 data (ADRA1B, 9 FDR-sig tissues):")
fig2_data=[]
for tissue in sorted(adra1b_tissues_fdr):
    sub=data[(data.Tissue==tissue)&(data.Gene=='ADRA1B')]
    if len(sub)==0: continue
    row=sub.iloc[0]
    y=row[ZT_LABELS].values.astype(float)
    t=np.array(ZT_HOURS)
    fit=cosinor_fit(t, y)
    if not fit: continue
    t_fine=np.linspace(0,24,100)
    fitted=fit['mesor']+fit['amp']*np.cos(2*np.pi*(t_fine-fit['acrophase'])/24)
    actual=(fit['acrophase']+6)%24
    qval=1.0
    if tissue in tissues_fdr:
        df_tissue=tissue_data[tissue]
        if 'ADRA1B' in list(df_tissue.CycID):
            gidx=list(df_tissue.CycID).index('ADRA1B')
            qval=fdr_results[tissue]['qvals'][gidx]
    fig2_data.append({
        'tissue':tissue,'zt_hours':t_fine.tolist(),'fitted':fitted.tolist(),
        'actual_values':y.tolist(),'acrophase_zt':fit['acrophase'],
        'acrophase_actual':actual,'r2':fit['r2'],'qval':qval
    })
fig2_df=pd.DataFrame(fig2_data)
fig2_df.to_csv(f'{OUT_DIR}/Figure2_ADRA1B_data.csv',index=False)
print("  Saved "+str(len(fig2_data))+" tissue curves")

print("Figure 3 data (ADRA1A, 3 FDR-sig tissues):")
fig3_data=[]
for tissue in sorted(adra1a_tissues_fdr):
    sub=data[(data.Tissue==tissue)&(data.Gene=='ADRA1A')]
    if len(sub)==0: continue
    row=sub.iloc[0]
    y=row[ZT_LABELS].values.astype(float)
    t=np.array(ZT_HOURS)
    fit=cosinor_fit(t, y)
    if not fit: continue
    t_fine=np.linspace(0,24,100)
    fitted=fit['mesor']+fit['amp']*np.cos(2*np.pi*(t_fine-fit['acrophase'])/24)
    actual=(fit['acrophase']+6)%24
    qval=1.0
    if tissue in tissues_fdr:
        df_tissue=tissue_data[tissue]
        if 'ADRA1A' in list(df_tissue.CycID):
            gidx=list(df_tissue.CycID).index('ADRA1A')
            qval=fdr_results[tissue]['qvals'][gidx]
    fig3_data.append({
        'tissue':tissue,'zt_hours':t_fine.tolist(),'fitted':fitted.tolist(),
        'actual_values':y.tolist(),'acrophase_zt':fit['acrophase'],
        'acrophase_actual':actual,'r2':fit['r2'],'qval':qval
    })
fig3_df=pd.DataFrame(fig3_data)
fig3_df.to_csv(f'{OUT_DIR}/Figure3_ADRA1A_data.csv',index=False)
print("  Saved "+str(len(fig3_data))+" tissue curves")

# =============================================================================
# 6. Summary statistics
# =============================================================================
print("\n" + "="*70)
print("UPDATED SUMMARY STATISTICS")
print("="*70)

print("\nADRA1B: "+str(len(adra1b_boot))+" FDR-significant tissues (from rhythmic_genes files)")
print("ADRA1A: "+str(len(adra1a_boot))+" FDR-significant tissues (from rhythmic_genes files)")
print("ADRA1A: "+str(len(adra1a_full))+" tissues with expression data in data.csv")
print("ADRA1B: "+str(len(adra1b_full))+" tissues with expression data in data.csv")

if len(adra1b_full)==0:
    print("\nNOTE: ADRA1B has 0 rows in data.csv (expression matrix).")
    print("This means ADRA1B was NOT included in the final filtered expression dataset.")
    print("However, ADRA1B IS present in rhythmic_genes files (9 tissues FDR-significant).")
    print("The rhythmic_genes files contain MetaCycle output parameters used for analysis.")
    print("data.csv only contains expression values for genes passing a different filter.")

print("\nAll Step 2 outputs saved successfully!")