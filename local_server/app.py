"""
ChronoTheraAtlas - Local Web Server
A local version of the ChronoTheraAtlas platform for chronopharmacology analysis.
Supports drug-target rhythmic analysis, pathway analysis, and visualization.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request, send_from_directory
from scipy import stats
from scipy.optimize import curve_fit
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='static', template_folder='templates')

# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
EXPR_DIR = os.path.join(DATA_DIR, 'rhythmic_expr')
GENE_DIR = os.path.join(DATA_DIR, 'rhythmic_genes')

ZT_HOURS = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
ACTUAL_HOURS = (ZT_HOURS + 6) % 24
TIME_LABELS = [f'ZT{h:02d}' for h in ZT_HOURS]
ACTUAL_TIME_LABELS = [f'{int(h):02d}:{int((h % 1)*60):02d}' for h in ACTUAL_HOURS]

# Tissue name mapping
TISSUE_NAMES = {
    'ADC': 'Adrenal Cortex', 'ADM': 'Adrenal Medulla', 'AMY': 'Amygdala',
    'ANT': 'Anterior Nucleus', 'AOR': 'Aorta', 'ARC': 'Arcuate Nucleus',
    'ASC': 'Ascending Colon', 'AXL': 'Axillary Lymph Node', 'BLA': 'Bladder',
    'BOM': 'Bone Marrow', 'CEC': 'Cecum', 'CER': 'Cerebellum',
    'COR': 'Cortex', 'DEC': 'Descending Colon', 'DMH': 'Dorsomedial Hypothalamus',
    'DUO': 'Duodenum', 'HAB': 'Habenula', 'HEA': 'Heart',
    'HIP': 'Hippocampus', 'ILE': 'Ileum', 'KIC': 'Kidney Cortex',
    'KIM': 'Kidney Medulla', 'LGP': 'Lateral Globus Pallidus', 'LH': 'Lateral Hypothalamus',
    'LIV': 'Liver', 'LUN': 'Lung', 'MEL': 'Melatonin (Pineal)',
    'MGP': 'Medial Globus Pallidus', 'MMB': 'Mammary Body', 'MUA': 'Muscle (Arm)',
    'MUG': 'Muscle (Leg)', 'OES': 'Esophagus', 'OLB': 'Olfactory Bulb',
    'OMF': 'Omental Fat', 'ONH': 'Optic Nerve Head', 'PAN': 'Pancreas',
    'PIN': 'Pineal', 'PIT': 'Pituitary', 'PON': 'Pons',
    'PRA': 'Preoptic Area', 'PRC': 'Precentral Cortex', 'PRE': 'Prefrontal Cortex',
    'PRO': 'Prostate', 'PUT': 'Putamen', 'PVN': 'Paraventricular Nucleus',
    'RET': 'Retina', 'SCN': 'Suprachiasmatic Nucleus', 'SKI': 'Skin',
    'SMM': 'Skeletal Muscle', 'SON': 'Supraoptic Nucleus', 'SPL': 'Spleen',
    'STF': 'Subcutaneous Fat', 'SUN': 'Substantia Nigra', 'TES': 'Testis',
    'THA': 'Thalamus', 'THR': 'Thyroid', 'VIC': 'Vena Cava',
    'VMH': 'Ventromedial Hypothalamus', 'WAM': 'White Adipose (Mesenteric)',
    'WAP': 'White Adipose (Perirenal)', 'WAR': 'White Adipose (Retroperitoneal)',
    'WAS': 'White Adipose (Subcutaneous)', 'WAT': 'White Adipose (Visceral)'
}

# Organ system mapping
ORGAN_SYSTEMS = {
    'Cardiovascular': ['AOR', 'VIC', 'HEA'],
    'Nervous': ['AMY', 'ANT', 'ARC', 'CER', 'COR', 'DMH', 'HAB', 'HIP',
                'LGP', 'LH', 'MGP', 'OLB', 'ONH', 'PIN', 'PIT', 'PON',
                'PRA', 'PRC', 'PRE', 'PUT', 'PVN', 'RET', 'SCN', 'SON',
                'SUN', 'THA', 'VMH'],
    'Digestive': ['ASC', 'CEC', 'DEC', 'DUO', 'ILE', 'OES', 'PAN', 'LIV'],
    'Endocrine': ['ADC', 'ADM', 'MEL', 'THR', 'TES', 'PRO', 'PIT'],
    'Musculoskeletal': ['MUA', 'MUG', 'SMM'],
    'Immune': ['AXL', 'BOM', 'SPL'],
    'Respiratory': ['LUN'],
    'Renal': ['KIC', 'KIM'],
    'Adipose': ['OMF', 'STF', 'WAM', 'WAP', 'WAR', 'WAS', 'WAT'],
    'Reproductive': ['TES', 'PRO'],
    'Integumentary': ['SKI'],
    'Sensory': ['RET', 'ONH', 'OLB'],
    'Excretory': ['BLA']
}

# Color palettes
TISSUE_COLORS = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel1 + px.colors.qualitative.Set2

SYSTEM_COLORS = {
    'Cardiovascular': '#e74c3c', 'Nervous': '#3498db', 'Digestive': '#2ecc71',
    'Endocrine': '#f39c12', 'Musculoskeletal': '#9b59b6', 'Immune': '#1abc9c',
    'Respiratory': '#e67e22', 'Renal': '#16a085', 'Adipose': '#f1c40f',
    'Reproductive': '#d35400', 'Integumentary': '#c0392b', 'Sensory': '#2980b9',
    'Excretory': '#7f8c8d'
}

# ============================================================================
# Drug Database (curated from DrugBank for the case study)
# ============================================================================
DRUG_DB = {
    'Midodrine': {
        'drugbank_id': 'DB00211',
        'cas': '42794-76-3',
        'targets': ['ADRA1A', 'ADRA1B'],
        'indication': 'Orthostatic hypotension',
        'kegg_id': 'D08223',
        'pathways': {
            'Vascular Smooth Muscle Contraction': 'hsa04270',
            'Calcium Signaling Pathway': 'hsa04020',
            'Neuroactive ligand-receptor interaction': 'hsa04080',
            'Salivary secretion': 'hsa04970'
        },
        'description': 'An alpha-1 adrenergic agonist used to treat orthostatic hypotension.'
    }
}


# ============================================================================
# Core Analysis Functions
# ============================================================================
def cosinor_func(t, mesor, amplitude, acrophase_zt):
    return mesor + amplitude * np.cos(2 * np.pi / 24 * (t - acrophase_zt))


def fit_cosinor(expression_values, time_points=ZT_HOURS):
    """Fit cosinor model to expression data"""
    expr = np.array(expression_values, dtype=float)
    mask = expr > 0
    if np.sum(mask) < 4:
        return None

    t = time_points[mask]
    y = expr[mask]
    min_val = np.min(y[y > 0])
    y = np.where(y > 0, y, min_val)

    try:
        mesor_g = np.mean(y)
        amp_g = (np.max(y) - np.min(y)) / 2
        phase_g = t[np.argmax(y)]

        popt, _ = curve_fit(
            cosinor_func, t, y,
            p0=[mesor_g, amp_g, phase_g],
            bounds=([0, 0, 0], [np.inf, np.inf, 24]),
            maxfev=5000
        )
        mesor, amplitude, acrophase_zt = popt

        fitted = cosinor_func(time_points, mesor, amplitude, acrophase_zt)
        residuals = expr - fitted
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((expr - np.mean(expr))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        n, p = len(t), 3
        if n > p and ss_tot > 0 and ss_res > 0:
            f_stat = ((ss_tot - ss_res) / (p - 1)) / (ss_res / (n - p))
            p_value = 1 - stats.f.cdf(f_stat, p - 1, n - p)
        else:
            p_value = 1

        acrophase_actual = (acrophase_zt + 6) % 24
        h = int(acrophase_actual)
        m = int((acrophase_actual - h) * 60)

        return {
            'mesor': mesor, 'amplitude': amplitude,
            'acrophase_zt': acrophase_zt,
            'acrophase_actual': acrophase_actual,
            'acrophase_time': f'{h:02d}:{m:02d}',
            'r_squared': r_squared, 'p_value': p_value,
            'fitted': fitted, 'success': True
        }
    except Exception:
        return None


def get_gene_expression(gene_symbol):
    """Get expression data for a gene across all 63 tissues"""
    results = []
    for tissue_file in os.listdir(EXPR_DIR):
        if not tissue_file.endswith('_processed.csv'):
            continue
        tissue = tissue_file.replace('_processed.csv', '')
        df = pd.read_csv(os.path.join(EXPR_DIR, tissue_file))
        gene_data = df[df['Human_Symbol'] == gene_symbol]
        if gene_data.empty:
            continue
        expr_values = gene_data.iloc[0][TIME_LABELS].values.astype(float)
        results.append({'tissue': tissue, 'expression': expr_values})
    return results


def analyze_gene_rhythmicity(gene_symbol):
    """Full rhythmic analysis of a gene across all tissues"""
    expr_data = get_gene_expression(gene_symbol)
    results = []
    for item in expr_data:
        fit = fit_cosinor(item['expression'])
        if fit and fit['success']:
            fit['tissue'] = item['tissue']
            fit['tissue_name'] = TISSUE_NAMES.get(item['tissue'], item['tissue'])
            fit['expression'] = item['expression']
            fit['organ_system'] = get_organ_system(item['tissue'])
            results.append(fit)
    results.sort(key=lambda x: x['p_value'])
    return results


def get_organ_system(tissue):
    for system, tissues in ORGAN_SYSTEMS.items():
        if tissue in tissues:
            return system
    return 'Other'


def get_rhythmic_genes(tissue, p_threshold=0.05):
    """Get list of rhythmic genes for a tissue"""
    filepath = os.path.join(GENE_DIR, f'rhythmic_genes_{tissue}_p0.05.csv')
    if not os.path.exists(filepath):
        return []
    df = pd.read_csv(filepath)
    return df['CycID'].tolist()


def count_tissue_rhythmic_genes():
    """Count rhythmic genes per tissue"""
    counts = {}
    for f in sorted(os.listdir(GENE_DIR)):
        if f.startswith('rhythmic_genes_') and f.endswith('_p0.05.csv'):
            tissue = f.replace('rhythmic_genes_', '').replace('_p0.05.csv', '')
            df = pd.read_csv(os.path.join(GENE_DIR, f))
            counts[tissue] = len(df)
    return counts


# ============================================================================
# Visualization Functions (Plotly)
# ============================================================================
def create_fit_curves_plot(analysis_results, gene_name):
    """Create multi-tissue rhythmic fitting curves"""
    n = len(analysis_results)
    if n == 0:
        return go.Figure()

    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"{r['tissue']} (R²={r['r_squared']:.3f}, p={r['p_value']:.4f})"
                        for r in analysis_results],
        vertical_spacing=0.08, horizontal_spacing=0.06
    )

    t_fine = np.linspace(0, 22, 200)
    for i, result in enumerate(analysis_results):
        row, col = (i // cols) + 1, (i % cols) + 1
        color = TISSUE_COLORS[i % len(TISSUE_COLORS)]

        fitted_fine = cosinor_func(t_fine, result['mesor'], result['amplitude'], result['acrophase_zt'])
        fig.add_trace(go.Scatter(x=t_fine, y=fitted_fine, mode='lines',
                                  line=dict(color=color, width=2),
                                  showlegend=False), row=row, col=col)
        fig.add_trace(go.Scatter(x=ZT_HOURS, y=result['expression'], mode='markers',
                                  marker=dict(color=color, size=6),
                                  showlegend=False), row=row, col=col)

    fig.update_layout(height=300 * rows, title_text=f'{gene_name} Multi-Tissue Rhythmic Fitting',
                      template='plotly_white')
    for i in range(n):
        row, col = (i // cols) + 1, (i % cols) + 1
        fig.update_xaxes(title_text='ZT (hours)', row=row, col=col)
        fig.update_yaxes(title_text='Expression', row=row, col=col)
    return fig


def create_polar_plot(analysis_results, gene_name):
    """Create polar acrophase distribution plot"""
    fig = go.Figure()
    for i, r in enumerate(analysis_results):
        phase = r['acrophase_actual']
        system = r.get('organ_system', 'Other')
        color = SYSTEM_COLORS.get(system, '#95a5a6')
        fig.add_trace(go.Barpolar(
            r=[r['r_squared']],
            theta=[phase / 24 * 360],
            width=[15],
            marker_color=color,
            name=f"{r['tissue']} ({r['acrophase_time']})",
            hovertemplate=f"<b>{r['tissue']}</b><br>Peak: {r['acrophase_time']}<br>R²: {r['r_squared']:.3f}<extra></extra>"
        ))

    fig.update_layout(
        title=f'{gene_name} Acrophase Distribution',
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            angularaxis=dict(
                direction='clockwise', rotation=90,
                tickmode='array',
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']
            )
        ),
        template='plotly_white', height=600
    )
    return fig


def create_heatmap(analysis_results, gene_name):
    """Create temporal expression heatmap"""
    if not analysis_results:
        return go.Figure()

    sorted_results = sorted(analysis_results, key=lambda x: x['acrophase_actual'])
    tissues = [r['tissue'] for r in sorted_results]
    expr_matrix = np.array([r['expression'] for r in sorted_results])

    # Normalize each row
    for i in range(expr_matrix.shape[0]):
        row = expr_matrix[i]
        rmin, rmax = np.min(row), np.max(row)
        if rmax > rmin:
            expr_matrix[i] = (row - rmin) / (rmax - rmin)
        else:
            expr_matrix[i] = 0

    fig = go.Figure(go.Heatmap(
        z=expr_matrix,
        x=[f'{int(h):02d}:00' for h in ACTUAL_HOURS],
        y=tissues,
        colorscale='RdYlBu_r',
        colorbar=dict(title='Normalized\nExpression')
    ))
    fig.update_layout(
        title=f'{gene_name} Temporal Expression Heatmap (sorted by acrophase)',
        xaxis_title='Time (hours)', yaxis_title='Tissue',
        template='plotly_white', height=max(400, len(tissues) * 25 + 100)
    )
    return fig


def create_r_squared_bar(analysis_results, gene_name):
    """Create R-squared distribution bar chart"""
    tissues = [r['tissue'] for r in analysis_results]
    r_squared = [r['r_squared'] for r in analysis_results]
    colors = []
    for r in analysis_results:
        system = r.get('organ_system', 'Other')
        colors.append(SYSTEM_COLORS.get(system, '#95a5a6'))

    fig = go.Figure(go.Bar(
        x=r_squared, y=tissues, orientation='h',
        marker_color=colors,
        text=[f'{v:.3f}' for v in r_squared],
        textposition='auto'
    ))
    fig.update_layout(
        title=f'{gene_name} Model Performance (R²)',
        xaxis_title='R-squared', yaxis_title='Tissue',
        xaxis=dict(range=[0, 1]),
        template='plotly_white', height=max(400, len(tissues) * 25 + 100)
    )
    return fig


def create_atlas_overview():
    """Create atlas statistics overview"""
    counts = count_tissue_rhythmic_genes()

    # Bar chart of rhythmic gene counts per tissue
    tissues = sorted(counts.keys(), key=lambda x: counts[x], reverse=True)
    values = [counts[t] for t in tissues]

    fig = go.Figure(go.Bar(
        x=values, y=tissues, orientation='h',
        marker_color=[SYSTEM_COLORS.get(get_organ_system(t), '#95a5a6') for t in tissues]
    ))
    fig.update_layout(
        title='Number of Rhythmic Genes per Tissue (p<0.05)',
        xaxis_title='Number of Rhythmic Genes',
        yaxis_title='Tissue',
        template='plotly_white', height=1200
    )
    return fig


# ============================================================================
# Flask Routes
# ============================================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query', '').strip()
    # Search in drug database
    results = []
    for name, info in DRUG_DB.items():
        if query.lower() in name.lower() or query.lower() in info.get('drugbank_id', '').lower():
            results.append({'name': name, **info})
    # Also search by gene symbol
    for name, info in DRUG_DB.items():
        for target in info.get('targets', []):
            if query.upper() == target:
                if not any(r['name'] == name for r in results):
                    results.append({'name': name, **info})
    return jsonify(results)


@app.route('/analyze_gene/<gene_symbol>')
def analyze_gene(gene_symbol):
    gene_symbol = gene_symbol.upper()
    results = analyze_gene_rhythmicity(gene_symbol)

    significant = [r for r in results if r['p_value'] < 0.05]
    response = {
        'gene': gene_symbol,
        'total_tissues': len(results),
        'significant_tissues': len(significant),
        'results': [{
            'tissue': r['tissue'],
            'tissue_name': r.get('tissue_name', r['tissue']),
            'organ_system': r.get('organ_system', 'Other'),
            'mesor': round(r['mesor'], 4),
            'amplitude': round(r['amplitude'], 4),
            'acrophase_zt': round(r['acrophase_zt'], 2),
            'acrophase_actual': round(r['acrophase_actual'], 2),
            'acrophase_time': r['acrophase_time'],
            'r_squared': round(r['r_squared'], 4),
            'p_value': r['p_value'],
            'expression': r['expression'].tolist()
        } for r in significant]
    }
    return jsonify(response)


@app.route('/plot/fit_curves/<gene_symbol>')
def plot_fit_curves(gene_symbol):
    gene_symbol = gene_symbol.upper()
    results = [r for r in analyze_gene_rhythmicity(gene_symbol) if r['p_value'] < 0.05]
    fig = create_fit_curves_plot(results, gene_symbol)
    return json.loads(plotly.io.to_json(fig))


@app.route('/plot/polar/<gene_symbol>')
def plot_polar(gene_symbol):
    gene_symbol = gene_symbol.upper()
    results = [r for r in analyze_gene_rhythmicity(gene_symbol) if r['p_value'] < 0.05]
    fig = create_polar_plot(results, gene_symbol)
    return json.loads(plotly.io.to_json(fig))


@app.route('/plot/heatmap/<gene_symbol>')
def plot_heatmap(gene_symbol):
    gene_symbol = gene_symbol.upper()
    results = [r for r in analyze_gene_rhythmicity(gene_symbol) if r['p_value'] < 0.05]
    fig = create_heatmap(results, gene_symbol)
    return json.loads(plotly.io.to_json(fig))


@app.route('/plot/r_squared/<gene_symbol>')
def plot_r_squared(gene_symbol):
    gene_symbol = gene_symbol.upper()
    results = [r for r in analyze_gene_rhythmicity(gene_symbol) if r['p_value'] < 0.05]
    fig = create_r_squared_bar(results, gene_symbol)
    return json.loads(plotly.io.to_json(fig))


@app.route('/atlas_overview')
def atlas_overview():
    counts = count_tissue_rhythmic_genes()
    total = sum(counts.values())
    tissues = len(counts)
    return jsonify({
        'total_rhythmic_genes': total,
        'tissue_count': tissues,
        'avg_genes_per_tissue': round(total / tissues, 1),
        'counts': counts
    })


@app.route('/plot/atlas_overview')
def plot_atlas_overview():
    fig = create_atlas_overview()
    return json.loads(plotly.io.to_json(fig))


@app.route('/drug_info/<drug_name>')
def drug_info(drug_name):
    drug = DRUG_DB.get(drug_name)
    if not drug:
        return jsonify({'error': 'Drug not found'}), 404
    return jsonify(drug)


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("ChronoTheraAtlas Local Server")
    print("Open http://127.0.0.1:5000 in your browser")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
