import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize, signal
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
sns.set_style("whitegrid")

# Read CSV file
try:
    df = pd.read_csv('data.csv')
    print("Data loaded successfully!")
    print(f"Data shape: {df.shape}")
    print(f"Number of tissues: {df['Tissue'].nunique()}")
except FileNotFoundError:
    print("Error: data.csv file not found")
    exit()

# Get time points (ZT in hours) and convert to actual time
zt_hours = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
actual_hours = (zt_hours + 6) % 24
time_labels = ['ZT00', 'ZT02', 'ZT04', 'ZT06', 'ZT08', 'ZT10', 
               'ZT12', 'ZT14', 'ZT16', 'ZT18', 'ZT20', 'ZT22']
actual_time_labels = ['6:00', '8:00', '10:00', '12:00', '14:00', '16:00',
                     '18:00', '20:00', '22:00', '0:00', '2:00', '4:00']

# ============================================================================
# META CYCLE METHODS IMPLEMENTATION
# ============================================================================

# 1. COSINOR (ARSER-like) Method
def cosinor_fit(time_points, expression_values, period=24):
    """Cosinor regression similar to ARSER method in MetaCycle"""
    # Remove zeros/missing values
    expr_array = np.array(expression_values)
    mask = expr_array > 0
    if np.sum(mask) < 4:
        return None
    
    t = time_points[mask]
    y = expr_array[mask]
    
    # Design matrix for cosinor regression
    cos_term = np.cos(2 * np.pi * t / period)
    sin_term = np.sin(2 * np.pi * t / period)
    
    X = np.column_stack([np.ones_like(t), cos_term, sin_term])
    
    try:
        # Solve linear regression
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # Extract parameters
        mesor = beta[0]
        amplitude = np.sqrt(beta[1]**2 + beta[2]**2)
        acrophase = np.arctan2(beta[2], beta[1]) * period / (2 * np.pi)
        if acrophase < 0:
            acrophase += period
        
        # Calculate fitted values
        fitted = mesor + amplitude * np.cos(2 * np.pi * (t - acrophase) / period)
        
        # Calculate R² and p-value
        residuals = y - fitted
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # F-test for significance
        n = len(y)
        p = 3  # number of parameters (mesor, amplitude, acrophase)
        if n > p and ss_tot > 0:
            f_stat = ((ss_tot - ss_res) / (p - 1)) / (ss_res / (n - p))
            p_value = 1 - stats.f.cdf(f_stat, p-1, n-p)
        else:
            p_value = 1
        
        return {
            'method': 'Cosinor',
            'period': period,
            'mesor': mesor,
            'amplitude': amplitude,
            'acrophase': acrophase,
            'r_squared': r_squared,
            'p_value': p_value,
            'fitted': fitted,
            'success': True
        }
    except:
        return None

# 2. LOMB-SCARGLE Method
def lomb_scargle_fit(time_points, expression_values, min_period=20, max_period=28):
    """Lomb-Scargle periodogram analysis"""
    from scipy.signal import lombscargle
    
    expr_array = np.array(expression_values)
    mask = expr_array > 0
    if np.sum(mask) < 4:
        return None
    
    t = time_points[mask]
    y = expr_array[mask]
    
    # Normalize
    y_norm = (y - np.mean(y)) / np.std(y)
    
    # Calculate frequencies
    frequencies = np.linspace(1/max_period, 1/min_period, 1000)
    power = lombscargle(t, y_norm, 2 * np.pi * frequencies)
    
    # Find dominant frequency
    dominant_idx = np.argmax(power)
    dominant_freq = frequencies[dominant_idx]
    period = 1 / dominant_freq
    
    # Calculate p-value using false alarm probability
    M = len(frequencies)
    z = power[dominant_idx]
    p_value = M * np.exp(-z)
    p_value = min(p_value, 1)
    
    # Fit cosinor with detected period
    cosinor_result = cosinor_fit(time_points, expression_values, period=period)
    
    if cosinor_result:
        cosinor_result['method'] = 'Lomb-Scargle'
        cosinor_result['power'] = z
        return cosinor_result
    return None

# 3. Combined MetaCycle Analysis
def metacycle_analysis(time_points, expression_values):
    """Combine multiple methods like MetaCycle R package"""
    results = {}
    
    # Try cosinor with 24-hour period (default)
    cosinor_24 = cosinor_fit(time_points, expression_values, period=24)
    if cosinor_24:
        results['Cosinor-24h'] = cosinor_24
    
    # Try Lomb-Scargle to find best period
    ls_result = lomb_scargle_fit(time_points, expression_values)
    if ls_result:
        results['Lomb-Scargle'] = ls_result
    
    # Select best result (lowest p-value)
    best_result = None
    best_p = 1
    
    for method, result in results.items():
        if result['p_value'] < best_p:
            best_p = result['p_value']
            best_result = result
            best_result['method_name'] = method
    
    if best_result is None:
        # Fallback to simple cosinor
        best_result = {
            'method': 'Simple Cosinor',
            'period': 24,
            'mesor': np.mean(expression_values[expression_values > 0]),
            'amplitude': 0,
            'acrophase': 12,
            'r_squared': 0,
            'p_value': 1,
            'success': False
        }
    
    return best_result

# ============================================================================
# Analyze all tissues using MetaCycle methods
# ============================================================================
print("\nAnalyzing tissues using MetaCycle methods...")

all_results = []
for idx, row in df.iterrows():
    tissue = row['Tissue']
    expression = row[time_labels].values.astype(float)
    
    # Basic statistics
    expr_array = np.array(expression)
    mask = expr_array > 0
    mean_expr = np.mean(expr_array[mask]) if np.sum(mask) > 0 else 0
    max_expr = np.max(expr_array)
    amplitude_raw = max_expr - np.min(expr_array[mask]) if np.sum(mask) > 0 else 0
    
    # MetaCycle analysis
    result = metacycle_analysis(zt_hours, expression)
    
    # Calculate rhythm score
    if result['success']:
        # Combine R², p-value, and amplitude for rhythm score
        rhythm_score = (0.4 * result['r_squared'] + 
                       0.3 * (1 - min(result['p_value'], 1)) +
                       0.3 * min(result['amplitude'] / (mean_expr + 1e-10), 1))
    else:
        rhythm_score = 0
    
    # Convert acrophase to actual time
    if result['success']:
        acrophase_actual = (result['acrophase'] + 6) % 24
    else:
        acrophase_actual = None
    
    tissue_result = {
        'Tissue': tissue,
        'Method': result['method_name'] if 'method_name' in result else result['method'],
        'Period': result['period'],
        'Acrophase_ZT': result['acrophase'] if result['success'] else None,
        'Acrophase_Actual': acrophase_actual,
        'Amplitude': result['amplitude'] if result['success'] else 0,
        'Mesor': result['mesor'] if result['success'] else mean_expr,
        'R_squared': result['r_squared'],
        'P_value': result['p_value'],
        'Rhythm_Score': rhythm_score,
        'Mean_Expression': mean_expr,
        'Max_Expression': max_expr,
        'Amplitude_Raw': amplitude_raw,
        'Success': result['success']
    }
    
    all_results.append(tissue_result)

# Create results DataFrame
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('Rhythm_Score', ascending=False)

print("\nMetaCycle Analysis Results (Top 15):")
print("=" * 100)
print(f"{'Tissue':<10} {'Method':<15} {'Period':<8} {'R²':<8} {'P-value':<12} {'Rhythm Score':<12} {'Peak Time':<12}")
print("-" * 100)

for _, row in results_df.head(15).iterrows():
    if row['Acrophase_Actual'] is not None:
        peak_hour = int(row['Acrophase_Actual'])
        peak_minute = int((row['Acrophase_Actual'] - peak_hour) * 60)
        peak_time = f"{peak_hour:02d}:{peak_minute:02d}"
    else:
        peak_time = "N/A"
    
    print(f"{row['Tissue']:<10} {row['Method']:<15} {row['Period']:<8.1f} {row['R_squared']:<8.3f} "
          f"{row['P_value']:<12.3e} {row['Rhythm_Score']:<12.3f} {peak_time:<12}")

# ============================================================================
# CREATE THE 4x3 PLOT WITH METACYCLE FITS
# ============================================================================
print("\nGenerating 4x3 plot with MetaCycle fits...")

# Select top 12 rhythmic tissues
top_tissues = results_df.head(12)['Tissue'].values

fig, axes = plt.subplots(4, 3, figsize=(20, 16))
axes = axes.flatten()

for i, tissue in enumerate(top_tissues[:12]):
    ax = axes[i]
    tissue_data = df[df['Tissue'] == tissue]
    expression = tissue_data[time_labels].values.flatten().astype(float)
    result = results_df[results_df['Tissue'] == tissue].iloc[0]
    
    # Plot raw data points
    ax.plot(actual_hours, expression, 'bo', markersize=6, label='Raw Data', alpha=0.7)
    
    # Plot MetaCycle fitted curve
    if result['Success']:
        # Generate smooth curve using the fitted parameters
        period = result['Period']
        mesor = result['Mesor']
        amplitude = result['Amplitude']
        acrophase_zt = result['Acrophase_ZT']
        
        # Generate time points for smooth curve (extended to handle wrap-around)
        t_zt = np.linspace(-6, 30, 400)
        fitted = mesor + amplitude * np.cos(2 * np.pi * (t_zt - acrophase_zt) / period)
        
        # Convert to actual time
        t_actual = (t_zt + 6) % 24
        
        # Sort for proper plotting
        sort_idx = np.argsort(t_actual)
        t_sorted = t_actual[sort_idx]
        fitted_sorted = fitted[sort_idx]
        
        # Plot only the 24-hour window
        mask = (t_sorted >= 0) & (t_sorted <= 24)
        ax.plot(t_sorted[mask], fitted_sorted[mask], 'r-', linewidth=2, 
                label=f"{result['Method']} Fit", alpha=0.8)
        
        # Mark peak phase
        peak_time_actual = result['Acrophase_Actual']
        if peak_time_actual is not None:
            # Calculate peak value
            peak_value = mesor + amplitude * np.cos(2 * np.pi * (acrophase_zt - acrophase_zt) / period)
            
            # Format peak time strings
            peak_hour = int(peak_time_actual)
            peak_minute = int((peak_time_actual - peak_hour) * 60)
            peak_time_str = f"{peak_hour:02d}:{peak_minute:02d}"
            
            peak_hour_zt = int(acrophase_zt)
            peak_minute_zt = int((acrophase_zt - peak_hour_zt) * 60)
            peak_phase_str = f"ZT{peak_hour_zt:02d}:{peak_minute_zt:02d}"
            
            # Plot peak marker
            ax.plot(peak_time_actual, peak_value, 'r*', markersize=12, label='Peak')
            
            # Add peak annotation
            ax.annotate(f'Peak: {peak_time_str}\n({peak_phase_str})',
                       xy=(peak_time_actual, peak_value),
                       xytext=(peak_time_actual + 1, peak_value * 1.1),
                       fontsize=7,
                       arrowprops=dict(arrowstyle='->', color='red', alpha=0.6, lw=1))
    
    # Add fit information box
    info_text = f"Method: {result['Method']}\n"
    info_text += f"Period: {result['Period']:.1f}h\n"
    info_text += f"R² = {result['R_squared']:.3f}\n"
    info_text += f"p = {result['P_value']:.3f}"
    
    if result['P_value'] < 0.05:
        info_text += " *"
    if result['P_value'] < 0.01:
        info_text += "*"
    if result['P_value'] < 0.001:
        info_text += "*"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Set axis labels and properties
    ax.set_xlabel('Time of Day', fontsize=10)
    ax.set_ylabel('Expression Level', fontsize=10)
    
    rhythm_score = result['Rhythm_Score']
    ax.set_title(f'{tissue}\nRhythm Score: {rhythm_score:.3f}',
                 fontsize=11, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    
    # Set x-ticks
    xticks = np.arange(0, 25, 2)
    xtick_labels = [f'{int(h):02d}:00' for h in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=45, fontsize=9)
    ax.set_xlim(0, 24)
    
    # Set y-limits based on data
    y_data = expression[expression > 0]
    if len(y_data) > 0:
        y_min = max(0, np.min(y_data) * 0.8)
        y_max = np.max(y_data) * 1.2
        ax.set_ylim(y_min, y_max)
    
    # Add day/night shading
    ax.axvspan(6, 18, alpha=0.1, color='yellow', label='Day')
    ax.axvspan(0, 6, alpha=0.1, color='blue', label='Night')
    ax.axvspan(18, 24, alpha=0.1, color='blue')
    
    # Add legend (only for first subplot)
    if i == 0:
        ax.legend(loc='upper right', fontsize=8)

# Adjust layout
plt.suptitle('ADRA1B Gene Expression - MetaCycle Analysis\nZT0 = 6:00, ZT12 = 18:00',
             fontsize=16, fontweight='bold')
plt.tight_layout()

# Save figure
plt.savefig('img1_metacycle.png', dpi=300, bbox_inches='tight')
print("\nFigure saved as img1_metacycle.png")

# Set style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
sns.set_style("whitegrid")

# Read CSV file
try:
    df = pd.read_csv('data.csv')
    print("Data loaded successfully!")
    print(f"Data shape: {df.shape}")
    print(f"Number of tissues: {df['Tissue'].nunique()}")
except FileNotFoundError:
    print("Error: data.csv file not found")
    print("Please ensure data.csv is in the current directory")
    exit()

# Get time points (ZT in hours) and convert to actual time
zt_hours = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])  # ZT times
actual_hours = (zt_hours + 6) % 24  # Convert ZT to actual time: ZT0 = 6:00
time_labels = ['ZT00', 'ZT02', 'ZT04', 'ZT06', 'ZT08', 'ZT10', 
               'ZT12', 'ZT14', 'ZT16', 'ZT18', 'ZT20', 'ZT22']
actual_time_labels = ['6:00', '8:00', '10:00', '12:00', '14:00', '16:00',
                     '18:00', '20:00', '22:00', '0:00', '2:00', '4:00']

# Cosinor model function
def cosinor_func(t, mesor, amplitude, acrophase_zt):
    """
    Cosinor model: y = mesor + amplitude * cos(2π/24 * (t - acrophase))
    t: time in ZT hours
    mesor: midline estimating statistic of rhythm (mean expression)
    amplitude: amplitude of rhythm
    acrophase_zt: peak phase in ZT hours
    """
    return mesor + amplitude * np.cos(2 * np.pi / 24 * (t - acrophase_zt))

def fit_cosinor_model(expression_values, time_points_zt):
    """Fit cosinor model to expression data"""
    expression_array = np.array(expression_values)
    min_nonzero = min([v for v in expression_array if v > 0], default=1e-10)
    expression_processed = np.where(expression_array > 0, expression_array, min_nonzero)
    
    try:
        # Initial parameter estimates
        mesor_guess = np.mean(expression_processed)
        amplitude_guess = (np.max(expression_processed) - np.min(expression_processed)) / 2
        peak_idx = np.argmax(expression_processed)
        acrophase_guess = time_points_zt[peak_idx]
        
        # Parameter bounds
        bounds = ([0, 0, 0], [np.inf, np.inf, 24])
        
        # Curve fitting
        popt, pcov = curve_fit(cosinor_func, time_points_zt, expression_processed, 
                               p0=[mesor_guess, amplitude_guess, acrophase_guess],
                               bounds=bounds, maxfev=5000)
        
        mesor, amplitude, acrophase_zt = popt
        fitted_values = cosinor_func(time_points_zt, mesor, amplitude, acrophase_zt)
        
        # Calculate R²
        residuals = expression_processed - fitted_values
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((expression_processed - np.mean(expression_processed))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Calculate p-value (F-test)
        n = len(time_points_zt)
        p = len(popt)
        if n > p and ss_tot > 0:
            f_value = (ss_tot - ss_res) / (p - 1) / (ss_res / (n - p)) if ss_res > 0 else 0
            p_value = 1 - stats.f.cdf(f_value, p-1, n-p) if f_value > 0 else 1
        else:
            p_value = 1
            
        # Convert acrophase to actual time
        acrophase_actual = (acrophase_zt + 6) % 24
        
        return {
            'success': True,
            'params': {'mesor': mesor, 'amplitude': amplitude, 
                      'acrophase_zt': acrophase_zt, 'acrophase_actual': acrophase_actual},
            'fitted_values': fitted_values,
            'r_squared': r_squared,
            'p_value': p_value,
            'expression_processed': expression_processed
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'params': None,
            'fitted_values': None,
            'r_squared': 0,
            'p_value': 1
        }

# Analyze each tissue
results = []
for _, row in df.iterrows():
    tissue = row['Tissue']
    expression = row[time_labels].values.astype(float)
    
    # Basic statistics
    mean_expr = np.mean(expression[expression > 0]) if np.any(expression > 0) else 0
    max_expr = np.max(expression)
    min_expr = np.min(expression[expression > 0]) if np.any(expression > 0) else 0
    amplitude = max_expr - min_expr
    
    # Fit cosinor model
    fit_result = fit_cosinor_model(expression, zt_hours)
    
    # Rhythm score
    if fit_result['success']:
        rhythm_score = 0.7 * fit_result['r_squared'] + 0.3 * min(amplitude / (mean_expr + 1e-10), 1)
        acrophase_zt = fit_result['params']['acrophase_zt']
        acrophase_actual = fit_result['params']['acrophase_actual']
    else:
        rhythm_score = 0
        acrophase_zt = None
        acrophase_actual = None
    
    results.append({
        'Tissue': tissue,
        'mean_expression': mean_expr,
        'max_expression': max_expr,
        'amplitude': amplitude,
        'rhythm_score': rhythm_score,
        'fit_success': fit_result['success'],
        'r_squared': fit_result['r_squared'] if fit_result['success'] else 0,
        'p_value': fit_result['p_value'] if fit_result['success'] else 1,
        'acrophase_zt': acrophase_zt,
        'acrophase_actual': acrophase_actual,
        'fit_result': fit_result
    })

# Create results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('rhythm_score', ascending=False)

print("\nCurve Fitting Results Summary:")
print("=" * 80)
print(f"{'Tissue':<10} {'Rhythm Score':<12} {'R²':<10} {'p-value':<10} {'Amplitude':<12} {'Acrophase (ZT)':<14} {'Acrophase (Actual)':<15}")
print("-" * 80)

for _, row in results_df.head(15).iterrows():
    if row['acrophase_actual'] is not None:
        actual_time_str = f"{int(row['acrophase_actual']):02d}:00"
        print(f"{row['Tissue']:<10} {row['rhythm_score']:.4f}      {row['r_squared']:.4f}    "
              f"{row['p_value']:.4f}    {row['amplitude']:.4f}     "
              f"ZT{row['acrophase_zt']:.1f}          {actual_time_str}")
    else:
        print(f"{row['Tissue']:<10} {row['rhythm_score']:.4f}      {row['r_squared']:.4f}    "
              f"{row['p_value']:.4f}    {row['amplitude']:.4f}     N/A             N/A")

# 1. Plot curve fitting for all tissues with correct time axis - FIXED VERSION
print("\nGenerating curve fitting plots with corrected time axis...")

# Select top rhythmic tissues for detailed display
top_tissues = results_df.head(12)['Tissue'].values

# Select top rhythmic tissues for detailed display
top_tissues = results_df.head(12)['Tissue'].values

fig, axes = plt.subplots(4, 3, figsize=(18, 15))
axes = axes.flatten()

for i, tissue in enumerate(top_tissues[:12]):
    ax = axes[i]
    tissue_data = df[df['Tissue'] == tissue]
    expression = tissue_data[time_labels].values.flatten().astype(float)
    fit_result = results_df[results_df['Tissue'] == tissue]['fit_result'].iloc[0]
    
    # Plot raw data points using actual time
    ax.plot(actual_hours, expression, 'bo', markersize=6, label='Raw Data')
    
    # Plot fitted curve - EXTENDED to cover full 24-hour cycle
    if fit_result['success']:
        # Generate dense ZT time points for smooth curve covering TWO cycles
        zt_dense = np.linspace(-6, 30, 400)  # Extend beyond 0-24 to ensure continuous curve
        fitted_dense = cosinor_func(zt_dense, 
                                   fit_result['params']['mesor'],
                                   fit_result['params']['amplitude'],
                                   fit_result['params']['acrophase_zt'])
        
        # Convert ZT dense to actual time for plotting
        actual_dense = (zt_dense + 6) % 24
        
        # Sort for proper plotting
        sort_idx = np.argsort(actual_dense)
        actual_dense_sorted = actual_dense[sort_idx]
        fitted_dense_sorted = fitted_dense[sort_idx]
        
        # Plot only the 24-hour window
        mask = (actual_dense_sorted >= 0) & (actual_dense_sorted <= 24)
        actual_dense_plot = actual_dense_sorted[mask]
        fitted_dense_plot = fitted_dense_sorted[mask]
        
        # Plot the curve without connecting lines across day boundaries
        # We need to handle the wrap-around at midnight
        if len(actual_dense_plot) > 0:
            # Find where the curve wraps around
            diffs = np.diff(actual_dense_plot)
            wrap_idx = np.where(diffs < 0)[0]
            
            if len(wrap_idx) > 0:
                # Split at wrap-around point
                split_idx = wrap_idx[0] + 1
                ax.plot(actual_dense_plot[:split_idx], fitted_dense_plot[:split_idx], 
                       'r-', linewidth=2, label='Cosinor Fit')
                ax.plot(actual_dense_plot[split_idx:], fitted_dense_plot[split_idx:], 
                       'r-', linewidth=2)
            else:
                ax.plot(actual_dense_plot, fitted_dense_plot, 'r-', linewidth=2, label='Cosinor Fit')
        
        # Mark peak phase in actual time
        peak_time_actual = fit_result['params']['acrophase_actual']
        peak_time_zt = fit_result['params']['acrophase_zt']
        peak_value = cosinor_func(peak_time_zt,
                                 fit_result['params']['mesor'],
                                 fit_result['params']['amplitude'],
                                 fit_result['params']['acrophase_zt'])
        ax.plot(peak_time_actual, peak_value, 'r*', markersize=12, 
                label=f'Peak ({int(peak_time_actual):02d}:00)')
        
        # Add fit information WITHOUT horizontal lines
        info_text = f"R² = {fit_result['r_squared']:.3f}\n"
        info_text += f"p = {fit_result['p_value']:.3f}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Set x-axis with actual time labels
    ax.set_xlabel('Time of Day', fontsize=10)
    ax.set_ylabel('Expression Level', fontsize=10)
    ax.set_title(f'{tissue}\nRhythm Score: {results_df[results_df["Tissue"]==tissue]["rhythm_score"].iloc[0]:.3f}', 
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set x-ticks at 2-hour intervals
    xticks = np.arange(0, 25, 2)
    xtick_labels = [f'{int(h):02d}:00' for h in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=45, fontsize=9)
    ax.set_xlim(0, 24)
    
    # Add day/night shading
    # Day: 6:00-18:00
    # Night: 18:00-6:00 (split into two parts)
    ax.axvspan(6, 18, alpha=0.1, color='yellow', label='Day')
    ax.axvspan(0, 6, alpha=0.1, color='blue', label='Night')
    ax.axvspan(18, 24, alpha=0.1, color='blue')
    
    # Add legend (only for first subplot to avoid repetition)
    if i == 0:
        ax.legend(loc='upper right', fontsize=8)

# Adjust layout
plt.suptitle('ADRA1B Gene Expression Curve Fitting (Cosinor Model) by Tissue\nZT0 = 6:00, ZT12 = 18:00', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
plt.savefig('img1.png', dpi=300, bbox_inches='tight')
# Show plot
plt.show()

# ============================================================================
# ADDITIONAL ANALYSIS: Method comparison
# ============================================================================
print("\n" + "=" * 80)
print("MetaCycle Method Comparison")
print("=" * 80)

# Count methods used
