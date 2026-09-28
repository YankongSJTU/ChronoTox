import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import stats
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
plt.savefig('imgneed1.png', dpi=300, bbox_inches='tight')


# 2. Plot overlay of top rhythmic tissues - CLEAN VERSION (no horizontal lines)
plt.figure(figsize=(14, 8))
top_n = 6
top_tissues_6 = results_df.head(top_n)['Tissue'].values

# Color mapping
colors = plt.cm.Set2(np.linspace(0, 1, top_n))

for i, tissue in enumerate(top_tissues_6):
    tissue_data = df[df['Tissue'] == tissue]
    expression = tissue_data[time_labels].values.flatten().astype(float)
    fit_result = results_df[results_df['Tissue'] == tissue]['fit_result'].iloc[0]
    
    # Plot raw data points
    plt.plot(actual_hours, expression, 'o', color=colors[i], markersize=8, alpha=0.6, label=f'{tissue} (Data)')
    
    # Plot fitted curve - EXTENDED to cover full 24-hour cycle
    if fit_result['success']:
        # Generate dense ZT time points for smooth curve
        zt_dense = np.linspace(-6, 30, 400)
        fitted_dense = cosinor_func(zt_dense, 
                                   fit_result['params']['mesor'],
                                   fit_result['params']['amplitude'],
                                   fit_result['params']['acrophase_zt'])
        
        # Convert to actual time
        actual_dense = (zt_dense + 6) % 24
        
        # Sort for proper plotting
        sort_idx = np.argsort(actual_dense)
        actual_dense_sorted = actual_dense[sort_idx]
        fitted_dense_sorted = fitted_dense[sort_idx]
        
        # Plot only the 24-hour window
        mask = (actual_dense_sorted >= 0) & (actual_dense_sorted <= 24)
        actual_dense_plot = actual_dense_sorted[mask]
        fitted_dense_plot = fitted_dense_sorted[mask]
        
        # Plot without connecting lines across day boundaries
        if len(actual_dense_plot) > 0:
            # Find where the curve wraps around
            diffs = np.diff(actual_dense_plot)
            wrap_idx = np.where(diffs < 0)[0]
            
            if len(wrap_idx) > 0:
                # Split at wrap-around point
                split_idx = wrap_idx[0] + 1
                plt.plot(actual_dense_plot[:split_idx], fitted_dense_plot[:split_idx], 
                        '-', color=colors[i], linewidth=2.5, alpha=0.8)
                plt.plot(actual_dense_plot[split_idx:], fitted_dense_plot[split_idx:], 
                        '-', color=colors[i], linewidth=2.5, alpha=0.8, 
                        label=f'{tissue} (R²={fit_result["r_squared"]:.3f})')
            else:
                plt.plot(actual_dense_plot, fitted_dense_plot, '-', color=colors[i], 
                        linewidth=2.5, alpha=0.8, 
                        label=f'{tissue} (R²={fit_result["r_squared"]:.3f})')

plt.xlabel('Time of Day', fontsize=12)
plt.ylabel('Expression Level', fontsize=12)
plt.title(f'ADRA1B Expression Curve Fitting in Top {top_n} Rhythmic Tissues\n(ZT0 = 6:00, ZT12 = 18:00)', 
          fontsize=14, fontweight='bold')

# Add day/night shading
plt.axvspan(6, 18, alpha=0.1, color='yellow', label='Day')
plt.axvspan(0, 6, alpha=0.1, color='blue', label='Night')
plt.axvspan(18, 24, alpha=0.1, color='blue')

plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
plt.grid(True, alpha=0.3)
plt.xticks(np.arange(0, 25, 2), [f'{int(h):02d}:00' for h in np.arange(0, 25, 2)], rotation=45)
plt.xlim(0, 24)
plt.tight_layout()
plt.show()
plt.savefig('img2.png', dpi=300, bbox_inches='tight')


# 3. Alternative: Simple overlay plot with normalized expression
plt.figure(figsize=(14, 8))

for i, tissue in enumerate(top_tissues_6):
    tissue_data = df[df['Tissue'] == tissue]
    expression = tissue_data[time_labels].values.flatten().astype(float)
    fit_result = results_df[results_df['Tissue'] == tissue]['fit_result'].iloc[0]
    
    # Normalize expression for comparison
    if np.max(expression) > 0:
        expression_norm = expression / np.max(expression)
    else:
        expression_norm = expression
    
    # Plot normalized raw data
    plt.plot(actual_hours, expression_norm, 'o', color=colors[i], markersize=8, alpha=0.6)
    
    # Plot normalized fitted curve
    if fit_result['success']:
        # Generate dense ZT time points
        zt_dense = np.linspace(-6, 30, 400)
        fitted_dense = cosinor_func(zt_dense, 
                                   fit_result['params']['mesor'],
                                   fit_result['params']['amplitude'],
                                   fit_result['params']['acrophase_zt'])
        
        # Normalize fitted curve
        if np.max(fitted_dense) > 0:
            fitted_norm = fitted_dense / np.max(fitted_dense)
        else:
            fitted_norm = fitted_dense
        
        # Convert to actual time
        actual_dense = (zt_dense + 6) % 24
        
        # Sort for proper plotting
        sort_idx = np.argsort(actual_dense)
        actual_dense_sorted = actual_dense[sort_idx]
        fitted_norm_sorted = fitted_norm[sort_idx]
        
        # Plot only the 24-hour window
        mask = (actual_dense_sorted >= 0) & (actual_dense_sorted <= 24)
        actual_dense_plot = actual_dense_sorted[mask]
        fitted_norm_plot = fitted_norm_sorted[mask]
        
        # Handle wrap-around at midnight
        if len(actual_dense_plot) > 0:
            diffs = np.diff(actual_dense_plot)
            wrap_idx = np.where(diffs < 0)[0]
            
            if len(wrap_idx) > 0:
                split_idx = wrap_idx[0] + 1
                plt.plot(actual_dense_plot[:split_idx], fitted_norm_plot[:split_idx], 
                        '-', color=colors[i], linewidth=2.5, alpha=0.8)
                plt.plot(actual_dense_plot[split_idx:], fitted_norm_plot[split_idx:], 
                        '-', color=colors[i], linewidth=2.5, alpha=0.8, 
                        label=f'{tissue} (R²={fit_result["r_squared"]:.3f})')
            else:
                plt.plot(actual_dense_plot, fitted_norm_plot, '-', color=colors[i], 
                        linewidth=2.5, alpha=0.8, 
                        label=f'{tissue} (R²={fit_result["r_squared"]:.3f})')

plt.xlabel('Time of Day', fontsize=12)
plt.ylabel('Normalized Expression', fontsize=12)
plt.title(f'Normalized ADRA1B Expression in Top {top_n} Rhythmic Tissues\n(ZT0 = 6:00, ZT12 = 18:00)', 
          fontsize=14, fontweight='bold')

# Add day/night shading
plt.axvspan(6, 18, alpha=0.1, color='yellow', label='Day')
plt.axvspan(0, 6, alpha=0.1, color='blue', label='Night')
plt.axvspan(18, 24, alpha=0.1, color='blue')

plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
plt.grid(True, alpha=0.3)
plt.xticks(np.arange(0, 25, 2), [f'{int(h):02d}:00' for h in np.arange(0, 25, 2)], rotation=45)
plt.xlim(0, 24)
plt.tight_layout()
plt.show()
plt.savefig('imgneed3.png', dpi=300, bbox_inches='tight')


# 4. Single tissue detailed example - PIN tissue (highest expression)
print("\nDetailed example: PIN (Pineal gland) tissue analysis")
pin_data = df[df['Tissue'] == 'PIN']
pin_expression = pin_data[time_labels].values.flatten().astype(float)
pin_fit = results_df[results_df['Tissue'] == 'PIN']['fit_result'].iloc[0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Raw data with fitted curve
ax1.plot(actual_hours, pin_expression, 'bo-', markersize=8, linewidth=1.5, label='Raw Data')

if pin_fit['success']:
    # Generate extended curve
    zt_dense = np.linspace(-6, 30, 400)
    fitted_dense = cosinor_func(zt_dense, 
                               pin_fit['params']['mesor'],
                               pin_fit['params']['amplitude'],
                               pin_fit['params']['acrophase_zt'])
    actual_dense = (zt_dense + 6) % 24
    
    # Sort and plot
    sort_idx = np.argsort(actual_dense)
    actual_dense_sorted = actual_dense[sort_idx]
    fitted_dense_sorted = fitted_dense[sort_idx]
    
    mask = (actual_dense_sorted >= 0) & (actual_dense_sorted <= 24)
    actual_plot = actual_dense_sorted[mask]
    fitted_plot = fitted_dense_sorted[mask]
    
    # Handle wrap-around
    diffs = np.diff(actual_plot)
    wrap_idx = np.where(diffs < 0)[0]
    
    if len(wrap_idx) > 0:
        split_idx = wrap_idx[0] + 1
        ax1.plot(actual_plot[:split_idx], fitted_plot[:split_idx], 'r-', linewidth=3, label='Cosinor Fit')
        ax1.plot(actual_plot[split_idx:], fitted_plot[split_idx:], 'r-', linewidth=3)
    else:
        ax1.plot(actual_plot, fitted_plot, 'r-', linewidth=3, label='Cosinor Fit')
    
    # Mark peak
    peak_time_actual = pin_fit['params']['acrophase_actual']
    peak_time_zt = pin_fit['params']['acrophase_zt']
    peak_value = cosinor_func(peak_time_zt,
                             pin_fit['params']['mesor'],
                             pin_fit['params']['amplitude'],
                             pin_fit['params']['acrophase_zt'])
    ax1.plot(peak_time_actual, peak_value, 'r*', markersize=15, label=f'Peak ({int(peak_time_actual):02d}:00)')

ax1.set_xlabel('Time of Day', fontsize=12)
ax1.set_ylabel('Expression Level', fontsize=12)
ax1.set_title('PIN Tissue: ADRA1B Expression and Cosinor Fit\n(ZT0 = 6:00, ZT12 = 18:00)', 
              fontsize=14, fontweight='bold')

# Add day/night shading
ax1.axvspan(6, 18, alpha=0.1, color='yellow', label='Day')
ax1.axvspan(0, 6, alpha=0.1, color='blue', label='Night')
ax1.axvspan(18, 24, alpha=0.1, color='blue')

ax1.grid(True, alpha=0.3)
ax1.set_xticks(np.arange(0, 25, 2))
ax1.set_xticklabels([f'{int(h):02d}:00' for h in np.arange(0, 25, 2)], rotation=45)
ax1.set_xlim(0, 24)
ax1.legend(loc='upper right')

# Plot 2: Residual analysis
if pin_fit['success']:
    residuals = pin_fit['expression_processed'] - pin_fit['fitted_values']
    
    ax2.plot(actual_hours, residuals, 'go-', markersize=8, linewidth=1.5)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Time of Day', fontsize=12)
    ax2.set_ylabel('Residuals (Observed - Fitted)', fontsize=12)
    ax2.set_title(f'Residual Analysis\nR² = {pin_fit["r_squared"]:.3f}, p = {pin_fit["p_value"]:.3f}', 
                  fontsize=14, fontweight='bold')
    
    # Add day/night shading
    ax2.axvspan(6, 18, alpha=0.1, color='yellow')
    ax2.axvspan(0, 6, alpha=0.1, color='blue')
    ax2.axvspan(18, 24, alpha=0.1, color='blue')
    
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(np.arange(0, 25, 2))
    ax2.set_xticklabels([f'{int(h):02d}:00' for h in np.arange(0, 25, 2)], rotation=45)
    ax2.set_xlim(0, 24)

plt.tight_layout()
plt.show()
plt.savefig('img4.png', dpi=300, bbox_inches='tight')


# Save detailed results
detailed_results = []
for _, row in results_df.iterrows():
    tissue = row['Tissue']
    fit_result = row['fit_result']
    
    if fit_result['success']:
        detailed_results.append({
            'Tissue': tissue,
            'Mean_Expression': row['mean_expression'],
            'Max_Expression': row['max_expression'],
            'Amplitude': row['amplitude'],
            'Rhythm_Score': row['rhythm_score'],
            'R_squared': row['r_squared'],
            'P_value': row['p_value'],
            'Mesor': fit_result['params']['mesor'],
            'Amplitude_fit': fit_result['params']['amplitude'],
            'Acrophase_ZT': fit_result['params']['acrophase_zt'],
            'Acrophase_Actual': fit_result['params']['acrophase_actual'],
            'Acrophase_Actual_Time': f"{int(fit_result['params']['acrophase_actual']):02d}:00",
            'Fit_Success': True
        })
    else:
        detailed_results.append({
            'Tissue': tissue,
            'Mean_Expression': row['mean_expression'],
            'Max_Expression': row['max_expression'],
            'Amplitude': row['amplitude'],
            'Rhythm_Score': row['rhythm_score'],
            'R_squared': row['r_squared'],
            'P_value': row['p_value'],
            'Mesor': np.nan,
            'Amplitude_fit': np.nan,
            'Acrophase_ZT': np.nan,
            'Acrophase_Actual': np.nan,
            'Acrophase_Actual_Time': 'N/A',
            'Fit_Success': False
        })

detailed_df = pd.DataFrame(detailed_results)
detailed_df.to_csv('ADRA1B_cosinor_fit_results_final.csv', index=False)
print(f"\nDetailed fitting results saved to: ADRA1B_cosinor_fit_results_final.csv")

print("\nAnalysis complete! Issues fixed:")
print("1. Curves now extend through 4-6 AM (0:00-6:00)")
print("2. Horizontal lines removed from fitted curves")
print("3. Continuous curves properly handle wrap-around at midnight")
