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
zt_hours = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
actual_hours = (zt_hours + 6) % 24
time_labels = ['ZT00', 'ZT02', 'ZT04', 'ZT06', 'ZT08', 'ZT10', 
               'ZT12', 'ZT14', 'ZT16', 'ZT18', 'ZT20', 'ZT22']
actual_time_labels = ['6:00', '8:00', '10:00', '12:00', '14:00', '16:00',
                     '18:00', '20:00', '22:00', '0:00', '2:00', '4:00']

# Cosinor model function
def cosinor_func(t, mesor, amplitude, acrophase_zt):
    return mesor + amplitude * np.cos(2 * np.pi / 24 * (t - acrophase_zt))

def fit_cosinor_model(expression_values, time_points_zt):
    expression_array = np.array(expression_values)
    min_nonzero = min([v for v in expression_array if v > 0], default=1e-10)
    expression_processed = np.where(expression_array > 0, expression_array, min_nonzero)
    
    try:
        mesor_guess = np.mean(expression_processed)
        amplitude_guess = (np.max(expression_processed) - np.min(expression_processed)) / 2
        peak_idx = np.argmax(expression_processed)
        acrophase_guess = time_points_zt[peak_idx]
        
        bounds = ([0, 0, 0], [np.inf, np.inf, 24])
        
        popt, pcov = curve_fit(cosinor_func, time_points_zt, expression_processed, 
                               p0=[mesor_guess, amplitude_guess, acrophase_guess],
                               bounds=bounds, maxfev=5000)
        
        mesor, amplitude, acrophase_zt = popt
        fitted_values = cosinor_func(time_points_zt, mesor, amplitude, acrophase_zt)
        
        residuals = expression_processed - fitted_values
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((expression_processed - np.mean(expression_processed))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        n = len(time_points_zt)
        p = len(popt)
        if n > p and ss_tot > 0:
            f_value = (ss_tot - ss_res) / (p - 1) / (ss_res / (n - p)) if ss_res > 0 else 0
            p_value = 1 - stats.f.cdf(f_value, p-1, n-p) if f_value > 0 else 1
        else:
            p_value = 1
            
        acrophase_actual = (acrophase_zt + 6) % 24
        
        return {
            'success': True,
            'params': {'mesor': mesor, 'amplitude': amplitude, 
                      'acrophase_zt': acrophase_zt, 'acrophase_actual': acrophase_actual},
            'fitted_values': fitted_values,
            'r_squared': r_squared,
            'p_value': p_value
        }
    except Exception as e:
        return {
            'success': False,
            'params': None,
            'fitted_values': None,
            'r_squared': 0,
            'p_value': 1
        }

# Analyze all tissues
results = []
for _, row in df.iterrows():
    tissue = row['Tissue']
    expression = row[time_labels].values.astype(float)
    
    mean_expr = np.mean(expression[expression > 0]) if np.any(expression > 0) else 0
    max_expr = np.max(expression)
    amplitude = max_expr - np.min(expression[expression > 0]) if np.any(expression > 0) else 0
    
    fit_result = fit_cosinor_model(expression, zt_hours)
    
    if fit_result['success']:
        rhythm_score = 0.7 * fit_result['r_squared'] + 0.3 * min(amplitude / (mean_expr + 1e-10), 1)
        acrophase_actual = fit_result['params']['acrophase_actual']
    else:
        rhythm_score = 0
        acrophase_actual = None
    
    results.append({
        'Tissue': tissue,
        'mean_expression': mean_expr,
        'rhythm_score': rhythm_score,
        'r_squared': fit_result['r_squared'],
        'p_value': fit_result['p_value'],
        'acrophase_actual': acrophase_actual,
        'fit_result': fit_result
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('rhythm_score', ascending=False)

print("\nTop 15 most rhythmic tissues:")
print("=" * 80)
print(f"{'Tissue':<10} {'Rhythm Score':<12} {'R²':<10} {'p-value':<10} {'Peak Time':<10}")
print("-" * 80)
for _, row in results_df.head(15).iterrows():
    peak_time = f"{int(row['acrophase_actual']):02d}:00" if row['acrophase_actual'] is not None else "N/A"
    print(f"{row['Tissue']:<10} {row['rhythm_score']:.4f}      {row['r_squared']:.4f}    "
          f"{row['p_value']:.4f}    {peak_time}")

# ============================================================================
# 1. INTEGRATED MULTI-CURVE PLOT: All top rhythmic tissues in one figure
# ============================================================================
print("\nGenerating integrated multi-curve plot...")

# Select top tissues for the integrated plot
top_n = 12
top_tissues = results_df.head(top_n)['Tissue'].values

# Create a figure with subplots for different views
fig = plt.figure(figsize=(20, 16))

# Color palette for tissues
colors = plt.cm.tab20(np.linspace(0, 1, top_n))

# 1A. Main integrated plot: Raw data with fitted curves
ax1 = plt.subplot(2, 2, (1, 2))  # Span first row

# Plot each tissue's data and fitted curve
for i, tissue in enumerate(top_tissues):
    tissue_data = df[df['Tissue'] == tissue]
    expression = tissue_data[time_labels].values.flatten().astype(float)
    fit_result = results_df[results_df['Tissue'] == tissue]['fit_result'].iloc[0]
    
    # Normalize expression for better visualization
    if np.max(expression) > 0:
        expression_norm = expression / np.max(expression)
    else:
        expression_norm = expression
    
    # Plot raw data points (smaller, semi-transparent)
    ax1.plot(actual_hours, expression_norm, 'o', color=colors[i], 
             markersize=4, alpha=0.5, label=f'{tissue} (data)')
    
    # Plot fitted curve
    if fit_result['success']:
        # Generate dense curve
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
        
        # Sort and handle wrap-around
        sort_idx = np.argsort(actual_dense)
        actual_sorted = actual_dense[sort_idx]
        fitted_sorted = fitted_norm[sort_idx]
        
        # Plot curve
        mask = (actual_sorted >= 0) & (actual_sorted <= 24)
        ax1.plot(actual_sorted[mask], fitted_sorted[mask], '-', 
                color=colors[i], linewidth=2, alpha=0.8, label=f'{tissue}')

# Add day/night shading
ax1.axvspan(6, 18, alpha=0.1, color='yellow', label='Day')
ax1.axvspan(0, 6, alpha=0.1, color='blue', label='Night')
ax1.axvspan(18, 24, alpha=0.1, color='blue')

ax1.set_xlabel('Time of Day', fontsize=14)
ax1.set_ylabel('Normalized Expression Level', fontsize=14)
ax1.set_title(f'Integrated ADRA1B Expression Curves: Top {top_n} Rhythmic Tissues\n(ZT0 = 6:00, ZT12 = 18:00)', 
              fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks([0, 4, 8, 12, 16, 20, 24])
ax1.set_xticklabels(['0:00', '4:00', '8:00', '12:00', '16:00', '20:00', '24:00'], rotation=45)
ax1.set_xlim(0, 24)
ax1.set_ylim(-0.1, 1.1)

# Create a compact legend (outside plot)
ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, 
          fontsize=9, ncol=2)

# 1B. Rhythm score bar plot
ax2 = plt.subplot(2, 2, 3)
rhythm_scores = results_df.head(top_n).sort_values('rhythm_score', ascending=True)
bars = ax2.barh(rhythm_scores['Tissue'], rhythm_scores['rhythm_score'], 
                color=colors[np.argsort(rhythm_scores['rhythm_score'])])
ax2.set_xlabel('Rhythm Score', fontsize=12)
ax2.set_title('Rhythmicity Ranking', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# Add R² values as text
for i, bar in enumerate(bars):
    width = bar.get_width()
    tissue = rhythm_scores.iloc[i]['Tissue']
    r2 = rhythm_scores.iloc[i]['r_squared']
    ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
             f'R²={r2:.2f}', va='center', fontsize=8)

# 1C. Peak time distribution
ax3 = plt.subplot(2, 2, 4)
peak_times = []
tissue_names = []

for tissue in top_tissues:
    peak_time = results_df[results_df['Tissue'] == tissue]['acrophase_actual'].iloc[0]
    if peak_time is not None:
        peak_times.append(peak_time)
        tissue_names.append(tissue)

if peak_times:
    # Create circular histogram
    ax3_circular = ax3  # We'll create a polar subplot later
    
    # Create inset for linear histogram
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax3_inset = inset_axes(ax3, width="60%", height="40%", loc='upper right')
    
    # Linear histogram
    ax3_inset.hist(peak_times, bins=12, range=(0, 24), alpha=0.7, 
                  color='steelblue', edgecolor='black')
    ax3_inset.axvspan(6, 18, alpha=0.1, color='yellow')
    ax3_inset.axvspan(0, 6, alpha=0.1, color='blue')
    ax3_inset.axvspan(18, 24, alpha=0.1, color='blue')
    ax3_inset.set_xlabel('Peak Time', fontsize=8)
    ax3_inset.set_ylabel('Count', fontsize=8)
    ax3_inset.set_xticks([0, 6, 12, 18, 24])
    ax3_inset.set_xticklabels(['0', '6', '12', '18', '24'], fontsize=7)
    ax3_inset.grid(True, alpha=0.3)
    
    # Create polar plot for circular distribution
    ax3.remove()
    ax3 = fig.add_subplot(2, 2, 4, projection='polar')
    
    # Convert to radians
    peak_radians = [p / 12 * np.pi for p in peak_times]
    
    # Plot each tissue's peak time
    for j, (tissue, peak_rad) in enumerate(zip(tissue_names, peak_radians)):
        color_idx = np.where(top_tissues == tissue)[0][0]
        ax3.plot([peak_rad, peak_rad], [0, 1], '-', color=colors[color_idx], 
                linewidth=2, alpha=0.7)
        ax3.plot(peak_rad, 1, 'o', color=colors[color_idx], markersize=8)
    
    # Set up polar plot
    ax3.set_theta_zero_location("N")
    ax3.set_theta_direction(-1)
    ax3.set_xticks(np.linspace(0, 2*np.pi, 12, endpoint=False))
    ax3.set_xticklabels(['0:00', '2:00', '4:00', '6:00', '8:00', '10:00',
                        '12:00', '14:00', '16:00', '18:00', '20:00', '22:00'])
    ax3.set_title('Peak Time Distribution (Polar)', fontsize=14, fontweight='bold', pad=20)
    ax3.grid(True)

ax3.set_title('Peak Time Analysis', fontsize=14, fontweight='bold')

plt.suptitle('Comprehensive ADRA1B Gene Expression Rhythm Analysis', 
             fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
plt.savefig('img1.png', dpi=300, bbox_inches='tight')


# ============================================================================
# 2. SIMPLIFIED MULTI-CURVE PLOT: Focus on the curves only
# ============================================================================
print("\nGenerating simplified multi-curve plot...")

plt.figure(figsize=(16, 10))

# Plot only the fitted curves (no data points)
for i, tissue in enumerate(top_tissues):
    fit_result = results_df[results_df['Tissue'] == tissue]['fit_result'].iloc[0]
    
    if fit_result['success']:
        # Generate dense curve
        zt_dense = np.linspace(-6, 30, 400)
        fitted_dense = cosinor_func(zt_dense, 
                                   fit_result['params']['mesor'],
                                   fit_result['params']['amplitude'],
                                   fit_result['params']['acrophase_zt'])
        
        # Normalize
        if np.max(fitted_dense) > 0:
            fitted_norm = fitted_dense / np.max(fitted_dense)
        else:
            fitted_norm = fitted_dense
        
        # Convert to actual time
        actual_dense = (zt_dense + 6) % 24
        
        # Sort and plot
        sort_idx = np.argsort(actual_dense)
        actual_sorted = actual_dense[sort_idx]
        fitted_sorted = fitted_norm[sort_idx]
        
        # Get rhythm score for line width
        rhythm_score = results_df[results_df['Tissue'] == tissue]['rhythm_score'].iloc[0]
        linewidth = 1 + rhythm_score * 3  # Thicker lines for more rhythmic tissues
        
        # Plot curve
        mask = (actual_sorted >= 0) & (actual_sorted <= 24)
        plt.plot(actual_sorted[mask], fitted_sorted[mask], '-', 
                color=colors[i], linewidth=linewidth, alpha=0.8, label=f'{tissue}')

# Add day/night shading
plt.axvspan(6, 18, alpha=0.1, color='yellow', label='Day')
plt.axvspan(0, 6, alpha=0.1, color='blue', label='Night')
plt.axvspan(18, 24, alpha=0.1, color='blue')

plt.xlabel('Time of Day', fontsize=14)
plt.ylabel('Normalized Expression Level', fontsize=14)
plt.title(f'ADRA1B Expression Rhythm Patterns: Top {top_n} Tissues\n(Line thickness indicates rhythm strength)', 
          fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks([0, 4, 8, 12, 16, 20, 24], 
           ['0:00', '4:00', '8:00', '12:00', '16:00', '20:00', '24:00'], 
           rotation=45)
plt.xlim(0, 24)
plt.ylim(-0.1, 1.1)

# Create a clean legend
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, 
          fontsize=10, ncol=1, title='Tissues', title_fontsize=12)

plt.tight_layout()
plt.show()
plt.savefig('img2.png', dpi=300, bbox_inches='tight')


# ============================================================================
# 3. GROUPED MULTI-CURVE PLOT: By peak time clusters
# ============================================================================
print("\nGenerating grouped multi-curve plot by peak time...")

# Group tissues by peak time
day_tissues = []
night_tissues = []
other_tissues = []

for tissue in top_tissues:
    peak_time = results_df[results_df['Tissue'] == tissue]['acrophase_actual'].iloc[0]
    if peak_time is not None:
        if 6 <= peak_time < 18:
            day_tissues.append(tissue)
        else:
            night_tissues.append(tissue)
    else:
        other_tissues.append(tissue)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Day-peaking tissues
ax1 = axes[0]
for i, tissue in enumerate(day_tissues):
    fit_result = results_df[results_df['Tissue'] == tissue]['fit_result'].iloc[0]
    
    if fit_result['success']:
        zt_dense = np.linspace(-6, 30, 400)
        fitted_dense = cosinor_func(zt_dense, 
                                   fit_result['params']['mesor'],
                                   fit_result['params']['amplitude'],
                                   fit_result['params']['acrophase_zt'])
        
        if np.max(fitted_dense) > 0:
            fitted_norm = fitted_dense / np.max(fitted_dense)
        else:
            fitted_norm = fitted_dense
        
        actual_dense = (zt_dense + 6) % 24
        sort_idx = np.argsort(actual_dense)
        actual_sorted = actual_dense[sort_idx]
        fitted_sorted = fitted_norm[sort_idx]
        
        mask = (actual_sorted >= 0) & (actual_sorted <= 24)
        color_idx = np.where(top_tissues == tissue)[0][0]
        ax1.plot(actual_sorted[mask], fitted_sorted[mask], '-', 
                color=colors[color_idx], linewidth=2, alpha=0.8, label=tissue)

ax1.axvspan(6, 18, alpha=0.2, color='yellow')
ax1.set_xlabel('Time of Day', fontsize=12)
ax1.set_ylabel('Normalized Expression', fontsize=12)
ax1.set_title(f'Day-peaking Tissues (n={len(day_tissues)})', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks([0, 6, 12, 18, 24])
ax1.set_xticklabels(['0:00', '6:00', '12:00', '18:00', '24:00'], rotation=45)
ax1.set_xlim(0, 24)
ax1.legend(loc='upper right', fontsize=9)

# Night-peaking tissues
ax2 = axes[1]
for i, tissue in enumerate(night_tissues):
    fit_result = results_df[results_df['Tissue'] == tissue]['fit_result'].iloc[0]
    
    if fit_result['success']:
        zt_dense = np.linspace(-6, 30, 400)
        fitted_dense = cosinor_func(zt_dense, 
                                   fit_result['params']['mesor'],
                                   fit_result['params']['amplitude'],
                                   fit_result['params']['acrophase_zt'])
        
        if np.max(fitted_dense) > 0:
            fitted_norm = fitted_dense / np.max(fitted_dense)
        else:
            fitted_norm = fitted_dense
        
        actual_dense = (zt_dense + 6) % 24
        sort_idx = np.argsort(actual_dense)
        actual_sorted = actual_dense[sort_idx]
        fitted_sorted = fitted_norm[sort_idx]
        
        mask = (actual_sorted >= 0) & (actual_sorted <= 24)
        color_idx = np.where(top_tissues == tissue)[0][0]
        ax2.plot(actual_sorted[mask], fitted_sorted[mask], '-', 
                color=colors[color_idx], linewidth=2, alpha=0.8, label=tissue)

ax2.axvspan(0, 6, alpha=0.2, color='blue')
ax2.axvspan(18, 24, alpha=0.2, color='blue')
ax2.set_xlabel('Time of Day', fontsize=12)
ax2.set_ylabel('Normalized Expression', fontsize=12)
ax2.set_title(f'Night-peaking Tissues (n={len(night_tissues)})', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xticks([0, 6, 12, 18, 24])
ax2.set_xticklabels(['0:00', '6:00', '12:00', '18:00', '24:00'], rotation=45)
ax2.set_xlim(0, 24)
ax2.legend(loc='upper right', fontsize=9)

# All tissues together
ax3 = axes[2]
for i, tissue in enumerate(top_tissues):
    fit_result = results_df[results_df['Tissue'] == tissue]['fit_result'].iloc[0]
    
    if fit_result['success']:
        zt_dense = np.linspace(-6, 30, 400)
        fitted_dense = cosinor_func(zt_dense, 
                                   fit_result['params']['mesor'],
                                   fit_result['params']['amplitude'],
                                   fit_result['params']['acrophase_zt'])
        
        if np.max(fitted_dense) > 0:
            fitted_norm = fitted_dense / np.max(fitted_dense)
        else:
            fitted_norm = fitted_dense
        
        actual_dense = (zt_dense + 6) % 24
        sort_idx = np.argsort(actual_dense)
        actual_sorted = actual_dense[sort_idx]
        fitted_sorted = fitted_norm[sort_idx]
        
        mask = (actual_sorted >= 0) & (actual_sorted <= 24)
        ax3.plot(actual_sorted[mask], fitted_sorted[mask], '-', 
                color=colors[i], linewidth=1, alpha=0.6)

ax3.axvspan(6, 18, alpha=0.1, color='yellow', label='Day')
ax3.axvspan(0, 6, alpha=0.1, color='blue', label='Night')
ax3.axvspan(18, 24, alpha=0.1, color='blue')
ax3.set_xlabel('Time of Day', fontsize=12)
ax3.set_ylabel('Normalized Expression', fontsize=12)
ax3.set_title(f'All {top_n} Top Tissues (Overlay)', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xticks([0, 6, 12, 18, 24])
ax3.set_xticklabels(['0:00', '6:00', '12:00', '18:00', '24:00'], rotation=45)
ax3.set_xlim(0, 24)
ax3.legend()

plt.suptitle('ADRA1B Expression Rhythms Grouped by Peak Time', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
plt.savefig('img3.png', dpi=300, bbox_inches='tight')


# ============================================================================
# 4. DENSITY PLOT: Showing distribution of curves
# ============================================================================
print("\nGenerating density plot of expression curves...")

plt.figure(figsize=(14, 8))

# Generate time points for density calculation
time_grid = np.linspace(0, 24, 241)  # Every 0.1 hour
density_matrix = np.zeros((len(top_tissues), len(time_grid)))

# Fill density matrix
for i, tissue in enumerate(top_tissues):
    fit_result = results_df[results_df['Tissue'] == tissue]['fit_result'].iloc[0]
    
    if fit_result['success']:
        # Calculate fitted values at each time point
        for j, t_actual in enumerate(time_grid):
            # Convert actual time to ZT
            t_zt = (t_actual - 6) % 24
            fitted_value = cosinor_func(t_zt, 
                                       fit_result['params']['mesor'],
                                       fit_result['params']['amplitude'],
                                       fit_result['params']['acrophase_zt'])
            
            # Normalize
            if fit_result['params']['amplitude'] > 0:
                # Find max and min for normalization
                zt_test = np.linspace(0, 24, 241)
                fitted_test = cosinor_func(zt_test, 
                                          fit_result['params']['mesor'],
                                          fit_result['params']['amplitude'],
                                          fit_result['params']['acrophase_zt'])
                if np.max(fitted_test) > np.min(fitted_test):
                    norm_value = (fitted_value - np.min(fitted_test)) / (np.max(fitted_test) - np.min(fitted_test))
                else:
                    norm_value = 0.5
            else:
                norm_value = 0.5
            
            density_matrix[i, j] = norm_value

# Create heatmap
plt.imshow(density_matrix, aspect='auto', cmap='viridis', 
           extent=[0, 24, 0, len(top_tissues)], origin='lower')
plt.savefig('img4.png', dpi=300, bbox_inches='tight')


plt.colorbar(label='Normalized Expression Level')

# Set y-ticks
plt.yticks(np.arange(len(top_tissues)) + 0.5, top_tissues, fontsize=9)

# Set x-ticks
plt.xticks([0, 6, 12, 18, 24], ['0:00', '6:00', '12:00', '18:00', '24:00'], rotation=45)

# Add day/night shading
plt.axvspan(6, 18, alpha=0.1, color='yellow')
plt.axvspan(0, 6, alpha=0.1, color='blue')
plt.axvspan(18, 24, alpha=0.1, color='blue')

plt.xlabel('Time of Day', fontsize=12)
plt.ylabel('Tissue', fontsize=12)
plt.title(f'Density Plot of ADRA1B Expression Rhythms\nTop {top_n} Most Rhythmic Tissues', 
          fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
plt.savefig('img5.png', dpi=300, bbox_inches='tight')


print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)
print(f"\nGenerated multiple multi-curve plots showing:")
print(f"1. Integrated plot with {top_n} tissues (data + curves + statistics)")
print(f"2. Simplified curve-only plot")
print(f"3. Grouped plots by peak time")
print(f"4. Density heatmap of expression patterns")
print(f"\nTotal tissues analyzed: {len(df)}")
print(f"Most rhythmic tissue: {results_df.iloc[0]['Tissue']} (Score: {results_df.iloc[0]['rhythm_score']:.3f})")
print(f"Least rhythmic tissue: {results_df.iloc[-1]['Tissue']} (Score: {results_df.iloc[-1]['rhythm_score']:.3f})")
