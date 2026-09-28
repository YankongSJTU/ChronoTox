import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
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

# ============================================================================
# META CYCLE METHODS IMPLEMENTATION (替换原始的curve_fit方法)
# ============================================================================

def cosinor_fit(time_points, expression_values, period=24):
    """Cosinor regression from MetaCycle"""
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
        
        # Calculate fitted values at original time points
        fitted = mesor + amplitude * np.cos(2 * np.pi * (time_points - acrophase) / period)
        
        # Calculate R²
        residuals = expr_array - fitted
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((expr_array - np.mean(expr_array))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # F-test for significance
        n = len(t)
        p = 3  # number of parameters
        if n > p and ss_tot > 0:
            f_stat = ((ss_tot - ss_res) / (p - 1)) / (ss_res / (n - p))
            p_value = 1 - stats.f.cdf(f_stat, p-1, n-p)
        else:
            p_value = 1
        
        # Convert acrophase to actual time
        acrophase_actual = (acrophase + 6) % 24
        
        return {
            'success': True,
            'params': {
                'mesor': mesor, 
                'amplitude': amplitude, 
                'acrophase_zt': acrophase, 
                'acrophase_actual': acrophase_actual
            },
            'fitted_values': fitted,
            'r_squared': r_squared,
            'p_value': p_value,
            'expression_processed': expr_array,
            'method': 'Cosinor-24h'
        }
    except Exception as e:
        return None

def lomb_scargle_fit(time_points, expression_values, min_period=20, max_period=28):
    """Lomb-Scargle periodogram analysis"""
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
    power = signal.lombscargle(t, y_norm, 2 * np.pi * frequencies)
    
    # Find dominant frequency
    dominant_idx = np.argmax(power)
    dominant_freq = frequencies[dominant_idx]
    period = 1 / dominant_freq
    
    # Calculate p-value
    M = len(frequencies)
    z = power[dominant_idx]
    p_value = M * np.exp(-z)
    p_value = min(p_value, 1)
    
    # Fit cosinor with detected period
    cosinor_result = cosinor_fit(time_points, expression_values, period=period)
    
    if cosinor_result:
        cosinor_result['method'] = 'Lomb-Scargle'
        return cosinor_result
    return None

def metacycle_analysis(time_points, expression_values):
    """MetaCycle analysis combining multiple methods"""
    results = {}
    
    # Try cosinor with 24-hour period
    cosinor_24 = cosinor_fit(time_points, expression_values, period=24)
    if cosinor_24:
        results['Cosinor-24h'] = cosinor_24
    
    # Try Lomb-Scargle
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
        # Fallback
        expr_array = np.array(expression_values)
        mask = expr_array > 0
        mean_expr = np.mean(expr_array[mask]) if np.sum(mask) > 0 else 0
        
        best_result = {
            'success': False,
            'params': {
                'mesor': mean_expr,
                'amplitude': 0,
                'acrophase_zt': 12,
                'acrophase_actual': (12 + 6) % 24
            },
            'fitted_values': np.full_like(expression_values, mean_expr),
            'r_squared': 0,
            'p_value': 1,
            'expression_processed': expression_values,
            'method': 'None',
            'method_name': 'No fit'
        }
    
    return best_result

# ============================================================================
# Cosinor model function (保持相同接口，内部调用MetaCycle)
# ============================================================================
def cosinor_func(t, mesor, amplitude, acrophase_zt):
    """Cosinor model function for curve plotting"""
    return mesor + amplitude * np.cos(2 * np.pi / 24 * (t - acrophase_zt))

def fit_cosinor_model(expression_values, time_points_zt):
    """Wrapper function that uses MetaCycle analysis"""
    # Use MetaCycle analysis instead of curve_fit
    result = metacycle_analysis(time_points_zt, expression_values)
    
    # Format to match original output structure
    return result

# ============================================================================
# 保持完全相同的分析流程（只替换了拟合方法）
# ============================================================================
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
    
    # Fit cosinor model (现在使用MetaCycle)
    fit_result = fit_cosinor_model(expression, zt_hours)
    
    # Rhythm score (保持相同公式)
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
        'fit_result': fit_result,
        'method': fit_result.get('method_name', fit_result.get('method', 'Unknown'))
    })

# Create results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('rhythm_score', ascending=False)

print("\nMetaCycle Curve Fitting Results Summary:")
print("=" * 90)
print(f"{'Tissue':<10} {'Method':<12} {'Rhythm Score':<12} {'R²':<10} {'p-value':<10} {'Amplitude':<12} {'Acrophase (ZT)':<14} {'Acrophase (Actual)':<15}")
print("-" * 90)

for _, row in results_df.head(15).iterrows():
    if row['acrophase_actual'] is not None:
        actual_time_str = f"{int(row['acrophase_actual']):02d}:00"
        print(f"{row['Tissue']:<10} {row['method'][:12]:<12} {row['rhythm_score']:.4f}      {row['r_squared']:.4f}    "
              f"{row['p_value']:.4f}    {row['amplitude']:.4f}     "
              f"ZT{row['acrophase_zt']:.1f}          {actual_time_str}")
    else:
        print(f"{row['Tissue']:<10} {row['method'][:12]:<12} {row['rhythm_score']:.4f}      {row['r_squared']:.4f}    "
              f"{row['p_value']:.4f}    {row['amplitude']:.4f}     N/A             N/A")

# ============================================================================
# 创建全局颜色映射字典 (新增代码)
# ============================================================================
print("\nCreating global color mapping for tissues...")

# 选择要显示的组织数量
top_n = 11
top_tissues = results_df.head(top_n)['Tissue'].values

# 创建颜色映射 - 使用tab20色彩映射，确保足够的颜色区分度
tissue_colors = {}
cmap = plt.cm.tab20  # 20种明显区分的颜色

for i, tissue in enumerate(top_tissues):
    # 循环使用颜色，如果有超过20个组织，会重复使用颜色
    color_idx = i % cmap.N
    tissue_colors[tissue] = cmap(color_idx)

print(f"Created color mapping for {len(tissue_colors)} tissues")

# ============================================================================
# 1. Plot curve fitting for all tissues (使用统一颜色)
# ============================================================================
print("\nGenerating curve fitting plots with MetaCycle method...")

fig, axes = plt.subplots(4, 3, figsize=(18, 15))
axes = axes.flatten()

for i, tissue in enumerate(top_tissues[:11]):
    ax = axes[i]
    tissue_data = df[df['Tissue'] == tissue]
    expression = tissue_data[time_labels].values.flatten().astype(float)
    fit_result = results_df[results_df['Tissue'] == tissue]['fit_result'].iloc[0]
    
    # 使用统一的颜色
    color = tissue_colors[tissue]
    
    # Plot raw data points using actual time
    ax.plot(actual_hours, expression, 'o', color=color, markersize=6, label='Raw Data')
    
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
                       '-', color=color, linewidth=2, label='MetaCycle Fit')
                ax.plot(actual_dense_plot[split_idx:], fitted_dense_plot[split_idx:], 
                       '-', color=color, linewidth=2)
            else:
                ax.plot(actual_dense_plot, fitted_dense_plot, '-', color=color, linewidth=2, label='MetaCycle Fit')
        
        # Mark peak phase in actual time
        peak_time_actual = fit_result['params']['acrophase_actual']
        peak_time_zt = fit_result['params']['acrophase_zt']
        peak_value = cosinor_func(peak_time_zt,
                                 fit_result['params']['mesor'],
                                 fit_result['params']['amplitude'],
                                 fit_result['params']['acrophase_zt'])
        ax.plot(peak_time_actual, peak_value, '*', color=color, markersize=12, 
                label=f'Peak ({int(peak_time_actual):02d}:00)')
        
        # 修改信息框：显示peak时间点和统计信息
        method_name = fit_result.get('method_name', fit_result.get('method', 'MetaCycle'))
        
        # 格式化peak时间
        peak_hour = int(peak_time_actual)
        peak_minute = int((peak_time_actual - peak_hour) * 60)
        peak_time_str = f"{peak_hour:02d}:{peak_minute:02d}"
        
        # 创建信息文本 - 重点显示peak时间
        info_text = f"Peak: {peak_time_str}\n"
        info_text += f"R² = {fit_result['r_squared']:.3f}\n"
        info_text += f"p = {fit_result['p_value']:.3f}\n"
        info_text += f"Method: {method_name[:8]}"
        
        # 根据p值添加显著性标记
        if fit_result['p_value'] < 0.001:
            info_text += "***"
        elif fit_result['p_value'] < 0.01:
            info_text += "**"
        elif fit_result['p_value'] < 0.05:
            info_text += "*"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.1))
    
    # Set axis labels and properties
    ax.set_xlabel('Time of Day', fontsize=10)
    ax.set_ylabel('Expression Level', fontsize=10)
    
    # 在标题中也显示rhythm score
    rhythm_score = results_df[results_df["Tissue"]==tissue]["rhythm_score"].iloc[0]
    ax.set_title(f'{tissue}\nRhythm Score: {rhythm_score:.3f}', 
                 fontsize=11, fontweight='bold', color=color)
    
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
    
    # 设置y轴范围，确保peak标记可见
    y_data = expression[expression > 0]
    if len(y_data) > 0 and fit_result['success']:
        y_min =  max(0, np.min(y_data) * 0.8)
        y_max = max(np.max(y_data), peak_value) * 1.2
        if y_max <2:
            y_min=-0.5
        if y_max<0.5:
            y_min=-0.1
        ax.set_ylim(y_min, y_max)
    
    # Add legend (only for first subplot to avoid repetition)
    if i == 0:
        # 简化图例，只显示必要的项目
        handles, labels = ax.get_legend_handles_labels()
        # 只保留第一个出现的每个标签
        unique_labels = []
        unique_handles = []
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)
        ax.legend(unique_handles, unique_labels, loc='upper right', fontsize=8)

# Adjust layout
plt.suptitle('ADRA1B Gene Expression Curve Fitting (MetaCycle Method) by Tissue\nZT0 = 6:00, ZT12 = 18:00', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('imgneed1.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 2. Simple overlay plot with normalized expression (使用统一颜色)
# ============================================================================
print("\nGenerating normalized overlay plot with MetaCycle...")

plt.figure(figsize=(15, 6))

for i, tissue in enumerate(top_tissues):
    tissue_data = df[df['Tissue'] == tissue]
    expression = tissue_data[time_labels].values.flatten().astype(float)
    fit_result = results_df[results_df['Tissue'] == tissue]['fit_result'].iloc[0]
    
    # 使用统一的颜色
    color = tissue_colors[tissue]
    
    # Normalize expression for comparison
    if np.max(expression) > 0:
        expression_norm = expression / np.max(expression)
    else:
        expression_norm = expression
    
    # Plot normalized raw data
    plt.plot(actual_hours, expression_norm, 'o', color=color, markersize=8, alpha=0.6)
    
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
                        '-', color=color, linewidth=2.5, alpha=0.8)
                plt.plot(actual_dense_plot[split_idx:], fitted_norm_plot[split_idx:], 
                        '-', color=color, linewidth=2.5, alpha=0.8, 
                        label=f'{tissue} ({fit_result.get("method_name", "MetaCycle")}, R²={fit_result["r_squared"]:.3f})')
            else:
                plt.plot(actual_dense_plot, fitted_norm_plot, '-', color=color, 
                        linewidth=2.5, alpha=0.8, 
                        label=f'{tissue} ({fit_result.get("method_name", "MetaCycle")}, R²={fit_result["r_squared"]:.3f})')

plt.xlabel('Time of Day', fontsize=12)
plt.ylabel('Normalized Expression', fontsize=12)
plt.title(f'Normalized ADRA1B Expression in Top {top_n} Rhythmic Tissues\nMetaCycle Analysis (ZT0 = 6:00, ZT12 = 18:00)', 
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
plt.savefig('imgneed3.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 3. 其他图形（保持原样但使用统一颜色）
# ============================================================================
print("\nGenerating additional plots...")

# Plot overlay of top rhythmic tissues - CLEAN VERSION (no horizontal lines)
plt.figure(figsize=(14, 8))

for i, tissue in enumerate(top_tissues):
    tissue_data = df[df['Tissue'] == tissue]
    expression = tissue_data[time_labels].values.flatten().astype(float)
    fit_result = results_df[results_df['Tissue'] == tissue]['fit_result'].iloc[0]
    
    # 使用统一的颜色
    color = tissue_colors[tissue]
    
    # Plot raw data points
    plt.plot(actual_hours, expression, 'o', color=color, markersize=8, alpha=0.6, label=f'{tissue} (Data)')
    
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
                        '-', color=color, linewidth=2.5, alpha=0.8)
                plt.plot(actual_dense_plot[split_idx:], fitted_dense_plot[split_idx:], 
                        '-', color=color, linewidth=2.5, alpha=0.8, 
                        label=f'{tissue} (R²={fit_result["r_squared"]:.3f})')
            else:
                plt.plot(actual_dense_plot, fitted_dense_plot, '-', color=color, 
                        linewidth=2.5, alpha=0.8, 
                        label=f'{tissue} (R²={fit_result["r_squared"]:.3f})')

plt.xlabel('Time of Day', fontsize=12)
plt.ylabel('Expression Level', fontsize=12)
plt.title(f'ADRA1B Expression Curve Fitting in Top {top_n} Rhythmic Tissues\nMetaCycle Analysis (ZT0 = 6:00, ZT12 = 18:00)', 
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
plt.savefig('img2.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 4. Single tissue detailed example (使用最高表达的组织)
# ============================================================================
print("\nDetailed example analysis")
# 使用最高节律评分的组织
best_tissue = results_df.iloc[0]['Tissue']
best_fit = results_df.iloc[0]['fit_result']

tissue_data = df[df['Tissue'] == best_tissue]
expression = tissue_data[time_labels].values.flatten().astype(float)

# 获取最佳组织的颜色
best_color = tissue_colors[best_tissue]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Raw data with fitted curve
ax1.plot(actual_hours, expression, 'o-', color=best_color, markersize=8, linewidth=1.5, label='Raw Data')

if best_fit['success']:
    # Generate extended curve
    zt_dense = np.linspace(-6, 30, 400)
    fitted_dense = cosinor_func(zt_dense, 
                               best_fit['params']['mesor'],
                               best_fit['params']['amplitude'],
                               best_fit['params']['acrophase_zt'])
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
        ax1.plot(actual_plot[:split_idx], fitted_plot[:split_idx], '-', color=best_color, linewidth=3, label='MetaCycle Fit')
        ax1.plot(actual_plot[split_idx:], fitted_plot[split_idx:], '-', color=best_color, linewidth=3)
    else:
        ax1.plot(actual_plot, fitted_plot, '-', color=best_color, linewidth=3, label='MetaCycle Fit')
    
    # Mark peak
    peak_time_actual = best_fit['params']['acrophase_actual']
    peak_time_zt = best_fit['params']['acrophase_zt']
    peak_value = cosinor_func(peak_time_zt,
                             best_fit['params']['mesor'],
                             best_fit['params']['amplitude'],
                             best_fit['params']['acrophase_zt'])
    ax1.plot(peak_time_actual, peak_value, '*', color=best_color, markersize=15, label=f'Peak ({int(peak_time_actual):02d}:00)')

ax1.set_xlabel('Time of Day', fontsize=12)
ax1.set_ylabel('Expression Level', fontsize=12)
method_name = best_fit.get('method_name', best_fit.get('method', 'MetaCycle'))
ax1.set_title(f'{best_tissue} Tissue: ADRA1B Expression and {method_name} Fit\n(ZT0 = 6:00, ZT12 = 18:00)', 
              fontsize=14, fontweight='bold', color=best_color)

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
if best_fit['success']:
    residuals = best_fit['expression_processed'] - best_fit['fitted_values']
    
    ax2.plot(actual_hours, residuals, 'o-', color=best_color, markersize=8, linewidth=1.5)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Time of Day', fontsize=12)
    ax2.set_ylabel('Residuals (Observed - Fitted)', fontsize=12)
    ax2.set_title(f'Residual Analysis\nR² = {best_fit["r_squared"]:.3f}, p = {best_fit["p_value"]:.3f}', 
                  fontsize=14, fontweight='bold', color=best_color)
    
    # Add day/night shading
    ax2.axvspan(6, 18, alpha=0.1, color='yellow')
    ax2.axvspan(0, 6, alpha=0.1, color='blue')
    ax2.axvspan(18, 24, alpha=0.1, color='blue')
    
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(np.arange(0, 25, 2))
    ax2.set_xticklabels([f'{int(h):02d}:00' for h in np.arange(0, 25, 2)], rotation=45)
    ax2.set_xlim(0, 24)

plt.tight_layout()
plt.savefig('img4.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 5. 综合多曲线图 (使用统一颜色)
# ============================================================================
print("\nGenerating integrated multi-curve plot...")

# Create a figure with subplots for different views
fig = plt.figure(figsize=(16, 12))

# 1A. Main integrated plot: Raw data with fitted curves
ax1 = plt.subplot(2, 3, (1, 3))  # Span first row

# Plot each tissue's data and fitted curve
for i, tissue in enumerate(top_tissues):
    tissue_data = df[df['Tissue'] == tissue]
    expression = tissue_data[time_labels].values.flatten().astype(float)
    fit_result = results_df[results_df['Tissue'] == tissue]['fit_result'].iloc[0]

    # 使用统一的颜色
    color = tissue_colors[tissue]

    # Normalize expression for better visualization
    if np.max(expression) > 0:
        expression_norm = expression / np.max(expression)
    else:
        expression_norm = expression

    # Plot raw data points (smaller, semi-transparent)
    ax1.plot(actual_hours, expression_norm, 'o', color=color,
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
                color=color, linewidth=2, alpha=0.8, label=f'{tissue}')

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

# 2A. 节律评分条形图 (第2行，第1列) - 使用统一颜色
ax2 = plt.subplot(2, 3, 4)  # 第2行，第1列

# 获取数据并排序
rhythm_scores = results_df.head(top_n).sort_values('rhythm_score', ascending=True)

# 设置条形图参数
bar_height = 0.6  # 条形的宽度（控制窄度）
y_positions = np.arange(len(rhythm_scores))  # y轴位置

# 创建水平条形图（更窄的条形）- 使用统一颜色
bars = []
for i, (idx, row) in enumerate(rhythm_scores.iterrows()):
    tissue = row['Tissue']
    color = tissue_colors[tissue]
    
    bar = ax2.barh(i, row['rhythm_score'],
                   height=bar_height,
                   color=color,
                   edgecolor='black', linewidth=0.5,
                   alpha=0.8)
    bars.append(bar[0])

# 设置y轴标签（组织名称）
ax2.set_yticks(y_positions)
ax2.set_yticklabels(rhythm_scores['Tissue'], fontsize=9)

# 设置x轴
ax2.set_xlabel('Rhythm Score', fontsize=11)
ax2.set_xlim(0, max(rhythm_scores['rhythm_score']) * 1.15)  # 为文本留空间

# 标题和网格
ax2.set_title('Rhythmicity Ranking', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x', linestyle='--')

# 添加R²值和统计信息
for i, bar in enumerate(bars):
    width = bar.get_width()
    tissue = rhythm_scores.iloc[i]['Tissue']
    r2 = rhythm_scores.iloc[i]['r_squared']
    p_value = rhythm_scores.iloc[i]['p_value']
    
    # 获取方法信息
    method_data = results_df[results_df['Tissue'] == tissue]
    if not method_data.empty:
        method = method_data['method'].iloc[0]
        method_short = method.split('-')[0] if '-' in method else method[:6]
    else:
        method_short = 'MetaC'
    
    # 计算文本位置
    text_x = width + 0.005  # 稍微远离条形
    
    # 创建文本内容
    text_content = f'R²={r2:.2f}'
    
    # 根据p值添加显著性标记
    if p_value < 0.001:
        text_content += '***'
    elif p_value < 0.01:
        text_content += '**'
    elif p_value < 0.05:
        text_content += '*'
    
    # 在条形右侧添加文本
    ax2.text(text_x, bar.get_y() + bar.get_height()/2,
             text_content, va='center', fontsize=7.5,
             fontweight='bold' if p_value < 0.05 else 'normal')

# 1C. Peak time distribution - 使用统一颜色
ax3 = plt.subplot(2, 3, 5)
peak_times = []
tissue_names = []
colors_list = []

for tissue in top_tissues:
    peak_time = results_df[results_df['Tissue'] == tissue]['acrophase_actual'].iloc[0]
    if peak_time is not None:
        peak_times.append(peak_time)
        tissue_names.append(tissue)
        colors_list.append(tissue_colors[tissue])

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

    mean_peak = np.mean(peak_times)
    std_peak = np.std(peak_times)
    ax3.text(0.02, 0.98, f'Mean: {mean_peak:.1f}h\nStd: {std_peak:.1f}h',
             transform=ax3.transAxes, verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # 创建极坐标图
    ax4 = plt.subplot(2, 3, 6, projection='polar')

    # Convert to radians
    peak_radians = [p / 12 * np.pi for p in peak_times]

    # Plot each tissue's peak time - 使用统一颜色
    for j, (tissue, peak_rad, color) in enumerate(zip(tissue_names, peak_radians, colors_list)):
        ax4.plot([peak_rad, peak_rad], [0, 1], '-', color=color,
                linewidth=2, alpha=0.7)
        ax4.plot(peak_rad, 1, 'o', color=color, markersize=8)

    # Set up polar plot
    ax4.set_theta_zero_location("N")
    ax4.set_theta_direction(-1)
    ax4.set_xticks(np.linspace(0, 2*np.pi, 12, endpoint=False))
    ax4.set_xticklabels(['0:00', '2:00', '4:00', '6:00', '8:00', '10:00',
                        '12:00', '14:00', '16:00', '18:00', '20:00', '22:00'])
    ax4.set_title('Peak Time Distribution (Polar)', fontsize=14, fontweight='bold', pad=20)
    ax4.grid(True)

    ax4.text(np.pi/2, 1.4, 'Day', ha='center', va='center', fontsize=10, 
             fontweight='bold', color='darkorange')
    ax4.text(3*np.pi/2, 1.4, 'Night', ha='center', va='center', fontsize=10,
             fontweight='bold', color='darkblue')

plt.suptitle('Comprehensive ADRA1B Gene Expression Rhythm Analysis (MetaCycle)',
             fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('img1.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 6. DENSITY PLOT: Showing distribution of curves (密度热力图) - 使用统一颜色
# ============================================================================
print("\nGenerating density plot of expression curves...")

plt.figure(figsize=(6, 6))

# Generate time points for density calculation
time_grid = np.linspace(0, 24, 241)  # Every 0.1 hour
density_matrix = np.zeros((len(top_tissues), len(time_grid)))

# Fill density matrix
for i, tissue in enumerate(top_tissues):
    # 获取结果
    result_row = results_df[results_df['Tissue'] == tissue].iloc[0]
    
    # 检查是否有成功的拟合
    if 'Success' in result_row:
        success = result_row['Success']
    elif 'fit_success' in result_row:
        success = result_row['fit_success']
    else:
        success = False
    
    if success:
        # 获取参数
        if 'Period' in result_row:
            period = result_row['Period']
            mesor = result_row['Mesor']
            amplitude = result_row['Amplitude']
            acrophase_zt = result_row['Acrophase_ZT']
        elif 'full_result' in result_row and result_row['full_result'] is not None:
            period = result_row['full_result'].get('period', 24)
            mesor = result_row['full_result'].get('mesor', 0)
            amplitude = result_row['full_result'].get('amplitude', 0)
            acrophase_zt = result_row['full_result'].get('acrophase', 12)
        else:
            period = 24
            mesor = result_row.get('mean_expression', 0)
            amplitude = result_row.get('amplitude', 0)
            acrophase_zt = result_row.get('acrophase_zt', 12)
        
        # Calculate fitted values at each time point
        for j, t_actual in enumerate(time_grid):
            # Convert actual time to ZT
            t_zt = (t_actual - 6) % 24
            fitted_value = mesor + amplitude * np.cos(2 * np.pi * (t_zt - acrophase_zt) / period)
            
            # Normalize
            if amplitude > 0:
                # Find max and min for normalization
                zt_test = np.linspace(0, 24, 241)
                fitted_test = mesor + amplitude * np.cos(2 * np.pi * (zt_test - acrophase_zt) / period)
                if np.max(fitted_test) > np.min(fitted_test):
                    norm_value = (fitted_value - np.min(fitted_test)) / (np.max(fitted_test) - np.min(fitted_test))
                else:
                    norm_value = 0.5
            else:
                norm_value = 0.5
            
            density_matrix[i, j] = norm_value
    else:
        # 如果拟合不成功，填充0.5（中间值）
        density_matrix[i, :] = 0.5

# 创建热力图
try:
    heatmap = plt.imshow(density_matrix, aspect='auto', cmap='viridis', 
                         extent=[0, 24, 0, len(top_tissues)], origin='lower',
                         interpolation='bilinear')
    
    # 添加colorbar
    cbar = plt.colorbar(heatmap, label='Normalized Expression Level', 
                        shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=9)
    
    # Set y-ticks - 确保使用正确的顺序
    plt.yticks(np.arange(len(top_tissues)) + 0.5, top_tissues, fontsize=9)
    
    # Set x-ticks
    plt.xticks([0, 6, 12, 18, 24], ['0:00', '6:00', '12:00', '18:00', '24:00'], 
               rotation=45, fontsize=10)
    
    # 添加昼夜背景色
    ax = plt.gca()
    ylim = ax.get_ylim()
    
    ax.axvspan(6, 18, alpha=0.15, color='yellow', zorder=0)
    ax.axvspan(0, 6, alpha=0.15, color='lightblue', zorder=0)
    ax.axvspan(18, 24, alpha=0.15, color='lightblue', zorder=0)
    
    ax.set_ylim(ylim)
    
    # 添加标签
    plt.xlabel('Time of Day', fontsize=12, fontweight='bold')
    plt.ylabel('Tissue', fontsize=12, fontweight='bold')
    plt.title(f'Expression Rhythm Heatmap: Top {top_n} Most Rhythmic Tissues\nMetaCycle Analysis', 
              fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('img5.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Density heatmap saved as img5.png")
    
except Exception as e:
    print(f"Error creating heatmap: {e}")

# ============================================================================
# Save results (添加MetaCycle方法信息)
# ============================================================================
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
            'Method': fit_result.get('method_name', fit_result.get('method', 'MetaCycle')),
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
            'Method': 'None',
            'Mesor': np.nan,
            'Amplitude_fit': np.nan,
            'Acrophase_ZT': np.nan,
            'Acrophase_Actual': np.nan,
            'Acrophase_Actual_Time': 'N/A',
            'Fit_Success': False
        })

detailed_df = pd.DataFrame(detailed_results)
detailed_df.to_csv('ADRA1B_metacycle_fit_results_final.csv', index=False)
print(f"\nDetailed fitting results saved to: ADRA1B_metacycle_fit_results_final.csv")

# 打印颜色映射信息供参考
print("\n" + "="*80)
print("COLOR MAPPING SUMMARY:")
print("="*80)
for i, tissue in enumerate(top_tissues):
    color = tissue_colors[tissue]
    # 将RGBA颜色转换为十六进制
    hex_color = '#{:02x}{:02x}{:02x}'.format(int(color[0]*255), int(color[1]*255), int(color[2]*255))
    print(f"{tissue:<20}: RGB({int(color[0]*255):3d}, {int(color[1]*255):3d}, {int(color[2]*255):3d}) Hex: {hex_color}")

print("\nMetaCycle Analysis complete!")
print("=" * 80)
print(f"Top rhythmic tissue: {best_tissue}")
print(f"Method: {best_fit.get('method_name', best_fit.get('method', 'MetaCycle'))}")
print(f"Rhythm Score: {results_df.iloc[0]['rhythm_score']:.3f}")
print(f"R²: {best_fit['r_squared']:.3f}")
print(f"p-value: {best_fit['p_value']:.3f}")
print(f"Peak time: {int(best_fit['params']['acrophase_actual']):02d}:00")
print("\nImages saved:")
print("imgneed1.png - Individual tissue curve fitting plots")
print("imgneed3.png - Normalized overlay plot")
print("img1.png - Comprehensive multi-curve plot")
print("img2.png - Raw overlay plot")
print("img4.png - Detailed example plot")
print("img5.png - Density heatmap plot")