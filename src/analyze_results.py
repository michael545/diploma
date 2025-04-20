import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind  # Changed from ttest_rel to ttest_ind

# Define directories
results_dir = '../results'  # Assuming src is the current directory
figures_dir = '../analysis_results'
os.makedirs(figures_dir, exist_ok=True)

# Function to get all CSV files
def get_all_csv(root_dir):
    all_csv = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.csv'):
                all_csv.append(os.path.join(dirpath, filename))
    return all_csv

# Function to extract model, interval, ticker, and window from CSV path
def extract_info(csv_path):
    dir_name = os.path.basename(os.path.dirname(csv_path))
    parts = dir_name.split('_')
    if len(parts) == 3:
        model, interval, ticker = parts
        filename = os.path.basename(csv_path)
        window = filename.split('_')[2].split('.')[0]  # e.g., 'window_1.csv' -> '1'
        return model, interval, ticker, int(window)
    return None, None, None, None

# Get all CSV files and filter for relevant models
all_csv = get_all_csv(results_dir)
data_info = [extract_info(csv) for csv in all_csv]
data_info = [info for info in data_info if info[0] in ['chronos', 'timesfm']]

# Calculate metrics per window
metrics_list = []
for csv_path in all_csv:
    model, interval, ticker, window = extract_info(csv_path)
    if model not in ['chronos', 'timesfm']:
        continue
    df = pd.read_csv(csv_path)
    errors = df['actual'] - df['forecast']
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    mape = np.mean(np.abs(errors / df['actual'])) * 100 if (df['actual'] != 0).all() else np.nan
    metrics_list.append({
        'model': model,
        'interval': interval,
        'ticker': ticker,
        'window': window,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    })

metrics_df = pd.DataFrame(metrics_list)

# Function to create histograms for metrics
def create_metric_histogram(data, metric, save_dir):
    """Create histograms for a given metric across models and intervals"""
    plt.figure(figsize=(15, 10))
    
    for idx, (name, group) in enumerate(data.groupby(['model', 'interval'])):
        model, interval = name
        plt.subplot(2, 3, idx + 1)
        sns.histplot(data=group, x=metric, kde=True)
        plt.title(f'{model.capitalize()} - {interval}\n{metric} Distribution')
        plt.xlabel(metric)
        plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{metric.lower()}_distributions.png'))
    plt.close()

# Create histograms directory
histograms_dir = os.path.join(figures_dir, 'histograms')
os.makedirs(histograms_dir, exist_ok=True)

# Generate histograms for each metric
for metric in ['MAE', 'RMSE', 'MAPE']:
    create_metric_histogram(metrics_df, metric, histograms_dir)

# Create combined distribution plots
plt.figure(figsize=(15, 5))
for idx, metric in enumerate(['MAE', 'RMSE', 'MAPE']):
    plt.subplot(1, 3, idx + 1)
    sns.boxplot(data=metrics_df, x='interval', y=metric, hue='model')
    plt.title(f'{metric} by Model and Interval')
    plt.xticks(rotation=45)
    
plt.tight_layout()
plt.savefig(os.path.join(histograms_dir, 'combined_distributions.png'))
plt.close()

# Create NVDA-specific analysis directory
nvda_analysis_dir = os.path.join(figures_dir, 'nvda_analysis')
os.makedirs(nvda_analysis_dir, exist_ok=True)

# Analyze NVDA across all intervals and models
nvda_metrics = metrics_df[metrics_df['ticker'] == 'NVDA']

# 1. Create interval comparison plots for each model
for model in ['chronos', 'timesfm']:
    model_data = nvda_metrics[nvda_metrics['model'] == model]
    
    # Boxplot for MAE across intervals
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=model_data, x='interval', y='MAE')
    plt.title(f'{model.capitalize()} Model - MAE Distribution Across Intervals (NVDA)')
    plt.xlabel('Time Interval')
    plt.ylabel('Mean Absolute Error')
    plt.tight_layout()
    plt.savefig(os.path.join(nvda_analysis_dir, f'{model}_intervals_mae_dist.png'))
    plt.close()

# 2. Create model comparison plots for each interval
for interval in ['5M', '15M', '1H']:
    interval_data = nvda_metrics[nvda_metrics['interval'] == interval]
    
    # Performance comparison boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=interval_data, x='model', y='MAE')
    plt.title(f'Model Comparison - {interval} Interval (NVDA)')
    plt.xlabel('Model')
    plt.ylabel('Mean Absolute Error')
    plt.tight_layout()
    plt.savefig(os.path.join(nvda_analysis_dir, f'{interval}_model_comparison.png'))
    plt.close()

# 3. Create comprehensive metrics table for NVDA
nvda_summary = nvda_metrics.groupby(['model', 'interval']).agg({
    'MAE': ['mean', 'std'],
    'RMSE': ['mean', 'std'],
    'MAPE': ['mean', 'std']
}).round(4)

# Save NVDA summary to CSV
nvda_summary.to_csv(os.path.join(nvda_analysis_dir, 'nvda_metrics_summary.csv'))

# 4. Create heatmap of metrics across models and intervals
metrics_pivot = pd.pivot_table(
    nvda_metrics,
    values=['MAE', 'RMSE', 'MAPE'],
    index='interval',
    columns='model',
    aggfunc='mean'
)

# Plot heatmaps for each metric
for metric in ['MAE', 'RMSE', 'MAPE']:
    plt.figure(figsize=(8, 4))
    sns.heatmap(metrics_pivot.loc[:, metric], annot=True, fmt='.4f', cmap='YlOrRd')
    plt.title(f'NVDA {metric} Comparison - Models vs Intervals')
    plt.tight_layout()
    plt.savefig(os.path.join(nvda_analysis_dir, f'nvda_{metric.lower()}_heatmap.png'))
    plt.close()

# 5. Statistical tests for model comparison at each interval
statistical_tests = []
for interval in ['5M', '15M', '1H']:
    interval_data = nvda_metrics[nvda_metrics['interval'] == interval]
    chronos_data = interval_data[interval_data['model'] == 'chronos']['MAE'].values
    timesfm_data = interval_data[interval_data['model'] == 'timesfm']['MAE'].values
    
    if len(chronos_data) > 0 and len(timesfm_data) > 0:
        # Use independent t-test instead of paired test since we might have different numbers of samples
        t_stat, p_value = ttest_ind(chronos_data, timesfm_data, equal_var=False)  # Using Welch's t-test
        statistical_tests.append({
            'interval': interval,
            'chronos_samples': len(chronos_data),
            'timesfm_samples': len(timesfm_data),
            't_statistic': t_stat,
            'p_value': p_value
        })
    else:
        print(f"Warning: Missing data for interval {interval}")
        print(f"Chronos samples: {len(chronos_data)}, TimesFM samples: {len(timesfm_data)}")

# Save statistical test results with sample sizes
statistical_results = pd.DataFrame(statistical_tests)
if not statistical_results.empty:
    statistical_results.to_csv(os.path.join(nvda_analysis_dir, 'statistical_tests.csv'), index=False)
    
    # Print summary of findings with sample sizes
    print("\nNVDA Analysis Summary:")
    print("=" * 50)
    print("\nMean MAE by Model and Interval:")
    print(metrics_pivot['MAE'].round(4))
    print("\nStatistical Test Results:")
    print(statistical_results.round(4))
else:
    print("\nNo statistical tests could be performed due to insufficient data")

# Aggregate metrics by model, interval, and ticker
agg_metrics = metrics_df.groupby(['model', 'interval', 'ticker']).agg({
    'MAE': ['mean', 'std'],
    'RMSE': ['mean', 'std'],
    'MAPE': ['mean', 'std']
}).reset_index()

# Fix column names while preserving the original column structure
agg_metrics.columns = [f"{col[0]}{'_' + col[1] if col[1] else ''}" for col in agg_metrics.columns]

# Compare models for 1H interval
one_hour_metrics = agg_metrics[agg_metrics['interval'] == '1H']
common_tickers = np.intersect1d(
    one_hour_metrics[one_hour_metrics['model'] == 'chronos']['ticker'].unique(),
    one_hour_metrics[one_hour_metrics['model'] == 'timesfm']['ticker'].unique()
)
comparison_df = one_hour_metrics[one_hour_metrics['ticker'].isin(common_tickers)]

# Visualize MAPE comparison across tickers
plt.figure(figsize=(12, 6))
sns.barplot(data=comparison_df, x='ticker', y='MAPE_mean', hue='model', errorbar='sd')
plt.title('Mean MAPE (%) with Standard Deviation for 1H Interval by Ticker and Model')
plt.xlabel('Ticker')
plt.ylabel('Mean Absolute Percentage Error (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'ticker_comparison.png'))
plt.close()

# Statistical test on MAPE
paired_mape = comparison_df.pivot(index='ticker', columns='model', values='MAPE_mean')
t_stat, p_value = ttest_ind(paired_mape['chronos'], paired_mape['timesfm'], equal_var=False)
print(f'Independent t-test p-value for MAPE comparison: {p_value}')

# Create MAPE comparisons for each interval
for interval in ['5M', '15M', '1H']:
    interval_metrics = agg_metrics[agg_metrics['interval'] == interval]
    common_tickers = np.intersect1d(
        interval_metrics[interval_metrics['model'] == 'chronos']['ticker'].unique(),
        interval_metrics[interval_metrics['model'] == 'timesfm']['ticker'].unique()
    )
    comparison_df = interval_metrics[interval_metrics['ticker'].isin(common_tickers)]

    # Visualize MAPE comparison across tickers
    plt.figure(figsize=(12, 6))
    sns.barplot(data=comparison_df, x='ticker', y='MAPE_mean', hue='model', errorbar='sd')
    plt.title(f'Mean MAPE (%) with Standard Deviation for {interval} Interval by Ticker and Model')
    plt.xlabel('Ticker')
    plt.ylabel('Mean Absolute Percentage Error (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'ticker_comparison_{interval}.png'))
    plt.close()

    # Statistical test on MAPE for each interval
    paired_mape = comparison_df.pivot(index='ticker', columns='model', values='MAPE_mean')
    t_stat, p_value = ttest_ind(paired_mape['chronos'], paired_mape['timesfm'], equal_var=False)
    print(f'Independent t-test p-value for MAPE comparison ({interval}): {p_value}')

# Distribution of MAPE per window for 1H interval
one_hour_windows = metrics_df[metrics_df['interval'] == '1H']
plt.figure(figsize=(8, 6))
sns.boxplot(data=one_hour_windows, x='model', y='MAPE')
plt.title('Distribution of MAPE (%) per Window for 1H Interval by Model')
plt.ylabel('Mean Absolute Percentage Error (%)')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'model_comparison_1h.png'))
plt.close()

# Analyze chronos across intervals for NVDA
nvda_chronos = metrics_df[(metrics_df['ticker'] == 'NVDA') & (metrics_df['model'] == 'chronos')]
nvda_by_interval = nvda_chronos.groupby('interval').agg({
    'MAE': ['mean', 'std'],
    'RMSE': ['mean', 'std'],
    'MAPE': ['mean', 'std']
}).reset_index()

# Fix column names for NVDA analysis in the same way as before
nvda_by_interval.columns = [f"{col[0]}{'_' + col[1] if col[1] else ''}" for col in nvda_by_interval.columns]

plt.figure(figsize=(8, 6))
sns.barplot(data=nvda_by_interval, x='interval', y='MAE_mean', errorbar='sd')
plt.title('Mean MAE with Standard Deviation for Chronos on NVDA by Interval')
plt.xlabel('Interval')
plt.ylabel('Mean MAE')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'chronos_nvda_intervals.png'))
plt.close()