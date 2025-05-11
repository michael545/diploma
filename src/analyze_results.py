import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind


results_dir = '../results'
figures_dir = '../analysis_results'

# Clean up existing analysis results
if os.path.exists(figures_dir):
    shutil.rmtree(figures_dir)

# Create fresh directory
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
        window = filename.split('_')[2].split('.')[0]
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
    
    # Calculate basic error metrics
    errors = df['actual'] - df['forecast']
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    mape = np.mean(np.abs(errors / df['actual'])) * 100 if (df['actual'] != 0).all() else np.nan
    
    # Calculate error volatility
    error_volatility = np.std(errors)
    
    # Calculate directional accuracy
    df['actual_change'] = df['actual'].diff()
    df['predicted_change'] = df['forecast'] - df['actual'].shift(1)
    
    correct_direction = ((df['actual_change'] > 0) & (df['predicted_change'] > 0)) | \
                       ((df['actual_change'] < 0) & (df['predicted_change'] < 0))
    
    directional_accuracy = np.mean(correct_direction[1:]) * 100
    
    metrics_list.append({
        'model': model,
        'interval': interval,
        'ticker': ticker,
        'window': window,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'error_volatility': error_volatility,
        'directional_accuracy': directional_accuracy
    })

metrics_df = pd.DataFrame(metrics_list)

# Function to create histograms for metrics
def create_metric_histogram(data, metric, save_dir):
    plt.figure(figsize=(15, 10))
    
    for idx, (name, group) in enumerate(data.groupby(['model', 'interval'])):
        model, interval = name
        plt.subplot(2, 3, idx + 1)
        sns.histplot(data=group, x=metric, kde=True)
        # Remove the median line by setting spines color to none
        plt.gca().spines['bottom'].set_color('lightgray')
        plt.gca().spines['left'].set_color('lightgray')
        plt.gca().spines['top'].set_color('none')
        plt.gca().spines['right'].set_color('none')
        plt.title(f'{model.capitalize()} - {interval}\n{metric} Distribution')
        plt.xlabel(metric)
        plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{metric.lower()}_distributions.png'))
    plt.close()

# Create histograms directory
histograms_dir = os.path.join(figures_dir, 'histograms')
os.makedirs(histograms_dir, exist_ok=True)

# Generate histograms for MAPE only
create_metric_histogram(metrics_df, 'MAPE', histograms_dir)

# Create combined distribution plot for MAPE only
plt.figure(figsize=(10, 5))
sns.boxplot(data=metrics_df, x='interval', y='MAPE', hue='model',
            whiskerprops=dict(color="lightgray"),  # Keep whisker lines
            medianprops=dict(color="none"),  # Remove median line
            boxprops=dict(alpha=0.5),  # Make boxes slightly transparent
            capprops=dict(color="none"),  # Remove cap lines
            showfliers=False)  # Remove outlier points
plt.title('MAPE by Model and Interval')
plt.ylabel('Mean Absolute Percentage Error (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(histograms_dir, 'combined_distributions.png'))
plt.close()

# Create comparisons directory
comparisons_dir = os.path.join(figures_dir, 'comparisons')
os.makedirs(comparisons_dir, exist_ok=True)

# Create ticker comparisons for each interval
for interval in ['5M', '15M', '1H']:
    # Filter out VKTX for 1H interval
    interval_metrics = metrics_df[metrics_df['interval'] == interval]
    if interval == '1H':
        interval_metrics = interval_metrics[interval_metrics['ticker'] != 'VKTX']

    # Create plots for each metric
    for metric, metric_label in [
        ('MAPE', 'Mean Absolute Percentage Error (%)'),
        ('error_volatility', 'Error Volatility'),
        ('directional_accuracy', 'Directional Accuracy (%)')
    ]:
        plt.figure(figsize=(15, 8))
        # Modified barplot to remove error bars
        sns.barplot(
            data=interval_metrics, 
            x='ticker', 
            y=metric, 
            hue='model',
            ci=None,  # Remove error bars completely
            saturation=0.7  # Slightly reduce color saturation for better visibility
        )
        
        plt.title(f'{metric_label} Comparison - {interval} Interval')
        plt.xlabel('Ticker')
        plt.ylabel(metric_label)
        plt.xticks(rotation=45)
        plt.legend(title='Model')
        plt.tight_layout()
        plt.savefig(os.path.join(comparisons_dir, f'{metric.lower()}_comparison_{interval}.png'))
        plt.close()

        # Handle duplicates by aggregating before statistical test
        agg_data = interval_metrics.groupby(['ticker', 'model'])[metric].mean().reset_index()
        stats_data = agg_data.pivot(index='ticker', columns='model', values=metric)
        t_stat, p_value = ttest_ind(
            stats_data['chronos'].dropna(),
            stats_data['timesfm'].dropna(),
            equal_var=False
        )
        print(f'\nStatistical test for {metric} ({interval}):')
        print(f'T-statistic: {t_stat:.4f}')
        print(f'P-value: {p_value:.4f}')

# Distribution of MAPE per window for 1H interval
one_hour_windows = metrics_df[metrics_df['interval'] == '1H']
one_hour_windows = one_hour_windows[one_hour_windows['ticker'] != 'VKTX']  # Filter out VKTX
plt.figure(figsize=(8, 6))
sns.boxplot(data=one_hour_windows, x='model', y='MAPE', 
            showfliers=False,  # Remove outlier points
            whiskerprops=dict(color="lightgray"),  # Keep whisker lines
            medianprops=dict(color="none"),  # Remove median line
            boxprops=dict(alpha=0.5),  # Make boxes slightly transparent
            capprops=dict(color="none"),  # Remove cap lines
            width=0.5)  # Adjust width of boxes
plt.title('Distribution of MAPE (%) per Window for 1H Interval by Model')
plt.ylabel('Mean Absolute Percentage Error (%)')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'model_comparison_1h.png'))
plt.close()