import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

results_dir = '../results'
figures_dir = '../analysis_results'
data_export_dir = os.path.join(figures_dir, 'data_exports') # Mapa za izvoz podatkov

# Clean up existing analysis results
if os.path.exists(figures_dir):
    shutil.rmtree(figures_dir)

# Create fresh directories
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(data_export_dir, exist_ok=True)

# Function to get all CSV files
def get_all_csv(root_dir):
    all_csv_files = [] # Preimenovano, da se ne prekriva z globalno spremenljivko
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.csv'):
                all_csv_files.append(os.path.join(dirpath, filename))
    return all_csv_files

# Function to extract model, interval, ticker, and window from CSV path
def extract_info(csv_path):
    dir_name = os.path.basename(os.path.dirname(csv_path))
    parts = dir_name.split('_')
    if len(parts) == 3:
        model, interval, ticker = parts
        filename = os.path.basename(csv_path)
        try:
            # Poskusimo prebrati okno kot celo število, če je to pričakovano
            window_str = filename.split('_')[2].split('.')[0]
            window = int(window_str)
            return model, interval, ticker, window
        except (IndexError, ValueError):
            # Če pride do napake pri razčlenjevanju ali pretvorbi, vrnemo None za okno
            print(f"Opozorilo: Ni mogoče razčleniti okna iz imena datoteke: {filename}")
            return model, interval, ticker, None 
    return None, None, None, None

# Get all CSV files
all_csv_files_list = get_all_csv(results_dir) # Preimenovano

# Calculate metrics per window
metrics_list = []
for csv_path in all_csv_files_list: # Uporabimo preimenovano spremenljivko
    model, interval, ticker, window = extract_info(csv_path)
    if model not in ['chronos', 'timesfm'] or window is None: # Dodan preizkus za window
        continue
    
    try:
        df = pd.read_csv(csv_path)
        if df.empty or 'actual' not in df.columns or 'forecast' not in df.columns:
            print(f"Opozorilo: Datoteka {csv_path} je prazna ali nima potrebnih stolpcev.")
            continue
        if df[['actual', 'forecast']].isnull().any().any():
            print(f"Opozorilo: Manjkajoče vrednosti v 'actual' ali 'forecast' v datoteki {csv_path}. Preskakujem.")
            continue

    except pd.errors.EmptyDataError:
        print(f"Opozorilo: Datoteka {csv_path} je prazna. Preskakujem.")
        continue
    
    # Calculate basic error metrics
    errors = df['actual'] - df['forecast']
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    
    # MAPE - izboljšano ravnanje z ničlami in zelo majhnimi vrednostmi
    actual_abs = np.abs(df['actual'])
    mape_numerator = np.abs(errors)
    # Izključi vrstice, kjer je dejanska vrednost zelo blizu ničle, da se prepreči deljenje z ničlo ali ekstremne vrednosti
    valid_for_mape = actual_abs > 1e-8 
    if np.sum(valid_for_mape) > 0:
        mape = np.mean(mape_numerator[valid_for_mape] / actual_abs[valid_for_mape]) * 100
    else:
        mape = np.nan

    # sMAPE (Symmetric Mean Absolute Percentage Error)
    smape_numerator = 2 * np.abs(errors)
    smape_denominator = actual_abs + np.abs(df['forecast'])
    valid_for_smape = smape_denominator > 1e-8
    if np.sum(valid_for_smape) > 0:
        smape = np.mean(smape_numerator[valid_for_smape] / smape_denominator[valid_for_smape]) * 100
    else:
        smape = np.nan
        
    # Calculate error volatility
    error_volatility = np.std(errors)
    
    # Calculate directional accuracy
    df['actual_change'] = df['actual'].diff()
    df['predicted_change'] = df['forecast'] - df['actual'].shift(1) # Napoved spremembe glede na zadnjo dejansko vrednost
    
    # Za prvo napoved v oknu ni mogoče izračunati spremembe, zato jo izpustimo
    actual_change_valid = df['actual_change'].iloc[1:]
    predicted_change_valid = df['predicted_change'].iloc[1:]

    if not actual_change_valid.empty:
        correct_direction = ((actual_change_valid > 0) & (predicted_change_valid > 0)) | \
                            ((actual_change_valid < 0) & (predicted_change_valid < 0)) | \
                            ((actual_change_valid == 0) & (predicted_change_valid == 0)) # Upoštevamo tudi, če ni spremembe
        directional_accuracy = np.mean(correct_direction) * 100
    else:
        directional_accuracy = np.nan
        
    metrics_list.append({
        'model': model,
        'interval': interval,
        'ticker': ticker,
        'window': window,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'sMAPE': smape, # Dodana metrika
        'error_volatility': error_volatility,
        'directional_accuracy': directional_accuracy
    })

metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv(os.path.join(data_export_dir, 'all_metrics_per_window.csv'), index=False)
print("DataFrame z vsemi metrikami shranjen v 'all_metrics_per_window.csv'")

# --- VIZUALIZACIJE ---

# Funkcija za ustvarjanje histogramov za metrike
def create_metric_histogram(data, metric, save_dir, file_prefix=''):
    plt.figure(figsize=(18, 12)) # Povečana velikost za boljšo berljivost
    
    grouped_data = data.groupby(['model', 'interval'])
    num_groups = len(grouped_data)
    
    # Dinamična postavitev subplotov (npr. 2 stolpca)
    cols = 2
    rows = (num_groups + cols - 1) // cols

    for idx, (name, group) in enumerate(grouped_data):
        model, interval = name
        plt.subplot(rows, cols, idx + 1)
        sns.histplot(data=group, x=metric, kde=True, bins=20) # Dodano bins za boljšo kontrolo
        plt.gca().spines['bottom'].set_color('lightgray')
        plt.gca().spines['left'].set_color('lightgray')
        plt.gca().spines['top'].set_color('none')
        plt.gca().spines['right'].set_color('none')
        plt.title(f'{model.capitalize()} - {interval}\nPorazdelitev {metric}')
        plt.xlabel(metric)
        plt.ylabel('Število opazovanj (oken)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{file_prefix}{metric.lower()}_distributions.png'))
    plt.close()

# Kreiranje map za slike
histograms_dir = os.path.join(figures_dir, 'histograms')
os.makedirs(histograms_dir, exist_ok=True)
comparisons_dir = os.path.join(figures_dir, 'comparisons')
os.makedirs(comparisons_dir, exist_ok=True)
forecast_plots_dir = os.path.join(figures_dir, 'forecast_examples') # Mapa za primere napovedi
os.makedirs(forecast_plots_dir, exist_ok=True)

# Generiranje histogramov za MAPE in sMAPE
metrics_to_plot_hist = ['MAPE', 'sMAPE', 'directional_accuracy']
for metric in metrics_to_plot_hist:
    create_metric_histogram(metrics_df, metric, histograms_dir)
    print(f"Histogram za {metric} ustvarjen.")

# Izboljšan združen škatlasti diagram (boxplot) z medianami
plt.figure(figsize=(12, 7)) # Prilagojena velikost
sns.boxplot(data=metrics_df, x='interval', y='MAPE', hue='model',
            whiskerprops=dict(color="gray", linestyle='--'), # Spremenjen stil brkov
            medianprops=dict(color="black", linewidth=1.5), # Prikazane mediane
            boxprops=dict(alpha=0.7), 
            capprops=dict(color="gray"), 
            showfliers=False) 
plt.title('Primerjava MAPE po Modelu in Intervalu')
plt.ylabel('Povprečna Absolutna Odstotna Napaka (%)')
plt.xlabel('Časovni Interval')
plt.xticks(rotation=30, ha='right')
plt.legend(title='Model')
plt.tight_layout()
plt.savefig(os.path.join(histograms_dir, 'combined_mape_boxplot.png'))
plt.close()
print("Združen škatlasti diagram za MAPE ustvarjen.")

# Primer scatter plota za primerjavo MAPE (agregirano po tickerju in intervalu)
# Najprej agregiramo podatke
agg_metrics_scatter = metrics_df.groupby(['model', 'interval', 'ticker'])['MAPE'].mean().unstack(level=0)
agg_metrics_scatter.columns = [f'MAPE_{col}' for col in agg_metrics_scatter.columns]
agg_metrics_scatter.reset_index(inplace=True)

if 'MAPE_chronos' in agg_metrics_scatter.columns and 'MAPE_timesfm' in agg_metrics_scatter.columns:
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=agg_metrics_scatter, x='MAPE_chronos', y='MAPE_timesfm', hue='interval', style='ticker', s=100, alpha=0.7)
    min_val = min(agg_metrics_scatter['MAPE_chronos'].min(), agg_metrics_scatter['MAPE_timesfm'].min())
    max_val = max(agg_metrics_scatter['MAPE_chronos'].max(), agg_metrics_scatter['MAPE_timesfm'].max())
    if pd.notna(min_val) and pd.notna(max_val): # Preverimo, če vrednosti niso NaN
      plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1, label='Linija enakosti')
    plt.xlabel('MAPE Chronos (%)')
    plt.ylabel('MAPE TimesFM (%)')
    plt.title('Neposredna Primerjava MAPE: Chronos vs. TimesFM (po Delnicah in Intervalih)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Prilagoditev za legendo
    plt.savefig(os.path.join(comparisons_dir, 'mape_scatter_comparison_detailed.png'))
    plt.close()
    print("Scatter plot za MAPE ustvarjen.")
else:
    print("Opozorilo: Manjkajo stolpci za scatter plot MAPE (MAPE_chronos ali MAPE_timesfm).")


# Primerjalni stolpični diagrami in t-testi
ttest_results_list = []
for interval_val in ['5M', '15M', '1H']: # Preimenovano iz 'interval'
    interval_metrics_df = metrics_df[metrics_df['interval'] == interval_val] # Preimenovano iz 'interval_metrics'
    if interval_val == '1H':
        interval_metrics_df = interval_metrics_df[interval_metrics_df['ticker'] != 'VKTX']

    if interval_metrics_df.empty:
        print(f"Ni podatkov za interval {interval_val} po filtriranju. Preskakujem primerjalne diagrame in t-teste.")
        continue

    metrics_for_barplot = [
        ('MAPE', 'Povprečna Absolutna Odstotna Napaka (%)'),
        ('sMAPE', 'Simetrična Povprečna Absolutna Odstotna Napaka (%)'), # Dodana sMAPE
        ('error_volatility', 'Volatilnost Napake'),
        ('directional_accuracy', 'Smerna Natančnost (%)')
    ]

    for metric_col, metric_label_str in metrics_for_barplot: # Preimenovano
        if metric_col not in interval_metrics_df.columns:
            print(f"Metrika {metric_col} ni na voljo za interval {interval_val}. Preskakujem.")
            continue

        plt.figure(figsize=(16, 9)) # Povečana velikost
        sns.barplot(
            data=interval_metrics_df, 
            x='ticker', 
            y=metric_col, 
            hue='model',
            estimator=np.mean, # Eksplicitno navedemo povprečje
            errorbar=('ci', 95), # Dodani intervali zaupanja namesto ci=None
            saturation=0.75,
            capsize=.05 # Dodane kapice na intervale zaupanja
        )
        
        plt.title(f'Primerjava Metrike: {metric_label_str} - Interval {interval_val}')
        plt.xlabel('Simbol Delnice')
        plt.ylabel(metric_label_str)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Model', loc='upper right')
        plt.grid(axis='y', linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(comparisons_dir, f'{metric_col.lower()}_comparison_{interval_val}.png'))
        plt.close()
        print(f"Stolpčni diagram za {metric_col} ({interval_val}) ustvarjen.")

        # T-test (ostaja enak, na agregiranih podatkih po tickerju)
        agg_data = interval_metrics_df.groupby(['ticker', 'model'])[metric_col].mean().reset_index()
        stats_data = agg_data.pivot(index='ticker', columns='model', values=metric_col)
        
        if 'chronos' in stats_data.columns and 'timesfm' in stats_data.columns:
            chronos_metric_values = stats_data['chronos'].dropna()
            timesfm_metric_values = stats_data['timesfm'].dropna()

            if len(chronos_metric_values) >= 2 and len(timesfm_metric_values) >= 2: # t-test zahteva vsaj 2 opazovanji
                mean_chronos = chronos_metric_values.mean()
                mean_timesfm = timesfm_metric_values.mean()
                
                t_stat, p_value = ttest_ind(
                    chronos_metric_values,
                    timesfm_metric_values,
                    equal_var=False, # Welchov t-test
                    nan_policy='omit' 
                )
                print(f'\nStatistični test (Welchov t-test) za {metric_col} ({interval_val}):')
                print(f'  Povprečje Chronos: {mean_chronos:.4f}')
                print(f'  Povprečje TimesFM: {mean_timesfm:.4f}')
                print(f'  T-statistika: {t_stat:.4f}')
                print(f'  P-vrednost: {p_value:.4f}')
                ttest_results_list.append({
                    'metric': metric_col, 
                    'interval': interval_val, 
                    'mean_chronos': mean_chronos,
                    'mean_timesfm': mean_timesfm,
                    't_statistic': t_stat, 
                    'p_value': p_value
                })
            else:
                print(f"Opozorilo: Ni dovolj podatkov za t-test za {metric_col} ({interval_val}) po odstranitvi NaN vrednosti.")
        else:
            print(f"Opozorilo: Manjkajo podatki za 'chronos' ali 'timesfm' za {metric_col} ({interval_val}) pri t-testu.")

# Shranjevanje rezultatov t-testov
if ttest_results_list:
    ttest_df = pd.DataFrame(ttest_results_list)
    ttest_df.to_csv(os.path.join(data_export_dir, 't_test_summary_results.csv'), index=False)
    print("\nRezultati t-testov shranjeni v 't_test_summary_results.csv'")


# Izboljšan boxplot za 1H interval (že obstoječ v vaši kodi, tukaj z medianami)
one_hour_windows = metrics_df[metrics_df['interval'] == '1H']
if 'VKTX' in one_hour_windows['ticker'].unique(): # Preverimo, ali je VKTX sploh prisoten
    one_hour_windows = one_hour_windows[one_hour_windows['ticker'] != 'VKTX']

if not one_hour_windows.empty:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=one_hour_windows, x='model', y='MAPE', 
                showfliers=False, 
                whiskerprops=dict(color="gray", linestyle='--'), 
                medianprops=dict(color="black", linewidth=1.5), # Prikazane mediane
                boxprops=dict(alpha=0.7), 
                capprops=dict(color="gray"), 
                width=0.5) 
    plt.title('Porazdelitev MAPE (\%) po Oknh za 1H Interval (po Modelih)')
    plt.ylabel('Povprečna Absolutna Odstotna Napaka (%)')
    plt.xlabel('Model')
    plt.tight_layout()
    plt.savefig(os.path.join(histograms_dir, 'model_comparison_mape_1h_boxplot.png')) # Malce drugačno ime, da se ne prepiše
    plt.close()
    print("Škatlasti diagram za MAPE (1H interval) ustvarjen.")
else:
    print("Ni podatkov za 1H interval za generiranje boxplota MAPE po modelih.")


# --- Primer funkcije za risanje posameznih napovedi ---
# To funkcijo bi klicali znotraj zanke ali pa ji posredovali specifične poti do CSV datotek.
# Za demonstracijo jo definiramo tukaj, vi pa jo lahko vključite v svoj potek dela.

def plot_forecast_example(actual_values, forecast_values_chronos, forecast_values_timesfm, 
                          ticker, interval, window_id, save_dir):
    plt.figure(figsize=(15, 7))
    plt.plot(actual_values.index, actual_values, label='Dejanske vrednosti', color='black', linewidth=1.5)
    if forecast_values_chronos is not None:
        plt.plot(forecast_values_chronos.index, forecast_values_chronos, label='Napoved Chronos', linestyle='--', color='blue', alpha=0.8)
    if forecast_values_timesfm is not None:
        plt.plot(forecast_values_timesfm.index, forecast_values_timesfm, label='Napoved TimesFM', linestyle=':', color='red', alpha=0.8)
    
    plt.title(f'Primer Napovedi: {ticker} ({interval}, Okno {window_id})')
    plt.xlabel('Časovni korak v napovednem horizontu')
    plt.ylabel('Cena')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    filename = f"forecast_example_{ticker}_{interval}_win{window_id}.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
    print(f"Primer napovedi shranjen: {filename}")

# Primer klica funkcije plot_forecast_example (potrebno prilagoditi, da najdete ustrezne CSV datoteke):
# Izberite nekaj reprezentativnih primerov za prikaz.
# Na primer, poženite analizo in poglejte, kje so največje razlike ali zanimivi vzorci.
# Nato ročno določite poti do teh CSV datotek.

# example_ticker = 'NVDA'
# example_interval = '15M'
# example_window = 10 # Neki primer okna

# path_chronos_example = f"../results/chronos_{example_interval}_{example_ticker}/forecast_window_{example_window}_len_128.csv"
# path_timesfm_example = f"../results/timesfm_{example_interval}_{example_ticker}/forecast_window_{example_window}_len_128.csv"

# try:
#     df_c_ex = pd.read_csv(path_chronos_example)
#     df_t_ex = pd.read_csv(path_timesfm_example) # Predpostavka, da imata enak 'actual' in indeks
    
#     # Za poravnavo indeksov, če se napoved začne od 0
#     # To je odvisno od tega, kako shranjujete indekse v CSV. Tukaj predpostavljam preprost zaporedni indeks.
#     horizon_len = len(df_c_ex['actual'])
#     plot_index = pd.RangeIndex(start=0, stop=horizon_len, step=1)

#     actual_series = pd.Series(df_c_ex['actual'].values, index=plot_index)
#     chronos_forecast_series = pd.Series(df_c_ex['forecast'].values, index=plot_index)
#     timesfm_forecast_series = pd.Series(df_t_ex['forecast'].values, index=plot_index)
    
#     plot_forecast_example(actual_series, chronos_forecast_series, timesfm_forecast_series,
#                           example_ticker, example_interval, example_window, forecast_plots_dir)
# except FileNotFoundError:
#     print(f"Primerne CSV datoteke za primer napovedi ({example_ticker}, {example_interval}, okno {example_window}) niso bile najdene.")
# except Exception as e:
#     print(f"Napaka pri generiranju primera napovedi: {e}")

print("\nAnaliza zaključena. Slike so shranjene v mapi:", figures_dir)
print("Podatkovni izvozi so shranjeni v mapi:", data_export_dir)