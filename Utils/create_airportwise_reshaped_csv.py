import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from data_formatter import parse_custom_datetime

def log_progress(message):
    print(message)

def impute_column(series):
    """Impute a single column using forward fill then backward fill"""
    return series.ffill().bfill()

RAW_PATH = '../datastep2.csv'
RESHAPED_PATH = 'data_by_airfield_reshaped.csv'
IMPUTED_PATH = 'data_by_airfield_reshaped_imputed.csv'
N_LAGS = 24

# Use only specified airports
REGRESSOR_AIRPORTS = ['KRZL', 'KMWC', 'KIGQ', 'KPNT', 'KOXI', 'KIKK', 'KDKB', 'KPPO', 'KBUU', 'KMDW']
TARGET_AIRPORT = 'KORD'
ALL_AIRPORTS = [TARGET_AIRPORT] + REGRESSOR_AIRPORTS

log_progress('Loading raw data...')
df = pd.read_csv(RAW_PATH)
log_progress(f'Raw data shape: {df.shape}')

# Filter for our selected airports
df = df[df['id'].isin(ALL_AIRPORTS)].copy()
log_progress(f'After filtering for {len(ALL_AIRPORTS)} airports: {df.shape}')
log_progress(f'Airports in dataset: {ALL_AIRPORTS}')
log_progress(f'Number of regressor airports: {len(REGRESSOR_AIRPORTS)}')

# Sort by datetime
df['datetime'] = parse_custom_datetime(df['datetime'])
df = df.sort_values('datetime')
log_progress(f'Unique datetimes: {df["datetime"].nunique()}')

# Impute raw data
log_progress('Imputing raw data...')
for col in tqdm(df.columns, desc='Imputing raw data'):
    if col != 'datetime' and col != 'id':
        df[col] = impute_column(df[col])

# Create pivot table
log_progress('Creating airport-specific columns...')
pivot_df = df.pivot(index='datetime', columns='id')
log_progress(f'Pivot shape before column renaming: {pivot_df.shape}')

# Rename columns
pivot_df.columns = [f'{feature}_{airport}' for feature, airport in pivot_df.columns]
log_progress(f'Pivot shape after column renaming: {pivot_df.shape}')
log_progress(f'Number of features per airport: {len(pivot_df.columns) // len(ALL_AIRPORTS)}')

# Delta features
delta_features = {}
for airport in tqdm(REGRESSOR_AIRPORTS, desc='Delta features (airports)'):
    temp_col = f'temp_{airport}'
    if temp_col in pivot_df.columns:
        delta = pivot_df[temp_col].diff(2).shift(1)
        delta = impute_column(delta)
        delta_features[f'{airport}_temp_2h_delta'] = delta
    wind_dir_col = f'winddirection_{airport}'
    if wind_dir_col in pivot_df.columns:
        delta = pivot_df[wind_dir_col].diff(2).shift(1)
        delta.loc[delta > 180] -= 360
        delta.loc[delta < -180] += 360
        delta = impute_column(delta)
        delta_features[f'{airport}_wind_dir_2h_delta'] = delta
    wind_vel_col = f'windspeed_{airport}'
    if wind_vel_col in pivot_df.columns:
        delta = pivot_df[wind_vel_col].diff(2).shift(1)
        delta = impute_column(delta)
        delta_features[f'{airport}_wind_vel_2h_delta'] = delta
delta_df = pd.DataFrame(delta_features, index=pivot_df.index)
log_progress(f'Delta features shape: {delta_df.shape}')

# Lagged features
lagged_features = {}
base_features = ['temp', 'windspeed', 'winddirection', 'humidity', 'dew', 'sealevel', 'visibility', 'mincloud', 'maxcloud']
for airport in tqdm(REGRESSOR_AIRPORTS, desc='Lagged features (airports)'):
    for feature in tqdm(base_features, desc=f'Lagged features ({airport})', leave=False):
        feature_col = f'{feature}_{airport}'
        if feature_col in pivot_df.columns:
            for lag in range(1, N_LAGS + 1):
                lagged = pivot_df[feature_col].shift(lag)
                lagged = impute_column(lagged)
                lagged_features[f'{airport}_{feature}_lag_{lag}'] = lagged
    # Special lags for temperature
    temp_col = f'temp_{airport}'
    if temp_col in pivot_df.columns:
        # 48-hour lag
        lagged = pivot_df[temp_col].shift(48)
        lagged = impute_column(lagged)
        lagged_features[f'{airport}_temp_lag_48'] = lagged
        # 120-hour lag
        lagged = pivot_df[temp_col].shift(120)
        lagged = impute_column(lagged)
        lagged_features[f'{airport}_temp_lag_120'] = lagged
        # 5-day rolling average
        rolling = pivot_df[temp_col].shift(1).rolling(window=120, min_periods=1).mean()
        rolling = impute_column(rolling)
        lagged_features[f'{airport}_temp_rolling_5d_avg'] = rolling
lagged_df = pd.DataFrame(lagged_features, index=pivot_df.index)
log_progress(f'Lagged features shape: {lagged_df.shape}')

# Combine all features
log_progress('Combining features...')
df_lagged = pd.concat([pivot_df, delta_df, lagged_df], axis=1)
log_progress(f'Final combined shape: {df_lagged.shape}')

# Final imputation check
log_progress('Performing final imputation check...')
missing_before = df_lagged.isnull().sum().sum()
if missing_before > 0:
    log_progress(f'Found {missing_before} missing values, imputing...')
    for col in tqdm(df_lagged.columns, desc='Final imputation'):
        df_lagged[col] = impute_column(df_lagged[col])
    missing_after = df_lagged.isnull().sum().sum()
    log_progress(f'Missing values after imputation: {missing_after}')

# Save the fully imputed data
log_progress(f'Saving imputed data to {IMPUTED_PATH} ...')
df_lagged.reset_index().to_csv(IMPUTED_PATH, index=False)
log_progress('Save complete.')

log_progress(f'Final feature count: {len(df_lagged.columns)}')
log_progress(f'Final row count: {len(df_lagged)}')
log_progress('Data preparation complete!')
