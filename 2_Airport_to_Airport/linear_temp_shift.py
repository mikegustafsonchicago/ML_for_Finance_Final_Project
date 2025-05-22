import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import sys
from tqdm import tqdm
import time
from datetime import datetime
sys.path.append('..')  # Add parent directory to path to import html_manager
from html_manager import HTMLManager
from Utils.data_formatter import parse_custom_datetime
from Utils.latex_utility import save_latex_file, latex_section, latex_subsection, df_to_latex_table, latex_list

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def log_progress(message, start_time=None):
    """Helper function to log progress with timing information"""
    if start_time:
        elapsed = time.time() - start_time
        logger.info(f"{message} (took {elapsed:.2f} seconds)")
    else:
        logger.info(message)
    return time.time()

def calculate_wind_direction_delta(df):
    """
    Calculate the change in wind direction between consecutive hours
    Handles the circular nature of wind direction (0-360 degrees)
    """
    df['wind_dir_delta'] = df['winddirection'].diff()
    df.loc[df['wind_dir_delta'] > 180, 'wind_dir_delta'] -= 360
    df.loc[df['wind_dir_delta'] < -180, 'wind_dir_delta'] += 360
    return df

def calculate_2hour_deltas(df, airport_code):
    """
    Calculate 2-hour changes for temperature, wind direction, and wind velocity for a specific airport
    """
    # Create a dictionary to store all deltas
    deltas = {}
    
    # Temperature delta
    deltas[f'{airport_code}_temp_2h_delta'] = df[f'{airport_code}_temp'].diff(2)
    
    # Wind direction delta (handling circular nature)
    deltas[f'{airport_code}_wind_dir_2h_delta'] = df[f'{airport_code}_winddirection'].diff(2)
    deltas[f'{airport_code}_wind_dir_2h_delta'].loc[deltas[f'{airport_code}_wind_dir_2h_delta'] > 180] -= 360
    deltas[f'{airport_code}_wind_dir_2h_delta'].loc[deltas[f'{airport_code}_wind_dir_2h_delta'] < -180] += 360
    
    # Wind velocity delta
    deltas[f'{airport_code}_wind_vel_2h_delta'] = df[f'{airport_code}_windspeed'].diff(2)
    
    return pd.DataFrame(deltas)

def load_and_prepare_data(file_path='../datastep2.csv', n_lags=24, chunk_size=50000):
    """
    Load and prepare data for time series regression
    """
    start_time = time.time()
    log_progress("Starting data preparation...")
    
    # Read data in chunks with progress bar
    log_progress("Loading data in chunks...")
    chunks = []
    chunk_start = time.time()
    for i, chunk in enumerate(tqdm(pd.read_csv(file_path, chunksize=chunk_size), desc="Loading data chunks")):
        chunks.append(chunk)
        if (i + 1) % 5 == 0:  # Log every 5 chunks
            log_progress(f"Loaded {i + 1} chunks", chunk_start)
            chunk_start = time.time()
    
    log_progress("Concatenating chunks...")
    df = pd.concat(chunks)
    log_progress(f"Loaded data shape: {df.shape}", start_time)
    log_progress(f"Columns in loaded data: {list(df.columns)}")
    
    # Get unique airport codes
    airport_codes = df['id'].unique()
    log_progress(f"Found {len(airport_codes)} airports: {airport_codes}")
    
    # Create a pivot table with datetime as index and airport-specific columns
    log_progress("Processing datetime and sorting data...")
    df['datetime'] = parse_custom_datetime(df['datetime'])
    df = df.sort_values('datetime')
    
    # Pivot the data to create airport-specific columns
    log_progress("Creating airport-specific columns...")
    pivot_start = time.time()
    
    # First, ensure we have a clean datetime index
    df = df.set_index(['datetime', 'id']).sort_index()
    
    # Create a list to store pivoted DataFrames for each feature
    pivoted_dfs = []
    features = ['temp', 'windspeed', 'winddirection', 'humidity', 'dew', 'sealevel', 'visibility', 'mincloud', 'maxcloud']
    
    for feature in tqdm(features, desc="Pivoting features"):
        if feature in df.columns:
            # Pivot each feature
            pivoted = df[feature].unstack(level='id')
            # Rename columns to include feature name
            pivoted.columns = [f'{col}_{feature}' for col in pivoted.columns]
            pivoted_dfs.append(pivoted)
    
    # Combine all pivoted features
    df = pd.concat(pivoted_dfs, axis=1)
    log_progress(f"Data shape after pivoting: {df.shape}")
    log_progress(f"Pivoting completed in {time.time() - pivot_start:.2f} seconds")
    
    # Save intermediate results
    log_progress("Saving intermediate results...")
    temp_file = Path('temp_airport_data.csv')
    df.to_csv(temp_file, index=True)
    log_progress(f"Saved intermediate data to {temp_file}")
    
    # Calculate 2-hour deltas for each airport
    log_progress("Calculating 2-hour deltas...")
    delta_dfs = []
    delta_start = time.time()
    
    for airport in tqdm(airport_codes, desc="Calculating deltas"):
        # Calculate deltas for each feature
        deltas = {}
        for feature in ['temp', 'winddirection', 'windspeed']:
            col = f'{airport}_{feature}'
            if col in df.columns:
                if feature == 'winddirection':
                    # Handle circular nature of wind direction
                    delta = df[col].diff(2)
                    delta.loc[delta > 180] -= 360
                    delta.loc[delta < -180] += 360
                else:
                    delta = df[col].diff(2)
                deltas[f'{airport}_{feature}_2h_delta'] = delta
        
        if deltas:
            delta_dfs.append(pd.DataFrame(deltas))
    
    # Combine all delta features
    if delta_dfs:
        delta_df = pd.concat(delta_dfs, axis=1)
        df = pd.concat([df, delta_df], axis=1)
    
    log_progress(f"Data shape after calculating deltas: {df.shape}")
    
    # Save intermediate results
    log_progress("Saving data with deltas...")
    df.to_csv(temp_file, index=True)
    log_progress(f"Updated intermediate data with deltas")

    # Process data in chunks to reduce memory usage
    chunk_size = 50000  # Reduced chunk size
    n_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size else 0)
    log_progress(f"Will process {n_chunks} chunks of size {chunk_size}")
    
    log_progress("Creating lagged features...")
    all_lagged_features = []
    chunk_start = time.time()
    
    for i in tqdm(range(n_chunks), desc="Processing chunks"):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        chunk = df.iloc[start_idx:end_idx].copy()
        
        # Create lagged features for this chunk
        chunk_lagged_features = {}
        
        # Process each airport's features
        for airport in airport_codes:
            # Create lagged features for each base feature
            for feature in features:
                col = f'{airport}_{feature}'
                if col in chunk.columns:
                    # Create hourly lags
                    for lag in range(1, n_lags + 1):
                        chunk_lagged_features[f'{col}_lag_{lag}'] = chunk[col].shift(lag)
            
            # Special handling for temperature
            temp_col = f'{airport}_temp'
            if temp_col in chunk.columns:
                # Add specific temperature lags
                chunk_lagged_features[f'{temp_col}_lag_48'] = chunk[temp_col].shift(48)
                chunk_lagged_features[f'{temp_col}_lag_120'] = chunk[temp_col].shift(120)
                chunk_lagged_features[f'{temp_col}_rolling_5d_avg'] = chunk[temp_col].shift(1).rolling(window=120, min_periods=1).mean()
        
        # Convert to DataFrame and append
        chunk_lagged_df = pd.DataFrame(chunk_lagged_features)
        all_lagged_features.append(chunk_lagged_df)
        
        # Save intermediate results
        if i % 5 == 0:  # Save every 5 chunks
            temp_lagged_file = Path(f'temp_lagged_features_{i}.csv')
            chunk_lagged_df.to_csv(temp_lagged_file, index=True)
            log_progress(f"Saved intermediate lagged features to {temp_lagged_file}")
        
        # Clear memory
        del chunk
        del chunk_lagged_features
        del chunk_lagged_df
        
        # Log progress
        if (i + 1) % 5 == 0:  # Log every 5 chunks
            log_progress(f"Processed {i + 1}/{n_chunks} chunks", chunk_start)
            chunk_start = time.time()
    
    # Combine all chunks
    log_progress("Combining all chunks...")
    lagged_df = pd.concat(all_lagged_features, axis=0)
    log_progress(f"Lagged DataFrame shape: {lagged_df.shape}")
    
    # Save final lagged features
    log_progress("Saving final lagged features...")
    lagged_df.to_csv('final_lagged_features.csv', index=True)
    log_progress("Saved final lagged features")
    
    # Combine with original data
    log_progress("Combining with original data...")
    df_lagged = pd.concat([df, lagged_df], axis=1)
    log_progress(f"Shape after combining lagged features: {df_lagged.shape}")
    
    # Drop rows with NaN values
    log_progress("Dropping NaN values...")
    df_lagged = df_lagged.dropna().reset_index(drop=True)
    log_progress(f"Shape after dropping NaNs: {df_lagged.shape}")

    # Prepare features and targets
    log_progress("Preparing features and targets...")
    feature_cols = [col for col in df_lagged.columns if (
        col.endswith(tuple([f'_lag_{i}' for i in range(1, n_lags + 1)]) + 
        ('_lag_48', '_lag_120', '_rolling_5d_avg')) or
        col.endswith(tuple(['_2h_delta'])))]
    
    X = df_lagged[feature_cols]
    
    # Use KORD temperature as target
    log_progress("Creating target variables...")
    y_1h = df_lagged['KORD_temp'].shift(-1)    # 1 hour ahead
    y_24h = df_lagged['KORD_temp'].shift(-24)  # 24 hours ahead
    y_120h = df_lagged['KORD_temp'].shift(-120)  # 120 hours ahead (5 days)
    
    # Calculate 5-day and 30-day average targets
    y_5d_avg = df_lagged['KORD_temp'].rolling(window=120, min_periods=1).mean().shift(-120)  # 5-day average ahead
    y_30d_avg = df_lagged['KORD_temp'].rolling(window=720, min_periods=1).mean().shift(-720)  # 30-day average ahead
    
    datetime = df_lagged.index

    # Drop rows where targets are nan (due to shifting)
    log_progress("Dropping rows with NaN targets...")
    valid_idx = (~y_1h.isna()) & (~y_24h.isna()) & (~y_120h.isna()) & (~y_5d_avg.isna()) & (~y_30d_avg.isna())
    X = X[valid_idx].reset_index(drop=True)
    y_1h = y_1h[valid_idx].reset_index(drop=True)
    y_24h = y_24h[valid_idx].reset_index(drop=True)
    y_120h = y_120h[valid_idx].reset_index(drop=True)
    y_5d_avg = y_5d_avg[valid_idx].reset_index(drop=True)
    y_30d_avg = y_30d_avg[valid_idx].reset_index(drop=True)
    datetime = datetime[valid_idx].reset_index(drop=True)

    # Clean up temporary files
    log_progress("Cleaning up temporary files...")
    for temp_file in Path('.').glob('temp_*.csv'):
        temp_file.unlink()
    log_progress("Cleaned up temporary files")

    total_time = time.time() - start_time
    log_progress(f"Data preparation completed in {total_time:.2f} seconds")

    return X, y_1h, y_24h, y_120h, y_5d_avg, y_30d_avg, datetime, feature_cols

def train_model(X, y, datetime):
    """
    Train a linear regression model
    Returns X_test, y_test, y_pred, datetime_test, metrics
    """
    logger.info("Training model...")
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, X.index, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    })
    feature_importance['Abs_Coefficient'] = abs(feature_importance['Coefficient'])
    feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Feature_Importance': feature_importance
    }
    datetime_test = datetime.iloc[idx_test].reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    return model, X_test, y_test, y_pred, datetime_test, metrics

def plot_results(datetime_test, y_test, y_pred, output_dir, suffix):
    """
    Plot actual vs predicted values with appropriate time windows for different prediction horizons
    """
    results_df = pd.DataFrame({
        'datetime': datetime_test,
        'actual': y_test,
        'predicted': y_pred
    })
    
    # Define time windows based on prediction type
    if suffix == "0":  # 1 hour ahead
        start_date = pd.Timestamp('2020-04-01')
        end_date = pd.Timestamp('2020-04-07 23:59:59')
        title = "Temperature Prediction: April 1-7, 2020 (1 Hour Ahead)"
    elif suffix == "1":  # 24 hours ahead
        start_date = pd.Timestamp('2020-04-01')
        end_date = pd.Timestamp('2020-04-07 23:59:59')
        title = "Temperature Prediction: April 1-7, 2020 (24 Hours Ahead)"
    elif suffix == "2":  # 120 hours ahead
        start_date = pd.Timestamp('2020-04-01')
        end_date = pd.Timestamp('2020-04-15 23:59:59')
        title = "Temperature Prediction: April 1-15, 2020 (5 Days Ahead)"
    elif suffix == "3":  # 5-day average
        start_date = pd.Timestamp('2020-04-01')
        end_date = pd.Timestamp('2020-04-30 23:59:59')
        title = "Temperature Prediction: April 2020 (5-Day Average)"
    else:  # 30-day average
        start_date = pd.Timestamp('2020-04-01')
        end_date = pd.Timestamp('2020-05-31 23:59:59')
        title = "Temperature Prediction: April-May 2020 (30-Day Average)"
    
    period_data = results_df[(results_df['datetime'] >= start_date) & (results_df['datetime'] <= end_date)]
    if period_data.empty:
        logger.warning(f"No data found for the specified period ({start_date} to {end_date}).")
        return None
    
    period_data = period_data.sort_values('datetime')
    plt.figure(figsize=(12, 6))
    plt.plot(period_data['datetime'], period_data['actual'], label='Actual', alpha=0.7)
    plt.plot(period_data['datetime'], period_data['predicted'], label='Predicted', alpha=0.7)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = output_dir / f'1-{suffix}-linear_temp_shift_results.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    return plot_path

def plot_feature_importance(feature_importance, output_dir, suffix):
    """
    Plot feature importance based on coefficient magnitudes
    """
    plt.figure(figsize=(8, 4))
    top_features = feature_importance.head(10)
    plt.barh(top_features['Feature'], top_features['Abs_Coefficient'])
    plt.title(f'Top 10 Feature Importance (1-{suffix})')
    plt.xlabel('Absolute Coefficient Value')
    plt.tight_layout()
    plot_path = output_dir / f'1-{suffix}-linear_temp_shift_feature_importance.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    return plot_path

def create_html_report(all_results, html_manager):
    """
    Create a single HTML report containing all temperature predictions
    Args:
        all_results: Dictionary containing results for all prediction types
        html_manager: HTMLManager instance
    """
    model_desc = """
    <div class="model-report">
        <h2>KORD Linear Regression with Multi-Airport Features: Temperature Prediction Analysis</h2>
        <div class="model-description">
            <h3>Model Architecture</h3>
            <p>This analysis implements multivariate time series regression models to predict temperature at Chicago O'Hare International Airport (KORD) using weather data from multiple airports. The model incorporates both historical data and recent changes in weather patterns across all airports.</p>
            
            <h4>Data Structure</h4>
            <p>The dataset contains weather measurements from multiple airports, with each measurement prefixed by its airport code (e.g., KORD, KZZZ). For each airport, we track:</p>
            <ul>
                <li>Current weather measurements</li>
                <li>2-hour changes in key weather variables</li>
                <li>Historical measurements through lagged features</li>
            </ul>

            <h4>Feature Engineering</h4>
            <p>For each airport in the dataset, we create the following feature groups:</p>
            
            <h5>1. Base Weather Features</h5>
            <div class="data-list">
                <table class="feature-table">
                    <thead>
                        <tr>
                            <th>Feature Name</th>
                            <th>Description</th>
                            <th>Units</th>
                            <th>Time Window</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>{airport}_temp</td>
                            <td>Temperature measurement</td>
                            <td>°C</td>
                            <td>Current</td>
                        </tr>
                        <tr>
                            <td>{airport}_windspeed</td>
                            <td>Wind speed measurement</td>
                            <td>m/s</td>
                            <td>Current</td>
                        </tr>
                        <tr>
                            <td>{airport}_winddirection</td>
                            <td>Wind direction measurement</td>
                            <td>degrees</td>
                            <td>Current</td>
                        </tr>
                        <tr>
                            <td>{airport}_humidity</td>
                            <td>Humidity measurement</td>
                            <td>%</td>
                            <td>Current</td>
                        </tr>
                        <tr>
                            <td>{airport}_dew</td>
                            <td>Dew point measurement</td>
                            <td>°C</td>
                            <td>Current</td>
                        </tr>
                        <tr>
                            <td>{airport}_sealevel</td>
                            <td>Sea level pressure</td>
                            <td>hPa</td>
                            <td>Current</td>
                        </tr>
                        <tr>
                            <td>{airport}_visibility</td>
                            <td>Visibility measurement</td>
                            <td>km</td>
                            <td>Current</td>
                        </tr>
                        <tr>
                            <td>{airport}_mincloud</td>
                            <td>Minimum cloud coverage</td>
                            <td>ft</td>
                            <td>Current</td>
                        </tr>
                        <tr>
                            <td>{airport}_maxcloud</td>
                            <td>Maximum cloud coverage</td>
                            <td>ft</td>
                            <td>Current</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <h5>2. 2-Hour Change Features</h5>
            <div class="data-list">
                <table class="feature-table">
                    <thead>
                        <tr>
                            <th>Feature Name</th>
                            <th>Description</th>
                            <th>Units</th>
                            <th>Time Window</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>{airport}_temp_2h_delta</td>
                            <td>Change in temperature over last 2 hours</td>
                            <td>°C</td>
                            <td>2-hour window</td>
                        </tr>
                        <tr>
                            <td>{airport}_wind_dir_2h_delta</td>
                            <td>Change in wind direction over last 2 hours</td>
                            <td>degrees</td>
                            <td>2-hour window</td>
                        </tr>
                        <tr>
                            <td>{airport}_wind_vel_2h_delta</td>
                            <td>Change in wind velocity over last 2 hours</td>
                            <td>m/s</td>
                            <td>2-hour window</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <h5>3. Historical Features</h5>
            <p>For each base feature, we create lagged versions at multiple time intervals:</p>
            <ul>
                <li>Hourly lags (1-24 hours)</li>
                <li>2-day lag (48 hours)</li>
                <li>5-day lag (120 hours)</li>
            </ul>
            <p>Additionally, for temperature specifically, we include:</p>
            <ul>
                <li>5-day rolling average (120-hour window)</li>
            </ul>

            <h4>Target Variable</h4>
            <p>The model predicts KORD's temperature at various time horizons:</p>
            <ul>
                <li>1 hour ahead</li>
                <li>24 hours ahead</li>
                <li>5 days ahead</li>
                <li>5-day average ahead</li>
                <li>30-day average ahead</li>
            </ul>

            <h4>Training Configuration</h4>
            <ul>
                <li><strong>Train/Test Split:</strong> 80/20</li>
                <li><strong>Random State:</strong> 42 (for reproducibility)</li>
                <li><strong>Validation:</strong> Standard train-test split</li>
            </ul>
        </div>
    </div>
    """
    
    # Create sections for each prediction type
    prediction_sections = []
    for horizon, results in all_results.items():
        metrics = results['metrics']
        plot_path = results['plot_path']
        feature_importance_plot = results['feature_importance_plot']
        
        section = f"""
        <div class="prediction-section">
            <h3>{horizon} Temperature Prediction</h3>
            <div class="metrics-section">
                <h4>Model Performance</h4>
                <div class="data-list">
                    <table class="metrics-table">
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Root Mean Squared Error (RMSE)</td>
                                <td>{metrics['RMSE']:.2f}</td>
                            </tr>
                            <tr>
                                <td>Mean Absolute Error (MAE)</td>
                                <td>{metrics['MAE']:.2f}</td>
                            </tr>
                            <tr>
                                <td>R² Score</td>
                                <td>{metrics['R2']:.2f}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        """
        
        # Add prediction plot
        section += html_manager.create_section_with_image(
            plot_path,
            f"{horizon} Predictions",
            "The following plot shows the actual vs predicted temperatures. The blue line represents actual measurements, while the orange line shows our model's predictions."
        )
        
        # Add feature importance plot
        section += html_manager.create_section_with_image(
            feature_importance_plot,
            f"{horizon} Feature Importance",
            "This plot shows the top 10 most important features based on their absolute coefficient values in the regression model."
        )
        
        section += "</div>"
        prediction_sections.append(section)
    
    # Combine all sections
    content = model_desc + "\n".join(prediction_sections)
    
    # Add interpretation section
    interpretation = """
    <div class="interpretation-section">
        <h3>Model Interpretation</h3>
        <p>These multivariate regression models capture the complex relationships between weather patterns across multiple airports and temperature at KORD. The models:</p>
        <ul>
            <li>Consider the impact of weather conditions at all available airports</li>
            <li>Account for recent changes in weather patterns through 2-hour delta features</li>
            <li>Incorporate historical weather patterns through lagged features</li>
            <li>Provide insights into which airports and features most influence KORD's temperature</li>
        </ul>
        <p>Future improvements could include:</p>
        <ul>
            <li>Feature selection to reduce dimensionality</li>
            <li>Non-linear models to capture complex relationships</li>
            <li>Time series specific models (ARIMA, LSTM)</li>
            <li>Spatial relationship modeling between airports</li>
        </ul>
    </div>
    """
    content += interpretation
    
    html_content = html_manager.template.format(
        title="KORD Linear Regression with Multi-Airport Features: Temperature Prediction Analysis",
        content=content,
        additional_js=""
    )
    
    html_filename = "1-linear_temp_shift.html"
    output_path = html_manager.save_section_html("Airport_to_Airport", html_content, html_filename)
    return html_content, output_path

def create_latex_report(all_results, output_dir):
    """
    Create a LaTeX report containing all temperature predictions
    Args:
        all_results: Dictionary containing results for all prediction types
        output_dir: Path to the output directory
    """
    model_desc = latex_section("KORD Linear Regression with Multi-Airport Features: Temperature Prediction Analysis",
        "This analysis implements multivariate time series regression models to predict temperature at Chicago O'Hare International Airport (KORD) using weather data from multiple airports. The model incorporates both historical data and recent changes in weather patterns across all airports.\\\n" +
        latex_list([
            "Base Model: Multivariate Linear Regression",
            "Feature Engineering: Airport-specific features with time lags and 2-hour changes",
            "Prediction Horizons: Multiple time windows (1h, 24h, 5d, 5d avg, 30d avg)",
            "Train/Test Split: 80/20",
            "Random State: 42 (for reproducibility)",
            "Validation: Standard train-test split"
        ]) +
        "\n\n\\subsection{Data Structure}\n" +
        "The dataset contains weather measurements from multiple airports, with each measurement prefixed by its airport code (e.g., KORD, KZZZ). For each airport, we track current weather measurements, 2-hour changes in key weather variables, and historical measurements through lagged features.\n\n" +
        "\\subsection{Feature Engineering}\n" +
        "For each airport in the dataset, we create the following feature groups:\n\n" +
        "\\subsubsection{Base Weather Features}\n" +
        "Each airport's current weather measurements include:\n" +
        "\\begin{itemize}\n" +
        "  \\item Temperature (°C)\n" +
        "  \\item Wind speed (m/s)\n" +
        "  \\item Wind direction (degrees)\n" +
        "  \\item Humidity (\\%)\n" +
        "  \\item Dew point (°C)\n" +
        "  \\item Sea level pressure (hPa)\n" +
        "  \\item Visibility (km)\n" +
        "  \\item Cloud coverage (min/max, ft)\n" +
        "\\end{itemize}\n\n" +
        "\\subsubsection{2-Hour Change Features}\n" +
        "For each airport, we calculate 2-hour changes in:\n" +
        "\\begin{itemize}\n" +
        "  \\item Temperature (\\textdegree C)\n" +
        "  \\item Wind direction (degrees)\n" +
        "  \\item Wind velocity (m/s)\n" +
        "\\end{itemize}\n\n" +
        "\\subsubsection{Historical Features}\n" +
        "For each base feature, we create lagged versions at multiple time intervals:\n" +
        "\\begin{itemize}\n" +
        "  \\item Hourly lags (1-24 hours)\n" +
        "  \\item 2-day lag (48 hours)\n" +
        "  \\item 5-day lag (120 hours)\n" +
        "\\end{itemize}\n" +
        "Additionally, for temperature specifically, we include a 5-day rolling average (120-hour window).\n\n" +
        "\\subsection{Target Variable}\n" +
        "The model predicts KORD's temperature at various time horizons:\n" +
        "\\begin{itemize}\n" +
        "  \\item 1 hour ahead\n" +
        "  \\item 24 hours ahead\n" +
        "  \\item 5 days ahead\n" +
        "  \\item 5-day average ahead\n" +
        "  \\item 30-day average ahead\n" +
        "\\end{itemize}\n"
    )
    
    prediction_sections = []
    for horizon, results in all_results.items():
        metrics = results['metrics']
        plot_path = results['plot_path']
        feature_importance_plot = results['feature_importance_plot']
        # Metrics table
        metrics_df = pd.DataFrame({
            'Metric': ['Root Mean Squared Error (RMSE)', 'Mean Absolute Error (MAE)', 'R^2 Score'],
            'Value': [metrics['RMSE'], metrics['MAE'], metrics['R2']]
        })
        section = latex_subsection(f"{horizon} Temperature Prediction",
            latex_subsection("Model Performance",
                df_to_latex_table(metrics_df, caption=f"{horizon} Metrics", label=f"tab:{horizon.replace(' ', '_').lower()}_metrics")
            ) +
            latex_subsection(f"{horizon} Predictions",
                f"\\begin{{figure}}[htbp]\n\\centering\n\\includegraphics[width=0.7\\textwidth]{{{Path(plot_path).name}}}\n\\caption{{{horizon} Predictions}}\n\\label{{fig:{horizon.replace(' ', '_').lower()}_pred}}\n\\end{{figure}}\n"
            ) +
            latex_subsection(f"{horizon} Feature Importance",
                f"\\begin{{figure}}[htbp]\n\\centering\n\\includegraphics[width=0.7\\textwidth]{{{Path(feature_importance_plot).name}}}\n\\caption{{{horizon} Feature Importance}}\n\\label{{fig:{horizon.replace(' ', '_').lower()}_featimp}}\n\\end{{figure}}\n"
            )
        )
        prediction_sections.append(section)
    
    interpretation = latex_section(
        "Model Interpretation",
        """
These multivariate regression models capture the complex relationships between weather patterns across multiple airports and temperature at KORD. The models:\\\\
\\begin{itemize}
  \\item Consider the impact of weather conditions at all available airports
  \\item Account for recent changes in weather patterns through 2-hour delta features
  \\item Incorporate historical weather patterns through lagged features
  \\item Provide insights into which airports and features most influence KORD's temperature
\\end{itemize}
Future improvements could include:\\\\
\\begin{itemize}
  \\item Feature selection to reduce dimensionality
  \\item Non-linear models to capture complex relationships
  \\item Time series specific models (ARIMA, LSTM)
  \\item Spatial relationship modeling between airports
\\end{itemize}
"""
    )
    latex_content = model_desc + "\n".join(prediction_sections) + interpretation
    latex_output_path = output_dir / '1-linear_temp_shift.tex'
    save_latex_file(latex_content, latex_output_path)
    logger.info(f"LaTeX linear regression report created: {latex_output_path}")
    return latex_output_path

def main():
    try:
        output_dir = Path(__file__).parent / 'outputs'
        output_dir.mkdir(exist_ok=True)
        manager = HTMLManager()
        manager.register_section("Airport_to_Airport", Path(__file__).parent)
        X, y_1h, y_24h, y_120h, y_5d_avg, y_30d_avg, datetime, feature_cols = load_and_prepare_data()
        
        # Dictionary to store all results
        all_results = {}
        
        # 1 hour ahead
        model_1h, X_test_1h, y_test_1h, y_pred_1h, datetime_test_1h, metrics_1h = train_model(X, y_1h, datetime)
        plot_path_1h = plot_results(datetime_test_1h, y_test_1h, y_pred_1h, output_dir, suffix="0")
        feature_importance_plot_1h = plot_feature_importance(metrics_1h['Feature_Importance'], output_dir, suffix="0")
        all_results["1 Hour Ahead"] = {
            'metrics': metrics_1h,
            'plot_path': plot_path_1h,
            'feature_importance_plot': feature_importance_plot_1h
        }
        logger.info("1-hour prediction analysis complete")
        
        # 24 hours ahead
        model_24h, X_test_24h, y_test_24h, y_pred_24h, datetime_test_24h, metrics_24h = train_model(X, y_24h, datetime)
        plot_path_24h = plot_results(datetime_test_24h, y_test_24h, y_pred_24h, output_dir, suffix="1")
        feature_importance_plot_24h = plot_feature_importance(metrics_24h['Feature_Importance'], output_dir, suffix="1")
        all_results["24 Hours Ahead"] = {
            'metrics': metrics_24h,
            'plot_path': plot_path_24h,
            'feature_importance_plot': feature_importance_plot_24h
        }
        logger.info("24-hour prediction analysis complete")
        
        # 120 hours ahead
        model_120h, X_test_120h, y_test_120h, y_pred_120h, datetime_test_120h, metrics_120h = train_model(X, y_120h, datetime)
        plot_path_120h = plot_results(datetime_test_120h, y_test_120h, y_pred_120h, output_dir, suffix="2")
        feature_importance_plot_120h = plot_feature_importance(metrics_120h['Feature_Importance'], output_dir, suffix="2")
        all_results["120 Hours (5 Days) Ahead"] = {
            'metrics': metrics_120h,
            'plot_path': plot_path_120h,
            'feature_importance_plot': feature_importance_plot_120h
        }
        logger.info("120-hour prediction analysis complete")
        
        # 5-day average
        model_5d_avg, X_test_5d_avg, y_test_5d_avg, y_pred_5d_avg, datetime_test_5d_avg, metrics_5d_avg = train_model(X, y_5d_avg, datetime)
        plot_path_5d_avg = plot_results(datetime_test_5d_avg, y_test_5d_avg, y_pred_5d_avg, output_dir, suffix="3")
        feature_importance_plot_5d_avg = plot_feature_importance(metrics_5d_avg['Feature_Importance'], output_dir, suffix="3")
        all_results["5-Day Average Ahead"] = {
            'metrics': metrics_5d_avg,
            'plot_path': plot_path_5d_avg,
            'feature_importance_plot': feature_importance_plot_5d_avg
        }
        logger.info("5-day average prediction analysis complete")
        
        # 30-day average
        model_30d_avg, X_test_30d_avg, y_test_30d_avg, y_pred_30d_avg, datetime_test_30d_avg, metrics_30d_avg = train_model(X, y_30d_avg, datetime)
        plot_path_30d_avg = plot_results(datetime_test_30d_avg, y_test_30d_avg, y_pred_30d_avg, output_dir, suffix="4")
        feature_importance_plot_30d_avg = plot_feature_importance(metrics_30d_avg['Feature_Importance'], output_dir, suffix="4")
        all_results["30-Day Average Ahead"] = {
            'metrics': metrics_30d_avg,
            'plot_path': plot_path_30d_avg,
            'feature_importance_plot': feature_importance_plot_30d_avg
        }
        logger.info("30-day average prediction analysis complete")
        
        # Create single HTML report with all results
        html_content, output_path = create_html_report(all_results, manager)
        logger.info(f"All temperature prediction analyses complete. Results saved to {output_path}")
        # Create LaTeX report
        create_latex_report(all_results, output_dir)
        
    except Exception as e:
        logger.error(f"Error in temperature prediction analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
