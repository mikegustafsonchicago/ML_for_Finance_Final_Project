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

def load_and_prepare_data(file_path='../datastep2.csv', n_lags=24):
    """
    Load and prepare data for time series regression, focusing only on KORD data
    """
    start_time = time.time()
    log_progress("Starting data preparation...")
    
    # Load only KORD data
    log_progress("Loading KORD data...")
    df = pd.read_csv(file_path)
    df = df[df['id'] == 'KORD'].copy()
    log_progress(f"Loaded KORD data shape: {df.shape}")
    
    # Process datetime
    log_progress("Processing datetime...")
    df['datetime'] = parse_custom_datetime(df['datetime'])
    df = df.sort_values('datetime')
    
    # Calculate 2-hour deltas
    log_progress("Calculating 2-hour deltas...")
    # Temperature delta
    df['temp_2h_delta'] = df['temp'].diff(2)
    
    # Wind direction delta (handling circular nature)
    df['wind_dir_2h_delta'] = df['winddirection'].diff(2)
    df.loc[df['wind_dir_2h_delta'] > 180, 'wind_dir_2h_delta'] -= 360
    df.loc[df['wind_dir_2h_delta'] < -180, 'wind_dir_2h_delta'] += 360
    
    # Wind velocity delta
    df['wind_vel_2h_delta'] = df['windspeed'].diff(2)
    
    # Create lagged features
    log_progress("Creating lagged features...")
    lagged_features = {}
    
    # Base features to create lags for
    base_features = ['temp', 'windspeed', 'winddirection', 'humidity', 'dew', 'sealevel', 'visibility', 'mincloud', 'maxcloud']
    
    # Create hourly lags for each base feature
    for feature in base_features:
        if feature in df.columns:
            for lag in range(1, n_lags + 1):
                lagged_features[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
    
    # Add specific temperature lags
    lagged_features['temp_lag_48'] = df['temp'].shift(48)
    lagged_features['temp_lag_120'] = df['temp'].shift(120)
    lagged_features['temp_rolling_5d_avg'] = df['temp'].shift(1).rolling(window=120, min_periods=1).mean()
    
    # Convert lagged features to DataFrame
    lagged_df = pd.DataFrame(lagged_features)
    
    # Combine with original data
    log_progress("Combining features...")
    df_lagged = pd.concat([df, lagged_df], axis=1)
    
    # Drop rows with NaN values
    log_progress("Dropping NaN values...")
    df_lagged = df_lagged.dropna().reset_index(drop=True)
    log_progress(f"Final data shape: {df_lagged.shape}")
    
    # Prepare features and targets
    log_progress("Preparing features and targets...")
    feature_cols = [col for col in df_lagged.columns if (
        col.endswith(tuple([f'_lag_{i}' for i in range(1, n_lags + 1)]) + 
        ('_lag_48', '_lag_120', '_rolling_5d_avg')) or
        col.endswith(tuple(['_2h_delta'])))]
    
    X = df_lagged[feature_cols]
    
    # Create target variables
    log_progress("Creating target variables...")
    y_1h = df_lagged['temp'].shift(-1)    # 1 hour ahead
    y_24h = df_lagged['temp'].shift(-24)  # 24 hours ahead
    y_120h = df_lagged['temp'].shift(-120)  # 120 hours ahead (5 days)
    
    # Calculate 5-day and 30-day average targets
    y_5d_avg = df_lagged['temp'].rolling(window=120, min_periods=1).mean().shift(-120)  # 5-day average ahead
    y_30d_avg = df_lagged['temp'].rolling(window=720, min_periods=1).mean().shift(-720)  # 30-day average ahead
    
    datetime = df_lagged['datetime']
    
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
            
            <h4>Regressors (Predictor Variables)</h4>
            <p>Our model uses the following variables to predict KORD's temperature:</p>
            <ul>
                <li><strong>KORD's Own Historical Data:</strong>
                    <ul>
                        <li>Temperature measurements from the past 24 hours (hourly lags)</li>
                        <li>Temperature measurements from 48 hours ago (2-day lag)</li>
                        <li>Temperature measurements from 120 hours ago (5-day lag)</li>
                        <li>5-day rolling average temperature (excluding current hour)</li>
                    </ul>
                </li>
                <li><strong>Other Airports' Current and Historical Data:</strong>
                    <ul>
                        <li>Temperature measurements and their 24-hour lags</li>
                        <li>Wind speed measurements and their 24-hour lags</li>
                        <li>Wind direction measurements and their 24-hour lags</li>
                        <li>Humidity measurements and their 24-hour lags</li>
                        <li>Dew point measurements and their 24-hour lags</li>
                        <li>Sea level pressure measurements and their 24-hour lags</li>
                        <li>Visibility measurements and their 24-hour lags</li>
                        <li>Cloud coverage measurements and their 24-hour lags</li>
                    </ul>
                </li>
                <li><strong>2-Hour Change Features:</strong>
                    <ul>
                        <li>Temperature changes over the last 2 hours</li>
                        <li>Wind direction changes over the last 2 hours</li>
                        <li>Wind velocity changes over the last 2 hours</li>
                    </ul>
                </li>
            </ul>
            <p>Each of these features is calculated for every airport in our dataset, creating a rich set of predictors that capture both local and regional weather patterns.</p>
            
            <h4>Data Structure</h4>
            <p>The dataset contains weather measurements from multiple airports, with each measurement prefixed by its airport code (e.g., KORD, KZZZ). For each airport, we track:</p>
            <ul>
                <li>Current weather measurements</li>
                <li>2-hour changes in key weather variables</li>
                <li>Historical measurements through lagged features</li>
            </ul>
            <p><strong>Note:</strong> While we use data from multiple airports, our target variable is always KORD's temperature. The other airports' data serve as potential predictors for KORD's temperature.</p>

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
                            <td>Temperature measurement at {airport}</td>
                            <td>°C</td>
                            <td>Current</td>
                        </tr>
                        <tr>
                            <td>{airport}_windspeed</td>
                            <td>Wind speed measurement at {airport}</td>
                            <td>m/s</td>
                            <td>Current</td>
                        </tr>
                        <tr>
                            <td>{airport}_winddirection</td>
                            <td>Wind direction measurement at {airport}</td>
                            <td>degrees</td>
                            <td>Current</td>
                        </tr>
                        <tr>
                            <td>{airport}_humidity</td>
                            <td>Humidity measurement at {airport}</td>
                            <td>%</td>
                            <td>Current</td>
                        </tr>
                        <tr>
                            <td>{airport}_dew</td>
                            <td>Dew point measurement at {airport}</td>
                            <td>°C</td>
                            <td>Current</td>
                        </tr>
                        <tr>
                            <td>{airport}_sealevel</td>
                            <td>Sea level pressure at {airport}</td>
                            <td>hPa</td>
                            <td>Current</td>
                        </tr>
                        <tr>
                            <td>{airport}_visibility</td>
                            <td>Visibility measurement at {airport}</td>
                            <td>km</td>
                            <td>Current</td>
                        </tr>
                        <tr>
                            <td>{airport}_mincloud</td>
                            <td>Minimum cloud coverage at {airport}</td>
                            <td>ft</td>
                            <td>Current</td>
                        </tr>
                        <tr>
                            <td>{airport}_maxcloud</td>
                            <td>Maximum cloud coverage at {airport}</td>
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
                            <td>Change in temperature over last 2 hours at {airport}</td>
                            <td>°C</td>
                            <td>2-hour window</td>
                        </tr>
                        <tr>
                            <td>{airport}_wind_dir_2h_delta</td>
                            <td>Change in wind direction over last 2 hours at {airport}</td>
                            <td>degrees</td>
                            <td>2-hour window</td>
                        </tr>
                        <tr>
                            <td>{airport}_wind_vel_2h_delta</td>
                            <td>Change in wind velocity over last 2 hours at {airport}</td>
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
            <h3>{horizon} Temperature Prediction for KORD</h3>
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
            f"{horizon} Predictions for KORD",
            "The following plot shows the actual vs predicted temperatures for KORD. The blue line represents actual measurements at KORD, while the orange line shows our model's predictions using data from all airports."
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
            <li>Use weather conditions from all available airports to predict KORD's temperature</li>
            <li>Account for recent changes in weather patterns through 2-hour delta features</li>
            <li>Incorporate historical weather patterns through lagged features</li>
            <li>Help identify which airports' weather patterns most influence KORD's temperature</li>
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
        "\n\n\\subsection{Regressors (Predictor Variables)}\n" +
        "Our model uses the following variables to predict KORD's temperature:\n\n" +
        "\\subsubsection{KORD's Own Historical Data}\n" +
        "\\begin{itemize}\n" +
        "  \\item Temperature measurements from the past 24 hours (hourly lags)\n" +
        "  \\item Temperature measurements from 48 hours ago (2-day lag)\n" +
        "  \\item Temperature measurements from 120 hours ago (5-day lag)\n" +
        "  \\item 5-day rolling average temperature (excluding current hour)\n" +
        "\\end{itemize}\n\n" +
        "\\subsubsection{Other Airports' Current and Historical Data}\n" +
        "\\begin{itemize}\n" +
        "  \\item Temperature measurements and their 24-hour lags\n" +
        "  \\item Wind speed measurements and their 24-hour lags\n" +
        "  \\item Wind direction measurements and their 24-hour lags\n" +
        "  \\item Humidity measurements and their 24-hour lags\n" +
        "  \\item Dew point measurements and their 24-hour lags\n" +
        "  \\item Sea level pressure measurements and their 24-hour lags\n" +
        "  \\item Visibility measurements and their 24-hour lags\n" +
        "  \\item Cloud coverage measurements and their 24-hour lags\n" +
        "\\end{itemize}\n\n" +
        "\\subsubsection{2-Hour Change Features}\n" +
        "\\begin{itemize}\n" +
        "  \\item Temperature changes over the last 2 hours\n" +
        "  \\item Wind direction changes over the last 2 hours\n" +
        "  \\item Wind velocity changes over the last 2 hours\n" +
        "\\end{itemize}\n\n" +
        "Each of these features is calculated for every airport in our dataset, creating a rich set of predictors that capture both local and regional weather patterns.\n\n" +
        "\\subsection{Data Structure}\n" +
        "The dataset contains weather measurements from multiple airports, with each measurement prefixed by its airport code (e.g., KORD, KZZZ). For each airport, we track current weather measurements, 2-hour changes in key weather variables, and historical measurements through lagged features. While we use data from multiple airports, our target variable is always KORD's temperature. The other airports' data serve as potential predictors for KORD's temperature.\n\n" +
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
        section = latex_subsection(f"{horizon} Temperature Prediction for KORD",
            latex_subsection("Model Performance",
                df_to_latex_table(metrics_df, caption=f"{horizon} Metrics", label=f"tab:{horizon.replace(' ', '_').lower()}_metrics")
            ) +
            latex_subsection(f"{horizon} Predictions",
                f"\\begin{{figure}}[htbp]\n\\centering\n\\includegraphics[width=0.7\\textwidth]{{{Path(plot_path).name}}}\n\\caption{{{horizon} Predictions for KORD}}\n\\label{{fig:{horizon.replace(' ', '_').lower()}_pred}}\n\\end{{figure}}\n"
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
  \\item Use weather conditions from all available airports to predict KORD's temperature
  \\item Account for recent changes in weather patterns through 2-hour delta features
  \\item Incorporate historical weather patterns through lagged features
  \\item Help identify which airports' weather patterns most influence KORD's temperature
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
