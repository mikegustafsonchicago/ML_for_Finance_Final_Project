import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import sys
sys.path.append('..')  # Add parent directory to path to import html_manager
from html_manager import HTMLManager
from Utils.data_formatter import parse_custom_datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def calculate_wind_direction_delta(df):
    df['wind_dir_delta'] = df['winddirection'].diff()
    df.loc[df['wind_dir_delta'] > 180, 'wind_dir_delta'] -= 360
    df.loc[df['wind_dir_delta'] < -180, 'wind_dir_delta'] += 360
    return df

def load_and_prepare_data(file_path='../datastep2.csv', n_lags=24):
    logger.info("Loading and preparing data...")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded data shape: {df.shape}")
    logger.info(f"Columns in loaded data: {list(df.columns)}")
    if 'id' in df.columns:
        df = df[df['id'] == 'KORD']
        logger.info(f"Data shape after filtering for KORD: {df.shape}")
    else:
        logger.warning("No 'id' column found; cannot filter for KORD.")
    df['datetime'] = parse_custom_datetime(df['datetime'])
    df = df.sort_values('datetime')
    logger.info(f"Data shape after sorting: {df.shape}")
    df = calculate_wind_direction_delta(df)
    logger.info(f"Data shape after wind_dir_delta: {df.shape}")

    expected_numeric_cols = [
        'skydescriptor', 'temp_str', 'temp', 'windspeed', 'winddirection',
        'humidity', 'dew', 'sealevel', 'visibility', 'mincloud', 'maxcloud'
    ]
    numeric_cols = [col for col in expected_numeric_cols if col in df.columns]
    logger.info(f"Numeric columns used for lagged features: {numeric_cols}")
    logger.info(f"Number of numeric columns: {len(numeric_cols)}")

    lagged_features = []
    # Standard hourly lags
    for col in numeric_cols:
        for i in range(1, n_lags + 1):
            lagged = df[col].shift(i)
            lagged.name = f'{col}_lag_{i}'
            lagged_features.append(lagged)
    # Historical ±2 day (48h) and ±5 day (120h) temp (LAGGED ONLY)
    for offset in [48, 120]:
        lagged = df['temp'].shift(offset)
        lagged.name = f'temp_lag_{offset}'
        lagged_features.append(lagged)
    # Last 5 day temp avg (rolling mean of last 120 hours, excluding current)
    rolling_5d = df['temp'].shift(1).rolling(window=120, min_periods=1).mean()
    rolling_5d.name = 'temp_rolling_5d_avg'
    lagged_features.append(rolling_5d)

    lagged_df = pd.concat(lagged_features, axis=1)
    logger.info(f"Lagged DataFrame shape: {lagged_df.shape}")
    df_lagged = pd.concat([df, lagged_df], axis=1)
    logger.info(f"Shape after combining lagged features: {df_lagged.shape}")
    df_lagged = df_lagged.dropna().reset_index(drop=True)
    logger.info(f"Shape after dropping NaNs: {df_lagged.shape}")

    # Prepare features and targets
    feature_cols = [col for col in df_lagged.columns if (
        col.endswith(tuple([f'_lag_{i}' for i in range(1, n_lags + 1)]) +
        ('_lag_48', '_lag_120', 'temp_rolling_5d_avg')))]
    X = df_lagged[feature_cols]
    y_1h = df_lagged['temp'].shift(-1)    # 1 hour ahead
    y_120h = df_lagged['temp'].shift(-120)  # 120 hours ahead (5 days)
    datetime = df_lagged['datetime']

    # Drop rows where targets are nan (due to shifting)
    valid_idx = (~y_1h.isna()) & (~y_120h.isna())
    X = X[valid_idx].reset_index(drop=True)
    y_1h = y_1h[valid_idx].reset_index(drop=True)
    y_120h = y_120h[valid_idx].reset_index(drop=True)
    datetime = datetime[valid_idx].reset_index(drop=True)

    return X, y_1h, y_120h, datetime, feature_cols

def train_model(X, y, datetime):
    logger.info("Training Random Forest model...")
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, X.index, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
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
    results_df = pd.DataFrame({
        'datetime': datetime_test,
        'actual': y_test,
        'predicted': y_pred
    })
    start_date = pd.Timestamp('2020-04-01')
    end_date = pd.Timestamp('2020-04-07 23:59:59')
    week_data = results_df[(results_df['datetime'] >= start_date) & (results_df['datetime'] <= end_date)]
    if week_data.empty:
        logger.warning("No data found for the specified week (April 1-7, 2020).")
        return None
    week_data = week_data.sort_values('datetime')
    plt.figure(figsize=(10, 4))
    plt.plot(week_data['datetime'], week_data['actual'], label='Actual', alpha=0.7)
    plt.plot(week_data['datetime'], week_data['predicted'], label='Predicted', alpha=0.7)
    plt.title(f'Temperature Prediction: April 1-7, 2020 (7-{suffix})')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = output_dir / f'7-{suffix}-random_forest_temp_prediction_results.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    return plot_path

def plot_feature_importance(feature_importance, output_dir, suffix):
    plt.figure(figsize=(8, 4))
    top_features = feature_importance.head(10)
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.title(f'Top 10 Feature Importance (7-{suffix})')
    plt.xlabel('Feature Importance Score')
    plt.tight_layout()
    plot_path = output_dir / f'7-{suffix}-random_forest_temp_feature_importance.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    return plot_path

def create_html_report(metrics, plot_path, feature_importance_plot, html_manager, suffix, horizon_desc):
    model_desc = f"""
    <div class="model-report">
        <h2>KORD Random Forest: Temperature Prediction Analysis ({horizon_desc})</h2>
        <div class="model-description">
            <h3>Model Architecture</h3>
            <p>This analysis implements a multivariate time series regression model using Random Forest to predict temperature at Chicago O'Hare International Airport (KORD).</p>
            <h4>Model Type</h4>
            <ul>
                <li><strong>Base Model:</strong> Random Forest Regression</li>
                <li><strong>Feature Engineering:</strong> Time-lagged features for all numeric variables</li>
                <li><strong>Prediction Target:</strong> {horizon_desc} temperature</li>
            </ul>
            <h4>Feature Details</h4>
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
                            <td>Temperature</td>
                            <td>Current and historical temperature measurements</td>
                            <td>°C</td>
                            <td>24-hour lag, ±2/5 day, 5-day avg</td>
                        </tr>
                        <tr>
                            <td>Wind Speed</td>
                            <td>Current and historical wind speed measurements</td>
                            <td>m/s</td>
                            <td>24-hour lag</td>
                        </tr>
                        <tr>
                            <td>Wind Direction</td>
                            <td>Current and historical wind direction measurements</td>
                            <td>degrees</td>
                            <td>24-hour lag</td>
                        </tr>
                        <tr>
                            <td>Wind Direction Delta</td>
                            <td>Change in wind direction between consecutive hours</td>
                            <td>degrees</td>
                            <td>Current</td>
                        </tr>
                        <tr>
                            <td>Humidity</td>
                            <td>Current and historical humidity measurements</td>
                            <td>%</td>
                            <td>24-hour lag</td>
                        </tr>
                        <tr>
                            <td>Dew Point</td>
                            <td>Current and historical dew point measurements</td>
                            <td>°C</td>
                            <td>24-hour lag</td>
                        </tr>
                        <tr>
                            <td>Sea Level Pressure</td>
                            <td>Current and historical pressure measurements</td>
                            <td>hPa</td>
                            <td>24-hour lag</td>
                        </tr>
                        <tr>
                            <td>Visibility</td>
                            <td>Current and historical visibility measurements</td>
                            <td>km</td>
                            <td>24-hour lag</td>
                        </tr>
                        <tr>
                            <td>Cloud Coverage</td>
                            <td>Current and historical min/max cloud coverage</td>
                            <td>ft</td>
                            <td>24-hour lag</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            <h4>Training Configuration</h4>
            <ul>
                <li><strong>Train/Test Split:</strong> 80/20</li>
                <li><strong>Random State:</strong> 42 (for reproducibility)</li>
                <li><strong>Validation:</strong> Standard train-test split</li>
            </ul>
        </div>
    </div>
    """
    interpretation = f"""
    <div class="interpretation-section">
        <h3>Model Interpretation</h3>
        <p>This Random Forest model captures the complex relationships between various weather parameters and temperature. The model:</p>
        <ul>
            <li>Considers the impact of all available weather parameters</li>
            <li>Accounts for changes in wind direction through the wind_dir_delta feature</li>
            <li>Provides insights into which factors most influence temperature</li>
        </ul>
        <p>Future improvements could include:</p>
        <ul>
            <li>Feature selection to reduce dimensionality</li>
            <li>Non-linear models to capture complex relationships</li>
            <li>Time series specific models (ARIMA, LSTM)</li>
        </ul>
    </div>
    """
    metrics_section = f"""
    <div class="metrics-section">
        <h3>Model Performance</h3>
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
    time_series_section = html_manager.create_section_with_image(
        plot_path,
        "Sample Week Predictions",
        "The following plots show the actual vs predicted temperatures for sample weeks in different months. The blue line represents actual measurements, while the orange line shows our model's predictions."
    )
    feature_importance_section = html_manager.create_section_with_image(
        feature_importance_plot,
        "Feature Importance",
        "This plot shows the top 10 most important features based on their importance scores in the Random Forest model."
    )
    content = model_desc + interpretation + metrics_section + time_series_section + feature_importance_section
    html_content = html_manager.template.format(
        title=f"KORD Random Forest: Temperature Prediction Analysis ({horizon_desc})",
        content=content,
        additional_js=""
    )
    html_filename = f"7-{suffix}-random_forest_temp_regression.html"
    output_path = html_manager.save_section_html("KORD_Self_Regression", html_content, html_filename)
    return html_content, output_path

def main():
    try:
        output_dir = Path(__file__).parent / 'outputs'
        output_dir.mkdir(exist_ok=True)
        manager = HTMLManager()
        manager.register_section("KORD_Self_Regression", Path(__file__).parent)
        X, y_1h, y_120h, datetime, feature_cols = load_and_prepare_data()
        # 1 hour ahead (suffix 0)
        model_1h, X_test_1h, y_test_1h, y_pred_1h, datetime_test_1h, metrics_1h = train_model(X, y_1h, datetime)
        plot_path_1h = plot_results(datetime_test_1h, y_test_1h, y_pred_1h, output_dir, suffix="0")
        feature_importance_plot_1h = plot_feature_importance(metrics_1h['Feature_Importance'], output_dir, suffix="0")
        html_content_1h, output_path_1h = create_html_report(
            metrics_1h, plot_path_1h, feature_importance_plot_1h, manager, suffix="0", horizon_desc="1 Hour Ahead"
        )
        logger.info(f"1-hour prediction analysis complete. Results saved to {output_path_1h}")
        # 5 days ahead (suffix 1)
        model_120h, X_test_120h, y_test_120h, y_pred_120h, datetime_test_120h, metrics_120h = train_model(X, y_120h, datetime)
        plot_path_120h = plot_results(datetime_test_120h, y_test_120h, y_pred_120h, output_dir, suffix="1")
        feature_importance_plot_120h = plot_feature_importance(metrics_120h['Feature_Importance'], output_dir, suffix="1")
        html_content_120h, output_path_120h = create_html_report(
            metrics_120h, plot_path_120h, feature_importance_plot_120h, manager, suffix="1", horizon_desc="120 Hours (5 Days) Ahead"
        )
        logger.info(f"5-day prediction analysis complete. Results saved to {output_path_120h}")
    except Exception as e:
        logger.error(f"Error in temperature prediction analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
