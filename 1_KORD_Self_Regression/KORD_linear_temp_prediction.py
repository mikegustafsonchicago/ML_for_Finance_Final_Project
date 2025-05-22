import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import sys
sys.path.append('..')  # Add parent directory to path to import html_manager
from html_manager import HTMLManager
from Utils.data_formatter import parse_custom_datetime
from Utils.latex_utility import save_latex_file, latex_section, latex_subsection, df_to_latex_table, latex_list

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def calculate_wind_direction_delta(df):
    """
    Calculate the change in wind direction between consecutive hours
    Handles the circular nature of wind direction (0-360 degrees)
    """
    df['wind_dir_delta'] = df['winddirection'].diff()
    df.loc[df['wind_dir_delta'] > 180, 'wind_dir_delta'] -= 360
    df.loc[df['wind_dir_delta'] < -180, 'wind_dir_delta'] += 360
    return df

def load_and_prepare_data(file_path='../datastep2.csv', n_lags=24):
    """
    Load and prepare data for time series regression
    Args:
        file_path: Path to the data file
        n_lags: Number of lagged features to create (for hourly lags)
    Returns:
        X: Features matrix
        y_1h: Target variable (1 hour ahead)
        y_24h: Target variable (24 hours ahead)
        y_120h: Target variable (120 hours ahead)
        y_5d_avg: Target variable (5-day average ahead)
        y_30d_avg: Target variable (30-day average ahead)
        datetime: Datetime index aligned with X/y
        feature_cols: List of feature column names
    """
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
    y_24h = df_lagged['temp'].shift(-24)  # 24 hours ahead
    y_120h = df_lagged['temp'].shift(-120)  # 120 hours ahead (5 days)
    
    # Calculate 5-day and 30-day average targets
    y_5d_avg = df_lagged['temp'].rolling(window=120, min_periods=1).mean().shift(-120)  # 5-day average ahead
    y_30d_avg = df_lagged['temp'].rolling(window=720, min_periods=1).mean().shift(-720)  # 30-day average ahead
    
    datetime = df_lagged['datetime']

    # Drop rows where targets are nan (due to shifting)
    valid_idx = (~y_1h.isna()) & (~y_24h.isna()) & (~y_120h.isna()) & (~y_5d_avg.isna()) & (~y_30d_avg.isna())
    X = X[valid_idx].reset_index(drop=True)
    y_1h = y_1h[valid_idx].reset_index(drop=True)
    y_24h = y_24h[valid_idx].reset_index(drop=True)
    y_120h = y_120h[valid_idx].reset_index(drop=True)
    y_5d_avg = y_5d_avg[valid_idx].reset_index(drop=True)
    y_30d_avg = y_30d_avg[valid_idx].reset_index(drop=True)
    datetime = datetime[valid_idx].reset_index(drop=True)

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
    plot_path = output_dir / f'5-{suffix}-linear_temp_prediction_results.png'
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
    plt.title(f'Top 10 Feature Importance (5-{suffix})')
    plt.xlabel('Absolute Coefficient Value')
    plt.tight_layout()
    plot_path = output_dir / f'5-{suffix}-linear_temp_feature_importance.png'
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
        <h2>KORD Linear Regression: Temperature Prediction Analysis</h2>
        <div class="model-description">
            <h3>Model Architecture</h3>
            <p>This analysis implements multivariate time series regression models to predict temperature at Chicago O'Hare International Airport (KORD) across multiple time horizons.</p>
            <h4>Model Type</h4>
            <ul>
                <li><strong>Base Model:</strong> Multivariate Linear Regression</li>
                <li><strong>Feature Engineering:</strong> Time-lagged features for all numeric variables</li>
                <li><strong>Prediction Horizons:</strong> Multiple time windows (1h, 24h, 5d, 5d avg, 30d avg)</li>
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
        <p>These multivariate regression models capture the complex relationships between various weather parameters and temperature across different time horizons. The models:</p>
        <ul>
            <li>Consider the impact of all available weather parameters</li>
            <li>Account for changes in wind direction through the wind_dir_delta feature</li>
            <li>Provide insights into which factors most influence temperature at different time scales</li>
        </ul>
        <p>Future improvements could include:</p>
        <ul>
            <li>Feature selection to reduce dimensionality</li>
            <li>Non-linear models to capture complex relationships</li>
            <li>Time series specific models (ARIMA, LSTM)</li>
        </ul>
    </div>
    """
    content += interpretation
    
    html_content = html_manager.template.format(
        title="KORD Linear Regression: Temperature Prediction Analysis",
        content=content,
        additional_js=""
    )
    
    html_filename = "5-linear_temp_regression.html"
    output_path = html_manager.save_section_html("KORD_Self_Regression", html_content, html_filename)
    return html_content, output_path

def create_latex_report(all_results, output_dir):
    """
    Create a LaTeX report containing all temperature predictions
    Args:
        all_results: Dictionary containing results for all prediction types
        output_dir: Path to the output directory
    """
    model_desc = latex_section("KORD Linear Regression: Temperature Prediction Analysis",
        "This analysis implements multivariate time series regression models to predict temperature at Chicago O'Hare International Airport (KORD) across multiple time horizons.\\\n" +
        latex_list([
            "Base Model: Multivariate Linear Regression",
            "Feature Engineering: Time-lagged features for all numeric variables",
            "Prediction Horizons: Multiple time windows (1h, 24h, 5d, 5d avg, 30d avg)",
            "Train/Test Split: 80/20",
            "Random State: 42 (for reproducibility)",
            "Validation: Standard train-test split"
        ])
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
These multivariate regression models capture the complex relationships between various weather parameters and temperature across different time horizons. The models:\\
\\begin{itemize}
  \item Consider the impact of all available weather parameters
  \item Account for changes in wind direction through the wind\_dir\_delta feature
  \item Provide insights into which factors most influence temperature at different time scales
\\end{itemize}
Future improvements could include:\\
\\begin{itemize}
  \item Feature selection to reduce dimensionality
  \item Non-linear models to capture complex relationships
  \item Time series specific models (ARIMA, LSTM)
\\end{itemize}
"""
    )
    latex_content = model_desc + "\n".join(prediction_sections) + interpretation
    latex_output_path = output_dir / '5-linear_temp_regression.tex'
    save_latex_file(latex_content, latex_output_path)
    logger.info(f"LaTeX linear regression report created: {latex_output_path}")
    return latex_output_path

def main():
    try:
        output_dir = Path(__file__).parent / 'outputs'
        output_dir.mkdir(exist_ok=True)
        manager = HTMLManager()
        manager.register_section("KORD_Self_Regression", Path(__file__).parent)
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
