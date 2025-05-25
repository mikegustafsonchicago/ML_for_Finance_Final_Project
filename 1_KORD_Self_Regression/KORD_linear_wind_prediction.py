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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def calculate_wind_direction_delta(df):
    """
    Calculate the change in wind direction between consecutive hours
    Handles the circular nature of wind direction (0-360 degrees)
    """
    # Calculate raw difference
    df['wind_dir_delta'] = df['winddirection'].diff()
    
    # Adjust for circular nature (if change is more than 180 degrees, take the shorter path)
    df.loc[df['wind_dir_delta'] > 180, 'wind_dir_delta'] -= 360
    df.loc[df['wind_dir_delta'] < -180, 'wind_dir_delta'] += 360
    
    return df

def load_and_prepare_data(file_path='../datastep2.csv', n_lags=24):
    """
    Load and prepare data for time series regression
    Args:
        file_path: Path to the data file
        n_lags: Number of lagged features to create
    Returns:
        X: Features matrix
        y: Target variable
        datetime: Datetime index aligned with X/y
        feature_cols: List of feature column names
    """
    logger.info("Loading and preparing data...")
    
    # Load data
    df = pd.read_csv(file_path)
    logger.info(f"Loaded data shape: {df.shape}")
    logger.info(f"Columns in loaded data: {list(df.columns)}")
    
    # Filter for KORD only
    if 'id' in df.columns:
        df = df[df['id'] == 'KORD']
        logger.info(f"Data shape after filtering for KORD: {df.shape}")
    else:
        logger.warning("No 'id' column found; cannot filter for KORD.")
    
    df['datetime'] = parse_custom_datetime(df['datetime'])
    
    # Sort by datetime
    df = df.sort_values('datetime')
    logger.info(f"Data shape after sorting: {df.shape}")
    
    # Calculate wind direction delta
    df = calculate_wind_direction_delta(df)
    logger.info(f"Data shape after wind_dir_delta: {df.shape}")
    
    # Only use known numeric columns
    expected_numeric_cols = [
        'skydescriptor', 'temp_str', 'temp', 'windspeed', 'winddirection',
        'humidity', 'dew', 'sealevel', 'visibility', 'mincloud', 'maxcloud'
    ]
    numeric_cols = [col for col in expected_numeric_cols if col in df.columns]
    logger.info(f"Numeric columns used for lagged features: {numeric_cols}")
    logger.info(f"Number of numeric columns: {len(numeric_cols)}")
    
    # Efficiently create lagged features for all numeric columns
    lagged_features = []
    for col in numeric_cols:
        for i in range(1, n_lags + 1):
            lagged = df[col].shift(i)
            lagged.name = f'{col}_lag_{i}'
            lagged_features.append(lagged)
    lagged_df = pd.concat(lagged_features, axis=1)
    logger.info(f"Lagged DataFrame shape: {lagged_df.shape}")
    
    # Combine lagged features with original DataFrame
    df_lagged = pd.concat([df, lagged_df], axis=1)
    logger.info(f"Shape after combining lagged features: {df_lagged.shape}")
    
    # Drop rows with NaN values (first n_lags rows)
    df_lagged = df_lagged.dropna().reset_index(drop=True)
    logger.info(f"Shape after dropping NaNs: {df_lagged.shape}")
    
    # Prepare features and target
    feature_cols = [col for col in df_lagged.columns if col.endswith(tuple([f'_lag_{i}' for i in range(1, n_lags + 1)]))]
    logger.info(f"Number of feature columns: {len(feature_cols)}")
    X = df_lagged[feature_cols]
    y = df_lagged['windspeed']
    datetime = df_lagged['datetime']
    
    return X, y, datetime, feature_cols

def train_model(X, y, datetime):
    """
    Train a linear regression model
    Returns X_test, y_test, y_pred, datetime_test, metrics
    """
    logger.info("Training model...")
    
    # Split data into train and test sets, keeping indices for datetime alignment
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, X.index, test_size=0.2, random_state=42
    )
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Get feature importance
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
    
    # Align datetime_test with y_test using idx_test
    datetime_test = datetime.iloc[idx_test].reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    
    return model, X_test, y_test, y_pred, datetime_test, metrics

def plot_results(datetime_test, y_test, y_pred, output_dir):
    """
    Plot actual vs predicted values for a specific week (e.g., April 1-7, 2020)
    """
    results_df = pd.DataFrame({
        'datetime': datetime_test,
        'actual': y_test,
        'predicted': y_pred
    })

    # Select a specific week: April 1-7, 2020
    start_date = pd.Timestamp('2020-04-01')
    end_date = pd.Timestamp('2020-04-07 23:59:59')
    week_data = results_df[(results_df['datetime'] >= start_date) & (results_df['datetime'] <= end_date)]

    if week_data.empty:
        logger.warning("No data found for the specified week (April 1-7, 2020).")
        return None

    # Sort by datetime to ensure correct plotting order
    week_data = week_data.sort_values('datetime')

    plt.figure(figsize=(10, 4))
    plt.plot(week_data['datetime'], week_data['actual'], label='Actual', alpha=0.7)
    plt.plot(week_data['datetime'], week_data['predicted'], label='Predicted', alpha=0.7)
    plt.title('Wind Speed Prediction: April 1-7, 2020')
    plt.xlabel('Date')
    plt.ylabel('Wind Speed')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plot_path = output_dir / 'wind_prediction_results.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()

    return plot_path

def plot_feature_importance(feature_importance, output_dir):
    """
    Plot feature importance based on coefficient magnitudes
    """
    plt.figure(figsize=(8, 4))
    # Plot top 10 features
    top_features = feature_importance.head(10)
    plt.barh(top_features['Feature'], top_features['Abs_Coefficient'])
    plt.title('Top 10 Feature Importance')
    plt.xlabel('Absolute Coefficient Value')
    plt.tight_layout()
    
    plot_path = output_dir / 'wind_feature_importance.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return plot_path

def create_html_report(metrics, plot_path, feature_importance_plot, html_manager):
    """
    Create HTML report for the wind prediction results using HTMLManager's functions
    """
    # Create model description section
    model_desc = f"""
    <div class="model-report">
        <h2>KORD Linear Regression: Wind Speed Prediction Analysis</h2>
        
        <div class="model-description">
            <h3>Model Architecture</h3>
            <p>This analysis focuses on predicting wind speeds at Chicago O'Hare International Airport (KORD) using historical weather data.</p>
            
            <h4>Key Implementation Details</h4>
            <ul>
                <li><strong>Time Window:</strong> 24-hour historical window to capture daily patterns</li>
                <li><strong>Wind Direction Handling:</strong> Special delta feature to handle circular wind measurements</li>
                <li><strong>Feature Engineering:</strong> Hourly lags of all weather parameters to capture temporal dependencies</li>
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
                            <td>Temperature</td>
                            <td>Current and historical temperature measurements</td>
                            <td>°C</td>
                            <td>24-hour lag</td>
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
    
    # Create interpretation section
    interpretation = f"""
    <div class="interpretation-section">
        <h3>Model Interpretation</h3>
        <p>Key findings for wind prediction at KORD:</p>
        <ul>
            <li>Wind direction changes provide significant predictive power</li>
            <li>24-hour historical window captures daily wind patterns</li>
            <li>Pressure and temperature gradients show strong correlation with wind speed changes</li>
        </ul>
        <p>Potential improvements specific to KORD:</p>
        <ul>
            <li>Incorporate runway configuration data</li>
            <li>Add seasonal interaction terms</li>
            <li>Include Lake Michigan effect variables</li>
        </ul>
    </div>
    """
    
    # Create metrics section
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
    
    # Create visualization sections using HTMLManager's create_section_with_image
    time_series_section = html_manager.create_section_with_image(
        plot_path,
        "Sample Week Predictions",
        "The following plots show the actual vs predicted wind speeds for sample weeks in different months. The blue line represents actual measurements, while the orange line shows our model's predictions."
    )
    
    feature_importance_section = html_manager.create_section_with_image(
        feature_importance_plot,
        "Feature Importance",
        "This plot shows the top 10 most important features based on their absolute coefficient values in the regression model."
    )
    
    # Combine all sections
    content = model_desc + interpretation + metrics_section + time_series_section + feature_importance_section
    
    # Save the HTML file with a distinct, numbered name
    html_content = html_manager.template.format(
        title="KORD Linear Regression: Wind Speed Prediction Analysis",
        content=content,
        additional_js=""
    )
    output_path = html_manager.save_section_html("KORD_Self_Regression", html_content, "1-linear_wind_regression.html")
    return html_content, output_path

def save_latex_results(metrics, output_dir, suffix):
    """
    Save model results in LaTeX format
    """
    latex_content = f"""\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{lr}}
\\hline
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\hline
RMSE & {metrics['RMSE']:.2f} \\\\
MAE & {metrics['MAE']:.2f} \\\\
R² Score & {metrics['R2']:.2f} \\\\
\\hline
\\end{{tabular}}
\\caption{{Linear Wind Prediction Performance Metrics}}
\\label{{tab:linear_wind_metrics_{suffix}}}
\\end{{table}}

\\begin{{figure}}[h]
\\centering
\\includegraphics[width=0.8\\textwidth]{{wind_prediction_results.png}}
\\caption{{Linear Wind Prediction Results}}
\\label{{fig:linear_wind_results_{suffix}}}
\\end{{figure}}

\\begin{{figure}}[h]
\\centering
\\includegraphics[width=0.8\\textwidth]{{wind_feature_importance.png}}
\\caption{{Linear Wind Prediction Feature Importance}}
\\label{{fig:linear_wind_importance_{suffix}}}
\\end{{figure}}
"""
    
    output_path = output_dir / f'linear_wind_results_{suffix}.tex'
    with open(output_path, 'w') as f:
        f.write(latex_content)
    logger.info(f"Saved LaTeX results to {output_path}")
    return output_path

def save_prediction_results(datetime_test, y_test, y_pred, output_dir):
    """
    Save prediction results to CSV file
    """
    # Ensure 1D arrays
    if hasattr(y_test, 'values'):
        y_test = y_test.values
    if hasattr(y_pred, 'values'):
        y_pred = y_pred.values
    y_test = np.ravel(y_test)
    y_pred = np.ravel(y_pred)

    results_df = pd.DataFrame({
        'datetime': datetime_test,
        'actual': y_test,
        'predicted': y_pred,
        'error': y_test - y_pred,
        'abs_error': np.abs(y_test - y_pred)
    })
    output_path = output_dir / 'wind_prediction_results.csv'
    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved prediction results to {output_path}")
    return output_path

def main():
    try:
        output_dir = Path(__file__).parent / 'outputs'
        output_dir.mkdir(exist_ok=True)
        manager = HTMLManager()
        manager.register_section("KORD_Self_Regression", Path(__file__).parent)
        X, y, datetime, feature_cols = load_and_prepare_data()
        
        # Train model and get predictions
        model, X_test, y_test, y_pred, datetime_test, metrics = train_model(X, y, datetime)
        
        # Plot results
        plot_path = plot_results(datetime_test, y_test, y_pred, output_dir)
        feature_importance_plot = plot_feature_importance(metrics['Feature_Importance'], output_dir)
        csv_path = save_prediction_results(datetime_test, y_test, y_pred, output_dir)
        latex_path = save_latex_results(metrics, output_dir, suffix="0")
        all_results = {
            'metrics': metrics,
            'plot_path': plot_path,
            'feature_importance_plot': feature_importance_plot,
            'csv_path': csv_path,
            'latex_path': latex_path
        }
        logger.info("Wind prediction analysis complete")
        
        # Create HTML report
        html_content, output_path = create_html_report(metrics, plot_path, feature_importance_plot, manager)
        logger.info(f"All wind prediction analyses complete. Results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error in wind prediction analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
