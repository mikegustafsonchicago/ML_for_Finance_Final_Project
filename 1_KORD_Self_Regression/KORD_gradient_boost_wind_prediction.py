import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import sys
import xgboost as xgb
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
    Train an XGBoost model
    Returns X_test, y_test, y_pred, datetime_test, metrics
    """
    logger.info("Training XGBoost model...")
    
    # Split data into train and test sets, keeping indices for datetime alignment
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, X.index, test_size=0.2, random_state=42
    )
    
    # Define XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.1,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 100,
        'random_state': 42
    }
    
    # Train model
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train
    )
    
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
        'Importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
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

    # Section-specific plot name
    plot_path = output_dir / '2-gradient_boosted_wind_regression_results.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()

    return plot_path

def plot_feature_importance(feature_importance, output_dir):
    """
    Plot feature importance based on XGBoost feature importance scores
    """
    plt.figure(figsize=(8, 4))
    # Plot top 10 features
    top_features = feature_importance.head(10)
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.title('Top 10 Feature Importance')
    plt.xlabel('Feature Importance Score')
    plt.tight_layout()
    
    # Section-specific plot name
    plot_path = output_dir / '2-gradient_boosted_feature_importance.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return plot_path

def create_html_report(metrics, plot_path, feature_importance_plot, html_manager):
    """
    Create HTML report for the wind prediction results using HTMLManager's functions
    """
    # Create model description section
    model_desc = f"""
    <div class="section" style="border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin: 20px 0; background-color: #f9f9f9;">
        <h2 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">Wind Speed Prediction Analysis</h2>
        
        <div class="model-description" style="background-color: #fff; padding: 20px; border-radius: 6px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h3 style="color: #2c3e50;">Model Architecture</h3>
            <p>This analysis implements a multivariate time series regression model using XGBoost to predict wind speed at Chicago O'Hare International Airport (KORD).</p>
            
            <h4 style="color: #34495e; margin-top: 15px;">Model Type</h4>
            <ul style="list-style-type: none; padding-left: 0;">
                <li>• <strong>Base Model:</strong> XGBoost (Gradient Boosting)</li>
                <li>• <strong>Feature Engineering:</strong> Time-lagged features for all numeric variables</li>
                <li>• <strong>Prediction Target:</strong> Next hour's wind speed</li>
            </ul>
            
            <h4 style="color: #34495e; margin-top: 15px;">Feature Details</h4>
            <div class="data-list">
                <table>
                    <thead>
                        <tr>
                            <th>Feature Name</th>
                            <th>Description</th>
                            <th>Time Window</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Wind Speed</td>
                            <td>Current and historical wind speed measurements</td>
                            <td>24-hour lag</td>
                        </tr>
                        <tr>
                            <td>Wind Direction</td>
                            <td>Current and historical wind direction measurements</td>
                            <td>24-hour lag</td>
                        </tr>
                        <tr>
                            <td>Wind Direction Delta</td>
                            <td>Change in wind direction between consecutive hours</td>
                            <td>Current</td>
                        </tr>
                        <tr>
                            <td>Temperature</td>
                            <td>Current and historical temperature measurements</td>
                            <td>24-hour lag</td>
                        </tr>
                        <tr>
                            <td>Humidity</td>
                            <td>Current and historical humidity measurements</td>
                            <td>24-hour lag</td>
                        </tr>
                        <tr>
                            <td>Dew Point</td>
                            <td>Current and historical dew point measurements</td>
                            <td>24-hour lag</td>
                        </tr>
                        <tr>
                            <td>Sea Level Pressure</td>
                            <td>Current and historical pressure measurements</td>
                            <td>24-hour lag</td>
                        </tr>
                        <tr>
                            <td>Visibility</td>
                            <td>Current and historical visibility measurements</td>
                            <td>24-hour lag</td>
                        </tr>
                        <tr>
                            <td>Cloud Coverage</td>
                            <td>Current and historical min/max cloud coverage</td>
                            <td>24-hour lag</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <h4 style="color: #34495e; margin-top: 15px;">Training Configuration</h4>
            <ul style="list-style-type: none; padding-left: 0;">
                <li>• <strong>Train/Test Split:</strong> 80/20</li>
                <li>• <strong>Random State:</strong> 42 (for reproducibility)</li>
                <li>• <strong>Validation:</strong> Early stopping with 10 rounds patience</li>
                <li>• <strong>Learning Rate:</strong> 0.1</li>
                <li>• <strong>Max Depth:</strong> 6</li>
                <li>• <strong>Number of Trees:</strong> 100</li>
            </ul>
        </div>
    </div>
    """
    
    # Create interpretation section
    interpretation = f"""
    <div class="section" style="background-color: #fff; padding: 20px; border-radius: 6px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h3 style="color: #2c3e50;">Model Interpretation</h3>
        <p>This XGBoost model captures complex non-linear relationships between various weather parameters and wind speed. The model:</p>
        <ul>
            <li>Learns complex patterns through gradient boosting</li>
            <li>Automatically handles feature interactions</li>
            <li>Provides robust feature importance rankings</li>
            <li>Uses early stopping to prevent overfitting</li>
        </ul>
        <p>Future improvements could include:</p>
        <ul>
            <li>Hyperparameter tuning using grid search or Bayesian optimization</li>
            <li>Feature selection based on importance scores</li>
            <li>Ensemble methods combining multiple models</li>
        </ul>
    </div>
    """
    
    # Create metrics section
    metrics_section = f"""
    <div class="section" style="background-color: #fff; padding: 20px; border-radius: 6px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h3 style="color: #2c3e50;">Model Performance</h3>
        <div class="data-list">
            <table>
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
        "This plot shows the top 10 most important features based on their importance scores in the XGBoost model."
    )
    
    # Combine all sections
    content = model_desc + interpretation + metrics_section + time_series_section + feature_importance_section
    
    # Save the HTML file with a distinct, numbered name
    output_path = Path(__file__).parent / 'outputs' / '2-gradient_boosted_wind_regression.html'
    html_manager.save_section_html("KORD_Self_Regression", content, "2-gradient_boosted_wind_regression.html")
    
    return output_path

def main():
    try:
        # Create outputs directory
        output_dir = Path(__file__).parent / 'outputs'
        output_dir.mkdir(exist_ok=True)
        
        # Create HTML manager
        manager = HTMLManager()
        manager.register_section("KORD_Self_Regression", Path(__file__).parent)
        
        # Load and prepare data
        X, y, datetime, feature_cols = load_and_prepare_data()
        
        # Train model and get predictions
        model, X_test, y_test, y_pred, datetime_test, metrics = train_model(X, y, datetime)
        
        # Plot results
        plot_path = plot_results(datetime_test, y_test, y_pred, output_dir)
        
        # Plot feature importance
        feature_importance_plot = plot_feature_importance(metrics['Feature_Importance'], output_dir)
        
        # Create HTML report
        html_path = create_html_report(metrics, plot_path, feature_importance_plot, manager)
        
        logger.info(f"Analysis complete. Results saved to {html_path}")
        
    except Exception as e:
        logger.error(f"Error in wind prediction analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
