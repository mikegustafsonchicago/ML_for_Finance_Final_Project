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

def load_and_prepare_data(file_path='../datastep2.csv', n_lags=24):
    """
    Load and prepare data for time series regression
    Args:
        file_path: Path to the data file
        n_lags: Number of lagged features to create
    Returns:
        X: Features matrix
        y: Target variable
    """
    logger.info("Loading and preparing data...")
    
    # Load data
    df = pd.read_csv(file_path)
    df['datetime'] = parse_custom_datetime(df['datetime'])
    
    # Sort by datetime
    df = df.sort_values('datetime')
    
    # Create lagged features
    for i in range(1, n_lags + 1):
        df[f'windspeed_lag_{i}'] = df['windspeed'].shift(i)
    
    # Drop rows with NaN values (first n_lags rows)
    df = df.dropna()
    
    # Prepare features and target
    feature_cols = [f'windspeed_lag_{i}' for i in range(1, n_lags + 1)]
    X = df[feature_cols]
    y = df['windspeed']
    
    return X, y, df['datetime']

def train_model(X, y):
    """
    Train a linear regression model
    """
    logger.info("Training model...")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    return model, X_test, y_test, y_pred, metrics

def plot_results(datetime_test, y_test, y_pred, output_dir):
    """
    Plot actual vs predicted values
    """
    # Time series plot
    plt.figure(figsize=(8, 3))  # Further reduced size
    plt.plot(datetime_test, y_test, label='Actual', alpha=0.7)
    plt.plot(datetime_test, y_pred, label='Predicted', alpha=0.7)
    plt.title('Wind Speed: Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Wind Speed')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the time series plot
    plot_path = output_dir / 'wind_prediction_results.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')  # Further reduced DPI
    plt.close()
    
    return plot_path

def plot_regression_relationship(X, y, output_dir):
    """
    Plot the relationship between current and next hour's wind speed
    """
    plt.figure(figsize=(6, 4))  # Further reduced size
    plt.scatter(X['windspeed_lag_1'], y, alpha=0.5, s=5)  # Smaller points
    
    # Add regression line
    z = np.polyfit(X['windspeed_lag_1'], y, 1)
    p = np.poly1d(z)
    plt.plot(X['windspeed_lag_1'], p(X['windspeed_lag_1']), "r--", alpha=0.8)
    
    plt.title('Current vs Next Hour Wind Speed')
    plt.xlabel('Current Hour Wind Speed')
    plt.ylabel('Next Hour Wind Speed')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Save the regression plot
    plot_path = output_dir / 'wind_regression_relationship.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')  # Further reduced DPI
    plt.close()
    
    return plot_path

def create_html_report(metrics, plot_path, regression_plot_path, html_manager):
    """
    Create HTML report for the wind prediction results using HTMLManager's functions
    """
    # Create model description section
    model_desc = f"""
    <div class="section" style="border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin: 20px 0; background-color: #f9f9f9;">
        <h2 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">Wind Speed Prediction Analysis</h2>
        
        <div class="model-description" style="background-color: #fff; padding: 20px; border-radius: 6px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h3 style="color: #2c3e50;">Model Architecture</h3>
            <p>This analysis implements a time series regression model to predict wind speed at Chicago O'Hare International Airport (KORD).</p>
            
            <h4 style="color: #34495e; margin-top: 15px;">Model Type</h4>
            <ul style="list-style-type: none; padding-left: 0;">
                <li>• <strong>Base Model:</strong> Linear Regression</li>
                <li>• <strong>Feature Engineering:</strong> Time-lagged features</li>
                <li>• <strong>Prediction Target:</strong> Next hour's wind speed</li>
            </ul>
            
            <h4 style="color: #34495e; margin-top: 15px;">Feature Details</h4>
            <ul style="list-style-type: none; padding-left: 0;">
                <li>• <strong>Input Features:</strong> Previous 24 hours of wind speed measurements</li>
                <li>• <strong>Feature Window:</strong> Rolling 24-hour window</li>
                <li>• <strong>Time Resolution:</strong> Hourly measurements</li>
            </ul>
            
            <h4 style="color: #34495e; margin-top: 15px;">Training Configuration</h4>
            <ul style="list-style-type: none; padding-left: 0;">
                <li>• <strong>Train/Test Split:</strong> 80/20</li>
                <li>• <strong>Random State:</strong> 42 (for reproducibility)</li>
                <li>• <strong>Validation:</strong> Standard train-test split</li>
            </ul>
        </div>
    </div>
    """
    
    # Create interpretation section
    interpretation = f"""
    <div class="section" style="background-color: #fff; padding: 20px; border-radius: 6px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h3 style="color: #2c3e50;">Model Interpretation</h3>
        <p>This basic linear regression model serves as a baseline for wind speed prediction. The model:</p>
        <ul>
            <li>Captures the linear relationship between past and future wind speeds</li>
            <li>Provides a foundation for comparing more complex models</li>
            <li>Helps identify patterns in wind speed persistence</li>
        </ul>
        <p>Future improvements could include:</p>
        <ul>
            <li>Incorporating additional weather features (temperature, pressure, etc.)</li>
            <li>Implementing more sophisticated time series models (ARIMA, LSTM)</li>
            <li>Adding seasonal decomposition to handle periodic patterns</li>
        </ul>
    </div>
    """
    
    # Create metrics section
    metrics_section = f"""
    <div class="section" style="background-color: #fff; padding: 20px; border-radius: 6px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h3 style="color: #2c3e50;">Model Performance</h3>
        <div class="metrics-table">
            <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
                <tr style="background-color: #f8f9fa;">
                    <th style="text-align: left; padding: 12px; border-bottom: 2px solid #dee2e6;">Metric</th>
                    <th style="text-align: right; padding: 12px; border-bottom: 2px solid #dee2e6;">Value</th>
                </tr>
                <tr>
                    <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">Root Mean Squared Error (RMSE)</td>
                    <td style="text-align: right; padding: 12px; border-bottom: 1px solid #dee2e6;">{metrics['RMSE']:.2f}</td>
                </tr>
                <tr style="background-color: #f8f9fa;">
                    <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">Mean Absolute Error (MAE)</td>
                    <td style="text-align: right; padding: 12px; border-bottom: 1px solid #dee2e6;">{metrics['MAE']:.2f}</td>
                </tr>
                <tr>
                    <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">R² Score</td>
                    <td style="text-align: right; padding: 12px; border-bottom: 1px solid #dee2e6;">{metrics['R2']:.2f}</td>
                </tr>
            </table>
        </div>
    </div>
    """
    
    # Create visualization sections using HTMLManager's create_section_with_image
    time_series_section = html_manager.create_section_with_image(
        plot_path,
        "Time Series Prediction",
        "The following plot shows the actual vs predicted wind speeds for the test set. The blue line represents actual measurements, while the orange line shows our model's predictions."
    )
    
    regression_section = html_manager.create_section_with_image(
        regression_plot_path,
        "Regression Relationship",
        "This plot shows the relationship between current hour's wind speed and next hour's wind speed. The red dashed line represents the linear regression fit."
    )
    
    # Combine all sections
    content = model_desc + interpretation + metrics_section + time_series_section + regression_section
    
    # Save the HTML file
    output_path = Path(__file__).parent / 'outputs' / 'wind_prediction.html'
    html_manager.save_section_html("KORD_Self_Regression", content, "wind_prediction.html")
    
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
        X, y, datetime = load_and_prepare_data()
        
        # Train model and get predictions
        model, X_test, y_test, y_pred, metrics = train_model(X, y)
        
        # Plot results
        plot_path = plot_results(datetime[-len(y_test):], y_test, y_pred, output_dir)
        
        # Plot regression relationship
        regression_plot_path = plot_regression_relationship(X, y, output_dir)
        
        # Create HTML report
        html_path = create_html_report(metrics, plot_path, regression_plot_path, manager)
        
        logger.info(f"Analysis complete. Results saved to {html_path}")
        
    except Exception as e:
        logger.error(f"Error in wind prediction analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
