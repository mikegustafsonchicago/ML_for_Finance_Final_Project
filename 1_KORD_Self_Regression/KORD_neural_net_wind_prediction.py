import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

# Define the neural network architecture
class WindPredictionNN(nn.Module):
    def __init__(self, input_size):
        super(WindPredictionNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

# Custom Dataset class
class WindDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        # Ensure y is always 2D: shape [N, 1]
        if len(y.shape) == 1:
            self.y = torch.FloatTensor(y).unsqueeze(1)
        else:
            self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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
    Train a neural network model
    Returns X_test, y_test, y_pred, datetime_test, metrics
    """
    logger.info("Training neural network model...")
    
    # Split data into train and test sets, keeping indices for datetime alignment
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, X.index, test_size=0.2, random_state=42
    )
    logger.info(f"Shapes - X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

    # Create datasets and dataloaders
    train_dataset = WindDataset(X_train.values, y_train.values)
    test_dataset = WindDataset(X_test.values, y_test.values)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WindPredictionNN(input_size=X_train.shape[1]).to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    n_epochs = 50
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch).item()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, Val Loss = {val_loss/len(test_loader):.4f}")
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X_test.values).to(device)).cpu().numpy()
    
    logger.info(f"Diagnostics after prediction: y_pred shape: {y_pred.shape}, y_test shape: {y_test.shape}")
    logger.info(f"Sample y_pred: {y_pred[:5].flatten()}, Sample y_test: {y_test.values[:5]}")
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"Metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    
    # Get feature importance (using absolute weights from first layer)
    first_layer_weights = model.network[0].weight.data.abs().mean(dim=0).cpu().numpy()
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': first_layer_weights
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
    logger.info("Starting plot_results...")
    # Ensure all arrays are 1-dimensional
    if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
        y_pred = y_pred.ravel()
    if isinstance(y_test, (np.ndarray, pd.Series)) and hasattr(y_test, 'values') and y_test.ndim > 1:
        y_test = y_test.values.ravel()
    if isinstance(datetime_test, (np.ndarray, pd.Series)) and hasattr(datetime_test, 'values') and datetime_test.ndim > 1:
        datetime_test = datetime_test.values.ravel()

    logger.info(f"Diagnostics in plot_results: datetime_test len: {len(datetime_test)}, y_test len: {len(y_test)}, y_pred len: {len(y_pred)}")

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
        logger.error("No data found for the specified week (April 1-7, 2020). Plot will not be generated.")
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
    plot_path = output_dir / '4-neural_net_wind_regression_results.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()

    logger.info(f"Plot saved to {plot_path}")
    return plot_path

def plot_feature_importance(feature_importance, output_dir):
    """
    Plot feature importance based on neural network weights
    """
    plt.figure(figsize=(8, 4))
    # Plot top 10 features
    top_features = feature_importance.head(10)
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.title('Top 10 Feature Importance')
    plt.xlabel('Feature Importance Score')
    plt.tight_layout()
    
    # Section-specific plot name
    plot_path = output_dir / '4-neural_net_feature_importance.png'
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
        <h2>KORD Neural Network: Wind Speed Prediction Analysis</h2>
        
        <div class="model-description">
            <h3>Model Architecture</h3>
            <p>This analysis implements a deep neural network model to predict wind speed at Chicago O'Hare International Airport (KORD).</p>
            
            <h4>Model Type</h4>
            <ul>
                <li><strong>Base Model:</strong> Deep Neural Network (PyTorch)</li>
                <li><strong>Feature Engineering:</strong> Time-lagged features for all numeric variables</li>
                <li><strong>Prediction Target:</strong> Next hour's wind speed</li>
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
                <li><strong>Network Architecture:</strong> 4-layer neural network (128→64→32→1)</li>
                <li><strong>Activation:</strong> ReLU</li>
                <li><strong>Dropout:</strong> 0.2</li>
                <li><strong>Optimizer:</strong> Adam (lr=0.001)</li>
                <li><strong>Batch Size:</strong> 32</li>
                <li><strong>Early Stopping:</strong> 10 epochs patience</li>
            </ul>
        </div>
    </div>
    """
    
    # Create interpretation section
    interpretation = f"""
    <div class="interpretation-section">
        <h3>Model Interpretation</h3>
        <p>This deep neural network model captures complex non-linear relationships between various weather parameters and wind speed. The model:</p>
        <ul>
            <li>Learns hierarchical features through multiple layers</li>
            <li>Uses dropout for regularization to prevent overfitting</li>
            <li>Implements early stopping to optimize training duration</li>
            <li>Provides feature importance based on first layer weights</li>
        </ul>
        <p>Future improvements could include:</p>
        <ul>
            <li>LSTM layers to better capture temporal dependencies</li>
            <li>Hyperparameter tuning using grid search or Bayesian optimization</li>
            <li>Ensemble methods combining multiple neural network architectures</li>
            <li>Attention mechanisms to focus on relevant time steps</li>
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
        "This plot shows the top 10 most important features based on their weights in the first layer of the neural network."
    )
    
    # Combine all sections
    content = model_desc + interpretation + metrics_section + time_series_section + feature_importance_section
    
    # Save the HTML file with a distinct, numbered name using the template (links stylesheet)
    html_content = html_manager.template.format(
        title="KORD Neural Network: Wind Speed Prediction Analysis",
        content=content,
        additional_js=""
    )
    output_path = html_manager.save_section_html("KORD_Self_Regression", html_content, "4-neural_net_wind_regression.html")
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
\\caption{{Neural Network Wind Prediction Performance Metrics}}
\\label{{tab:neural_net_wind_metrics_{suffix}}}
\\end{{table}}

\\begin{{figure}}[h]
\\centering
\\includegraphics[width=0.8\\textwidth]{{4-neural_net_wind_regression_results.png}}
\\caption{{Neural Network Wind Prediction Results}}
\\label{{fig:neural_net_wind_results_{suffix}}}
\\end{{figure}}

\\begin{{figure}}[h]
\\centering
\\includegraphics[width=0.8\\textwidth]{{4-neural_net_wind_feature_importance.png}}
\\caption{{Neural Network Wind Prediction Feature Importance}}
\\label{{fig:neural_net_wind_importance_{suffix}}}
\\end{{figure}}
"""
    
    output_path = output_dir / f'4-neural_net_wind_results_{suffix}.tex'
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
    output_path = output_dir / '4-neural_net_wind_prediction_results.csv'
    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved prediction results to {output_path}")
    return output_path

def main():
    try:
        logger.info("Starting main wind prediction analysis pipeline...")
        # Create outputs directory
        output_dir = Path(__file__).parent / 'outputs'
        output_dir.mkdir(exist_ok=True)
        
        # Create HTML manager
        manager = HTMLManager()
        manager.register_section("KORD_Self_Regression", Path(__file__).parent)
        
        # Load and prepare data
        logger.info("Loading and preparing data...")
        X, y, datetime, feature_cols = load_and_prepare_data()
        logger.info("Data loaded and prepared.")
        
        # Train model and get predictions
        logger.info("Training model...")
        model, X_test, y_test, y_pred, datetime_test, metrics = train_model(X, y, datetime)
        logger.info("Model trained and predictions made.")
        
        # Plot results
        logger.info("Plotting results...")
        plot_path = plot_results(datetime_test, y_test, y_pred, output_dir)
        if plot_path is None:
            logger.error("Plotting failed: No plot was generated for the specified week. Exiting.")
            raise RuntimeError("No data available for plotting for the specified week (April 1-7, 2020).")
        
        # Plot feature importance
        logger.info("Plotting feature importance...")
        feature_importance_plot = plot_feature_importance(metrics['Feature_Importance'], output_dir)
        if feature_importance_plot is None:
            logger.error("Feature importance plot was not generated. Exiting.")
            raise RuntimeError("Feature importance plot was not generated.")
        
        # Create HTML report
        logger.info("Creating HTML report...")
        html_content, output_path = create_html_report(metrics, plot_path, feature_importance_plot, manager)
        logger.info(f"Analysis complete. Results saved to {output_path}")
        
        # Save LaTeX results
        logger.info("Saving LaTeX results...")
        latex_path = save_latex_results(metrics, output_dir, suffix="0")
        logger.info(f"LaTeX results saved to {latex_path}")
        
        # Save prediction results
        logger.info("Saving prediction results...")
        prediction_path = save_prediction_results(datetime_test, y_test, y_pred, output_dir)
        logger.info(f"Prediction results saved to {prediction_path}")
        
    except Exception as e:
        logger.error(f"Error in wind prediction analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
