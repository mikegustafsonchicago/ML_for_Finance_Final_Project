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
class TempPredictionNN(nn.Module):
    def __init__(self, input_size):
        super(TempPredictionNN, self).__init__()
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
class TempDataset(Dataset):
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
    logger.info("Training neural network model...")
    
    # Split data into train and test sets, keeping indices for datetime alignment
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, X.index, test_size=0.2, random_state=42
    )
    logger.info(f"Shapes - X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

    # Create datasets and dataloaders
    train_dataset = TempDataset(X_train.values, y_train.values)
    test_dataset = TempDataset(X_test.values, y_test.values)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TempPredictionNN(input_size=X_train.shape[1]).to(device)
    
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
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
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

def plot_results(datetime_test, y_test, y_pred, output_dir, suffix):
    # Ensure all arrays are 1-dimensional
    y_test = y_test.flatten() if hasattr(y_test, 'flatten') else y_test
    y_pred = y_pred.flatten() if hasattr(y_pred, 'flatten') else y_pred
    
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
    plot_path = output_dir / f'6-3-{suffix}-neural_net_temp_prediction_results.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    return plot_path

def plot_feature_importance(feature_importance, output_dir, suffix):
    plt.figure(figsize=(8, 4))
    top_features = feature_importance.head(10)
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.title(f'Top 10 Feature Importance (6-3-{suffix})')
    plt.xlabel('Feature Importance Score')
    plt.tight_layout()
    plot_path = output_dir / f'6-3-{suffix}-neural_net_temp_feature_importance.png'
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
        <h2>KORD Neural Network: Temperature Prediction Analysis</h2>
        <div class="model-description">
            <h3>Model Architecture</h3>
            <p>This analysis implements deep neural network models to predict temperature at Chicago O'Hare International Airport (KORD) across multiple time horizons.</p>
            <h4>Model Type</h4>
            <ul>
                <li><strong>Base Model:</strong> Deep Neural Network (PyTorch)</li>
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
            "This plot shows the top 10 most important features based on their weights in the first layer of the neural network."
        )
        
        section += "</div>"
        prediction_sections.append(section)
    
    # Combine all sections
    content = model_desc + "\n".join(prediction_sections)
    
    # Add interpretation section
    interpretation = """
    <div class="interpretation-section">
        <h3>Model Interpretation</h3>
        <p>These deep neural network models capture complex non-linear relationships between various weather parameters and temperature across different time horizons. The models:</p>
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
    content += interpretation
    
    html_content = html_manager.template.format(
        title="KORD Neural Network: Temperature Prediction Analysis",
        content=content,
        additional_js=""
    )
    
    html_filename = "6-3-neural_net_temp_regression.html"
    output_path = html_manager.save_section_html("KORD_Self_Regression", html_content, html_filename)
    return html_content, output_path

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
        
    except Exception as e:
        logger.error(f"Error in temperature prediction analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
