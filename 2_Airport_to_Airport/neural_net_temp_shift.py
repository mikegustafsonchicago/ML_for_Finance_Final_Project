import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import sys
from tqdm import tqdm
import time
from datetime import datetime
sys.path.append('..')
from html_manager import HTMLManager
from Utils.data_formatter import parse_custom_datetime
from Utils.latex_utility import save_latex_file, latex_section, latex_subsection, df_to_latex_table, latex_list
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

REGRESSOR_AIRPORTS = ['KRZL', 'KMWC', 'KIGQ', 'KPNT', 'KOXI', 'KIKK', 'KDKB', 'KPPO', 'KBUU', 'KMDW']
TARGET_AIRPORT = 'KORD'
ALL_AIRPORTS = [TARGET_AIRPORT] + REGRESSOR_AIRPORTS

def log_progress(message, start_time=None):
    if start_time:
        elapsed = time.time() - start_time
        logger.info(f"{message} (took {elapsed:.2f} seconds)")
    else:
        logger.info(message)
    return time.time()

def load_and_prepare_data(file_path='../Utils/data_by_airfield_reshaped_imputed.csv'):
    start_time = time.time()
    log_progress("Loading reshaped data...")
    df = pd.read_csv(file_path)
    log_progress(f"Loaded data shape: {df.shape}")
    log_progress("\n=== Data Quality Report ===")
    log_progress(f"Total rows: {len(df)}")
    log_progress(f"Total columns: {len(df.columns)}")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    log_progress("\nMissing Values Summary:")
    log_progress(f"Columns with missing values: {missing_values[missing_values > 0].shape[0]}")
    log_progress(f"Total missing values: {missing_values.sum()}")
    log_progress("\nTop 10 columns with most missing values:")
    missing_report = pd.DataFrame({
        'Missing Values': missing_values[missing_values > 0],
        'Percentage': missing_percent[missing_values > 0]
    }).sort_values('Missing Values', ascending=False).head(10)
    log_progress(missing_report.to_string())
    log_progress(f"\nUsing airports: {ALL_AIRPORTS}")
    log_progress(f"Regressor airports: {REGRESSOR_AIRPORTS}")
    feature_cols = [
        col for col in df.columns 
        if any(col.startswith(f'{airport}_') for airport in REGRESSOR_AIRPORTS) and '_KORD' not in col
    ]
    kord_cols = [col for col in df.columns if '_KORD' in col]
    if kord_cols:
        log_progress(f"\n[DIAGNOSTIC] Columns with '_KORD' in name (should NOT be in features!): {kord_cols}")
    kord_in_features = [col for col in feature_cols if '_KORD' in col]
    if kord_in_features:
        log_progress(f"\n[WARNING] KORD columns in feature set! These will be removed: {kord_in_features}")
        feature_cols = [col for col in feature_cols if '_KORD' not in col]
    else:
        log_progress("\n[CHECK] No KORD columns in feature set. Good!")
    log_progress(f"\nFeature columns: {len(feature_cols)}")
    feature_types = df[feature_cols].dtypes.value_counts()
    log_progress("\nFeature data types:")
    log_progress(feature_types.to_string())
    kord_temp_col = f'temp_KORD'
    if kord_temp_col not in df.columns:
        raise ValueError("KORD temperature column not found in the data")
    y_1h = df[kord_temp_col].shift(-1)
    y_24h = df[kord_temp_col].shift(-24)
    y_120h = df[kord_temp_col].shift(-120)
    y_5d_avg = df[kord_temp_col].rolling(window=120, min_periods=1).mean().shift(-120)
    y_30d_avg = df[kord_temp_col].rolling(window=720, min_periods=1).mean().shift(-720)
    log_progress("\nDropping rows with NaN targets...")
    valid_idx = (~y_1h.isna()) & (~y_24h.isna()) & (~y_120h.isna()) & (~y_5d_avg.isna()) & (~y_30d_avg.isna())
    log_progress(f"Rows dropped: {len(df) - valid_idx.sum()}")
    log_progress(f"Remaining rows: {valid_idx.sum()}")
    X = df[feature_cols][valid_idx].reset_index(drop=True)
    y_1h = y_1h[valid_idx].reset_index(drop=True)
    y_24h = y_24h[valid_idx].reset_index(drop=True)
    y_120h = y_120h[valid_idx].reset_index(drop=True)
    y_5d_avg = y_5d_avg[valid_idx].reset_index(drop=True)
    y_30d_avg = y_30d_avg[valid_idx].reset_index(drop=True)
    datetime = pd.to_datetime(df['datetime'][valid_idx].reset_index(drop=True))
    log_progress("\n=== Final Data Quality ===")
    log_progress(f"Final feature count: {X.shape[1]}")
    log_progress(f"Final row count: {len(X)}")
    log_progress(f"Missing values in final dataset: {X.isnull().sum().sum()}")
    total_time = time.time() - start_time
    log_progress(f"\nData preparation completed in {total_time:.2f} seconds")
    return X, y_1h, y_24h, y_120h, y_5d_avg, y_30d_avg, datetime, feature_cols

def train_model(X, y, datetime):
    logger.info("Training Neural Network (MLPRegressor) model...")
    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=0.2, random_state=42
    )
    model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # For neural nets, use absolute value of weights as "importance"
    importances = np.abs(model.coefs_[0]).sum(axis=1)
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': importances
    })
    feature_importance['Abs_Coefficient'] = abs(feature_importance['Coefficient'])
    feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Feature_Importance': feature_importance
    }
    datetime_test = datetime[idx_test]
    return model, X_test, y_test, y_pred, datetime_test, metrics

def plot_results(datetime_test, y_test, y_pred, output_dir, suffix):
    plt.figure(figsize=(10, 6))
    plt.plot(datetime_test, y_test, label='Actual')
    plt.plot(datetime_test, y_pred, label='Predicted')
    plt.title('Actual vs. Predicted Temperatures')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plot_path = output_dir / f'4-{suffix}-neural_net_temp_shift_results.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    return plot_path

def plot_feature_importance(feature_importance, output_dir, suffix):
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Coefficient'])
    plt.title('Feature Importance')
    plt.xlabel('Coefficient')
    plt.ylabel('Feature')
    plt.grid(True)
    plot_path = output_dir / f'4-{suffix}-neural_net_temp_shift_feature_importance.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    return plot_path

def plot_feature_importance_map(feature_importance, output_path, top_n=10):
    """
    Plot a map with the top N regression features overlaid at their respective airports.
    Args:
        feature_importance: DataFrame with columns ['Feature', 'Coefficient', 'Abs_Coefficient']
        output_path: Path to save the image
        top_n: Number of top features to show
    """
    # Static airport coordinates (ICAO: (name, lat, lon))
    AIRPORT_COORDS = {
        'KRZL': ("Reedsburg Municipal", 43.1156, -90.6825),
        'KMWC': ("Milwaukee Timmerman", 43.1104, -88.0344),
        'KIGQ': ("Boone County", 41.5239, -85.7979),
        'KPNT': ("Pontiac Municipal", 40.9193, -88.6926),
        'KOXI': ("Starke County", 41.3533, -86.9989),
        'KIKK': ("Greater Kankakee", 40.9231, -87.8461),
        'KDKB': ("DeKalb Taylor", 41.9339, -88.7056),
        'KPPO': ("LaPorte Municipal", 41.3497, -87.4217),
        'KBUU': ("Burlington Municipal", 42.6906, -88.3042),
        'KMDW': ("Chicago Midway", 41.7868, -87.7524),
        'KORD': ("Chicago O'Hare", 41.9786, -87.9048),
    }

    # Get top N features and add ranking
    top_feats = feature_importance.head(top_n).copy()
    top_feats['Rank'] = range(1, len(top_feats) + 1)
    
    # Parse features to airport
    airport_feats = {}
    for _, row in top_feats.iterrows():
        feat = row['Feature']
        abs_coef = row['Abs_Coefficient']
        rank = row['Rank']
        match = re.match(r"([A-Z0-9]+)_([a-z]+)_(.*)", feat)
        if match:
            airport, feat_type, lag = match.groups()
            airport_feats.setdefault(airport, []).append((rank, feat_type + '_' + lag, abs_coef))

    # Create figure and axis with projection
    plt.figure(figsize=(15, 15))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Calculate map bounds (with some padding)
    lats = [coord[1] for coord in AIRPORT_COORDS.values()]
    lons = [coord[2] for coord in AIRPORT_COORDS.values()]
    padding = 1.0  # degrees
    ax.set_extent([
        min(lons) - padding,
        max(lons) + padding,
        min(lats) - padding,
        max(lats) + padding
    ])
    
    # Add map features with improved styling
    ax.add_feature(cfeature.STATES.with_scale('10m'), linewidth=0.5, edgecolor='gray')
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5, edgecolor='gray')
    ax.add_feature(cfeature.LAKES.with_scale('10m'), alpha=0.5, edgecolor='gray')
    ax.add_feature(cfeature.LAND.with_scale('10m'), alpha=0.3)
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), alpha=0.3)
    
    # Plot airports
    for ap, (name, lat, lon) in AIRPORT_COORDS.items():
        # Plot airport marker
        ax.plot(lon, lat, '^', markersize=10, color='blue', transform=ccrs.PlateCarree())
        
        # Add airport code
        ax.text(lon, lat - 0.08, ap, fontsize=9, color='black',
                transform=ccrs.PlateCarree(),
                horizontalalignment='center',
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray', pad=1, boxstyle='round,pad=0.2'))
        
        # Add feature importance text if this airport has a top feature
        feats = airport_feats.get(ap, [])
        if feats:
            # Sort features by rank
            feats.sort(key=lambda x: x[0])
            # Format feature text with rank
            feat_text = '\n'.join([f"#{f[0]}: {f[1]}\n({f[2]:.3f})" for f in feats])
            ax.text(lon, lat + 0.12, feat_text, fontsize=8, color='darkred',
                    transform=ccrs.PlateCarree(),
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='red', pad=2, boxstyle='round,pad=0.3'))
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    
    # Add title
    plt.title('Top 10 Feature Importance by Airport', pad=20, fontsize=16, fontweight='bold')
    
    # Save the map
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path

def create_html_report(all_results, html_manager):
    X, y_1h, y_24h, y_120h, y_5d_avg, y_30d_avg, datetime, feature_cols = load_and_prepare_data()
    model_desc = f"""
    <div class="model-report">
        <h2>KORD Temperature Prediction Using Selected Airport Features (Neural Net)</h2>
        <div class="model-description">
            <h3>Model Architecture</h3>
            <p>This analysis implements multivariate time series regression models to predict temperature at Chicago O'Hare International Airport (KORD) using weather data from <b>selected airports in the dataset</b>. The model incorporates both historical data and recent changes in weather patterns from these regressor airports.</p>
            <h4>Regressor Airports</h4>
            <p>The following airports are used as regressors:</p>
            <ul>
                {''.join([f'<li>{ap}</li>' for ap in REGRESSOR_AIRPORTS])}
            </ul>
            <h4>Data Structure</h4>
            <p>The reshaped data contains:</p>
            <ul>
                <li>Number of rows: {len(X)}</li>
                <li>Number of features: {len(feature_cols)}</li>
                <li>Time period: {datetime.min()} to {datetime.max()}</li>
            </ul>
        </div>
    </div>
    """
    prediction_sections = []
    for horizon, results in all_results.items():
        metrics = results['metrics']
        plot_path = results['plot_path']
        feature_importance_plot = results['feature_importance_plot']
        feature_importance_map = results.get('feature_importance_map')
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
        section += html_manager.create_section_with_image(
            plot_path,
            f"{horizon} Predictions for KORD",
            f"The following plot shows the actual vs predicted temperatures for KORD. The blue line represents actual measurements at KORD, while the orange line shows our model's predictions using data from {', '.join(REGRESSOR_AIRPORTS)}."
        )
        section += html_manager.create_section_with_image(
            feature_importance_plot,
            f"{horizon} Feature Importance",
            "This plot shows the top 10 most important features based on their absolute coefficient values in the regression model."
        )
        if feature_importance_map:
            section += html_manager.create_section_with_image(
                feature_importance_map,
                f"{horizon} Feature Importance Map",
                "This map shows the spatial distribution of the top 10 most important features across the selected airports."
            )
        section += "</div>"
        prediction_sections.append(section)
    content = model_desc + "\n".join(prediction_sections)
    html_content = html_manager.template.format(
        title=f"KORD Temperature Prediction Using {', '.join(REGRESSOR_AIRPORTS)} Features (Neural Net)",
        content=content,
        additional_js=""
    )
    html_filename = "4-neural_net_temp_shift.html"
    output_path = html_manager.save_section_html("Airport_to_Airport", html_content, html_filename)
    return html_content, output_path

def main():
    try:
        output_dir = Path(__file__).parent / 'outputs'
        output_dir.mkdir(exist_ok=True)
        manager = HTMLManager()
        manager.register_section("Airport_to_Airport", Path(__file__).parent)
        X, y_1h, y_24h, y_120h, y_5d_avg, y_30d_avg, datetime, feature_cols = load_and_prepare_data()
        all_results = {}
        # 1 hour ahead
        model_1h, X_test_1h, y_test_1h, y_pred_1h, datetime_test_1h, metrics_1h = train_model(X, y_1h, datetime)
        plot_path_1h = plot_results(datetime_test_1h, y_test_1h, y_pred_1h, output_dir, suffix="0")
        feature_importance_plot_1h = plot_feature_importance(metrics_1h['Feature_Importance'], output_dir, suffix="0")
        feature_importance_map_1h = plot_feature_importance_map(metrics_1h['Feature_Importance'], output_dir / '4-feature_importance_map_1h.png', top_n=10)
        all_results["1 Hour Ahead"] = {
            'metrics': metrics_1h,
            'plot_path': plot_path_1h,
            'feature_importance_plot': feature_importance_plot_1h,
            'feature_importance_map': feature_importance_map_1h
        }
        logger.info("1-hour prediction analysis complete")
        # 24 hours ahead
        model_24h, X_test_24h, y_test_24h, y_pred_24h, datetime_test_24h, metrics_24h = train_model(X, y_24h, datetime)
        plot_path_24h = plot_results(datetime_test_24h, y_test_24h, y_pred_24h, output_dir, suffix="1")
        feature_importance_plot_24h = plot_feature_importance(metrics_24h['Feature_Importance'], output_dir, suffix="1")
        feature_importance_map_24h = plot_feature_importance_map(metrics_24h['Feature_Importance'], output_dir / '4-feature_importance_map_24h.png', top_n=10)
        all_results["24 Hours Ahead"] = {
            'metrics': metrics_24h,
            'plot_path': plot_path_24h,
            'feature_importance_plot': feature_importance_plot_24h,
            'feature_importance_map': feature_importance_map_24h
        }
        logger.info("24-hour prediction analysis complete")
        # 120 hours ahead
        model_120h, X_test_120h, y_test_120h, y_pred_120h, datetime_test_120h, metrics_120h = train_model(X, y_120h, datetime)
        plot_path_120h = plot_results(datetime_test_120h, y_test_120h, y_pred_120h, output_dir, suffix="2")
        feature_importance_plot_120h = plot_feature_importance(metrics_120h['Feature_Importance'], output_dir, suffix="2")
        feature_importance_map_120h = plot_feature_importance_map(metrics_120h['Feature_Importance'], output_dir / '4-feature_importance_map_120h.png', top_n=10)
        all_results["120 Hours (5 Days) Ahead"] = {
            'metrics': metrics_120h,
            'plot_path': plot_path_120h,
            'feature_importance_plot': feature_importance_plot_120h,
            'feature_importance_map': feature_importance_map_120h
        }
        logger.info("120-hour prediction analysis complete")
        # 5-day average
        model_5d_avg, X_test_5d_avg, y_test_5d_avg, y_pred_5d_avg, datetime_test_5d_avg, metrics_5d_avg = train_model(X, y_5d_avg, datetime)
        plot_path_5d_avg = plot_results(datetime_test_5d_avg, y_test_5d_avg, y_pred_5d_avg, output_dir, suffix="3")
        feature_importance_plot_5d_avg = plot_feature_importance(metrics_5d_avg['Feature_Importance'], output_dir, suffix="3")
        feature_importance_map_5d_avg = plot_feature_importance_map(metrics_5d_avg['Feature_Importance'], output_dir / '4-feature_importance_map_5d_avg.png', top_n=10)
        all_results["5-Day Average Ahead"] = {
            'metrics': metrics_5d_avg,
            'plot_path': plot_path_5d_avg,
            'feature_importance_plot': feature_importance_plot_5d_avg,
            'feature_importance_map': feature_importance_map_5d_avg
        }
        logger.info("5-day average prediction analysis complete")
        # 30-day average
        model_30d_avg, X_test_30d_avg, y_test_30d_avg, y_pred_30d_avg, datetime_test_30d_avg, metrics_30d_avg = train_model(X, y_30d_avg, datetime)
        plot_path_30d_avg = plot_results(datetime_test_30d_avg, y_test_30d_avg, y_pred_30d_avg, output_dir, suffix="4")
        feature_importance_plot_30d_avg = plot_feature_importance(metrics_30d_avg['Feature_Importance'], output_dir, suffix="4")
        feature_importance_map_30d_avg = plot_feature_importance_map(metrics_30d_avg['Feature_Importance'], output_dir / '4-feature_importance_map_30d_avg.png', top_n=10)
        all_results["30-Day Average Ahead"] = {
            'metrics': metrics_30d_avg,
            'plot_path': plot_path_30d_avg,
            'feature_importance_plot': feature_importance_plot_30d_avg,
            'feature_importance_map': feature_importance_map_30d_avg
        }
        logger.info("30-day average prediction analysis complete")
        # HTML and LaTeX report generation (reuse from linear_temp_shift.py)
        from linear_temp_shift import create_html_report, create_latex_report
        html_content, output_path = create_html_report(all_results, manager)
        logger.info(f"All temperature prediction analyses complete. Results saved to {output_path}")
        create_latex_report(all_results, output_dir)
    except Exception as e:
        logger.error(f"Error in temperature prediction analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
