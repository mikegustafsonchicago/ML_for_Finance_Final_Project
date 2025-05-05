import pandas as pd
import logging
from pathlib import Path
import sys
sys.path.append('..')  # Add parent directory to path to import html_manager
from html_manager import HTMLManager
import numpy as np

logger = logging.getLogger(__name__)

def analyze_dataset(df, html_manager: HTMLManager = None):
    """Analyze the dataset and return key information"""
    logger.info("Analyzing dataset...")
    
    # Initialize HTML manager if not provided
    if html_manager is None:
        html_manager = HTMLManager()
        html_manager.register_section("Introduction", Path(__file__).parent)
    
    # Calculate basic statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = df[numeric_cols].describe()
    
    # Calculate missing values
    missing_values = df.isnull().sum()
    missing_percentages = (missing_values / len(df) * 100).round(2)
    
    # Create data overview using HTML manager
    content = f"""
    <div class="section">
        <h2>Dataset Overview</h2>
        
        <h3>Data Collection Summary</h3>
        <p>This dataset contains comprehensive weather measurements collected from multiple airports in the Chicago area. 
        The data spans a significant time period, with {len(df):,} individual weather observations recorded.</p>
        
        <h3>Time Coverage</h3>
        <p>The measurements were taken between {df['datetime'].min()} and {df['datetime'].max()}, 
        providing a detailed view of weather patterns in the region.</p>
        
        <h3>Weather Parameters</h3>
        <p>The dataset includes {len(df.columns)} different weather parameters, providing a comprehensive view of atmospheric conditions:</p>
        <div class="data-list">
            {', '.join(f'<span class="column-name">{col}</span>' for col in df.columns)}
        </div>
        
        <h3>Data Quality</h3>
        <p>Missing values analysis across parameters:</p>
        <div class="data-list">
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <th style="text-align: left; padding: 8px;">Parameter</th>
                    <th style="text-align: right; padding: 8px;">Missing Values</th>
                    <th style="text-align: right; padding: 8px;">Percentage</th>
                </tr>
                {''.join(f'<tr><td style="padding: 8px;">{col}</td><td style="text-align: right; padding: 8px;">{missing_values[col]:,}</td><td style="text-align: right; padding: 8px;">{missing_percentages[col]}%</td></tr>' for col in df.columns)}
            </table>
        </div>
        
        <h3>Statistical Summary</h3>
        <p>Key statistics for numerical weather parameters:</p>
        <div class="data-list">
            <pre>{stats.to_html()}</pre>
        </div>
        
        <h3>Data Types</h3>
        <p>Overview of data types for each parameter:</p>
        <div class="data-list">
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <th style="text-align: left; padding: 8px;">Parameter</th>
                    <th style="text-align: left; padding: 8px;">Data Type</th>
                </tr>
                {''.join(f'<tr><td style="padding: 8px;">{col}</td><td style="padding: 8px;">{dtype}</td></tr>' for col, dtype in df.dtypes.items())}
            </table>
        </div>
    </div>
    """
    
    # Save the HTML file
    output_path = Path(__file__).parent / 'outputs' / 'data_overview.html'
    html_manager.save_section_html("Introduction", content, "data_overview.html")
    
    logger.info(f"Data overview created: {output_path}")
    return output_path

if __name__ == "__main__":
    try:
        # Create HTML manager
        manager = HTMLManager()
        manager.register_section("Introduction", Path(__file__).parent)
        
        # Read and prepare data
        logger.info("Reading data...")
        df = pd.read_csv('../datastep2.csv')
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Create data overview
        analyze_dataset(df, manager)
        
    except Exception as e:
        logger.error(f"Error creating data overview: {str(e)}")
        raise
