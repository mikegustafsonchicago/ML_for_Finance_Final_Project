import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def analyze_dataset(df):
    """Analyze the dataset and return key information"""
    logger.info("Analyzing dataset...")
    
    # Get column information
    columns = df.columns.tolist()
    
    # Get unique airport codes
    airports = df['airport'].unique().tolist() if 'airport' in df.columns else []
    airports.sort()  # Sort alphabetically
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dataset Information</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                line-height: 1.6;
            }}
            .section {{
                margin: 20px;
                padding: 20px;
                background-color: white;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h2 {{
                color: #333;
                margin-bottom: 20px;
            }}
            .data-list {{
                background-color: #f5f5f5;
                padding: 15px;
                border-radius: 5px;
                font-family: monospace;
            }}
            .column-name {{
                color: #2c5282;
            }}
            .airport-code {{
                color: #2b6cb0;
                display: inline-block;
                margin-right: 10px;
                padding: 2px 6px;
                background-color: #ebf8ff;
                border-radius: 3px;
            }}
        </style>
    </head>
    <body>
        <div class="section">
            <h2>Dataset Overview</h2>
            <p>Number of records: {len(df):,}</p>
            <p>Time range: {df['datetime'].min()} to {df['datetime'].max()}</p>
            
            <h3>Available Columns ({len(columns)})</h3>
            <div class="data-list">
                {', '.join(f'<span class="column-name">{col}</span>' for col in columns)}
            </div>
            
            <h3>Airport Locations ({len(airports)})</h3>
            <div class="data-list">
                {' '.join(f'<span class="airport-code">{airport}</span>' for airport in airports)}
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save the HTML file
    output_path = Path(__file__).parent / 'outputs' / 'data_overview.html'
    with open(output_path, 'w') as f:
        f.write(html_content)
    logger.info(f"Data overview created: {output_path}")
    
    return output_path
