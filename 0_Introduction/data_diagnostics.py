import pandas as pd
import logging
from pathlib import Path
import sys
sys.path.append('..')  # Add parent directory to path to import html_manager
from html_manager import HTMLManager
import numpy as np
from Utils.data_formatter import format_datetime_axis, parse_custom_datetime
from Utils.latex_utility import df_to_latex_table, save_latex_file, latex_section, latex_subsection, latex_list

# Disable scientific notation
pd.set_option('display.float_format', lambda x: '%.2f' % x)

logger = logging.getLogger(__name__)

def analyze_dataset(df, html_manager: HTMLManager = None):
    """Analyze the dataset and return key information, outputting both HTML and LaTeX."""
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
    
    # Analyze time step
    df_sorted = df.sort_values('datetime')
    time_deltas = df_sorted['datetime'].diff().dropna()
    if not time_deltas.empty:
        most_common_step = time_deltas.mode()[0]
        step_str = str(most_common_step)
        prose_time_step = f"The most common time step between consecutive measurements is <b>{step_str}</b>."
        prose_time_step_latex = f"The most common time step between consecutive measurements is \textbf{{{step_str}}}."
    else:
        prose_time_step = "Time step information could not be determined."
        prose_time_step_latex = "Time step information could not be determined."
    
    # Create data overview using HTML manager
    content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link rel="stylesheet" href="../../style.css">
    </head>
    <body>
        <div class="container">
            <div class="section">
                <h2>Dataset Overview</h2>
                
                <h3>Data Collection Summary</h3>
                <p>This dataset contains comprehensive weather measurements collected from multiple airports in the Chicago area. 
                The data spans a significant time period, with {len(df):,} individual weather observations recorded.</p>
                
                <h3>Time Coverage</h3>
                <p>The measurements were taken between {df['datetime'].min().strftime('%Y-%m-%d %H:%M')} and {df['datetime'].max().strftime('%Y-%m-%d %H:%M')}, 
                providing a detailed view of weather patterns in the region.</p>
                <p>{prose_time_step}</p>
                
                <h3>Weather Parameters</h3>
                <p>The dataset includes {len(df.columns)} different weather parameters, providing a comprehensive view of atmospheric conditions:</p>
                <div class="data-list">
                    {', '.join(f'<span class="column-name">{col}</span>' for col in df.columns)}
                </div>
                
                <h3>Data Quality</h3>
                <p>Missing values analysis across parameters:</p>
                <div class="data-list">
                    <table class="feature-table">
                        <tr>
                            <th>Parameter</th>
                            <th>Missing Values</th>
                            <th>Percentage</th>
                        </tr>
                        {''.join(f'<tr><td>{col}</td><td>{missing_values[col]:,}</td><td>{missing_percentages[col]}%</td></tr>' for col in df.columns)}
                    </table>
                </div>
                
                <h3>Statistical Summary</h3>
                <p>Key statistics for numerical weather parameters:</p>
                <div class="data-list">
                    <table class="metrics-table">
                        {stats.to_html(classes='metrics-table')}
                    </table>
                </div>
                
                <h3>Data Types</h3>
                <p>Overview of data types for each parameter:</p>
                <div class="data-list">
                    <table class="feature-table">
                        <tr>
                            <th>Parameter</th>
                            <th>Data Type</th>
                        </tr>
                        {''.join(f'<tr><td>{col}</td><td>{dtype}</td></tr>' for col, dtype in df.dtypes.items())}
                    </table>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save the HTML file
    output_path = Path(__file__).parent / 'outputs' / 'data_overview.html'
    html_manager.save_section_html("Introduction", content, "data_overview.html")
    logger.info(f"Data overview created: {output_path}")

    # --- LaTeX Output ---
    latex_content = ""
    latex_content += latex_section("Dataset Overview", "")
    latex_content += latex_subsection("Data Collection Summary",
        f"This dataset contains comprehensive weather measurements collected from multiple airports in the Chicago area. The data spans a significant time period, with {len(df):,} individual weather observations recorded.")
    latex_content += latex_subsection("Time Coverage",
        f"The measurements were taken between {df['datetime'].min().strftime('%Y-%m-%d %H:%M')} and {df['datetime'].max().strftime('%Y-%m-%d %H:%M')}, providing a detailed view of weather patterns in the region. {prose_time_step_latex}")
    latex_content += latex_subsection("Weather Parameters",
        f"The dataset includes {len(df.columns)} different weather parameters, providing a comprehensive view of atmospheric conditions: " + latex_list([col for col in df.columns]))
    # Data Quality Table
    missing_df = pd.DataFrame({
        'Parameter': df.columns,
        'Missing Values': [missing_values[col] for col in df.columns],
        'Percentage': [missing_percentages[col] for col in df.columns]
    })
    latex_content += latex_subsection("Data Quality",
        "Missing values analysis across parameters:" + "\n" + df_to_latex_table(missing_df, caption="Missing Values Analysis", label="tab:missing_values"))
    # Statistical Summary Table
    latex_content += latex_subsection("Statistical Summary",
        "Key statistics for numerical weather parameters:" + "\n" + df_to_latex_table(stats, caption="Statistical Summary", label="tab:stats"))
    # Data Types Table
    dtypes_df = pd.DataFrame({
        'Parameter': df.columns,
        'Data Type': [str(dtype) for dtype in df.dtypes]
    })
    latex_content += latex_subsection("Data Types",
        "Overview of data types for each parameter:" + "\n" + df_to_latex_table(dtypes_df, caption="Data Types", label="tab:dtypes"))
    # Save the LaTeX file
    latex_output_path = Path(__file__).parent / 'outputs' / 'data_overview.tex'
    save_latex_file(latex_content, latex_output_path)
    logger.info(f"LaTeX data overview created: {latex_output_path}")
    return output_path

if __name__ == "__main__":
    try:
        # Create HTML manager
        manager = HTMLManager()
        manager.register_section("Introduction", Path(__file__).parent)
        
        # Read and prepare data
        logger.info("Reading data...")
        df = pd.read_csv('../datastep2.csv')
        df['datetime'] = parse_custom_datetime(df['datetime'])
        
        # Create data overview
        analyze_dataset(df, manager)
        
    except Exception as e:
        logger.error(f"Error creating data overview: {str(e)}")
        raise
