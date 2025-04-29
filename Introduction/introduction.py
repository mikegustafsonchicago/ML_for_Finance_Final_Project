import pandas as pd
from pathlib import Path
import logging
from html_combine import create_title_page, create_section_with_image, combine_html_files
from temperature import create_temperature_plot
from wind import create_wind_plot
from data_diagnostics import analyze_dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Create outputs directory if it doesn't exist
output_dir = Path(__file__).parent / 'outputs'
output_dir.mkdir(exist_ok=True)

def create_weather_plots(df):
    """Create all weather plots and their individual HTML pages"""
    html_files = []
    
    # Add title page
    html_files.append(create_title_page())
    
    # Add data overview
    data_overview = analyze_dataset(df)
    html_files.append(data_overview)
    
    # Create temperature plot
    temp_plot = create_temperature_plot(df, output_dir)
    html_files.append(create_section_with_image(
        temp_plot,
        'Temperature Analysis',
        output_dir / 'temperature.html'
    ))

    # Create wind plot
    wind_plot = create_wind_plot(df, output_dir)
    html_files.append(create_section_with_image(
        wind_plot,
        'Wind Analysis',
        output_dir / 'wind.html'
    ))
    
    # Combine all HTML files
    combine_html_files(html_files)
    logger.info("All sections created and combined!")

# Main execution
try:
    # Read the CSV file
    logger.info("Reading CSV data...")
    df = pd.read_csv('../datastep2.csv')
    logger.info(f"Loaded {len(df)} rows of data")

    # Convert datetime to proper datetime type
    logger.info("Converting datetime column...")
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Create all plots and sections
    create_weather_plots(df)
except Exception as e:
    logger.error(f"Error creating plots: {str(e)}")
    raise