import pandas as pd
from pathlib import Path
import logging
import sys
sys.path.append('..')  # Add parent directory to path to import html_manager
from html_manager import HTMLManager
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

def create_weather_plots(df, html_manager: HTMLManager):
    """Create all weather plots and their individual HTML pages"""
    logger.info("Creating weather plots...")
    
    # Create outputs directory if it doesn't exist
    output_dir = Path(__file__).parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    # Add data overview
    data_overview = analyze_dataset(df, html_manager)
    logger.info("Created data overview")
    
    # Create temperature plots
    temp_plot_paths = create_temperature_plot(df, output_dir)
    for i, plot_path in enumerate(temp_plot_paths, 1):
        html_manager.create_section_with_image(
            plot_path,
            f"Temperature Analysis - Group {i}",
            "Temperature comparison between O'Hare International Airport (KORD) and other regional airports.",
            f"temperature_group_{i}.html"
        )
    logger.info("Created temperature analysis sections")

    # Create wind plot
    wind_plot_path = create_wind_plot(df, output_dir)
    html_manager.create_section_with_image(
        wind_plot_path,
        "Wind Analysis",
        "Analysis of wind speed and direction patterns.",
        "wind.html"
    )
    logger.info("Created wind analysis section")

def main():
    try:
        # Create HTML manager
        manager = HTMLManager()
        manager.register_section("Introduction", Path(__file__).parent)
        
        # Read the CSV file
        logger.info("Reading CSV data...")
        df = pd.read_csv('../datastep2.csv')
        logger.info(f"Loaded {len(df)} rows of data")

        # Convert datetime to proper datetime type
        logger.info("Converting datetime column...")
        df['datetime'] = pd.to_datetime(df['datetime'])
        logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

        # Create all plots and sections
        create_weather_plots(df, manager)
        
        # Combine all sections into one file
        manager.combine_sections(["Introduction"], "introduction_complete.html")
        logger.info("Created complete introduction section")
        
    except Exception as e:
        logger.error(f"Error creating plots: {str(e)}")
        raise

if __name__ == "__main__":
    main()