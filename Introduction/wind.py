import matplotlib.pyplot as plt
from pathlib import Path
import logging
import pandas as pd
import sys
sys.path.append('..')  # Add parent directory to path to import html_manager
from html_manager import HTMLManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("wind")

def create_wind_plot(df, output_dir, html_manager: HTMLManager = None):
    """Create wind speed and direction plot"""
    logger.info("Creating wind plot...")
    
    # Initialize HTML manager if not provided
    if html_manager is None:
        html_manager = HTMLManager()
        html_manager.register_section("Introduction", Path(__file__).parent)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.scatter(df['winddirection'], df['windspeed'], alpha=0.5)
    plt.title('Wind Speed vs Direction')
    plt.xlabel('Wind Direction')
    plt.ylabel('Wind Speed')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    output_path = output_dir / 'wind.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Wind plot created: {output_path}")
    return output_path

if __name__ == "__main__":
    try:
        # Create outputs directory
        output_dir = Path(__file__).parent / 'outputs'
        output_dir.mkdir(exist_ok=True)
        
        # Create HTML manager
        manager = HTMLManager()
        manager.register_section("Introduction", Path(__file__).parent)
        
        # Read and prepare data
        logger.info("Reading data...")
        df = pd.read_csv('../datastep2.csv')
        df['datetime'] = pd.to_datetime(df['datetime'])
        logger.info(f"Loaded {len(df)} rows of data")
        
        # Create plot
        plot_path = create_wind_plot(df, output_dir, manager)
        
        # Create HTML section
        manager.create_section_with_image(
            plot_path,
            "Wind Analysis",
            "Analysis of wind speed and direction patterns across all airports.",
            "wind.html"
        )
        
        logger.info("Created wind analysis section")
        
    except Exception as e:
        logger.error(f"Error creating wind plot: {str(e)}")
        raise
