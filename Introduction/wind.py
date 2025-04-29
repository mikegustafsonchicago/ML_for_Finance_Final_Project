import matplotlib.pyplot as plt
from pathlib import Path
import logging
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def create_wind_plot(df, output_dir):
    """Create wind speed and direction plot"""
    logger.info("Creating wind plot...")
    plt.figure(figsize=(12, 6))
    plt.scatter(df['winddirection'], df['windspeed'], alpha=0.5)
    plt.title('Wind Speed vs Direction')
    plt.xlabel('Wind Direction')
    plt.ylabel('Wind Speed')
    plt.tight_layout()
    
    output_path = output_dir / 'wind.png'
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Wind plot created: {output_path}")
    return output_path

if __name__ == "__main__":
    # This code only runs when the script is run directly
    try:
        # Create outputs directory
        output_dir = Path(__file__).parent / 'outputs'
        output_dir.mkdir(exist_ok=True)
        
        # Read and prepare data
        logger.info("Reading data...")
        df = pd.read_csv('../datastep2.csv')
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Create plot
        plot_path = create_wind_plot(df, output_dir)
        
        # Create standalone HTML
        from html_combine import create_section_with_image
        html_path = output_dir / 'wind.html'
        create_section_with_image(plot_path, 'Wind Analysis', html_path)
        logger.info(f"Created standalone wind analysis at {html_path}")
        
    except Exception as e:
        logger.error(f"Error creating wind plot: {str(e)}")
        raise
