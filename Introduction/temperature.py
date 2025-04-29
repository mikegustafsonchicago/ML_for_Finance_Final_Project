import matplotlib.pyplot as plt
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def create_temperature_plot(df, output_dir):
    """Create temperature over time plot with separate lines for each airport"""
    logger.info("Creating temperature plot...")
    
    # Get unique airports and remove ORD (we'll add it to each plot)
    airports = sorted(list(set(df['id'].unique()) - {'KORD'}))
    
    # Create groups of 5 airports
    airport_groups = [airports[i:i+5] for i in range(0, len(airports), 5)]
    plot_paths = []
    
    # Create a plot for each group
    for i, airport_group in enumerate(airport_groups):
        plt.figure(figsize=(15, 8))
        
        # Plot ORD data first (bold and clearly visible)
        ord_data = df[df['id'] == 'KORD']
        plt.plot(ord_data['datetime'], 
                ord_data['temp'], 
                'k-', 
                label='KORD (O\'Hare)',
                linewidth=2,
                alpha=1.0)
        
        # Plot the group's airports
        colors = plt.cm.tab10(np.linspace(0, 1, len(airport_group)))
        for airport, color in zip(airport_group, colors):
            airport_data = df[df['id'] == airport]
            plt.plot(airport_data['datetime'], 
                    airport_data['temp'], 
                    '.', 
                    label=airport,
                    color=color,
                    alpha=0.6,
                    markersize=4)
        
        # Customize the plot
        plt.title(f'Temperature Over Time - Group {i+1}', fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Temperature (°C)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Format x-axis
        ax = plt.gca()
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45, ha='right')
        
        # Add legend
        plt.legend(title='Airports', 
                  bbox_to_anchor=(1.05, 1),
                  loc='upper left',
                  borderaxespad=0.)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot
        output_path = output_dir / f'temperature_group_{i+1}.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        plot_paths.append(output_path)
        logger.info(f"Temperature plot created for group {i+1}: {output_path}")
    
    return plot_paths

if __name__ == "__main__":
    try:
        # Create outputs directory
        output_dir = Path(__file__).parent / 'outputs'
        output_dir.mkdir(exist_ok=True)
        
        # Read and prepare data
        logger.info("Reading data...")
        df = pd.read_csv('../datastep2.csv')
        logger.info(f"Loaded {len(df)} rows of data")
        
        # Convert datetime using the correct format
        logger.info("Converting datetime column...")
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d%H%M')
        logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        # Create plots
        plot_paths = create_temperature_plot(df, output_dir)
        
        # Create single HTML with all plots
        from html_combine import create_section_with_image
        html_content = """
        <div class="section">
            <h2>Temperature Analysis</h2>
            <p>Temperature comparison between O'Hare International Airport (KORD) and other regional airports.</p>
        """
        
        for i, path in enumerate(plot_paths, 1):
            html_content += f"""
            <div class="plot-group">
                <h3>Group {i}</h3>
                <img src="{path.name}" alt="Temperature Group {i}" style="max-width: 100%; height: auto;">
            </div>
            """
        
        html_content += "</div>"
        
        html_path = output_dir / 'temperature.html'
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Created temperature analysis at {html_path}")
        
    except Exception as e:
        logger.error(f"Error creating temperature plot: {str(e)}")
        logger.debug("Error details:", exc_info=True)
        raise
