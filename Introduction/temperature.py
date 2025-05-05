import matplotlib.pyplot as plt
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter
import sys
sys.path.append('..')  # Add parent directory to path to import html_manager
from html_manager import HTMLManager
from Utils.data_formatter import parse_custom_datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("temperature")

def create_temperature_plot(df, output_dir, html_manager: HTMLManager = None):
    """Create temperature over time plot for each airport (excluding KORD), year-over-year visualization, plus an example week plot."""
    logger.info("Creating temperature plot...")
    
    # Initialize HTML manager if not provided
    if html_manager is None:
        html_manager = HTMLManager()
        html_manager.register_section("Introduction", Path(__file__).parent)
    
    # Get unique airports and remove KORD
    airports = sorted(list(set(df['id'].unique()) - {'KORD'}))
    plot_paths = []
    
    for airport in airports:
        # --- Full Year Plot ---
        plt.figure(figsize=(14, 5))
        cmap = plt.get_cmap('tab10')
        airport_data = df[df['id'] == airport].copy()
        airport_data['datetime'] = parse_custom_datetime(airport_data['datetime'])
        airport_data['year'] = airport_data['datetime'].dt.year
        dt = airport_data['datetime']
        doy = dt.dt.dayofyear
        seconds = dt.dt.hour * 3600 + dt.dt.minute * 60 + dt.dt.second
        frac_of_day = seconds / 86400
        airport_data['doy_frac'] = doy + frac_of_day
        years = sorted(airport_data['year'].unique())
        for j, year in enumerate(years):
            year_df = airport_data[airport_data['year'] == year]
            plt.scatter(
                year_df['doy_frac'],
                year_df['temp'],
                alpha=0.7,
                label=f'{year}',  # Only year in legend
                color=cmap(j % 10),
                s=16
            )
        plt.title(f'Temperature Over Time: {airport}', fontsize=14, pad=20)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Temperature (°C)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        import calendar
        month_starts = [pd.Timestamp(f'2000-{m:02d}-01').dayofyear for m in range(1, 13)]
        month_labels = [calendar.month_abbr[m] for m in range(1, 13)]
        plt.xticks(month_starts, month_labels)
        plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        output_path_full = output_dir / f'temperature_{airport}.png'
        plt.savefig(output_path_full, bbox_inches='tight', dpi=200)
        plt.close()
        logger.info(f"Temperature plot created for {airport}: {output_path_full}")

        # --- Example Week Plot (April 1-7) ---
        plt.figure(figsize=(14, 5))
        week_mask = (airport_data['datetime'].dt.month == 4) & (airport_data['datetime'].dt.day >= 1) & (airport_data['datetime'].dt.day <= 7)
        week_data = airport_data[week_mask]
        years_week = sorted(week_data['year'].unique())
        for j, year in enumerate(years_week):
            year_df = week_data[week_data['year'] == year]
            plt.scatter(
                year_df['doy_frac'],
                year_df['temp'],
                alpha=0.8,
                label=f'{year}',  # Only year in legend
                color=cmap(j % 10),
                s=32
            )
        plt.title(f'Temperature: Example Week (April 1-7) - {airport}', fontsize=14, pad=20)
        plt.xlabel('Day of Year', fontsize=12)
        plt.ylabel('Temperature (°C)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        # X-ticks for April 1-7
        week_days = [pd.Timestamp(f'2000-04-{d:02d}').dayofyear for d in range(1, 8)]
        week_labels = [f'Apr {d}' for d in range(1, 8)]
        plt.xticks(week_days, week_labels)
        plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        output_path_week = output_dir / f'temperature_{airport}_april1week.png'
        plt.savefig(output_path_week, bbox_inches='tight', dpi=200)
        plt.close()
        logger.info(f"Example week plot created for {airport}: {output_path_week}")

        plot_paths.append((airport, output_path_full, output_path_week))
    return plot_paths

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
        logger.info(f"Loaded {len(df)} rows of data")
        # Create plots
        plot_paths = create_temperature_plot(df, output_dir, manager)
        # Combine all plots into a single HTML section
        section_title = "Temperature Analysis: Individual Airports"
        description = (
            "Each plot shows temperature trends for a single regional airport, year-over-year. "
            "Temperature data is shown over the course of a year, with different colors representing different years. "
            "A second plot for each airport shows only the week of April 1-7 as an example of short-term variability."
        )
        plots_html = ""
        for airport, plot_path_full, plot_path_week in plot_paths:
            plots_html += f"""
                <div class=\"plot-group\" style=\"margin-bottom: 32px;\">
                    <h3>{airport} - Full Year</h3>
                    <img src=\"{plot_path_full.name}\" alt=\"Temperature {airport}\" style=\"max-width: 100%; height: auto;\">
                    <h4 style=\"margin-top: 18px;\">Example Week (April 1-7)</h4>
                    <img src=\"{plot_path_week.name}\" alt=\"Temperature {airport} April 1-7\" style=\"max-width: 100%; height: auto;\">
                </div>
            """
        content_html = f"""
            <div class=\"section\">
                <h2>{section_title}</h2>
                <p>{description}</p>
                {plots_html}
            </div>
        """
        manager.save_section_html("Introduction", content_html, "temperature.html")
        logger.info("Created all temperature analysis sections in one HTML file")
    except Exception as e:
        logger.error(f"Error creating temperature plot: {str(e)}")
        logger.debug("Error details:", exc_info=True)
        raise
