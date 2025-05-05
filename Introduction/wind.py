import matplotlib.pyplot as plt
from pathlib import Path
import logging
import pandas as pd
import sys
sys.path.append('..')  # Add parent directory to path to import html_manager
from html_manager import HTMLManager
from Utils.data_formatter import format_datetime_axis, parse_custom_datetime
import numpy as np

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

def create_wind_plots_by_airport(df, output_dir, html_manager: HTMLManager = None):
    """
    Create wind speed plots for each airport, overlaying all years on a single Jan-Dec x-axis.
    Each year is a different color; x-axis is day-of-year + time (ignoring year).
    Also creates a sample week plot (April 1-7) for each airport.
    """
    logger.info("Creating wind plots by airport (year-over-year overlay, single-year x-axis)...")
    
    if html_manager is None:
        html_manager = HTMLManager()
        html_manager.register_section("Introduction", Path(__file__).parent)
    
    airport_codes = df['id'].unique()
    plot_paths = []
    for airport in sorted(airport_codes):
        airport_df = df[df['id'] == airport].copy()
        airport_df['datetime'] = parse_custom_datetime(airport_df['datetime'])
        airport_df['year'] = airport_df['datetime'].dt.year

        # Create a "time of year" value: day-of-year + fraction of day
        dt = airport_df['datetime']
        doy = dt.dt.dayofyear
        seconds = dt.dt.hour * 3600 + dt.dt.minute * 60 + dt.dt.second
        frac_of_day = seconds / 86400
        airport_df['doy_frac'] = doy + frac_of_day

        # --- Full Year Plot ---
        plt.figure(figsize=(14, 5))
        years = sorted(airport_df['year'].unique())
        cmap = plt.get_cmap('tab10')
        for i, year in enumerate(years):
            year_df = airport_df[airport_df['year'] == year]
            plt.scatter(
                year_df['doy_frac'],
                year_df['windspeed'],
                alpha=0.7,
                label=str(year),
                color=cmap(i % 10),
                s=18
            )
        plt.title(f'Wind Speed by Day/Hour of Year - {airport}', fontsize=14, pad=20)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Wind Speed', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
        # Set x-ticks for months
        import calendar
        month_starts = [pd.Timestamp(f'2000-{m:02d}-01').dayofyear for m in range(1, 13)]
        month_labels = [calendar.month_abbr[m] for m in range(1, 13)]
        plt.xticks(month_starts, month_labels)
        plt.tight_layout()
        plot_path = output_dir / f'wind_{airport}.png'
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()

        # --- Example Week Plot (April 1-7) ---
        plt.figure(figsize=(14, 5))
        week_mask = (airport_df['datetime'].dt.month == 4) & (airport_df['datetime'].dt.day >= 1) & (airport_df['datetime'].dt.day <= 7)
        week_data = airport_df[week_mask]
        years_week = sorted(week_data['year'].unique())
        for i, year in enumerate(years_week):
            year_df = week_data[week_data['year'] == year].sort_values('doy_frac')  # Sort by time for line plot
            plt.plot(
                year_df['doy_frac'],
                year_df['windspeed'],
                alpha=0.8,
                label=str(year),
                color=cmap(i % 10),
                linewidth=2,
                marker='o',  # Add small markers at data points
                markersize=4
            )
        plt.title(f'Wind Speed: Example Week (April 1-7) - {airport}', fontsize=14, pad=20)
        plt.xlabel('Day of Year', fontsize=12)
        plt.ylabel('Wind Speed', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        # X-ticks for April 1-7
        week_days = [pd.Timestamp(f'2000-04-{d:02d}').dayofyear for d in range(1, 8)]
        week_labels = [f'Apr {d}' for d in range(1, 8)]
        plt.xticks(week_days, week_labels)
        plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plot_path_week = output_dir / f'wind_{airport}_april1week.png'
        plt.savefig(plot_path_week, dpi=200, bbox_inches='tight')
        plt.close()

        plot_paths.append((airport, plot_path, plot_path_week))
        logger.info(f"Wind plots created for {airport}: {plot_path} and {plot_path_week}")
    return plot_paths

def analyze_dataset(df, html_manager: HTMLManager = None):
    logger.info("Analyzing dataset...")

    if html_manager is None:
        html_manager = HTMLManager()
        html_manager.register_section("Introduction", Path(__file__).parent)

    # Calculate basic statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = df[numeric_cols].describe()

    # Format stats table as HTML with thousands separators and no scientific notation
    stats_html = stats.applymap(lambda x: f"{x:,.2f}").to_html()

    # Calculate missing values
    missing_values = df.isnull().sum()
    missing_percentages = (missing_values / len(df) * 100).round(2)

    # Analyze time step
    df_sorted = df.sort_values('datetime')
    time_deltas = df_sorted['datetime'].diff().dropna()
    if not time_deltas.empty:
        most_common_step = time_deltas.mode()[0]
        step_str = str(most_common_step)
        prose_time_step = (
            f"The most common time step between consecutive measurements is <b>{step_str}</b>."
        )
    else:
        prose_time_step = "Time step information could not be determined."

    # Format min/max dates for display
    min_date = df['datetime'].min().strftime('%Y-%m-%d %H:%M')
    max_date = df['datetime'].max().strftime('%Y-%m-%d %H:%M')

    # Create data overview using HTML manager
    content = f"""
    <div class="section">
        <h2>Dataset Overview</h2>
        
        <h3>Data Collection Summary</h3>
        <p>This dataset contains comprehensive weather measurements collected from multiple airports in the Chicago area. 
        The data spans a significant time period, with {len(df):,} individual weather observations recorded.</p>
        
        <h3>Time Coverage</h3>
        <p>The measurements were taken between {min_date} and {max_date}, 
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
            <pre>{stats_html}</pre>
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
        # Create outputs directory
        output_dir = Path(__file__).parent / 'outputs'
        output_dir.mkdir(exist_ok=True)
        
        # Create HTML manager
        manager = HTMLManager()
        manager.register_section("Introduction", Path(__file__).parent)
        
        # Read and prepare data
        logger.info("Reading data...")
        df = pd.read_csv('../datastep2.csv')
        df['datetime'] = parse_custom_datetime(df['datetime'])
        logger.info(f"Loaded {len(df)} rows of data")
        
        # Create plots by airport
        plot_paths = create_wind_plots_by_airport(df, output_dir, manager)
        
        # Create HTML section with all plots
        section_title = "Wind Speed Time Series by Airport"
        description = (
            "The following plots show wind speed over time for each airport in the dataset. "
            "Each airport has two plots: one showing the full year of data with different colors for each year, "
            "and another showing a detailed view of the week of April 1-7 as an example of short-term variability."
        )
        plots_html = ""
        for airport, plot_path, plot_path_week in plot_paths:
            plots_html += f"""
                <div class="plot-group" style="margin-bottom: 32px;">
                    <h3>{airport} - Full Year</h3>
                    <img src="{plot_path.name}" alt="Wind Speed - {airport}" style="max-width: 100%; height: auto;">
                    <h4 style="margin-top: 18px;">Example Week (April 1-7)</h4>
                    <img src="{plot_path_week.name}" alt="Wind Speed {airport} April 1-7" style="max-width: 100%; height: auto;">
                </div>
            """
        content_html = f"""
            <div class="section">
                <h2>{section_title}</h2>
                <p>{description}</p>
                {plots_html}
            </div>
        """
        manager.save_section_html("Introduction", content_html, "wind.html")
        
        logger.info("Created wind analysis section")
        
        # Create data overview
        analyze_dataset(df, manager)
        
    except Exception as e:
        logger.error(f"Error creating wind plot: {str(e)}")
        raise
