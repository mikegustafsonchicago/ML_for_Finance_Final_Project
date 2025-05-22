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
    """Create wind speed and direction plot using arrows"""
    logger.info("Creating wind plot...")
    
    # Initialize HTML manager if not provided
    if html_manager is None:
        html_manager = HTMLManager()
        html_manager.register_section("Introduction", Path(__file__).parent)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Convert wind direction from degrees to radians for arrow plotting
    angles_rad = np.radians(df['winddirection'])
    
    # Calculate arrow components (constant length)
    arrow_length = 0.5  # Constant length for all arrows
    dx = arrow_length * np.sin(angles_rad)
    dy = arrow_length * np.cos(angles_rad)
    
    # Plot arrows
    plt.quiver(df['winddirection'], df['windspeed'], dx, dy,
              scale=1, scale_units='inches', alpha=0.5,
              headwidth=3, headlength=4, width=0.002)
    
    plt.title('Wind Speed vs Direction')
    plt.xlabel('Wind Direction (degrees)')
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

        # --- Full Year Plot (Scatter) ---
        years = sorted(airport_df['year'].unique())
        n_years = len(years)
        
        # Create figure with subplots for each year
        fig, axes = plt.subplots(n_years, 1, figsize=(7, 1.0*n_years), sharex=True)
        if n_years == 1:
            axes = [axes]  # Make axes iterable for single year case
            
        # Add overall title at the top
        fig.suptitle(f'Wind Speed by Day/Hour of Year - {airport}', 
                    fontsize=11, y=0.98)
        
        # Get a color map for the years
        year_colors = plt.cm.tab10(np.linspace(0, 1, n_years))
        
        for i, (year, ax) in enumerate(zip(years, axes)):
            year_df = airport_df[airport_df['year'] == year]
            ax.scatter(
                year_df['doy_frac'],
                year_df['windspeed'],
                alpha=0.6,
                color=year_colors[i],
                s=12
            )
            
            # Add year label in upper right corner
            ax.text(0.98, 0.95, f'Year {year}', 
                   transform=ax.transAxes,
                   horizontalalignment='right',
                   verticalalignment='top',
                   fontsize=9,
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            
            # Customize subplot
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_ylabel('Wind Speed', fontsize=9)
            
            # Only show x-axis labels on bottom subplot
            if i == n_years - 1:
                import calendar
                month_starts = [pd.Timestamp(f'2000-{m:02d}-01').dayofyear for m in range(1, 13)]
                month_labels = [calendar.month_abbr[m] for m in range(1, 13)]
                ax.set_xticks(month_starts)
                ax.set_xticklabels(month_labels)
                ax.set_xlabel('Month', fontsize=9)
        
        # Adjust layout to be more compact
        plt.tight_layout(h_pad=0.5)
        plot_path = output_dir / f'wind_{airport}.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()

        # --- Example Week Plot (April 1-7) with Quiver ---
        plt.figure(figsize=(12, 4))  # Slightly smaller figure
        week_mask = (airport_df['datetime'].dt.month == 4) & (airport_df['datetime'].dt.day >= 1) & (airport_df['datetime'].dt.day <= 7)
        week_data = airport_df[week_mask]
        years_week = sorted(week_data['year'].unique())
        n_years = len(years_week)
        
        # Create figure with subplots for each year
        fig, axes = plt.subplots(n_years, 1, figsize=(7, 1.0*n_years), sharex=True)
        if n_years == 1:
            axes = [axes]  # Make axes iterable for single year case
        
        # Add overall title at the top
        fig.suptitle(f'Wind Speed and Direction: Example Week (April 1-7) - {airport}', 
                    fontsize=11, y=0.98)
        
        # Get a color map for the years
        year_colors = plt.cm.tab10(np.linspace(0, 1, n_years))
        
        for i, (year, ax) in enumerate(zip(years_week, axes)):
            year_df = week_data[week_data['year'] == year].sort_values('doy_frac')
            
            # Plot the connecting line
            ax.plot(
                year_df['doy_frac'],
                year_df['windspeed'],
                color=year_colors[i],
                alpha=0.3,
                linewidth=0.8,
                label='Wind Speed'
            )
            
            # Convert wind direction to radians for arrow plotting
            angles_rad = np.radians(year_df['winddirection'])
            
            # Calculate arrow components (constant length)
            arrow_length = 0.15
            dx = arrow_length * np.sin(angles_rad)
            dy = arrow_length * np.cos(angles_rad)
            
            # Plot arrows
            ax.quiver(
                year_df['doy_frac'],
                year_df['windspeed'],
                dx, dy,
                scale=1,
                scale_units='inches',
                alpha=0.56,
                headwidth=2.4,
                headlength=4,
                width=0.002,
                color=year_colors[i]
            )
            
            # Add year label in upper right corner
            ax.text(0.98, 0.95, f'Year {year}', 
                   transform=ax.transAxes,
                   horizontalalignment='right',
                   verticalalignment='top',
                   fontsize=9,
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            
            # Customize subplot
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_ylabel('Wind Speed', fontsize=9)
            
            # Only show x-axis labels on bottom subplot
            if i == n_years - 1:
                week_days = [pd.Timestamp(f'2000-04-{d:02d}').dayofyear for d in range(1, 8)]
                week_labels = [f'Apr {d}' for d in range(1, 8)]
                ax.set_xticks(week_days)
                ax.set_xticklabels(week_labels)
                ax.set_xlabel('Day of Year', fontsize=9)
        
        # Adjust layout to be more compact
        plt.tight_layout(h_pad=0.5)
        plot_path_week = output_dir / f'wind_{airport}_april1week.png'
        plt.savefig(plot_path_week, dpi=100, bbox_inches='tight')
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
        section_title = "Wind Speed and Direction Analysis"
        description = """
            <div class="intro-section section">
                <h2>Wind Speed and Direction Analysis</h2>
                
                <h3>Full Year Analysis</h3>
                <p>The full-year plots below show wind speed patterns across different years for each airport. 
                Each subplot represents a single year, allowing for clear comparison of seasonal patterns and 
                year-to-year variations. The x-axis spans the entire year (January through December), while 
                the y-axis shows wind speed measurements.</p>
                
                <h3>Weekly Analysis</h3>
                <p>The weekly plots focus on a specific time period (April 1-7) to examine short-term wind 
                patterns in detail. Each subplot shows both wind speed (y-axis) and wind direction (arrow 
                orientation) for a single year. The arrows indicate wind direction, with their length being 
                constant to avoid visual clutter. The thin connecting lines help track wind speed trends 
                throughout the week.</p>
                
                <h3>Key Observations</h3>
                <ul>
                    <li>Seasonal patterns in wind speed and direction can be observed in the full-year plots</li>
                    <li>The weekly plots reveal daily variations and potential diurnal patterns</li>
                    <li>Wind direction arrows help identify prevailing wind patterns and their changes</li>
                    <li>Year-to-year comparisons show both consistent patterns and notable variations</li>
                </ul>
            </div>
        """
        plots_html = ""
        for idx, (airport, plot_path, plot_path_week) in enumerate(plot_paths):
            if idx == 0:
                plots_html += '<hr class="section-divider">'
            else:
                plots_html += '\n<hr class="section-divider">\n'
            plots_html += f"""
                <div class="section airport-section">
                    <h3>{airport} Analysis</h3>
                    <div class="visualization-section">
                        <h4>Full Year Wind Patterns</h4>
                        <img src="{plot_path.name}" alt="Wind Speed - {airport}" style="max-width: 100%; height: auto;">
                        <p class="figure-caption">Full year wind speed patterns for {airport}, with each subplot showing data for a single year.</p>
                        
                        <h4>Weekly Wind Patterns (April 1-7)</h4>
                        <img src="{plot_path_week.name}" alt="Wind Speed {airport} April 1-7" style="max-width: 100%; height: auto;">
                        <p class="figure-caption">Detailed view of wind speed and direction for the week of April 1-7, with arrows indicating wind direction.</p>
                    </div>
                </div>
            """
        content_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <link rel="stylesheet" href="../style.css">
                <title>Wind Analysis</title>
            </head>
            <body>
                <div class="container">
                    {description}
                    {plots_html}
                </div>
            </body>
            </html>
        """
        manager.save_section_html("Introduction", content_html, "wind.html")
        
        logger.info("Created wind analysis section")
        
        # Create data overview
        analyze_dataset(df, manager)
        
    except Exception as e:
        logger.error(f"Error creating wind plot: {str(e)}")
        raise
