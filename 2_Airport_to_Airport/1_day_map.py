import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import sys
sys.path.append('..')  # Add parent directory to path to import html_manager
from html_manager import HTMLManager
from Utils.latex_utility import save_latex_file, latex_section, df_to_latex_table
from Utils.data_formatter import parse_custom_datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("1_day_map")

# Dictionary of ICAO codes, names, and their coordinates (latitude, longitude)
AIRPORT_COORDS = {
    'KRZL': ("Reedsburg Municipal", 43.1156, -90.6825),
    'KRYV': ("Watertown Municipal", 43.1711, -88.7243),
    'KMKE': ("Milwaukee", 42.9550, -87.8989),
    'KRPJ': ("Rochelle Municipal", 42.2461, -89.5821),
    'KORD': ("Chicago O'Hare", 41.9786, -87.9048),
    'KJOT': ("Joliet Regional", 41.5178, -88.1756),
    'KPNT': ("Pontiac Municipal", 40.9193, -88.6926),
    'KOXI': ("Starke County", 41.3533, -86.9989),
    'KIGQ': ("Boone County", 41.5239, -85.7979),
}

def load_weather_data(file_path='../datastep2.csv', date='2020-04-02'):
    """Load weather data for a specific date."""
    df = pd.read_csv(file_path)
    df['datetime'] = parse_custom_datetime(df['datetime'])
    filtered_df = df[df['datetime'].dt.date == pd.to_datetime(date).date()]
    logger.info(f"Filtered data shape for {date}: {filtered_df.shape}")
    logger.info(f"Available dates in data: {sorted(df['datetime'].dt.date.unique())}")
    logger.info(f"Available airports in data: {sorted(df['id'].unique())}")
    return filtered_df

def create_weather_map(weather_data, output_dir, time, html_manager: HTMLManager = None):
    """Create a map showing weather at all airports for a specific time."""
    logger.info(f"Creating weather map for {time}...")
    
    # Filter data for specific time
    time_hour = int(time.split(':')[0])
    time_data = weather_data[weather_data['datetime'].dt.hour == time_hour]
    logger.info(f"Data points for {time}: {len(time_data)}")
    
    # Initialize HTML manager if not provided
    if html_manager is None:
        html_manager = HTMLManager()
        html_manager.register_section("Weather Maps", Path(__file__).parent)
    
    # Calculate map bounds (with some padding)
    lats = [coord[1] for coord in AIRPORT_COORDS.values()]
    lons = [coord[2] for coord in AIRPORT_COORDS.values()]
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)
    
    # Create figure and axis with projection
    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set map bounds
    padding = 1.0  # degrees
    ax.set_extent([
        min(lons) - padding,
        max(lons) + padding,
        min(lats) - padding,
        max(lats) + padding
    ])
    
    # Add map features with improved styling
    ax.add_feature(cfeature.STATES.with_scale('10m'), linewidth=0.5, edgecolor='gray')
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5, edgecolor='gray')
    ax.add_feature(cfeature.LAKES.with_scale('10m'), alpha=0.5, edgecolor='gray')
    
    # Add terrain background
    ax.add_feature(cfeature.LAND.with_scale('10m'), alpha=0.3)
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), alpha=0.3)
    
    # Plot airports with improved styling
    for icao, (name, lat, lon) in AIRPORT_COORDS.items():
        # Get weather data for this airport and time
        airport_data = time_data[time_data['id'] == icao]
        if len(airport_data) == 0:
            logger.warning(f"No data found for {icao} at {time}")
            continue
            
        weather_info = airport_data.iloc[0]
        logger.info(f"Processing {icao} at {time}: Temp={weather_info['temp']}, Wind={weather_info['windspeed']} m/s")
        
        # Calculate color based on temperature
        temp = weather_info['temp']
        # Normalize temperature to 0-1 range (assuming typical range of -10 to 40°C)
        temp_norm = (temp + 10) / 50
        # Use a colormap (cool to warm)
        marker_color = plt.cm.coolwarm(temp_norm)
        
        # Plot airport marker with temperature-based color
        ax.plot(lon, lat, '^', markersize=10, color=marker_color, transform=ccrs.PlateCarree())
        
        # Add segmented sky map
        sky_condition = weather_info['skydescriptor']
        cloud_height = weather_info['mincloud']
        
        # Create sky condition visualization
        sky_x = lon + 0.15
        sky_y = lat
        sky_width = 0.1
        sky_height = 0.05
        
        # Draw sky condition box
        # Convert sky_condition to string and handle numeric values
        if isinstance(sky_condition, (int, float, np.number)):
            # If it's a number, use it to determine cloud coverage
            if sky_condition <= 0.1:
                color = 'lightblue'  # Clear
            elif sky_condition <= 0.3:
                color = 'skyblue'    # Scattered
            elif sky_condition <= 0.7:
                color = 'steelblue'  # Broken
            else:
                color = 'darkblue'   # Overcast
        else:
            # If it's a string, use the text-based logic
            sky_condition_str = str(sky_condition).lower()
            if 'clear' in sky_condition_str:
                color = 'lightblue'
            elif 'scattered' in sky_condition_str:
                color = 'skyblue'
            elif 'broken' in sky_condition_str:
                color = 'steelblue'
            elif 'overcast' in sky_condition_str:
                color = 'darkblue'
            else:
                color = 'gray'
            
        ax.add_patch(plt.Rectangle((sky_x, sky_y), sky_width, sky_height, 
                                  facecolor=color, alpha=0.7, 
                                  transform=ccrs.PlateCarree()))
        
        # Add weather data text box with improved styling and moved position
        # Move the text box significantly below the airport
        weather_text = f"{icao}\nSky: {sky_condition}\nCeiling: {cloud_height}ft\nWind: {weather_info['windspeed']} m/s\nDir: {weather_info['winddirection']}°\nTemp: {weather_info['temp']}°C"
        ax.text(lon, lat - 0.15, weather_text, fontsize=9, 
                transform=ccrs.PlateCarree(),
                horizontalalignment='center',
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8, 
                         edgecolor='gray', pad=2,
                         boxstyle='round,pad=0.5'))
    
    # Add wind barbs last so they're on top
    for icao, (name, lat, lon) in AIRPORT_COORDS.items():
        airport_data = time_data[time_data['id'] == icao]
        if len(airport_data) == 0:
            continue
            
        weather_info = airport_data.iloc[0]
        wind_speed = weather_info['windspeed']
        wind_dir = weather_info['winddirection']
        # Convert wind direction to radians and adjust for meteorological convention
        wind_dir_rad = np.radians(270 - wind_dir)  # 270 - dir because 0 is North in met convention
        # Scale wind speed for arrow length with enhanced size
        arrow_length = 0.25 * (1 + wind_speed/10)  # Increased base length and scaling
        
        # Calculate arrow end point
        arrow_x = lon + arrow_length * np.cos(wind_dir_rad)
        arrow_y = lat + arrow_length * np.sin(wind_dir_rad)
        
        # Draw arrow with head
        ax.annotate('', 
                   xy=(arrow_x, arrow_y),  # Arrow end point
                   xytext=(lon, lat),      # Arrow start point
                   arrowprops=dict(arrowstyle='->',  # Arrow style
                                 color='blue',
                                 linewidth=2,
                                 mutation_scale=15))  # Size of arrow head
    
    # Add gridlines with improved styling
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    
    # Add title with improved styling
    plt.title(f'Weather at Selected Airports: April 2, 2020 - {time}', pad=20, fontsize=16, fontweight='bold')
    
    # Add legend for sky conditions
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', alpha=0.7, label='Clear'),
        plt.Rectangle((0, 0), 1, 1, facecolor='skyblue', alpha=0.7, label='Scattered'),
        plt.Rectangle((0, 0), 1, 1, facecolor='steelblue', alpha=0.7, label='Broken'),
        plt.Rectangle((0, 0), 1, 1, facecolor='darkblue', alpha=0.7, label='Overcast')
    ]
    ax.legend(handles=legend_elements, loc='upper right', 
              title='Sky Conditions', framealpha=0.8)
    
    # Save the map
    output_path = output_dir / f'weather_map_{time.replace(":", "_")}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Build the HTML content
    content_html = f"""
        <div class="section">
            <div class="plot-group">
                <img src="{output_path.name}" alt="Weather Map" style="max-width: 100%; height: auto;">
            </div>
        </div>
    """
    
    # Add CSS styling
    css_path = Path(__file__).parent.parent / 'style.css'
    if css_path.exists():
        with open(css_path, 'r') as f:
            css_content = f.read()
        content_html = f"<style>{css_content}</style>{content_html}"
    
    manager.save_section_html("Weather Maps", content_html, f"weather_map_{time.replace(':', '_')}.html")
    logger.info(f"Weather map created for {time}: {output_path}")

    # --- LaTeX Output ---
    latex_content = latex_section(
        f"Weather at Airports: {time}",
        f"Map showing the weather at all airports for {time}."
    )
    latex_content += r"\begin{figure}[htbp]" + "\n" + \
        r"\centering" + "\n" + \
        f"\includegraphics[width=0.7\textwidth]{{{output_path.name}}}" + "\n" + \
        r"\caption{Weather at Airports: " + time + r"}" + "\n" + \
        r"\label{fig:weather_map_" + time.replace(":", "_") + r"}" + "\n" + \
        r"\end{figure}" + "\n"
    latex_output_path = output_dir / f'weather_map_{time.replace(":", "_")}.tex'
    save_latex_file(latex_content, latex_output_path)
    logger.info(f"LaTeX weather map created for {time}: {latex_output_path}")
    return output_path

def create_html_report(all_maps, html_manager):
    """
    Create a single HTML report containing all weather maps for the day
    Args:
        all_maps: Dictionary containing map paths for each time
        html_manager: HTMLManager instance
    """
    model_desc = """
    <div class="model-report">
        <h2>Weather Progression: April 2, 2020</h2>
        <div class="model-description">
            <h3>Analysis Overview</h3>
            <p>This analysis shows the progression of weather conditions across nine regional airports throughout April 2, 2020. The maps are shown at four key times during the day: midnight (00:00), early morning (06:00), noon (12:00), and evening (18:00).</p>
            <h4>Airports Included</h4>
            <ul>
                <li><strong>KRZL:</strong> Reedsburg Municipal</li>
                <li><strong>KRYV:</strong> Watertown Municipal</li>
                <li><strong>KMKE:</strong> Milwaukee</li>
                <li><strong>KRPJ:</strong> Rochelle Municipal</li>
                <li><strong>KORD:</strong> Chicago O'Hare</li>
                <li><strong>KJOT:</strong> Joliet Regional</li>
                <li><strong>KPNT:</strong> Pontiac Municipal</li>
                <li><strong>KOXI:</strong> Starke County</li>
                <li><strong>KIGQ:</strong> Boone County</li>
            </ul>
            <h4>Weather Information Displayed</h4>
            <ul>
                <li>🌡️ Temperature (airport marker color scales with temperature)</li>
                <li>🌤️ Sky Condition (color-coded boxes)</li>
                <li>💨 Wind Speed and Direction (blue arrows)</li>
                <li>📊 Ceiling Height</li>
            </ul>
        </div>
    </div>
    """
    
    # Create sections for each time snapshot
    prediction_sections = []
    for time, map_path in all_maps.items():
        section = f"""
        <div class="plot-group">
            <img src="{map_path.name}" alt="Weather Map" style="max-width: 100%; height: auto;">
        </div>
        """
        prediction_sections.append(section)
    
    # Combine all sections
    content = model_desc + "\n".join(prediction_sections)
    
    # Add interpretation section
    interpretation = """
    <div class="interpretation-section">
        <h3>Weather Pattern Analysis</h3>
        <p>These maps show how weather conditions evolved throughout April 2, 2020, across the Chicago metropolitan area and surrounding regions. The progression from midnight to evening reveals:</p>
        <ul>
            <li>🌡️ Daily temperature variations</li>
            <li>💨 Wind pattern changes</li>
            <li>☁️ Cloud cover development</li>
            <li>🌍 Regional weather system movements</li>
        </ul>
    </div>
    """
    content += interpretation
    
    html_content = html_manager.template.format(
        title="Weather Progression: April 2, 2020",
        content=content,
        additional_js=""
    )
    
    html_filename = "weather_progression.html"
    output_path = html_manager.save_section_html("Weather Maps", html_content, html_filename)
    return html_content, output_path

def create_latex_report(all_maps, output_dir):
    """
    Create a master LaTeX document containing all weather maps for the day
    Args:
        all_maps: Dictionary containing map paths for each time
        output_dir: Directory to save the LaTeX file
    """
    latex_content = r"""
\documentclass{article}
\usepackage{graphicx}
\usepackage{float}
\usepackage{hyperref}
\usepackage{geometry}
\geometry{a4paper, margin=1in}

\title{Weather Progression: April 2, 2020}
\author{Weather Analysis System}
\date{\today}

\begin{document}
\maketitle

\section{Analysis Overview}
This analysis shows the progression of weather conditions across nine regional airports throughout April 2, 2020. The maps are shown at four key times during the day: midnight (00:00), early morning (06:00), noon (12:00), and evening (18:00).

\subsection{Airports Included}
\begin{itemize}
    \item \textbf{KRZL}: Reedsburg Municipal
    \item \textbf{KRYV}: Watertown Municipal
    \item \textbf{KMKE}: Milwaukee
    \item \textbf{KRPJ}: Rochelle Municipal
    \item \textbf{KORD}: Chicago O'Hare
    \item \textbf{KJOT}: Joliet Regional
    \item \textbf{KPNT}: Pontiac Municipal
    \item \textbf{KOXI}: Starke County
    \item \textbf{KIGQ}: Boone County
\end{itemize}

\subsection{Weather Information Displayed}
\begin{itemize}
    \item Temperature (airport marker color scales with temperature)
    \item Sky Condition (color-coded boxes)
    \item Wind Speed and Direction (blue arrows)
    \item Ceiling Height
\end{itemize}

\section{Weather Progression}
"""
    
    # Add each map to the LaTeX document
    for time, map_path in all_maps.items():
        latex_content += f"""
\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=\\textwidth]{{{map_path.name}}}
    \\caption{{Weather at Selected Airports: April 2, 2020 - {time}}}
    \\label{{fig:weather_{time.replace(':', '_')}}}
\\end{{figure}}
"""
    
    # Add interpretation section
    latex_content += r"""
\section{Weather Pattern Analysis}
These maps show how weather conditions evolved throughout April 2, 2020, across the Chicago metropolitan area and surrounding regions. The progression from midnight to evening reveals:
\begin{itemize}
    \item Daily temperature variations
    \item Wind pattern changes
    \item Cloud cover development
    \item Regional weather system movements
\end{itemize}

\end{document}
"""
    
    # Save the LaTeX file
    latex_output_path = output_dir / 'weather_progression.tex'
    with open(latex_output_path, 'w') as f:
        f.write(latex_content)
    logger.info(f"Created LaTeX report: {latex_output_path}")
    return latex_output_path

if __name__ == "__main__":
    try:
        # Create outputs directory
        output_dir = Path(__file__).parent / 'outputs'
        output_dir.mkdir(exist_ok=True)
        
        # Create HTML manager
        manager = HTMLManager()
        manager.register_section("Weather Maps", Path(__file__).parent)
        
        # Load weather data for April 2, 2020
        weather_data = load_weather_data()
        
        # Dictionary to store all map paths
        all_maps = {}
        
        # Create maps for different times
        times = ['00:00', '06:00', '12:00', '18:00']
        for time in times:
            map_path = create_weather_map(weather_data, output_dir, time, manager)
            all_maps[time] = map_path
        
        # Create single HTML report with all maps
        html_content, output_path = create_html_report(all_maps, manager)
        logger.info(f"Created weather progression report: {output_path}")
        
        # Create single LaTeX report with all maps
        latex_output_path = create_latex_report(all_maps, output_dir)
        logger.info(f"Created LaTeX report: {latex_output_path}")
        
    except Exception as e:
        logger.error(f"Error creating weather maps: {str(e)}")
        raise
