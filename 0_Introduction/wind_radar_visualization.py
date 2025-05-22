import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import sys
sys.path.append('..')  # Add parent directory to path to import html_manager
from html_manager import HTMLManager
from Utils.latex_utility import save_latex_file, latex_section, latex_subsection, latex_list

logger = logging.getLogger(__name__)

def create_wind_radar_plot(df, date, output_path=None):
    """
    Create a radar (polar scatter) plot of wind speed by direction, colored by time-of-day bins.
    Data is for Chicago O'Hare (KORD) for a specific date.
    """
    try:
        # Filter for specific date
        target_date = pd.to_datetime(date)
        df = df[df['datetime'].dt.date == target_date.date()]
        
        if df.empty:
            logger.error(f"No data found for {date}.")
            return None

        # Define time-of-day bins and labels (covering all 24 hours)
        bins = [0, 6, 12, 18, 24]
        labels = ['Night', 'Morning', 'Afternoon', 'Evening']
        colors = {'Night': 'purple', 'Morning': 'orange', 'Afternoon': 'green', 'Evening': 'blue'}
        df['hour'] = df['datetime'].dt.hour
        df['time_bin'] = pd.cut(df['hour'] % 24, bins=bins, labels=labels, right=False, include_lowest=True)
        
        # Convert wind direction to radians for polar plot
        df['winddir_rad'] = np.deg2rad(df['winddirection'])
        
        # Create polar scatter plot with smaller figure size
        fig = plt.figure(figsize=(7, 7))  # Reduced from (8, 8)
        ax = plt.subplot(111, polar=True)
        for label in labels:
            subset = df[df['time_bin'] == label]
            ax.scatter(subset['winddir_rad'], subset['windspeed'], label=label, color=colors[label], alpha=0.8)
        
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        # Remove time from the title
        ax.set_title(f'Wind Speed by Direction and Time of Day (KORD) - {date.strftime("%B %d, %Y")}', fontsize=16, pad=20)
        ax.legend(title=None, loc='upper right', bbox_to_anchor=(1.2, 1.1))
        
        # Save the plot
        if output_path is None:
            output_path = Path(__file__).parent / 'outputs' / f'wind_radar_plot_{date.strftime("%Y%m%d")}.png'
        else:
            output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"Wind radar plot saved to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error creating wind radar plot: {str(e)}")
        raise

def create_html_report(plot_paths, html_manager):
    """
    Create HTML report for the wind radar visualization
    """
    # Create introduction section with enhanced explanation
    intro = """
    <div class="model-report">
        <h2>Wind Pattern Analysis: KORD Airport</h2>
        
        <div class="model-description">
            <h3>Analysis Overview</h3>
            <p>This analysis examines wind patterns at Chicago O'Hare International Airport (KORD) during the first week of April 2020. 
            The visualization shows wind speed and direction patterns throughout the day, categorized into four time periods:</p>
            
            <ul>
                <li><strong>Night:</strong> 00:00-06:00 (Purple)</li>
                <li><strong>Morning:</strong> 06:00-12:00 (Orange)</li>
                <li><strong>Afternoon:</strong> 12:00-18:00 (Green)</li>
                <li><strong>Evening:</strong> 18:00-24:00 (Blue)</li>
            </ul>
            
            <h4>Visualization Details</h4>
            <ul>
                <li><strong>Radial Distance:</strong> Represents wind speed in meters per second (m/s)</li>
                <li><strong>Angle (Wind Direction):</strong> Represents the direction the wind is coming from:
                    <ul>
                        <li>0° (North): Wind coming from the north</li>
                        <li>90° (East): Wind coming from the east</li>
                        <li>180° (South): Wind coming from the south</li>
                        <li>270° (West): Wind coming from the west</li>
                    </ul>
                </li>
                <li><strong>Color Coding:</strong> Indicates time of day, helping to identify daily wind patterns</li>
            </ul>

            <h4>Interpretation Guide</h4>
            <p>When reading these radar plots:</p>
            <ul>
                <li>Points further from the center indicate stronger winds</li>
                <li>The angle shows where the wind is coming from (not where it's going)</li>
                <li>Clusters of points in similar directions suggest prevailing wind patterns</li>
                <li>Different colors in the same direction indicate how wind patterns change throughout the day</li>
            </ul>
        </div>
    </div>
    """
    
    # Create visualization sections
    visualization_sections = ""
    for date, plot_path in plot_paths.items():
        section = html_manager.create_section_with_image(
            plot_path,
            f"Wind Patterns for {date.strftime('%B %d, %Y')}",
            f"Radar plot showing wind speed and direction patterns throughout the day on {date.strftime('%B %d, %Y')}. "
            "The plot reveals how wind patterns change across different times of day, with the angle indicating the direction "
            "the wind is coming from and the distance from the center showing the wind speed."
        )
        visualization_sections += section
    
    # Combine all sections
    content = intro + visualization_sections
    
    # Save the HTML file
    html_content = html_manager.template.format(
        title="Wind Pattern Analysis: KORD Airport",
        content=content,
        additional_js=""
    )
    output_path = html_manager.save_section_html("Wind_Radar_Analysis", html_content, "wind_radar_analysis.html")
    return html_content, output_path

def create_latex_report(plot_paths, output_dir):
    """
    Create LaTeX report for the wind radar visualization
    """
    intro = latex_section("Wind Pattern Analysis: KORD Airport",
        "This analysis examines wind patterns at Chicago O'Hare International Airport (KORD) during the first week of April 2020. "
        "The visualization shows wind speed and direction patterns throughout the day, categorized into four time periods:" +
        latex_list([
            "Night: 00:00-06:00 (Purple)",
            "Morning: 06:00-12:00 (Orange)",
            "Afternoon: 12:00-18:00 (Green)",
            "Evening: 18:00-24:00 (Blue)"
        ]) +
        latex_subsection("Visualization Details",
            latex_list([
                "Radial Distance: Represents wind speed in meters per second (m/s)",
                "Angle (Wind Direction): Represents the direction the wind is coming from:"
            ]) +
            latex_list([
                "0° (North): Wind coming from the north",
                "90° (East): Wind coming from the east",
                "180° (South): Wind coming from the south",
                "270° (West): Wind coming from the west"
            ]) +
            "Color Coding: Indicates time of day, helping to identify daily wind patterns"
        ) +
        latex_subsection("Interpretation Guide",
            "When reading these radar plots:" + latex_list([
                "Points further from the center indicate stronger winds",
                "The angle shows where the wind is coming from (not where it's going)",
                "Clusters of points in similar directions suggest prevailing wind patterns",
                "Different colors in the same direction indicate how wind patterns change throughout the day"
            ])
        )
    )
    # Create visualization sections
    visualization_sections = ""
    for date, plot_path in plot_paths.items():
        section = latex_subsection(
            f"Wind Patterns for {date.strftime('%B %d, %Y')}",
            f"\\begin{{figure}}[htbp]\n\\centering\n\\includegraphics[width=0.7\\textwidth]{{{Path(plot_path).name}}}\n\\caption{{Wind Patterns for {date.strftime('%B %d, %Y')}}}\n\\label{{fig:wind_{date.strftime('%Y%m%d')}}}\n\\end{{figure}}\n"
        )
        visualization_sections += section
    latex_content = intro + visualization_sections
    latex_output_path = output_dir / 'wind_radar_analysis.tex'
    save_latex_file(latex_content, latex_output_path)
    logger.info(f"LaTeX wind radar analysis report created: {latex_output_path}")
    return latex_output_path

def main():
    try:
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Create outputs directory
        output_dir = Path(__file__).parent / 'outputs'
        output_dir.mkdir(exist_ok=True)
        
        # Create HTML manager
        manager = HTMLManager()
        manager.register_section("Wind_Radar_Analysis", Path(__file__).parent)
        
        # Read the data with explicit datetime parsing
        df = pd.read_csv('../datastep2.csv', parse_dates=['datetime'])
        
        # Filter for KORD only
        if 'id' in df.columns:
            df = df[df['id'] == 'KORD']
        
        # Generate plots for April 1-7, 2020
        plot_paths = {}
        for day in range(1, 8):
            date = pd.Timestamp(f'2020-04-{day:02d}')
            plot_path = create_wind_radar_plot(df, date, output_dir / f'wind_radar_plot_{date.strftime("%Y%m%d")}.png')
            if plot_path:
                plot_paths[date] = plot_path
        
        # Create HTML report
        html_content, output_path = create_html_report(plot_paths, manager)
        logger.info(f"Analysis complete. Results saved to {output_path}")
        # Create LaTeX report
        create_latex_report(plot_paths, output_dir)
        
    except Exception as e:
        logger.error(f"Error in wind radar analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()