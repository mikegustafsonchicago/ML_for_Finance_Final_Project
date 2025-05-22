import pandas as pd
import logging
from pathlib import Path
import sys
sys.path.append('..')  # Add parent directory to path to import html_manager
from html_manager import HTMLManager
from Utils.latex_utility import save_latex_file, latex_section, latex_subsection, latex_list

logger = logging.getLogger(__name__)

def create_weather_explanation(html_manager: HTMLManager = None):
    """Create a layman-friendly explanation of the weather data, with both HTML and LaTeX outputs."""
    logger.info("Creating weather data explanation...")
    
    # Initialize HTML manager if not provided
    if html_manager is None:
        html_manager = HTMLManager()
        html_manager.register_section("Introduction", Path(__file__).parent)
    
    # --- HTML Content ---
    content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link rel="stylesheet" href="../../style.css">
    </head>
    <body>
        <div class="container">
            <div class="section">
                <h2>Understanding Weather Data</h2>
                
                <p>This dataset contains detailed weather observations from airports in the Chicago area, recorded in a special format called METAR (Meteorological Aerodrome Report). 
                METAR is the international standard code format for hourly weather observations used by meteorologists, pilots, and aviation professionals worldwide. 
                These reports are typically issued every hour, providing a snapshot of current weather conditions at airports.</p>

                <h3>What is METAR Data?</h3>
                <div class="data-list">
                    <p>METAR reports follow a standardized format that includes:</p>
                    <ul>
                        <li><strong>Time:</strong> Observations are typically taken every hour, on the hour</li>
                        <li><strong>Location:</strong> Each report is tagged with a unique identifier for the airport</li>
                        <li><strong>Weather Elements:</strong> Temperature, wind, visibility, cloud cover, and other conditions</li>
                        <li><strong>Special Conditions:</strong> Any significant weather phenomena like rain, snow, or fog</li>
                    </ul>
                    <p>This standardization ensures that weather information can be quickly understood and used by aviation professionals worldwide, regardless of language barriers.</p>
                </div>
                
                <h3>Basic Weather Measurements</h3>
                
                <div class="data-list">
                    <h4>Temperature and Dew Point</h4>
                    <p>Temperature measurements in this dataset are recorded in Celsius (°C):</p>
                    <ul>
                        <li><strong>Temperature (temp):</strong> The actual air temperature, ranging from -38.3°C to 38.3°C in our dataset. 
                        This is measured at a standard height of 2 meters above ground level.</li>
                        <li><strong>Dew Point (dew):</strong> The temperature at which water vapor in the air would start to condense into liquid water. 
                        When the temperature and dew point are close, it indicates high humidity and possible fog formation. 
                        In our data, dew points range from -45°C to 31.7°C.</li>
                    </ul>
                    <p>To convert to Fahrenheit: multiply by 1.8 and add 32.</p>
                </div>
                
                <div class="data-list">
                    <h4>Wind Measurements</h4>
                    <p>Wind is described by two main components:</p>
                    <ul>
                        <li><strong>Wind Speed (windspeed):</strong> Measured in knots (1 knot = 1.15 mph). 
                        In our dataset, wind speeds range from 0 to 124.2 knots, with an average of 13.2 knots. 
                        This is measured at a standard height of 10 meters above ground level.</li>
                        <li><strong>Wind Direction (winddirection):</strong> Measured in degrees from true north (0° = North, 90° = East, 180° = South, 270° = West). 
                        Our data shows directions ranging from 0° to 356°, with an average of 203.7° (southwest).</li>
                    </ul>
                </div>
                
                <div class="data-list">
                    <h4>Humidity and Pressure</h4>
                    <p>These measurements help us understand the moisture content and air pressure:</p>
                    <ul>
                        <li><strong>Humidity (humidity):</strong> Measured as a percentage (0-100%). 
                        Our data shows humidity ranging from 2% to 100%, with an average of 72.4%. 
                        This indicates the amount of water vapor in the air compared to the maximum possible at that temperature.</li>
                        <li><strong>Sea Level Pressure (sealevel):</strong> Measured in millibars (mb), ranging from 978.9 to 1048.3 mb in our dataset. 
                        This is the atmospheric pressure adjusted to sea level, which helps in comparing pressure readings from different altitudes.</li>
                        <li><strong>Barometric Tendency (barometric):</strong> Shows if the pressure is rising (R), falling (F), or steady (S). 
                        This helps predict weather changes, as falling pressure often indicates approaching storms.</li>
                    </ul>
                </div>
                
                <h3>Sky Conditions</h3>
                
                <div class="data-list">
                    <h4>Cloud Cover and Height</h4>
                    <p>The sky conditions are described using standardized terms and measurements:</p>
                    <ul>
                        <li><strong>Sky Descriptor (skydescriptor):</strong> A numerical code (1-33) that describes the overall sky condition. 
                        Our data shows values ranging from 1 to 33, with an average of 8.8.</li>
                        <li><strong>Cloud Heights (mincloud, maxcloud):</strong> Measured in meters above ground level:
                            <ul>
                                <li>Minimum cloud height (mincloud): Ranges from 0 to 14,326 meters</li>
                                <li>Maximum cloud height (maxcloud): Ranges from 0 to 29,535 meters</li>
                            </ul>
                        </li>
                        <li><strong>Sky Condition (skycondition):</strong> Describes the cloud coverage using standard terms:
                            <ul>
                                <li>Clear (C): No clouds</li>
                                <li>Few (F): 1-2/8 of the sky covered</li>
                                <li>Scattered (S): 3-4/8 of the sky covered</li>
                                <li>Broken (B): 5-7/8 of the sky covered</li>
                                <li>Overcast (O): 8/8 of the sky covered</li>
                                <li>Obscured (X): Sky is hidden by fog, smoke, etc.</li>
                            </ul>
                        </li>
                    </ul>
                </div>
                
                <h3>Weather Phenomena</h3>
                
                <div class="data-list">
                    <h4>Precipitation and Weather Types</h4>
                    <p>The data includes various types of weather conditions, recorded using standard METAR codes:</p>
                    <ul>
                        <li><strong>Rain (RA):</strong> Liquid precipitation</li>
                        <li><strong>Snow (SN):</strong> Frozen precipitation</li>
                        <li><strong>Thunderstorms (TS):</strong> Storms with lightning and thunder</li>
                        <li><strong>Fog (FG):</strong> Reduced visibility due to water droplets in the air</li>
                        <li><strong>Haze (HZ):</strong> Reduced visibility due to dust or smoke</li>
                    </ul>
                    <p>These conditions can be modified with intensity indicators:</p>
                    <ul>
                        <li><strong>Light (-):</strong> Light intensity</li>
                        <li><strong>Heavy (+):</strong> Heavy intensity</li>
                        <li><strong>Showers (SH):</strong> Brief periods of precipitation</li>
                        <li><strong>Freezing (FZ):</strong> Precipitation that freezes on contact</li>
                    </ul>
                </div>
                
                <div class="data-list">
                    <h4>Visibility</h4>
                    <p>Visibility is measured in miles, ranging from 0 to 32.19 miles in our dataset, with an average of 14.60 miles. 
                    This measurement is crucial for aviation safety and indicates how far a pilot can see clearly. 
                    When visibility drops below certain thresholds, it can affect airport operations and flight safety.</p>
                </div>
                
                <h3>Data Quality and Completeness</h3>
                
                <div class="data-list">
                    <p>Our dataset contains 1,900,862 weather observations, with most parameters having very few missing values (less than 0.1%). 
                    However, there are some notable exceptions:</p>
                    <ul>
                        <li>Maximum cloud height (maxcloud) has 17.51% missing values, which is common as this measurement is only taken when multiple cloud layers are present</li>
                        <li>Minimum cloud height (mincloud) has 0.16% missing values</li>
                        <li>Wind speed has 0.08% missing values</li>
                        <li>Temperature and temperature string have 0.02% missing values each</li>
                    </ul>
                    <p>These missing values are typical in weather data and can occur due to equipment malfunctions, maintenance periods, or when certain conditions are not present to measure.</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save the HTML file
    output_path = Path(__file__).parent / 'outputs' / 'weather_explanation.html'
    html_manager.save_section_html("Introduction", content, "weather_explanation.html")
    logger.info(f"Weather explanation created: {output_path}")

    # --- LaTeX Output ---
    latex_content = latex_section("Understanding Weather Data",
        "This dataset contains detailed weather observations from airports in the Chicago area, recorded in a special format called METAR (Meteorological Aerodrome Report). METAR is the international standard code format for hourly weather observations used by meteorologists, pilots, and aviation professionals worldwide. These reports are typically issued every hour, providing a snapshot of current weather conditions at airports.")
    latex_content += latex_subsection("What is METAR Data?",
        "METAR reports follow a standardized format that includes:" + latex_list([
            "Time: Observations are typically taken every hour, on the hour",
            "Location: Each report is tagged with a unique identifier for the airport",
            "Weather Elements: Temperature, wind, visibility, cloud cover, and other conditions",
            "Special Conditions: Any significant weather phenomena like rain, snow, or fog"
        ]) +
        "This standardization ensures that weather information can be quickly understood and used by aviation professionals worldwide, regardless of language barriers.")
    latex_content += latex_subsection("Basic Weather Measurements",
        latex_subsection("Temperature and Dew Point",
            "Temperature measurements in this dataset are recorded in Celsius (\textdegree C):" + latex_list([
                "Temperature (temp): The actual air temperature, ranging from -38.3\textdegree C to 38.3\textdegree C in our dataset. This is measured at a standard height of 2 meters above ground level.",
                "Dew Point (dew): The temperature at which water vapor in the air would start to condense into liquid water. When the temperature and dew point are close, it indicates high humidity and possible fog formation. In our data, dew points range from -45\textdegree C to 31.7\textdegree C."
            ]) + "To convert to Fahrenheit: multiply by 1.8 and add 32.") +
        latex_subsection("Wind Measurements",
            "Wind is described by two main components:" + latex_list([
                "Wind Speed (windspeed): Measured in knots (1 knot = 1.15 mph). In our dataset, wind speeds range from 0 to 124.2 knots, with an average of 13.2 knots. This is measured at a standard height of 10 meters above ground level.",
                "Wind Direction (winddirection): Measured in degrees from true north (0\textdegree = North, 90\textdegree = East, 180\textdegree = South, 270\textdegree = West). Our data shows directions ranging from 0\textdegree to 356\textdegree, with an average of 203.7\textdegree (southwest)."
            ])) +
        latex_subsection("Humidity and Pressure",
            "These measurements help us understand the moisture content and air pressure:" + latex_list([
                "Humidity (humidity): Measured as a percentage (0-100%). Our data shows humidity ranging from 2\% to 100\%, with an average of 72.4\%. This indicates the amount of water vapor in the air compared to the maximum possible at that temperature.",
                "Sea Level Pressure (sealevel): Measured in millibars (mb), ranging from 978.9 to 1048.3 mb in our dataset. This is the atmospheric pressure adjusted to sea level, which helps in comparing pressure readings from different altitudes.",
                "Barometric Tendency (barometric): Shows if the pressure is rising (R), falling (F), or steady (S). This helps predict weather changes, as falling pressure often indicates approaching storms."
            ]))
    )
    latex_content += latex_subsection("Sky Conditions",
        latex_subsection("Cloud Cover and Height",
            "The sky conditions are described using standardized terms and measurements:" + latex_list([
                "Sky Descriptor (skydescriptor): A numerical code (1-33) that describes the overall sky condition. Our data shows values ranging from 1 to 33, with an average of 8.8.",
                "Cloud Heights (mincloud, maxcloud): Measured in meters above ground level: Minimum cloud height (mincloud): Ranges from 0 to 14,326 meters; Maximum cloud height (maxcloud): Ranges from 0 to 29,535 meters.",
                "Sky Condition (skycondition): Describes the cloud coverage using standard terms:" + latex_list([
                    "Clear (C): No clouds",
                    "Few (F): 1-2/8 of the sky covered",
                    "Scattered (S): 3-4/8 of the sky covered",
                    "Broken (B): 5-7/8 of the sky covered",
                    "Overcast (O): 8/8 of the sky covered",
                    "Obscured (X): Sky is hidden by fog, smoke, etc."
                ])
            ]))
    )
    latex_content += latex_subsection("Weather Phenomena",
        latex_subsection("Precipitation and Weather Types",
            "The data includes various types of weather conditions, recorded using standard METAR codes:" + latex_list([
                "Rain (RA): Liquid precipitation",
                "Snow (SN): Frozen precipitation",
                "Thunderstorms (TS): Storms with lightning and thunder",
                "Fog (FG): Reduced visibility due to water droplets in the air",
                "Haze (HZ): Reduced visibility due to dust or smoke"
            ]) +
            "These conditions can be modified with intensity indicators:" + latex_list([
                "Light (-): Light intensity",
                "Heavy (+): Heavy intensity",
                "Showers (SH): Brief periods of precipitation",
                "Freezing (FZ): Precipitation that freezes on contact"
            ])
        ) +
        latex_subsection("Visibility",
            "Visibility is measured in miles, ranging from 0 to 32.19 miles in our dataset, with an average of 14.60 miles. This measurement is crucial for aviation safety and indicates how far a pilot can see clearly. When visibility drops below certain thresholds, it can affect airport operations and flight safety.")
    )
    latex_content += latex_subsection("Data Quality and Completeness",
        "Our dataset contains 1,900,862 weather observations, with most parameters having very few missing values (less than 0.1\%). However, there are some notable exceptions:" + latex_list([
            "Maximum cloud height (maxcloud) has 17.51\% missing values, which is common as this measurement is only taken when multiple cloud layers are present",
            "Minimum cloud height (mincloud) has 0.16\% missing values",
            "Wind speed has 0.08\% missing values",
            "Temperature and temperature string have 0.02\% missing values each"
        ]) + "These missing values are typical in weather data and can occur due to equipment malfunctions, maintenance periods, or when certain conditions are not present to measure.")
    # Save the LaTeX file
    latex_output_path = Path(__file__).parent / 'outputs' / 'weather_explanation.tex'
    save_latex_file(latex_content, latex_output_path)
    logger.info(f"LaTeX weather explanation created: {latex_output_path}")
    return output_path

if __name__ == "__main__":
    try:
        # Create HTML manager
        manager = HTMLManager()
        manager.register_section("Introduction", Path(__file__).parent)
        
        # Create weather explanation
        create_weather_explanation(manager)
        
    except Exception as e:
        logger.error(f"Error creating weather explanation: {str(e)}")
        raise
