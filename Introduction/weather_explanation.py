import pandas as pd
import logging
from pathlib import Path
import sys
sys.path.append('..')  # Add parent directory to path to import html_manager
from html_manager import HTMLManager

logger = logging.getLogger(__name__)

def create_weather_explanation(html_manager: HTMLManager = None):
    """Create a layman-friendly explanation of the weather data"""
    logger.info("Creating weather data explanation...")
    
    # Initialize HTML manager if not provided
    if html_manager is None:
        html_manager = HTMLManager()
        html_manager.register_section("Introduction", Path(__file__).parent)
    
    content = f"""
    <div class="section">
        <h2>Understanding Weather Data</h2>
        
        <p>This dataset contains detailed weather observations from airports in the Chicago area. 
        These observations are similar to what you might hear in a weather report, but with more technical detail. 
        Let's break down what each measurement means in simple terms.</p>
        
        <h3>Basic Weather Measurements</h3>
        
        <div class="weather-parameter">
            <h4>Temperature</h4>
            <p>Temperature is measured in Celsius (°C). This is the same as the temperature you see on your weather app, 
            just in a different scale (to convert to Fahrenheit, multiply by 1.8 and add 32).</p>
        </div>
        
        <div class="weather-parameter">
            <h4>Wind</h4>
            <p>Wind measurements include two main components:</p>
            <ul>
                <li><strong>Wind Speed:</strong> Measured in knots (1 knot = 1.15 mph). This tells us how fast the wind is blowing.</li>
                <li><strong>Wind Direction:</strong> Measured in degrees (0° = North, 90° = East, 180° = South, 270° = West). 
                This tells us which way the wind is blowing from.</li>
            </ul>
        </div>
        
        <div class="weather-parameter">
            <h4>Humidity and Dew Point</h4>
            <p>These measurements help us understand how much moisture is in the air:</p>
            <ul>
                <li><strong>Humidity:</strong> Measured as a percentage (0-100%). This tells us how much water vapor is in the air compared to the maximum possible.</li>
                <li><strong>Dew Point:</strong> Measured in Celsius. This is the temperature at which water vapor in the air would start to condense into liquid water (like dew on grass).</li>
            </ul>
        </div>
        
        <h3>Sky Conditions</h3>
        
        <div class="weather-parameter">
            <h4>Cloud Cover</h4>
            <p>The sky conditions are described using these terms:</p>
            <ul>
                <li><strong>Clear (C):</strong> No clouds</li>
                <li><strong>Few (F):</strong> 1-2/8 of the sky covered</li>
                <li><strong>Scattered (S):</strong> 3-4/8 of the sky covered</li>
                <li><strong>Broken (B):</strong> 5-7/8 of the sky covered</li>
                <li><strong>Overcast (O):</strong> 8/8 of the sky covered</li>
                <li><strong>Obscured (X):</strong> Sky is hidden by fog, smoke, etc.</li>
            </ul>
        </div>
        
        <div class="weather-parameter">
            <h4>Cloud Height</h4>
            <p>Cloud bases are measured in meters above ground level. This helps pilots and meteorologists understand how low the clouds are.</p>
        </div>
        
        <h3>Weather Phenomena</h3>
        
        <div class="weather-parameter">
            <h4>Precipitation and Weather Types</h4>
            <p>The data includes various types of weather conditions:</p>
            <ul>
                <li><strong>Rain (RA):</strong> Liquid precipitation</li>
                <li><strong>Snow (SN):</strong> Frozen precipitation</li>
                <li><strong>Thunderstorms (TS):</strong> Storms with lightning and thunder</li>
                <li><strong>Fog (FG):</strong> Reduced visibility due to water droplets in the air</li>
                <li><strong>Haze (HZ):</strong> Reduced visibility due to dust or smoke</li>
            </ul>
            <p>These conditions can be modified with terms like:</p>
            <ul>
                <li><strong>Light (-):</strong> Light intensity</li>
                <li><strong>Heavy (+):</strong> Heavy intensity</li>
                <li><strong>Showers (SH):</strong> Brief periods of precipitation</li>
                <li><strong>Freezing (FZ):</strong> Precipitation that freezes on contact</li>
            </ul>
        </div>
        
        <h3>Additional Measurements</h3>
        
        <div class="weather-parameter">
            <h4>Pressure and Visibility</h4>
            <ul>
                <li><strong>Sea Level Pressure:</strong> Measured in millibars. This is the atmospheric pressure adjusted to sea level.</li>
                <li><strong>Visibility:</strong> Measured in miles. This tells us how far we can see clearly.</li>
                <li><strong>Barometric Tendency:</strong> Shows if the pressure is rising (R), falling (F), or steady (S).</li>
            </ul>
        </div>
        
        <div class="weather-parameter">
            <h4>Precipitation Amounts</h4>
            <p>The dataset includes precipitation measurements for different time periods (1, 3, 6, 12, and 24 hours), 
            measured in centimeters. This helps track how much rain or snow has fallen over time.</p>
        </div>
        
        <h3>Time and Location</h3>
        
        <div class="weather-parameter">
            <h4>Time Measurements</h4>
            <p>All times are recorded in multiple formats:</p>
            <ul>
                <li><strong>UTC:</strong> Universal Coordinated Time (the standard time reference)</li>
                <li><strong>LST:</strong> Local Standard Time</li>
                <li><strong>Local Time:</strong> The actual time at the observation location</li>
            </ul>
        </div>
        
        <div class="weather-parameter">
            <h4>Location</h4>
            <p>Each observation is tagged with a unique identifier for the airport or weather station where it was recorded.</p>
        </div>
    </div>
    """
    
    # Save the HTML file
    output_path = Path(__file__).parent / 'outputs' / 'weather_explanation.html'
    html_manager.save_section_html("Introduction", content, "weather_explanation.html")
    
    logger.info(f"Weather explanation created: {output_path}")
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
