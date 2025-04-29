import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import base64
import os

# Create outputs directory if it doesn't exist
output_dir = Path(__file__).parent / 'outputs'
output_dir.mkdir(exist_ok=True)

def create_html_with_image(image_path, html_path='outputs/master.html'):
    # Convert PNG to base64 for embedding in HTML
    with open(image_path, 'rb') as img_file:
        img_data = base64.b64encode(img_file.read()).decode('utf-8')
    
    # Create or append to HTML file
    html_content = f"""
    <div style="margin: 20px;">
        <img src="data:image/png;base64,{img_data}" alt="Plot">
    </div>
    """
    
    # If file exists, insert new content before the closing body tag
    if Path(html_path).exists():
        with open(html_path, 'r') as file:
            existing_content = file.read()
        if '</body>' in existing_content:
            new_content = existing_content.replace('</body>', f'{html_content}</body>')
        else:
            new_content = existing_content + html_content
    else:
        new_content = f"""
        <html>
        <head><title>Data Visualization</title></head>
        <body>
        {html_content}
        </body>
        </html>
        """
    
    with open(html_path, 'w') as file:
        file.write(new_content)

# Read the CSV file
df = pd.read_csv('../datastep2.csv')

# Convert datetime to proper datetime type
df['datetime'] = pd.to_datetime(df['datetime'])

# Create multiple plots
def create_weather_plots():
    # 1. Temperature over time
    plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], df['temp'], 'r-')
    plt.title('Temperature Over Time')
    plt.xlabel('Date')
    plt.ylabel('Temperature')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/temperature.png')
    plt.close()
    create_html_with_image('outputs/temperature.png')

    # 2. Wind speed and direction
    plt.figure(figsize=(12, 6))
    plt.scatter(df['winddirection'], df['windspeed'], alpha=0.5)
    plt.title('Wind Speed vs Direction')
    plt.xlabel('Wind Direction')
    plt.ylabel('Wind Speed')
    plt.tight_layout()
    plt.savefig('outputs/wind.png')
    plt.close()
    create_html_with_image('outputs/wind.png')

    # 3. Humidity and Temperature relationship
    plt.figure(figsize=(12, 6))
    plt.scatter(df['temp'], df['humidity'], alpha=0.5)
    plt.title('Humidity vs Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Humidity (%)')
    plt.tight_layout()
    plt.savefig('outputs/humidity_temp.png')
    plt.close()
    create_html_with_image('outputs/humidity_temp.png')

    # 4. Barometric Pressure over time
    plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], df['barometric'], 'b-')
    plt.title('Barometric Pressure Over Time')
    plt.xlabel('Date')
    plt.ylabel('Pressure')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/pressure.png')
    plt.close()
    create_html_with_image('outputs/pressure.png')

# Create all plots
create_weather_plots()