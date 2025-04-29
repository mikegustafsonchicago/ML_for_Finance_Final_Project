import logging
from pathlib import Path
import base64

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def create_title_page(output_path='outputs/title.html'):
    """Create the title page HTML"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Weather Futures Analysis</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                line-height: 1.6;
            }
            .title-page {
                height: 100vh;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                text-align: center;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .title {
                font-size: 2.5em;
                margin-bottom: 20px;
                color: #333;
            }
            .subtitle {
                font-size: 1.5em;
                margin-bottom: 30px;
                color: #666;
            }
            .authors {
                font-size: 1.2em;
                margin-bottom: 20px;
                color: #444;
            }
            .date {
                font-size: 1.1em;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="title-page">
            <h1 class="title">Examining the Efficiency of Weather Futures</h1>
            <div class="subtitle">Data Analysis and Visualization Report</div>
            <div class="authors">
                Group 17:<br>
                Ashton Coates, Chad Schmerling, David Oropeza, Mike Gustafson
            </div>
            <div class="subtitle">BUSN 35137: Machine Learning in Finance</div>
            <div class="date">April 14, 2025</div>
        </div>
    </body>
    </html>
    """
    with open(output_path, 'w') as f:
        f.write(html_content)
    logger.info(f"Title page created: {output_path}")
    return output_path

def create_section_with_image(image_path, section_title, output_path):
    """Create an HTML section with an embedded image"""
    with open(image_path, 'rb') as img_file:
        img_data = base64.b64encode(img_file.read()).decode('utf-8')
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{section_title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                line-height: 1.6;
            }}
            .section {{
                margin: 20px;
                padding: 20px;
                background-color: white;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h2 {{
                color: #333;
                margin-bottom: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="section">
            <h2>{section_title}</h2>
            <img src="data:image/png;base64,{img_data}" alt="{section_title}">
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    logger.info(f"Section created: {output_path}")
    return output_path

def combine_html_files(html_files, output_path='outputs/master.html'):
    """Combine multiple HTML files into one"""
    combined_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Weather Futures Analysis</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                line-height: 1.6;
            }
            .section {
                margin: 20px;
                padding: 20px;
            }
        </style>
    </head>
    <body>
    """
    
    for html_file in html_files:
        with open(html_file, 'r') as f:
            content = f.read()
            body_start = content.find('<body')
            body_end = content.find('</body>')
            if body_start != -1 and body_end != -1:
                body_content = content[body_start:body_end + 7]
                combined_content += f"<div class='section'>{body_content}</div>"
    
    combined_content += "</body></html>"
    
    with open(output_path, 'w') as f:
        f.write(combined_content)
    logger.info(f"Combined HTML created: {output_path}")
    return output_path
