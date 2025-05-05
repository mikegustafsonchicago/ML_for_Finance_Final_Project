import os
from pathlib import Path
from typing import List, Dict, Optional, Union
import shutil
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import base64

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class HTMLManager:
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.sections = {}
        self.template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="styles/style.css">
    {additional_css}
</head>
<body>
    <div class="container">
        {content}
    </div>
    <script src="scripts/main.js"></script>
    {additional_js}
</body>
</html>
"""
        self._setup_directories()
        logger.info("HTMLManager initialized")

    def _setup_directories(self):
        """Create necessary directories for the project."""
        dirs = ['styles', 'scripts', 'assets']
        for dir_name in dirs:
            (self.base_dir / dir_name).mkdir(exist_ok=True)
        
        # Copy the CSS file to the styles directory
        css_source = Path(__file__).parent / 'style.css'
        css_target = self.base_dir / 'styles' / 'style.css'
        if css_source.exists():
            shutil.copy2(css_source, css_target)
            logger.info("Copied style.css to styles directory")
        else:
            logger.warning("style.css not found in the same directory as html_manager.py")
        
        logger.info("Project directories created")

    def register_section(self, section_name: str, section_dir: str):
        """Register a new section with its directory."""
        section_path = Path(section_dir)
        if not section_path.exists():
            section_path.mkdir(parents=True)
            (section_path / 'outputs').mkdir()
        self.sections[section_name] = section_path
        logger.info(f"Registered section: {section_name}")

    def create_title_page(self, title: str = "Weather Futures Analysis", 
                         subtitle: str = "Data Analysis and Visualization Report",
                         authors: List[str] = None,
                         course: str = "BUSN 35137: Machine Learning in Finance",
                         date: str = "April 14, 2025",
                         output_file: str = "title.html") -> str:
        """Create a title page for the report."""
        if authors is None:
            authors = ["Group 17:", "Ashton Coates", "Chad Schmerling", "David Oropeza", "Mike Gustafson"]
        
        content = f"""
        <div class="title-page">
            <h1 class="title">{title}</h1>
            <div class="subtitle">{subtitle}</div>
            <div class="authors">
                {'<br>'.join(authors)}
            </div>
            <div class="subtitle">{course}</div>
            <div class="date">{date}</div>
        </div>
        """
        
        html_content = self.template.format(
            title=title,
            content=content,
            additional_css="",
            additional_js=""
        )
        
        if output_file:
            self.save_section_html(
                "Introduction",  # Title page is always part of Introduction
                html_content,
                output_file
            )
        
        return content

    def create_section_html(self, section_name: str, content: str, 
                          additional_css: str = "", additional_js: str = "") -> str:
        """Create HTML for a single section."""
        return self.template.format(
            title=f"{section_name} - Report",
            content=content,
            additional_css=additional_css,
            additional_js=additional_js
        )

    def save_section_html(self, section_name: str, html_content: str, filename: str):
        """Save a section's HTML to its outputs directory."""
        if section_name not in self.sections:
            raise ValueError(f"Section {section_name} not registered")
        
        output_dir = self.sections[section_name] / 'outputs'
        output_file = output_dir / filename
        output_file.write_text(html_content)
        logger.info(f"Saved section HTML: {output_file}")

    def create_section_with_image(self, image_path: Union[str, Path], title: str, 
                                description: str = "", output_file: str = None) -> str:
        """Create a section with an image and optional description."""
        if isinstance(image_path, str):
            image_path = Path(image_path)
        
        content = f"""
        <div class="section">
            <h2>{title}</h2>
            {f'<p>{description}</p>' if description else ''}
            <div class="plot-group">
                <img src="{image_path.name}" alt="{title}" style="max-width: 100%; height: auto;">
            </div>
        </div>
        """
        
        if output_file:
            self.save_section_html(
                self._get_section_from_path(image_path),
                self.create_section_html(title, content),
                output_file
            )
        
        return content

    def create_data_overview(self, df: pd.DataFrame, section_name: str) -> str:
        """Create a data overview section with key statistics."""
        columns = df.columns.tolist()
        airports = df['airport'].unique().tolist() if 'airport' in df.columns else []
        airports.sort()

        content = f"""
        <div class="section">
            <h2>Dataset Overview</h2>
            <p>Number of records: {len(df):,}</p>
            <p>Time range: {df['datetime'].min()} to {df['datetime'].max()}</p>
            
            <h3>Available Columns ({len(columns)})</h3>
            <div class="data-list">
                {', '.join(f'<span class="column-name">{col}</span>' for col in columns)}
            </div>
            
            <h3>Airport Locations ({len(airports)})</h3>
            <div class="data-list">
                {' '.join(f'<span class="airport-code">{airport}</span>' for airport in airports)}
            </div>
        </div>
        """
        
        return content

    def combine_sections(self, section_order: List[str], output_file: str = "index.html"):
        """Combine multiple sections into a single HTML file."""
        combined_content = []
        
        for section in section_order:
            if section not in self.sections:
                raise ValueError(f"Section {section} not registered")
            
            section_dir = self.sections[section]
            section_files = sorted((section_dir / 'outputs').glob('*.html'))
            
            for file in section_files:
                content = file.read_text()
                # Extract content between container divs
                start = content.find('<div class="container">') + len('<div class="container">')
                end = content.find('</div>', start)
                if start > 0 and end > 0:
                    combined_content.append(content[start:end].strip())

        final_html = self.template.format(
            title="Complete Report",
            content="\n".join(combined_content),
            additional_css="",
            additional_js=""
        )

        (self.base_dir / output_file).write_text(final_html)
        logger.info(f"Combined sections into: {output_file}")

    def copy_assets(self, source_dir: str, asset_type: str):
        """Copy assets (images, etc.) to the main assets directory."""
        source_path = Path(source_dir)
        target_path = self.base_dir / 'assets' / asset_type
        
        if not target_path.exists():
            target_path.mkdir(parents=True)
            
        for file in source_path.glob('*'):
            if file.is_file():
                shutil.copy2(file, target_path)
        logger.info(f"Copied assets from {source_dir} to {target_path}")

    def _get_section_from_path(self, path: Path) -> str:
        """Get section name from a file path."""
        for section_name, section_path in self.sections.items():
            if path.is_relative_to(section_path):
                return section_name
        return "unknown"

# Example usage:
if __name__ == "__main__":
    manager = HTMLManager()
    
    # Register sections
    manager.register_section("Introduction", "Introduction")
    manager.register_section("Chapter1", "Chad")
    manager.register_section("Appendix", "Appendix")
    
    # Create title page
    manager.create_title_page()
    
    # Create data overview
    df = pd.read_csv('datastep2.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    overview_html = manager.create_data_overview(df, "Introduction")
    manager.save_section_html("Introduction", overview_html, "data_overview.html")
    
    # Create map
    map_path = create_airport_map(output_dir, manager)
    
    # Create HTML section with more discussion and a section title
    section_title = "Chicago Area Airport Locations and Coverage"
    discussion = (
        "This map displays the locations of major airports in the Chicago metropolitan area, "
        "with each airport marked by its ICAO code. The blue dashed circle represents a 100-mile "
        "radius centered on downtown Chicago, illustrating the region's air travel coverage. "
        "This visualization helps highlight the density and distribution of airports serving the "
        "greater Chicago area, which is a major hub for both passenger and cargo air traffic in the Midwest. "
        "Understanding the spatial arrangement of these airports is important for analyzing regional connectivity, "
        "logistics, and the potential impact of weather or other disruptions on air travel."
    )
    manager.create_section_with_image(
        map_path,
        section_title,
        discussion,
        "map.html"
    )
    
    # Combine sections
    manager.combine_sections(["Introduction", "Chapter1", "Appendix"])
