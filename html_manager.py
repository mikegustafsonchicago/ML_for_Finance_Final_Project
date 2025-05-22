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
    <link rel="stylesheet" href="../../style.css">
</head>
<body>
    <div class="container">
        {content}
    </div>
    {additional_js}
</body>
</html>
"""
        logger.info("HTMLManager initialized")

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
            additional_js=""
        )
        
        if output_file:
            self.save_section_html(
                "Introduction",  # Title page is always part of Introduction
                html_content,
                output_file
            )
        
        return content

    def create_section_with_image(self, image_path, title, description):
        """
        Create an HTML section with an image, a title, and a description.
        """
        # If image_path is a Path object, convert to string
        image_path_str = str(image_path)
        # Try to make the path relative to the base_dir for portability
        try:
            image_path_str = str(Path(image_path).relative_to(self.base_dir))
        except Exception:
            image_path_str = str(image_path)
        section_html = f"""
        <div class="section">
            <h3>{title}</h3>
            <p>{description}</p>
            <img src="{image_path_str}" alt="{title}">
        </div>
        """
        return section_html

    def save_section_html(self, section_name: str, html_content: str, output_file: str):
        """
        Save the HTML content for a section to the appropriate directory.
        """
        if section_name not in self.sections:
            raise ValueError(f"Section '{section_name}' is not registered.")
        section_path = self.sections[section_name]
        output_path = section_path / 'outputs' / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"Saved HTML for section '{section_name}' to {output_path}")
        return output_path
