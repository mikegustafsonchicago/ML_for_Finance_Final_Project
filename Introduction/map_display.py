import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import logging
import numpy as np
import sys
sys.path.append('..')  # Add parent directory to path to import html_manager
from html_manager import HTMLManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("map_display")

# Dictionary of ICAO codes, names, and their coordinates (latitude, longitude)
AIRPORT_COORDS = {
    'KORD': ("Chicago O'Hare", 41.9786, -87.9048),
    'KMDW': ("Chicago Midway", 41.7868, -87.7522),
    'KMKE': ("Milwaukee", 42.9550, -87.8989),
    'KPWK': ("Chicago Executive", 42.1142, -87.9015),
    'KDPA': ("DuPage", 41.9078, -88.2486),
    'KGYY': ("Gary", 41.6163, -87.4128),
    'KRFD': ("Chicago Rockford", 42.1954, -89.0972),
    'KARR': ("Aurora Municipal", 41.7714, -88.4813),
    'KUGN': ("Waukegan", 42.4222, -87.8679),
    'KVPZ': ("Porter County", 41.4539, -87.0071),
    'KJOT': ("Joliet Regional", 41.5178, -88.1756),
    'KIKK': ("Greater Kankakee", 41.0714, -87.8463),
    'KENW': ("Kenosha Regional", 42.5957, -87.9278),
    'KRAC': ("Racine", 42.7612, -87.8137),
    'KBUU': ("Burlington Municipal", 42.6907, -88.3047),
    'KJVL': ("Southern Wisconsin Regional", 42.6199, -89.0416),
    'KDKB': ("DeKalb Taylor Municipal", 41.9338, -88.7079),
    'KRYV': ("Watertown Municipal", 43.1711, -88.7243),
    'KRZL': ("Reedsburg Municipal", 43.1156, -90.6825),
    'KUES': ("Waukesha County", 43.0411, -88.2371),
    'KC09': ("Plainfield", 41.7433, -88.1216),
    'KIGQ': ("Boone County", 41.5239, -85.7979),
    'KLOT': ("Lewis University", 41.6072, -88.0961),
    'KMGC': ("Michigan City Municipal", 41.7033, -86.8219),
    'KMWC': ("Milwaukee-Timmerman", 43.1100, -88.0344),
    'KOXI': ("Starke County", 41.3533, -86.9989),
    'KPNT': ("Pontiac Municipal", 40.9193, -88.6926),
    'KPPO': ("Poplar Grove", 42.9097, -89.0326),
    'KRPJ': ("Rochelle Municipal", 42.2461, -89.5821),
}

def create_airport_map(output_dir, html_manager: HTMLManager = None):
    """Create a static map showing all airports in the Chicago area"""
    logger.info("Creating airport map...")
    
    # Initialize HTML manager if not provided
    if html_manager is None:
        html_manager = HTMLManager()
        html_manager.register_section("Introduction", Path(__file__).parent)
    
    # Calculate map bounds (with some padding)
    lats = [coord[1] for coord in AIRPORT_COORDS.values()]
    lons = [coord[2] for coord in AIRPORT_COORDS.values()]
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)
    
    # Create figure and axis with projection
    fig = plt.figure(figsize=(12, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set map bounds
    padding = 1.0  # degrees
    ax.set_extent([
        min(lons) - padding,
        max(lons) + padding,
        min(lats) - padding,
        max(lats) + padding
    ])
    
    # Add map features
    ax.add_feature(cfeature.STATES.with_scale('10m'), linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)
    ax.add_feature(cfeature.LAKES.with_scale('10m'), alpha=0.5)
    
    # Plot airports
    for icao, (name, lat, lon) in AIRPORT_COORDS.items():
        ax.plot(lon, lat, 'r^', markersize=8, transform=ccrs.PlateCarree())
        ax.text(lon, lat, icao, fontsize=8, 
                transform=ccrs.PlateCarree(),
                horizontalalignment='right',
                verticalalignment='bottom')
    
    # Add 100-mile radius circle around Chicago
    chicago_coords = (-87.6298, 41.8781)  # lon, lat
    circle_points = 100
    radius = 1.5  # degrees (approximately 100 miles)
    circle_lats = []
    circle_lons = []
    for i in range(circle_points):
        angle = 2 * np.pi * i / circle_points
        circle_lats.append(chicago_coords[1] + radius * np.cos(angle))
        circle_lons.append(chicago_coords[0] + radius * np.sin(angle))
    ax.plot(circle_lons, circle_lats, 'b--', alpha=0.5, transform=ccrs.PlateCarree())
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    
    # Add title
    plt.title('Chicago Area Airports', pad=20)
    
    # Save the map
    output_path = output_dir / 'airport_map.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Airport map created: {output_path}")
    return output_path

if __name__ == "__main__":
    try:
        # Create outputs directory
        output_dir = Path(__file__).parent / 'outputs'
        output_dir.mkdir(exist_ok=True)
        
        # Create HTML manager
        manager = HTMLManager()
        manager.register_section("Introduction", Path(__file__).parent)
        
        # Create map
        map_path = create_airport_map(output_dir, manager)
        
        # Build the HTML table
        table_html = "<table><tr><th>Code</th><th>Name</th><th>Latitude</th><th>Longitude</th></tr>"
        for code, (name, lat, lon) in AIRPORT_COORDS.items():
            table_html += f"<tr><td>{code}</td><td>{name}</td><td>{lat}</td><td>{lon}</td></tr>"
        table_html += "</table>"

        # Add this table to your HTML section content
        content_html = f"""
            <div class="section">
                <h2>Airport Locations</h2>
                <p>Map showing the locations of all airports in the Chicago area, with a 100-mile radius circle around Chicago.</p>
                <div class="plot-group">
                    <img src="{map_path.name}" alt="Airport Map" style="max-width: 100%; height: auto;">
                </div>
                <h3>Airport Table</h3>
                {table_html}
            </div>
        """
        manager.save_section_html("Introduction", content_html, "map.html")
        
        logger.info("Created airport map section")
        
    except Exception as e:
        logger.error(f"Error creating airport map: {str(e)}")
        raise
