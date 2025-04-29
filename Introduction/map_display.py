import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import logging
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Dictionary of ICAO codes and their coordinates (latitude, longitude)
AIRPORT_COORDS = {
    'KORD': (41.9786, -87.9048),  # Chicago O'Hare
    'KMDW': (41.7868, -87.7522),  # Chicago Midway
    'KMKE': (42.9550, -87.8989),  # Milwaukee
    'KPWK': (42.1142, -87.9015),  # Chicago Executive
    'KDPA': (41.9078, -88.2486),  # DuPage
    'KGYY': (41.6163, -87.4128),  # Gary
    'KRFD': (42.1954, -89.0972),  # Chicago Rockford
    'KARR': (41.7714, -88.4813),  # Aurora Municipal
    'KUGN': (42.4222, -87.8679),  # Waukegan
    'KVPZ': (41.4539, -87.0071),  # Porter County
    'KJOT': (41.5178, -88.1756),  # Joliet Regional
    'KIKK': (41.0714, -87.8463),  # Greater Kankakee
    'KENW': (42.5957, -87.9278),  # Kenosha Regional
    'KRAC': (42.7612, -87.8137),  # Racine
    'KBUU': (42.6907, -88.3047),  # Burlington Municipal
    'KJVL': (42.6199, -89.0416),  # Southern Wisconsin Regional
    'KDKB': (41.9338, -88.7079),  # DeKalb Taylor Municipal
    'KRYV': (43.1711, -88.7243),  # Watertown Municipal
    'KRZL': (43.1156, -90.6825),  # Reedsburg Municipal
    'KUES': (43.0411, -88.2371),  # Waukesha County
    'KC09': (41.7433, -88.1216),  # Plainfield
    'KIGQ': (41.5239, -85.7979),  # Boone County
    'KLOT': (41.6072, -88.0961),  # Lewis University
    'KMGC': (41.7033, -86.8219),  # Michigan City Municipal
    'KMWC': (43.1100, -88.0344),  # Milwaukee-Timmerman
    'KOXI': (41.3533, -86.9989),  # Starke County
    'KPNT': (40.9193, -88.6926),  # Pontiac Municipal
    'KPPO': (42.9097, -89.0326),  # Poplar Grove
    'KRPJ': (42.2461, -89.5821),  # Rochelle Municipal
}

def create_airport_map(output_dir):
    """Create a static map showing all airports in the Chicago area"""
    logger.info("Creating airport map...")
    
    # Calculate map bounds (with some padding)
    lats = [coord[0] for coord in AIRPORT_COORDS.values()]
    lons = [coord[1] for coord in AIRPORT_COORDS.values()]
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
    for icao, coords in AIRPORT_COORDS.items():
        ax.plot(coords[1], coords[0], 'r^', markersize=8, transform=ccrs.PlateCarree())
        ax.text(coords[1], coords[0], icao, fontsize=8, 
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
        
        # Create map
        map_path = create_airport_map(output_dir)
        
        # Create standalone HTML
        from html_combine import create_section_with_image
        html_path = output_dir / 'map.html'
        create_section_with_image(map_path, 'Airport Locations', html_path)
        logger.info(f"Created standalone map at {html_path}")
        
    except Exception as e:
        logger.error(f"Error creating airport map: {str(e)}")
        raise
