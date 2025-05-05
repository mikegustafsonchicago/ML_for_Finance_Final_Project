import pandas as pd
import logging
from pathlib import Path
import numpy as np
from data_formatter import parse_custom_datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def generate_llm_data_info():
    """Generate a text file with dataset information formatted for LLM prompts"""
    logger.info("Generating LLM data info...")
    
    try:
        # Read data files
        df = pd.read_csv('../datastep2.csv')
        df['datetime'] = parse_custom_datetime(df['datetime'])
        
        # Calculate key statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats = df[numeric_cols].describe()
        
        # Calculate missing values
        missing_values = df.isnull().sum()
        missing_percentages = (missing_values / len(df) * 100).round(2)
        
        # Analyze time step
        df_sorted = df.sort_values('datetime')
        time_deltas = df_sorted['datetime'].diff().dropna()
        most_common_step = time_deltas.mode()[0] if not time_deltas.empty else "Unknown"
        
        # Generate formatted text output
        output = []
        output.append("=== CHICAGO WEATHER DATASET OVERVIEW ===\n")
        
        output.append("BASIC INFORMATION:")
        output.append(f"- Total observations: {len(df):,}")
        output.append(f"- Date range: {df['datetime'].min().strftime('%Y-%m-%d %H:%M')} to {df['datetime'].max().strftime('%Y-%m-%d %H:%M')}")
        output.append(f"- Typical measurement interval: {most_common_step}")
        output.append(f"- Number of parameters: {len(df.columns)}")
        
        output.append("\nPARAMETERS:")
        for col in df.columns:
            output.append(f"- {col} ({df[col].dtype})")
            
        output.append("\nDATA QUALITY:")
        output.append("Missing values by parameter:")
        for col in df.columns:
            output.append(f"- {col}: {missing_values[col]:,} missing ({missing_percentages[col]:.1f}%)")
            
        output.append("\nSTATISTICAL SUMMARY:")
        output.append(stats.to_string())
        
        # Write to file
        output_path = Path("LLM_data_info.txt")
        with open(output_path, "w") as f:
            f.write("\n".join(output))
            
        logger.info(f"LLM data info written to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error generating LLM data info: {str(e)}")
        raise

if __name__ == "__main__":
    generate_llm_data_info()
