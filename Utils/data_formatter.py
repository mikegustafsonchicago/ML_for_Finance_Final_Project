import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

def format_datetime_axis(ax=None, date_format='%Y-%m-%d', rotation=45, tight_layout=True):
    """
    Format the x-axis of a plot for datetime values.
    
    Args:
        ax: The matplotlib Axes object. If None, uses current axes.
        date_format: The date format string.
        rotation: Degrees to rotate the x-axis labels.
        tight_layout: Whether to call plt.tight_layout().
    """
    if ax is None:
        ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha='right')
    if tight_layout:
        plt.tight_layout()

def parse_custom_datetime(series):
    """
    Parse a pandas Series of custom-formatted datetime strings (YYYYMMDDHHMM).
    """
    return pd.to_datetime(series, format='%Y%m%d%H%M')
