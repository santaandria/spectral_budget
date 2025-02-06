import pandas as pd
import numpy as np
import re
from os import makedirs


def load_txt_file(file):
    f_info = file.split("_")  # Name_Interval_Start_End
    datetime = pd.date_range(
        start=(f_info[2][-4:]),
        end=str(int(f_info[3][-4:]) + 1),
        freq=f_info[1],
        name="datetime",
        inclusive="left",
    )
    prcp = np.loadtxt("./data/" + file + ".txt")
    return pd.DataFrame(prcp, index=datetime, columns=["PRCP"])


def remove_missing_years(df_, columns, perc):
    """
    Remove years where the number of missing values in specified column(s) exceeds threshold.
    
    Args:
        df_ (pd.DataFrame): Input DataFrame with datetime index
        columns (str or list): Column(s) to check for missing values
        perc (float): Maximum allowed percentage of missing values (0 to 1)
        
    Returns:
        tuple: (cleaned DataFrame, list of removed years)
    """
    if not isinstance(columns, (str, list)):
        raise TypeError("columns must be string or list")
    if not 0 <= perc <= 1:
        raise ValueError("perc must be between 0 and 1")
    
    columns = [columns] if isinstance(columns, str) else columns
    df = df_.copy()
    if 'YEAR' not in df.columns:
        df['YEAR'] = df.index.year
    
    years = np.unique(df.index.year)
    min_required_values = 365 * (1 - perc)  
    removed = []
    
    for year in years:
        year_data = df[df.YEAR == year][columns]
        # Check if we have enough non-NA values in each specified column
        non_na_count = len(year_data.dropna())
        if non_na_count < min_required_values:
            df = df[df.YEAR != year]
            removed.append(year)
    
    df = df.dropna(subset=columns)
    return df.drop(columns=['YEAR']), removed


def load_arpav_5mn(file, reindex=True, remove_missing_years=False, perc_missing=0.1):
    """
    Load ARPAV 5-minute data from CSV file.

    Args:
        file (str): Path to CSV file
        reindex (bool): Reindex DataFrame to 5-minute frequency (default: True)

    Returns:
        pd.DataFrame: ARPAV data with datetime index
    """
    if remove_missing_years and not 0 <= perc_missing <= 1:
        raise ValueError("perc_missing must be between 0 and 1")

    df = pd.read_csv(
        file, sep=";", parse_dates=["dataora"], index_col="dataora"
    ).rename_axis("datetime")
    df = (
        df[df["qualita"] == 1]
        .rename(columns={"valore": "PRCP"})
        .drop(columns=["qualita"])
    )
    if remove_missing_years:
        df, _ = remove_missing_years(df, "PRCP", perc_missing)
    if reindex:
        df = df.asfreq("5min").reindex()
    return df


def index_to_time_str(di: float, freq: str) -> str:
    """
    Convert an index increment to string based on the sampling frequency.
    
    Args:
        di (float): Index increment/shift
        freq (str): Frequency string in pandas format (e.g., '5min', '1H', '2D')
        
    Returns:
        str: Formatted time string with appropriate unit (mn, h, d, mo, or y)
    """
    # Parse the frequency string
    
    match = re.match(r'(\d*\.?\d*)?\s*([A-Za-z]+)', freq)
    if not match:
        raise ValueError(f"Invalid frequency format: {freq}")
        
    value = float(match.group(1)) if match.group(1) else 1
    unit = match.group(2).lower()
    
    # Convert to minutes
    conversion = {
        'min': 1,
        't': 1,          # minute
        'h': 60,         # hour
        'd': 60 * 24,    # day
        'w': 60 * 24 * 7 # week
    }
    
    base_unit = unit.rstrip('s')  # handle plurals like 'mins'
    if base_unit not in conversion:
        raise ValueError(f"Unsupported frequency unit. Must be one of {list(conversion.keys())}")
    
    # Calculate total time in minutes
    tau = di * value * conversion[base_unit]
    
    # Time thresholds in minutes
    HOUR = 60
    DAY = HOUR * 24
    MONTH = DAY * 30
    YEAR = DAY * 365
    
    if tau < HOUR:
        return f"{tau:.1f}mn"
    elif tau < DAY:
        return f"{(tau/HOUR):.1f}h"
    elif tau < MONTH:
        return f"{(tau/DAY):.1f}d"
    elif tau < YEAR:
        return f"{(tau/MONTH):.1f}mo"
    else:
        return f"{(tau/YEAR):.1f}y"
    

def generate_increment_data(
    data,
    idx_shifts,
    sampling_freq,
    save_file,
    prefix = None
):
    """
    Generate increment data by calculating differences between shifted data points.
    
    Args:
        data: Input time series data array
        idx_shifts: List of index shifts to calculate increments for
        sampling_freq: Sampling frequency of the data
        save_file: If True, saves increments to files instead of returning them
        prefix: Optional prefix for saved file names
    
    Returns:
        List of increment arrays if save_file is False, None otherwise
    """
    if save_file:
        output_dir = "./data/increments_data"
        makedirs(output_dir, exist_ok=True)
        
    increments_list = []
    time_shifts = []
    
    for shift in idx_shifts:
        if shift >= len(data):
            break

        increments = data[shift:] - data[:-shift]
        increments = increments[~np.isnan(increments)]
        
        if save_file:
            filename = f"{output_dir}/{prefix+'_' if prefix else ''}{sampling_freq}_{shift}.npy"

            np.save(filename, increments)
            print(f"Saved increments for shift {shift} ({index_to_time_str(shift, sampling_freq)}) to {filename}")
        else:
            time_shifts.append(index_to_time_str(shift, sampling_freq))
            increments_list.append(increments)
    if not save_file:
        return increments_list, time_shifts
    