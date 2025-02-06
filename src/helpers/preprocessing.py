### This module deals with missing values.
### Author: Santa Andria (santa.andria@dicea.unipd.it)
from .analysis import acf_analysis
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
from collections import Counter


# Preprocessing helper.py
def compute_gap_lengths(timeseries, delta_t):
    """
    Returns the distribution of the gap length
    """

    gap_ts = np.isnan(timeseries)
    gap_lengths = []
    current_gap = 0

    for value in gap_ts:
        if value == 1:
            current_gap += 1
        elif current_gap > 0:
            gap_lengths.append(current_gap)
            current_gap = 0

    if current_gap > 0:  # Append the last gap if it ends at the sequence's end
        gap_lengths.append(current_gap)

    gap_lengths = delta_t * np.array(gap_lengths)
    gap_count = Counter(gap_lengths)
    total_gaps = sum(gap_count.values())
    return {length: count / total_gaps for length, count in gap_count.items()}


def plot_gap_distribution(data, delta_t, ax_dist, ax_acf):
    """
    Plot the distribution and the ACF of the missing data
    Parameters:
    gap_distribution (dict): Dictionary containing the PMF of the gap length {'x': PMF}
    integral_scale (float): Integral scale of the original PRCP data
    """
    integral_scale = acf_analysis(data, delta_t)[2]
    gap_ts = np.isnan(data)
    gap_distribution = compute_gap_lengths(data, delta_t)

    nlags = int(max(gap_distribution.keys()) / delta_t)
    gap_acf = acf(gap_ts, missing="conservative", nlags=nlags)
    lags = np.arange(len(gap_acf)) * delta_t

    # Plot the distribution
    markerline, stemlines, baseline = ax_dist.stem(
        gap_distribution.keys(), gap_distribution.values(), markerfmt="none"
    )
    baseline.set_color("k")
    stemlines.set_color("b")
    ax_dist.axvline(
        x=integral_scale,
        color="r",
        ls="--",
        label=f"PRCP Integral Scale = {integral_scale:.2f} h",
    )
    ax_dist.set_ylim(bottom=0)
    ax_dist.set_xlabel("Gap Duration [h]")
    ax_dist.set_ylabel("PMF")
    ax_dist.set_title("Gap Length Distribution")
    ax_dist.legend()
    ax_acf.plot(lags, gap_acf, "-", markerfacecolor="None")
    ax_acf.set_xlabel("lags [h]")
    ax_acf.set_ylabel("ACF")
    ax_acf.set_title("Missing Values ACF")
    return ax_dist, ax_acf


def replace_nans_by_sampling(df_, column):
    """
    Replace NaN values in the DataFrame at specific indices by sampling from
    non-missing values with the same month, day, and time across all years.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data with a datetime index.
        missing_indices (list): List of missing datetime indices (strings or pd.Timestamp).

    Returns:
        pd.DataFrame: Updated DataFrame with NaN values replaced.
    """
    df = df_.copy()

    df["Filled"] = 0
    df["Month"] = df.index.month
    df["Day"] = df.index.day
    df["Time"] = df.index.time

    # Group by (Month, Day, Time)
    grouped = df.groupby(["Month", "Day", "Time"])

    def fill_nan(group):
        non_nan_values = group[column].dropna()
        if not non_nan_values.empty:

            def replace_and_track(x):
                if pd.isna(x):
                    return np.random.choice(non_nan_values.values), 1
                else:
                    return x, 0

            group[[column, "Filled"]] = group[column].apply(
                lambda x: pd.Series(replace_and_track(x))
            )

        else:
            raise ValueError(
                f"No non-NaN values available for sampling for"
                f"({group['Month'].iloc[0]}, {group['Day'].iloc[0]}, {group['Time'].iloc[0]}). "
                f"First missing timestamp: {group.loc[group[column].isna()].index[0]}"
            )
        return group

    df = grouped.apply(fill_nan, include_groups=False)
    df = df.reset_index(level=["Month", "Day", "Time"], drop=True).sort_index()
    df.index.names = ["datetime"]
    df["Filled"] = df["Filled"].astype(np.int8)
    return df


def fill_missing_values(df, column, method="interpolation"):
    """
    Fill NaN values in a specified column using either interpolation or sampling.

    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Name of column to fill NaN values in
        method (str, optional): Method to use for filling NaNs.
            Either 'interpolation' or 'sampling'. Defaults to 'interpolation'.

    Returns:
        pd.DataFrame: DataFrame with NaN values filled in specified column
    """
    if method == "sampling":
        df = replace_nans_by_sampling(df, column=column)
    else:
        df[column] = df[column].interpolate(method="linear", limit_direction="both")
    return df
