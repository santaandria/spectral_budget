"""
Seasonal (JJA / DJF) spectral scaling analysis.

For each station the precipitation series is masked down to a single season,
transformed, and a power-law slope is fitted to the wavelet PSD over two
frequency bands. The masking leaves NaNs in the series, so the wavelet PSD is
computed in nan-safe mode (see ``analysis.wavelet_psd``).
"""

import numpy as np
import pandas as pd
import pywt
from scipy.stats import linregress

from .analysis import (
    acf_analysis,
    load_station_series,
    standardize,
    wavelet_psd,
    wavelet_transform,
)

# frequency bounds in h^-1 -> period ranges of [1day-1hour] and [1hour-1min]
FREQ_RANGES = {
    "1day-1hour": (1 / 24, 1),
    "1hour-1min": (1, 60),
}

EXAMPLE_STATIONS = ["003_BL_Ar", "127_VR_Bu", "168_VE_Ch"]


def fit_scaling_slope(f, psd, freq_range):
    """
    Slope (and its std. error) of log10(E) vs log10(f), restricted to
    frequencies within freq_range = (f_min, f_max).
    """
    mask = (f >= freq_range[0]) & (f <= freq_range[1])
    if mask.sum() < 2:
        return np.nan, np.nan
    slope, intercept, r, p, se = linregress(np.log10(f[mask]), np.log10(psd[mask]))
    return slope, se


def compute_seasonal_scaling(
    stations_df, folder, delta_t, save_path="output/seasonal_scaling_slopes.csv"
):
    """
    Compute summer/winter wavelet spectra and scaling slopes for every station.

    Returns:
        spectra: {station: {season: (psd, f)}}
        int_scale: {station: {season: integral scale}}
        scaling_df: one row per (station, season, frequency band), merged with
            the station coordinates
        tau: wavelet scales of the final transform, reused by the Fig 6
            secondary time axis (identical across stations, since it depends
            only on delta_t and the number of surviving levels)
    """
    psi = pywt.Wavelet("haar")
    spectra = {}
    int_scale = {}

    # All stations (already includes the three example stations used for the
    # spectra plot below)
    stations = stations_df["station_id"].values
    scaling_records = []
    tau = None

    for station in stations:
        df = load_station_series(folder, station)

        summer = df.copy()
        summer.loc[~summer.index.month.isin([6, 7, 8]), "PRCP"] = np.nan  # JJA

        winter = df.copy()
        winter.loc[~winter.index.month.isin([12, 1, 2]), "PRCP"] = np.nan  # DJF

        result = {}
        ints = {}

        for season, data in zip(["summer", "winter"], [summer, winter]):
            data = data["PRCP"].values

            # Normalize
            x = standardize(data)

            # ACF Analysis
            _, _, integral_scale, _ = acf_analysis(x, delta_t)

            # Wavelet Transform
            wT, tau = wavelet_transform(x, psi, delta_t, mode="per")

            # nan_safe: the season mask leaves NaNs in x, which spread through the
            # filter bank; average over the finite coefficients only. Fully masked
            # coarse levels stay NaN and are dropped by the psd > 0 filter below.
            psd, f = wavelet_psd(wT, tau, delta_t, angular=False, nan_safe=True)
            psd, f, tau = psd[psd > 0], f[psd > 0], tau[psd > 0]
            result.update({season: (psd, f)})
            ints.update({season: integral_scale})

            for range_name, rng in FREQ_RANGES.items():
                slope, se = fit_scaling_slope(f, psd, rng)
                scaling_records.append(
                    {
                        "station_id": station,
                        "season": season,
                        "freq_range": range_name,
                        "slope": -slope,  # c is plotted
                        "std_error": se,
                    }
                )

        spectra.update({station: result})
        int_scale.update({station: ints})

    scaling_df = pd.DataFrame(scaling_records)
    scaling_df.to_csv(save_path, index=False)
    scaling_df = scaling_df.merge(
        stations_df[["station_id", "Elv", "Lat", "Lon"]], on="station_id"
    )

    return spectra, int_scale, scaling_df, tau
