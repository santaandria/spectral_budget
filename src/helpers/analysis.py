"""
Spectral analysis of precipitation time series.

Wavelet and Fourier power spectra, autocorrelation and integral scales,
Lorentzian power-law fitting, and the shared series-loading and normalization
helpers used by the analysis scripts.

Author: Santa Andria (santa.andria@dicea.unipd.it)
"""
import numpy as np
import pandas as pd
import pywt
from scipy.integrate import trapezoid
from statsmodels.tsa.stattools import acf
from scipy.signal import welch
import piecewise_regression


def load_station_series(folder, station):
    """Load one station's 5-minute precipitation series, indexed by datetime."""
    return pd.read_csv(
        folder + station + ".csv",
        parse_dates=["datetime"],
        index_col="datetime",
    )


def standardize(data):
    """
    Centre a series on its mean and scale it to unit standard deviation,
    ignoring NaNs.

    Both statistics are nan-aware so that gappy or seasonally masked series
    are handled: the NaNs are preserved in the output and propagate to any
    subsequent transform. Note the standard deviation is taken on the already
    centred array, matching the original in-line form exactly.

    Parameters:
        data: 1-D array, may contain NaNs

    Return:
        Standardised array of the same shape
    """
    x = data - np.nanmean(data)
    return x / np.nanstd(x)


def acf_analysis(x, delta_t):
    """
    Compute the ACF and the integral scale of a time series and optionally plot them
    """

    nlags = 2 ** (int(np.log2(len(x))))
    autocorr = acf(x, missing="conservative", nlags=nlags)  # Until first zero crossing
    lags = np.arange(len(autocorr)) * delta_t
    first_zero_crossing = int(np.where(autocorr < 0)[0][0])
    integral_scale = trapezoid(
        autocorr[:first_zero_crossing], lags[:first_zero_crossing]
    )
    return autocorr, lags, integral_scale, first_zero_crossing


def wavelet_psd(wT, scales, delta_t, angular=True, nan_safe=False):
    """
    Compute the wavelet power spectral density E(K) where k is the wave number
    Parameters:
    wT: [wT1, wT2, ..., wT_levels] where wTi corresponds to the ith level (increasing scale)
    scales: corresponding scales in the time unit
    delta_t: Measurement spacing in physical space. Inverse of sampling frequency
    nan_safe: If True, average only over the finite coefficients at each level
        (np.nanmean instead of np.mean). Required for seasonally masked series,
        where out-of-season samples are set to NaN before the transform: the NaNs
        spread through the filter bank, so a plain np.mean returns NaN for every
        level. Levels that are entirely NaN (fully masked, i.e. the coarse scales
        exceeding the season length) still yield NaN and are meant to be dropped
        downstream by the caller's psd > 0 filter.

    Return:
    psk: Power spectral density
    K: scale

    """
    C = 2 * np.pi if angular else 1
    K = (C / scales) if angular else (1 / scales)
    mean = np.nanmean if nan_safe else np.mean
    psd = [
        mean(w**2) * delta_t / (C * np.log(2)) for w in wT
    ]  # Power spectral density (Energy in each wavenumber band dK)
    return np.asarray(psd), np.asarray(K)


def fourier_psd(x, delta_t, window="hann", nperseg=1024, angular=True):
    """
    Computes the Fourier power density spectrum
    Return:
    psd: power density spectrum
    K: Wave number

    """
    C = 2 * np.pi if angular else 1
    f, P = welch(x, window=window, nperseg=nperseg, fs=1)  # fs=assumed to 1
    K = C * f * (1 / delta_t)
    psd = P * delta_t / C  # psd = E/dK where dK = 2*np.pi*df*(1/delta_t)
    return np.asarray(psd), np.asarray(K)


def wavelet_transform(x, psi, delta_t, mode="per", useful_max_level=True):
    """
    Compute the discrete wavelet transform up to N levels, according to a dyadic scale arrangement, where N = np.floor(np.log2(len(x)))
    Parameters:
        x (array-like): Data
        psi (pywt Wavelet)
        delta_t: Measurement spacing in physical space. Inverse of sampling frequency
        mode: Mode for the boundary in pywt.wavedec
        useful_max_level: If True, stop decomposition when the signal becomes shorter than the FIR filter length
    Returns:
        wT: [wT1, wT2, ..., wT_levels] where wTi corresponds to the ith level (increasing scale)
        scales: corresponding scales in the time unit
    """
    J = (
        pywt.dwt_max_level(data_len=len(x), filter_len=psi.dec_len)
        if useful_max_level
        else int(np.log2(len(x)))
    )
    wT = pywt.wavedec(x, psi, level=J, mode=mode)[::-1][:-1]
    scales = 2 ** (np.arange(1, J + 1)) * delta_t
    return wT, scales


def lorentzian(x, A, B, c):
    """
    Calculate the Lorentzian function.
    Args:
        x: Input values
        A: Amplitude parameter
        B: scale parameter
        c: Power law exponent

    Returns:
        np.ndarray: Lorentzian function values
    """
    return A / (1 + (x / (B + 1e-15)) ** c)


def fit_psd_power_law(psd, K, integral_scale, angular=True):
    """
    Fit a power law to the power spectral density using piecewise regression in log-log space
    Args:
        psd: Power spectral density values
        K: Wavenumber values
        integral_scale: Integral scale parameter

    Returns:
        tuple: (power law exponent, confidence interval, breakpoint)
    """
    x = np.log10(K)
    y = np.log10(psd)
    C = 2 * np.pi if angular else 1
    pw_fit = piecewise_regression.Fit(
        x, y, n_breakpoints=1, start_values=[np.log10(C / integral_scale)]
    )
    results = pw_fit.get_results()["estimates"]

    c = results["alpha2"]["estimate"]
    c_ci = results["alpha2"]["confidence_interval"]
    breakpt = results["breakpoint1"]["estimate"]

    return c, c_ci, breakpt


##### Plotting Utils #####


###### Secondary time axis #######
# ---------------------------------------------------------------------------
# "Nice" intuitive time-tick machinery
# ---------------------------------------------------------------------------

_UNIT_SECONDS = {
    "yr": 365.25 * 86400,
    "mo": 30.44 * 86400,
    "d": 86400,
    "h": 3600,
    "mn": 60,
}

# Fallback "nice" multiples per unit, smallest -> largest. Used only when the
# bare "1x" tick of a unit does not fit inside (or near) the plotted range.
_NICE_MULTIPLIERS = {
    "yr": [1, 2, 5, 10, 20, 50, 100],
    "mo": [1, 2, 3, 6],
    "d": [1, 2, 3, 5],
    "h": [1, 2, 3, 6, 12],
    "mn": [1, 2, 5, 10, 15, 20, 30, 40, 50],
}

_UNIT_ORDER = ["yr", "mo", "d", "h", "mn"]

_FROM_UNIT_SECONDS = {
    "s": 1,
    "mn": 60,
    "h": 3600,
    "d": 86400,
    "mo": 30.44 * 86400,
    "yr": 365.25 * 86400,
}


def _unit_to_freq(unit, mult, from_unit, wavenumber):
    """Frequency (in from_unit**-1, or 2*pi/from_unit if wavenumber) for mult*unit."""
    t_seconds = mult * _UNIT_SECONDS[unit]
    t_in_from_unit = t_seconds / _FROM_UNIT_SECONDS[from_unit]
    return (2 * np.pi / t_in_from_unit) if wavenumber else (1.0 / t_in_from_unit)


def _label_for(unit, mult):
    return f"{mult}{unit}"


def _pick_intuitive_ticks(xlim, from_unit="h", wavenumber=False, pad_frac=0.12):
    """
    Choose intuitive time ticks (1yr, 1mo, 1d, 1h, 1mn) for a log
    frequency axis spanning `xlim` (in the same units as the primary axis).

    For each canonical unit:
      1. If the bare "1-unit" tick falls inside `xlim`, use it directly.
      2. Else if it falls just outside but within a small log-space padding
         (`pad_frac`), keep it and record the (slightly) extended limits so
         the caller can add a bit of blank margin to the axis to show it.
      3. Otherwise, fall back to the smallest "nice" multiple of that same
         unit (e.g. 5mn, 10mn, ...) that actually fits inside the original,
         unpadded range.
      4. If no multiple of the unit fits at all, that unit is skipped.

    Returns
    -------
    tick_positions : list of float
    tick_labels    : list of str
    new_xlim       : tuple (xmin, xmax), possibly slightly wider than the
                      input `xlim` to accommodate a near-miss "1-unit" tick.
    """
    xmin, xmax = min(xlim), max(xlim)

    log_xmin, log_xmax = np.log10(xmin), np.log10(xmax)
    span = log_xmax - log_xmin
    pad_xmin = 10 ** (log_xmin - pad_frac * span)
    pad_xmax = 10 ** (log_xmax + pad_frac * span)

    tick_positions, tick_labels = [], []
    new_xmin, new_xmax = xmin, xmax

    for unit in _UNIT_ORDER:
        f1 = _unit_to_freq(unit, 1, from_unit, wavenumber)

        if pad_xmin <= f1 <= pad_xmax:
            new_xmin = min(new_xmin, f1)
            new_xmax = max(new_xmax, f1)
            tick_positions.append(f1)
            tick_labels.append(_label_for(unit, 1))
            continue

        for m in _NICE_MULTIPLIERS[unit][1:]:
            f = _unit_to_freq(unit, m, from_unit, wavenumber)
            if xmin <= f <= xmax:
                tick_positions.append(f)
                tick_labels.append(_label_for(unit, m))
                break
        # else: unit skipped entirely, nothing of it fits in range

    order = np.argsort(tick_positions)
    tick_positions = [tick_positions[i] for i in order]
    tick_labels = [tick_labels[i] for i in order]
    return tick_positions, tick_labels, (new_xmin, new_xmax)


def add_time_axis(
    ax, scales, wavenumber=False, loc="bottom", from_unit="h", pad_frac=0.12
):
    """
    Adds a secondary time axis to the provided frequency axis of a plot.

    Places ticks at intuitive, human-readable
    time units (1yr, 1mo, 1wk, 1d, 1h, 1mn). If a canonical "1-unit" tick
    falls just outside the plotted frequency range, the axis is given a
    small amount of extra whitespace (controlled by `pad_frac`) so the
    label can still be shown; if it is too far outside to justify that,
    the nearest "nice" multiple of the same unit (e.g. 5mn, 10mn, 30mn)
    that does fit inside the range is used instead. Units with no fitting
    tick at all are simply omitted.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The primary axis object for the plot (frequency axis).
    scales : list or array-like
        The frequency scales used in the analysis (only used to fall back
        to axis limits if `ax` has no data yet; the actual tick choice is
        driven by `ax.get_xlim()`).
    wavenumber : bool, optional (default=False)
        If True, treats frequency as 2*pi/scale instead of 1/scale.
    loc : {'top', 'bottom'}, optional (default='bottom')
        Location of the secondary axis relative to the primary axis.
    from_unit : str, optional (default='h')
        Time unit in which `scales` (and the primary axis) are expressed.
    pad_frac : float, optional (default=0.12)
        Fractional log-space padding allowed when deciding whether to
        stretch the axis slightly to fit a near-miss "1-unit" tick.

    Returns
    -------
    t_ax : matplotlib.axes.Axes
        The secondary time axis.
    """
    t_ax = ax.twiny()

    xlim = ax.get_xlim()
    if xlim[0] <= 0 or xlim[1] <= 0:
        # fall back to data-derived limits if the axis has no valid log range yet
        freqs = (
            (2 * np.pi / np.asarray(scales))
            if wavenumber
            else (1.0 / np.asarray(scales))
        )
        xlim = (freqs.min(), freqs.max())

    tick_positions, tick_labels, new_xlim = _pick_intuitive_ticks(
        xlim, from_unit=from_unit, wavenumber=wavenumber, pad_frac=pad_frac
    )

    if loc == "bottom":
        t_ax.xaxis.set_ticks_position("bottom")
        t_ax.xaxis.set_label_position("bottom")
        t_ax.spines["bottom"].set_position(("axes", -0.25))
        t_ax.set_frame_on(True)
        t_ax.patch.set_visible(False)
        t_ax.spines["bottom"].set_visible(True)
        ax.tick_params(top=True, which="both")
    elif loc == "top":
        t_ax.xaxis.set_ticks_position("top")
        t_ax.xaxis.set_label_position("top")
        ax.tick_params(bottom=True, which="both")

    # apply the (possibly slightly widened) limits to both axes, so the
    # near-miss "1-unit" tick is visible with a bit of blank margin
    ax.set_xlim(new_xlim)
    t_ax.set_xscale("log")
    t_ax.set_xlim(new_xlim)
    t_ax.set_xticks(tick_positions)
    t_ax.minorticks_off()
    t_ax.set_xticklabels(tick_labels, fontdict={"fontsize": 6})

    return t_ax


##################################

