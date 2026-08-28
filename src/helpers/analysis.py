### This module deals with missing values.
### Author: Santa Andria (santa.andria@dicea.unipd.it)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import pandas as pd
import pywt
from scipy.integrate import trapezoid
from statsmodels.tsa.stattools import acf
from .Time_Unit_Converter import TimeUnitConverter
from statsmodels.tsa.stattools import acf
from scipy.signal import welch
from scipy.stats import kurtosis, skew
import piecewise_regression
from scipy.optimize import curve_fit
from .plotting_helper import configure_axis
from .pdf_helper import fit_scale_wise_pdf, plot_scale_wise_pdf
import os
import json


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


def compute_quantile_stats(w):
    """
    Compute quantile-based statistics for a single wavelet level. Schmid and Trede (2003)

    Returns:
        Tuple of (tail_index, peakedness_index, asymmetry_index)
    """
    percentiles = np.percentile(w, [2.5, 10, 12.5, 25, 50, 75, 87.5, 90, 97.5])
    tail = (percentiles[8] - percentiles[0]) / (percentiles[6] - percentiles[2])
    peakedness = (percentiles[6] - percentiles[2]) / (percentiles[5] - percentiles[3])
    asymmetry = (percentiles[4] - percentiles[1]) / (percentiles[7] - percentiles[4])
    return tail, peakedness, asymmetry


def compute_moment_stats(w):
    w = w[np.abs(w) > 1e-13]
    return np.std(w**2) / np.mean(w**2), skew(w), kurtosis(w)


def compute_wavelet_stats(wT, quantile_based=True):
    """
    Compute statistics for all wavelet levels.

    Args:
        wT: List of wavelet transforms at different levels
        quantile_based: If True, compute quantile-based statistics,
                        otherwise compute moment-based statistics

    Returns:
        Tuple of three arrays containing the computed statistics
    """
    compute_func = compute_quantile_stats if quantile_based else compute_moment_stats
    results = np.array([compute_func(w) for w in wT])
    return tuple(results.T)  # Transpose to get separate arrays for each statistic


def wavelet_psd(wT, scales, delta_t, angular=True):
    """
    Compute the wavelet power spectral density E(K) where k is the wave number
    Parameters:
    wT: [wT1, wT2, ..., wT_levels] where wTi corresponds to the ith level (increasing scale)
    scales: corresponding scales in the time unit
    delta_t: Measurement spacing in physical space. Inverse of sampling frequency

    Return:
    psk: Power spectral density
    K: scale

    """
    C = 2 * np.pi if angular else 1
    K = (C / scales) if angular else (1 / scales)
    psd = [
        np.mean(w**2) * delta_t / (C * np.log(2)) for w in wT
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


def lorentzian_spectra(
    wavelets, integral_scale, fourier=None, return_err=False, angular=True
):
    """
    Analyze both wavelet and Fourier spectra using Lorentzian fitting.

    Parameters
    wavelets : Tuple[np.ndarray, np.ndarray]
        Tuple of (power spectral density, wavenumbers) for wavelet analysis
    integral_scale : float
        Integral scale parameter for initialization
    fourier : Optional[Tuple[np.ndarray, np.ndarray]]
        Optional tuple of (power spectral density, wavenumbers) for Fourier analysis
    return_err : bool, optional
        If True, returns error estimates for fitted parameters

    Returns
        Dictionary containing results for wavelet analysis and optionally Fourier analysis
    """

    psd_w, K_w = map(np.asarray, wavelets)

    C = 2 * np.pi if angular else 1

    # Initial parameter estimation
    c, _, breakpt = fit_psd_power_law(psd_w, K_w, integral_scale, angular)
    params_init = [integral_scale / np.pi / C, 10**breakpt, -c]

    # Fit wavelet spectrum
    res_w = fit_lorentzian(
        psd_w, K_w, params_init, return_err=return_err, bootstrap_samples=1000
    )

    if return_err:
        res_w[0]["breakpoint"] = 10**breakpt
    else:
        res_w["breakpoint"] = 10**breakpt

    if fourier is None:
        return res_w

    # Process Fourier spectrum if provided
    psd_F, K_F = map(np.asarray, fourier)

    # Filter Fourier data to match wavelet range
    mask = K_F < max(K_w)
    psd_F, K_F = psd_F[mask], K_F[mask]
    K_F[K_F == 0] = 1e-15  # Avoid log(0)

    # Use wavelet results as initial parameters for Fourier fit
    wavelet_params = res_w[0] if return_err else res_w
    params_init = [wavelet_params["A"], wavelet_params["B"], wavelet_params["c"]]

    # Fit Fourier spectrum
    res_F = fit_lorentzian(
        psd_F, K_F, params_init, return_err=return_err, bootstrap_samples=1000
    )

    return {"wavelet": res_w, "fourier": res_F}


def fit_lorentzian(psd, K, params_init, return_err=True, bootstrap_samples=1000):
    """
    Fit a Lorentzian function to the power spectrum density using non-linear least squares.

    Parameters
    ----------
    psd : np.ndarray
        Power spectral density values
    K : np.ndarray
        Wavenumber values
    params_init : List[float]
        Initial parameter guesses [A, B, c] for the Lorentzian function
    return_err : bool, optional
        If True, returns standard error estimates via bootstrap resampling
    bootstrap_samples : int, optional
        Number of bootstrap resamples for error estimation

    Returns
    -------
    Union[Dict[str, float], Tuple[Dict[str, float], np.ndarray]]
        If return_err is False:
            Dictionary containing fitted parameters {'A', 'B', 'c'}
        If return_err is True:
            Tuple of (parameters dictionary, standard errors array)

    Notes
    -----
    The Lorentzian function is fitted in log-space for numerical stability.
    Error estimates are computed using bootstrap resampling when return_err=True.
    """
    psd, K = np.asarray(psd), np.asarray(K)

    def log_lorentzian(params: tuple, u):
        A, B, c = params
        eps = 1e-15
        # logaddexp(x1,x2) = log(exp(x1) + exp(x2)) is numerically more stable as it handles large values of x better.
        return np.log(max(A, eps)) - np.logaddexp(0, c * (u - np.log(max(B, eps))))

    popt, pcov = curve_fit(
        lambda u, A, B, c: log_lorentzian((A, B, c), u),
        np.log(K),
        np.log(psd),
        p0=params_init,
    )
    perr = np.sqrt(np.diag(pcov))

    params = {
        "A": popt[0],
        "B": popt[1],
        "c": popt[2],
    }

    if not return_err:
        return params

    if bootstrap_samples <= 0:
        return params, np.sqrt(np.diag(pcov))

    # Perform bootstrap resampling for error estimation
    boot_params = []
    rng = np.random.default_rng()

    for _ in range(bootstrap_samples):
        indices = rng.integers(0, len(psd), size=len(psd))
        psd_resampled = psd[indices]
        K_resampled = K[indices]

        try:
            popt_resampled, _ = curve_fit(
                lambda u, A, B, c: log_lorentzian((A, B, c), u),
                np.log(K_resampled),
                np.log(psd_resampled),
                p0=params_init,
            )
            boot_params.append(popt_resampled)
        except RuntimeError:
            continue

    boot_params = np.array(boot_params)
    boot_se = np.std(boot_params, axis=0)

    return params, boot_se


def spectral_analysis(wT, scales, timeseries, delta_t, integral_scale, ax=None):
    """
    Perform spectral analysis using both Fourier and Wavelet methods.

    Args:
        wT: Input time series
        scales: Scale values for wavelet transform
        delta_t: Time step
        timeseries: For Fourier transform

    Returns:
        dict: Lorentzian fit parameters
    """
    # Calculate PSDs
    psd_w, K_w = wavelet_psd(wT, scales, delta_t)
    psd_F, K_F = fourier_psd(timeseries, delta_t, window="hann", nperseg=8 * 8192)

    # Discard the smallest and the largest scale from the analysis
    psd_w, K_w, scales = psd_w[1:-1], K_w[1:-1], scales[1:-1]
    # Fit Lorentzian
    lorentzian_params = lorentzian_spectra(
        (psd_w, K_w),
        integral_scale=integral_scale,
        return_err=True,
        fourier=(psd_F, K_F),
    )

    # Create visualization
    if not ax:
        fig, ax = plt.subplots(figsize=(4, 3))

    # Plot Fourier and Wavelet PSDs
    ax.plot(K_F, psd_F, label="Fourier", alpha=0.8, linewidth=1, c="r")
    ax.plot(K_w, psd_w, "k-o", label="Wavelet", markersize=6, markerfacecolor="None")

    # Plot Lorentzian fit
    lorentzian_param = lorentzian_params["wavelet"][0]
    KK = np.logspace(np.log10(min(K_w)), np.log10(max(K_w)), 100)
    ax.plot(
        KK,
        lorentzian(
            KK, lorentzian_param["A"], lorentzian_param["B"], lorentzian_param["c"]
        ),
        "--b",
        linewidth=2,
        label=f"Lorentzian c={lorentzian_param['c']:.2f}, A={lorentzian_param['A']:.2f} h",
    )

    # Configure plot settings
    axis_settings = {
        "ylabel": "$\hat{E}(K) \ [h]$",
        "xlabel": "$K \ [\\text{rad}\cdot h^{-1}]$",
        "xscale": "log",
        "yscale": "log",
        "ylim": (1e-3, 2e1),
    }
    configure_axis(ax, axis_settings)
    add_time_axis(ax, scales, wavenumber=True, loc="top")
    ax.legend(fancybox=True, loc=3)

    return lorentzian_params, ax


##### Plotting Utils #####


def plot_acf(autocorr, lags, integral_scale, first_zero_crossing, ax, **kwargs):
    ax.plot(
        lags[:first_zero_crossing] / integral_scale,
        autocorr[:first_zero_crossing],
        **kwargs,
    )
    ax.set_xlabel("$\\tau$/$\\tau_0$")
    ax.set_ylabel("$\\rho(\\tau)$")
    ax.set_title(f"Autocorrelation Function ($\\tau_0$ = {integral_scale:.2f} h)")
    return ax


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


def plot_wavelet_stats(wT, scales, quantile_based=True, axs=None):
    """
    Plot the statistical measures for all wavelet levels.

    Args:
        wT: List of wavelet transforms at different levels
        quantile_based: If True, plot quantile-based statistics,
                        otherwise plot moment-based statistics
    """
    stats = compute_wavelet_stats(wT, quantile_based)
    K = 2 * np.pi / scales

    if quantile_based:
        titles = ["Tail Index", "Peakedness Index", "Asymmetry Index"]
        yscales = ["log", "log", "linear"]
    else:
        titles = ["CV", "$\\gamma_{1}$", "Kurt $\\, [W_{\\psi}^{(\\tau)}]$"]
        yscales = ["linear", "linear", "linear"]

    if not axs:
        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    for ax, stat, title, yscale in zip(axs, stats, titles, yscales):
        ax.plot(K, stat, "r-o", linewidth=1, markersize=3)
        ax.set_xlabel("$K \ [\\text{rad}\cdot h^{-1}]$")
        ax.set_ylabel(title)
        ax.set_xscale("log")
        ax.set_yscale(yscale)
        add_time_axis(ax, scales, wavenumber=True, loc="top")
    return axs


def plot_tsallis_params(params, axs=None):
    if not axs:
        fig, axs = plt.subplots(1, 2, figsize=(5, 2.5))
    a = [param[2] for param in params]
    c = [param[3] for param in params]
    q = [param[0] for param in params]
    axs[0].scatter(q, a, fc="None", ec="k")
    axs[1].scatter(q, c, fc="None", ec="k")
    for ax in axs:
        ax.set_xlabel("$q$")
    axs[0].set_ylabel("$a$")
    axs[1].set_ylabel("$c$")
    return axs


def create_combined_plot(
    station, map_fpath, delta_t, psi, data_folder="data/filled_sampling/"
):
    """
    Create a combined figure with multiple subplots including map, spectral analysis,
    wavelet statistics, and scale-wise evolution.

    Args:
        station (dict): Dictionary containing station information and ID

    Returns:
        matplotlib.figure.Figure: The complete figure with all subplots
    """
    # Create the main figure with GridSpec
    fig = plt.figure(figsize=(9, 14))

    # Add main title to the figure
    fig.suptitle(station["station_id"], fontsize=16, fontweight="bold", y=0.95)

    # Create GridSpec with adjusted heights
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1.5, 1, 1, 2])

    # Add section titles
    fig.text(
        0.5,
        0.69,
        "Wavelet Statistics",
        horizontalalignment="center",
        fontsize=14,
        fontweight="400",
    )

    fig.text(
        0.5,
        0.305,
        "Scale-wise PDF evolution",
        horizontalalignment="center",
        fontsize=14,
        fontweight="400",
    )

    ### Map
    ax1 = fig.add_subplot(gs[0, 0])
    img = mpimg.imread(map_fpath)
    ax1.imshow(img)
    ax1.axis("off")

    ### Data analysis
    # Loading and preprocessing data
    df = pd.read_csv(
        data_folder + station["station_id"] + ".csv",
        parse_dates=["datetime"],
        index_col="datetime",
    )

    # Normalize data
    data = df["PRCP"].values
    x = (data - np.nanmean(data)) / np.nanstd(data)

    # Create subplot axes
    ax2 = fig.add_subplot(gs[0, 1])

    # Create 3-column subgrids for wavelet statistics
    gs_wavelet_stats_1 = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[1, :], wspace=0.2
    )
    axs_wavelet_stats_1 = [fig.add_subplot(gs_wavelet_stats_1[0, i]) for i in range(3)]

    gs_wavelet_stats_2 = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[2, :], wspace=0.2
    )
    axs_wavelet_stats_2 = [fig.add_subplot(gs_wavelet_stats_2[0, i]) for i in range(3)]

    ax5 = fig.add_subplot(gs[3, 0])

    # Perform analyses and create plots
    _, _, integral_scale, _ = acf_analysis(x, delta_t)

    # Wavelet Transform
    wT, scales = wavelet_transform(x, psi, delta_t, mode="per")

    # Spectral analysis
    lorentzian_param, ax2 = spectral_analysis(
        wT, scales, x, delta_t, integral_scale, ax=ax2
    )

    # Non-quantile based wavelet stats
    _ = plot_wavelet_stats(
        wT[1:], scales[1:], quantile_based=False, axs=axs_wavelet_stats_1
    )

    # Quantile based wavelet stats
    _ = plot_wavelet_stats(
        wT[1:], scales[1:], quantile_based=True, axs=axs_wavelet_stats_2
    )

    # Scale-wise PDF
    params, bin_x_list, emp_pdf_list = fit_scale_wise_pdf(
        wT[1:], scales[1:], lorentzian_param["wavelet"][0]["breakpoint"]
    )
    _ = plot_scale_wise_pdf(params, bin_x_list, emp_pdf_list, scales[1:], ax=ax5)

    # q vs a and q vs c
    gs_tsallis = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[3, 1], wspace=0.3
    )
    axs_tsallis = [fig.add_subplot(gs_tsallis[i, 0]) for i in range(2)]
    _ = plot_tsallis_params(params, axs_tsallis)

    # Adjust spacing for titles
    plt.tight_layout()
    fig.subplots_adjust(top=0.92, hspace=0.35, wspace=0.3)

    # Save results
    print("------ Saving Results ------")
    save_analysis_results(
        station["station_id"],
        lorentzian_param,
        params,
        bin_x_list,
        emp_pdf_list,
        base_path="./output",
    )
    return fig


def save_analysis_results(
    station_id, lorentzian_param, params, bin_x_list, emp_pdf_list, base_path="./output"
):
    """
    Save analysis results to separate folders for each variable type.

    Args:
        station_id (str): Station identifier
        lorentzian_param (dict): Dictionary containing Lorentzian parameters
        params (list): List of parameter arrays
        bin_x_list (list): List of bin arrays
        emp_pdf_list (list): List of empirical PDF arrays
        base_path (str): Base directory for output files
    """
    # Create main directories if they don't exist
    directories = {
        "lorentzian": os.path.join(base_path, "lorentzian_parameters"),
        "params": os.path.join(base_path, "distribution_parameters"),
        "bins": os.path.join(base_path, "bin_values"),
        "pdf": os.path.join(base_path, "empirical_pdfs"),
    }

    for directory in directories.values():
        os.makedirs(directory, exist_ok=True)

    # Save lorentzian param as JSON
    save_lorentzian_param(
        directories["lorentzian"],
        lorentzian_param,
        station_id,
        fname_template="{station_id}_lorentzian.json",
    )

    # Helper function to save list of numpy arrays
    def save_array_list(data_list, directory, filename):
        file_path = os.path.join(directory, filename)
        np.savez(file_path, *data_list)

    # Save params, bin_x_list, and emp_pdf_list
    save_array_list(params, directories["params"], f"{station_id}_params.npz")
    save_array_list(bin_x_list, directories["bins"], f"{station_id}_bins.npz")
    save_array_list(emp_pdf_list, directories["pdf"], f"{station_id}_pdfs.npz")


def load_analysis_results(station_id, base_path="./output"):
    """
    Load previously saved analysis results.

    Args:
        station_id (str): Station identifier
        base_path (str): Base directory for input files

    Returns:
        tuple: (lorentzian_param, params, bin_x_list, emp_pdf_list)
    """
    directories = {
        "lorentzian": os.path.join(base_path, "lorentzian_parameters"),
        "params": os.path.join(base_path, "distribution_parameters"),
        "bins": os.path.join(base_path, "bin_values"),
        "pdf": os.path.join(base_path, "empirical_pdfs"),
    }

    # Load Lorentzian parameters
    lorentzian_file = os.path.join(
        directories["lorentzian"], f"{station_id}_lorentzian.json"
    )

    lorentzian_param = load_lorentzian_param(lorentzian_file)

    # Helper function to load list of numpy arrays
    def load_array_list(directory, filename):
        file_path = os.path.join(directory, filename)
        with np.load(file_path) as data:
            return [data[arr] for arr in data.files]

    # Load params, bin_x_list, and emp_pdf_list
    params = load_array_list(directories["params"], f"{station_id}_params.npz")
    bin_x_list = load_array_list(directories["bins"], f"{station_id}_bins.npz")
    emp_pdf_list = load_array_list(directories["pdf"], f"{station_id}_pdfs.npz")

    return lorentzian_param, params, bin_x_list, emp_pdf_list


def save_lorentzian_param(
    outdir, lorentzian_params, station_id, fname_template="{station_id}_lorentzian.json"
):
    """
    Save Lorentzian parameters to a JSON file, converting any NumPy arrays to lists.

    Parameters
    ----------
    outdir : str
        Directory where the JSON file will be saved
    lorentzian_params : Dict[str, Any]
        Dictionary containing Lorentzian parameters, potentially including nested
        NumPy arrays, lists, and dictionaries
    station_id : str
        Station identifier used in the filename
    filename_template : str, optional
        Template for the output filename, default: "{station_id}_lorentzian.json"
    """

    def convert_numpy_to_serializable(obj):
        """
        Recursively convert NumPy arrays to lists in nested data structures.
        Parameters
        obj : Any
            The object to convert. Can be a NumPy array, list, dictionary,
            or nested combination of these.
        Returns
        Any
            The converted object with all NumPy arrays converted to lists.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {
                key: convert_numpy_to_serializable(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [convert_numpy_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy_to_serializable(item) for item in obj)
        return obj

    os.makedirs(outdir, exist_ok=True)

    # Convert all NumPy arrays to lists recursively
    serializable_lorentzian = convert_numpy_to_serializable(lorentzian_params)
    output_file = os.path.join(outdir, fname_template.format(station_id=station_id))

    # Save to JSON file
    with open(output_file, "w") as f:
        json.dump(serializable_lorentzian, f, indent=4)


def load_lorentzian_param(filepath):
    """
    Load Lorentzian parameters from a JSON file converting lists to NumPy arrays.
    """

    def convert_lists_to_numpy(obj):
        """
        Recursively convert lists to NumPy arrays in nested data structures.
        """
        if isinstance(obj, dict):
            return {key: convert_lists_to_numpy(value) for key, value in obj.items()}

        if isinstance(obj, list):
            # Try to convert the entire list to a NumPy array
            try:
                # Check if all elements are numeric or nested lists of numeric values
                arr = np.array(obj)
                # Only convert if we don't end up with an array of objects
                if arr.dtype != object:
                    return arr
            except (ValueError, TypeError):
                pass

            # If conversion failed or produced object array, process elements individually
            return [convert_lists_to_numpy(item) for item in obj]

        return obj

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r") as f:
        data = json.load(f)

    return convert_lists_to_numpy(data)
