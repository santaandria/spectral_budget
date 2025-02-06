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
from scipy.optimize import least_squares
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


def wavelet_psd(wT, scales, delta_t):
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
    K = (
        2 * np.pi / scales
    )  # Wavenumber (necessary for comparing wavelet power spectrum with Fourier spectrum)
    psd = [
        np.mean(w**2) * delta_t / (2 * np.pi * np.log(2)) for w in wT
    ]  # Power spectral density (Energy in each wavenumber band dK)
    return psd, K


def fourier_psd(x, delta_t, window="hann", nperseg=1024):
    """
    Computes the Fourier power density spectrum
    Return:
    psd: power density spectrum
    K: Wave number

    """
    f, P = welch(x, window=window, nperseg=nperseg, fs=1)  # fs=assumed to 1
    K = 2 * np.pi * f * (1 / delta_t)
    psd = P * delta_t / (2 * np.pi)  # psd = E/dK where dK = 2*np.pi*df*(1/delta_t)
    return psd, K


def wavelet_transform(x, psi, delta_t, mode="per"):
    """
    Compute the discrete wavelet transform up to N levels, according to a dyadic scale arrangement, where N = np.floor(np.log2(len(x)))
    Parameters:
    x (array-like): Data
    psi (pywt Wavelet)
    levels: Number of levels
    delta_t: Measurement spacing in physical space. Inverse of sampling frequency
    mode: Mode for the boundary in pywt.wavedec

    Returns:
    wT: [wT1, wT2, ..., wT_levels] where wTi corresponds to the ith level (increasing scale)
    scales: corresponding scales in the time unit
    """
    J = int(np.log2(len(x)))
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


def fit_psd_power_law(psd, K, integral_scale):
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
    pw_fit = piecewise_regression.Fit(
        x, y, n_breakpoints=1, start_values=[np.log10(integral_scale)]
    )
    results = pw_fit.get_results()["estimates"]

    c = results["alpha2"]["estimate"]
    c_ci = results["alpha2"]["confidence_interval"]
    breakpt = results["breakpoint1"]["estimate"]

    return c, c_ci, breakpt


def fit_lorentzian(psd, K, integral_scale):
    """
    Fit a Lorentzian function to the power spectrum density.

    First finds an estimate of the breakpoint in log-log space using piecewise regression.
    Then finds a first guess of the power law exponent. The first guess for the numerator
    is given by E(0) ~ integral_scale/pi. Finally uses non-linear least squares to fit
    the parameters.

    Args:
        psd: Power spectral density values
        K: Wavenumber values
        integral_scale: Integral scale parameter

    Returns:
        dict: Fitted parameters including A, B, c, confidence interval, and breakpoint
    """
    c, c_ci, breakpt = fit_psd_power_law(psd, K, integral_scale)

    def log_lorentzian(params: tuple, u):
        A, B, c = params
        eps = 1e-15
        # logaddexp(x1,x2) = log(exp(x1) + exp(x2)) is numerically more stable as it handles large values of x better.
        return np.log(max(A, eps)) - np.logaddexp(0, c * (u - np.log(max(B, eps))))

    params_init = [integral_scale / np.pi, 1, -c]
    out = least_squares(
        lambda p, u, v: (v - log_lorentzian(p, u)),
        params_init,
        args=(np.log(K), np.log(psd)),
    )

    A, B, c = out.x
    return {
        "A": A,
        "B": B,
        "c": c,
        "c_CI": -1 * np.array(c_ci),
        "breakpoint": 10**breakpt,
    }


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
    psd_F, K_F = fourier_psd(timeseries, delta_t, window="hann", nperseg=8192)

    # Discard the smallest and the largest scale from the analysis
    psd_w, K_w, scales = psd_w[1:-1], K_w[1:-1], scales[1:-1]
    # Fit Lorentzian
    lorentzian_param = fit_lorentzian(psd_w, K_w, integral_scale)

    # Create visualization
    if not ax:
        fig, ax = plt.subplots(figsize=(4, 3))

    # Plot Fourier and Wavelet PSDs
    ax.plot(K_F, psd_F, label="Fourier", alpha=0.8, linewidth=1, c="r")
    ax.plot(K_w, psd_w, "k-o", label="Wavelet", markersize=6, markerfacecolor="None")

    # Plot Lorentzian fit
    KK = np.logspace(np.log10(min(K_w)), np.log10(max(K_w)), 100)
    ax.plot(
        KK,
        lorentzian(
            KK, lorentzian_param["A"], lorentzian_param["B"], lorentzian_param["c"]
        ),
        "--b",
        linewidth=2,
        label=f"Lorentzian c={lorentzian_param['c']:.2f} ($CI_{{.95}}$: {lorentzian_param['c_CI'][0]:.2f}, {lorentzian_param['c_CI'][1]:.2f}), A={lorentzian_param['A']:.2f} h",
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

    return lorentzian_param, ax


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


def add_time_axis(ax, scales, wavenumber=False, loc="bottom"):
    """
    Adds a secondary time axis to the provided frequency axis of a plot.
    The secondary axis displays time units corresponding to the given frequency scales. The labels on the time axis are based on the first occurrence of each time unit in the list of time scales. The user can position the secondary axis either at the top or bottom of the plot.

    Parameters:
    ax : matplotlib.axes.Axes
        The primary axis object for the plot (frequency axis).

    scales : list or array-like
        A list of frequency scales used in the analysis.

    wavenumber : bool, optional (default=False)
        If True, treats frequency as 2π/scales.

    loc : {'top', 'bottom'}, optional (default='bottom')
        The location of the secondary axis relative to the primary axis.

    Returns:
    t_ax : matplotlib.axes.Axes
        The secondary time axis.
    """
    # Create a secondary axis twinned to the primary axis
    t_ax = ax.twiny()

    # Convert frequency scales to time strings
    time_strings = [
        TimeUnitConverter.convert_time_units(t, from_unit="h") for t in scales
    ]

    # Get unique time labels and their corresponding indices
    time_labels, time_indices = TimeUnitConverter.find_first_unit_occurrences(
        time_strings, return_indices=True
    )

    # Adjust secondary axis placement and style based on location
    if loc == "bottom":
        t_ax.xaxis.set_ticks_position("bottom")
        t_ax.xaxis.set_label_position("bottom")

        # Offset the secondary axis below the primary axis
        t_ax.spines["bottom"].set_position(("axes", -0.25))

        # Configure the twin axis frame
        t_ax.set_frame_on(True)
        t_ax.patch.set_visible(False)
        t_ax.spines["bottom"].set_visible(True)

        # Adjust tick parameters on the primary axis
        ax.tick_params(top=True, which="both")

    # Calculate tick positions on the primary axis scale
    tick_positions = [
        2 * np.pi / scales[i] if wavenumber else 1 / scales[i] for i in time_indices
    ]

    # Set properties for the secondary axis
    t_ax.set_xscale("log")
    t_ax.set_xlim(ax.get_xlim())
    t_ax.set_xticks(tick_positions)
    t_ax.minorticks_off()
    t_ax.set_xticklabels(time_labels, fontdict={"fontsize": 6})


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
        wT[1:], scales[1:], lorentzian_param["breakpoint"]
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

    # Convert numpy arrays in lorentzian_param to lists before saving
    serializable_lorentzian = {}
    for key, value in lorentzian_param.items():
        if isinstance(value, np.ndarray):
            serializable_lorentzian[key] = value.tolist()
        else:
            serializable_lorentzian[key] = value

    # Save Lorentzian parameters (dictionary)
    lorentzian_file = os.path.join(
        directories["lorentzian"], f"{station_id}_lorentzian.json"
    )
    with open(lorentzian_file, "w") as f:
        json.dump(serializable_lorentzian, f, indent=4)

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
    with open(lorentzian_file, "r") as f:
        lorentzian_param = json.load(f)

    # Convert lists back to numpy arrays in lorentzian_param
    for key, value in lorentzian_param.items():
        if isinstance(value, list):
            lorentzian_param[key] = np.array(value)

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
