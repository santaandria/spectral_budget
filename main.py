# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.helpers import *
import pywt
from scipy.special import gamma
from scipy.optimize import minimize, curve_fit, least_squares
from scipy.integrate import quad, simpson, trapezoid
from scipy.stats import kurtosis, skew, bootstrap


from statsmodels.tsa.stattools import acf
import statsmodels.api as sm
from scipy.integrate import simpson, romb, quad
from scipy.signal import welch
from collections import Counter
from src.helpers.maps_helper import *
import warnings

import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import plotly.graph_objects as go

import os
import json

PRCP_FOLDER = "/mnt/d/climate_data/ARPAV_5min/filled_sampling/"
RECOMPUTE_LPARAMS = False


# %%
###################################################################################
###################################################################################
def conceptual_plot(savepath=None):
    def low_freq_correction_factor(A, integral_scale, alpha, f):
        E0 = integral_scale / np.pi
        return ((E0 / A) + alpha * f**2) / (1 + alpha * f**2)

    def cutoff(beta, f1, f, d):
        return np.exp(-beta * ((f / f1) ** d - (10 / f1) ** d))

    def generalized_spectrum(f, A, f0, c, integral_scale, alpha, beta, f1, d):
        Fl = low_freq_correction_factor(A, integral_scale, alpha, f)
        Fh = cutoff(beta, f1, f, d)
        lorentz = lorentzian(f, A, f0, c)
        return Fl * lorentz * Fh

    def compute_source(E, f, u0, D0):
        dE_df = np.gradient(E, f, edge_order=2)
        J = u0 * f**2 * E - D0 * f**3 * dE_df
        return np.gradient(J, f, edge_order=2)

    def dissipation(E, f, nu):
        dE_df = np.gradient(E, f, edge_order=2)
        return -nu * f**3 * np.gradient(dE_df, f, edge_order=2)

    # Parameters
    f = np.logspace(-8, 3, 10000)
    A, f0, c = 10, 0.01, 0.8
    alpha = 1e14  # transition Fh
    integral_scale = 3
    beta = 10
    f1 = 1e2
    d = 2
    u0 = 1
    D0 = 1
    nu = 0.8

    E = generalized_spectrum(f, A, f0, c, integral_scale, alpha, beta, f1, d)
    S = compute_source(E, f, u0, D0)
    D = dissipation(E, f, nu)

    fig, axs = plt.subplots(2, 1, figsize=(5, 6), tight_layout=True)
    axs[0].plot(f, E, "-r", linewidth=2, zorder=99)
    axs[0].axhline(y=integral_scale / np.pi, linestyle="--", c="grey", linewidth=0.5)

    upper_axis_settings = {
        "ylabel": "$\log \hat{E} \, (\\tilde{f\,}) $",
        "xlabel": "$\log \\tilde{f\,}$",
        "xscale": "log",
        "yscale": "log",
        "ylim": (1e-3, 5e1),
        "title": "(a)",
    }
    configure_axis(axs[0], upper_axis_settings)

    axs[1].plot(f, -S, "-k", linewidth=2)
    axs[1].plot(f, D, "-r", linewidth=1)
    axs[1].plot(f, S - D, "-b", linewidth=1)

    lower_axis_settings = {
        "ylabel": "$\partial_{f} \ J$",
        "xlabel": "$\log \\tilde{f\,}$",
        "xscale": "log",
        "title": "(b)",
    }
    configure_axis(axs[1], lower_axis_settings)

    for ax in axs:
        ax.axvline(x=f0, linestyle="--", c="grey", linewidth=0.5)
        ax.axvspan(1e-6, 10, color="0.90")
        ax.tick_params(
            labelleft=False,
            labelbottom=False,
            bottom=False,
            top=False,
            left=False,
            right=False,
            which="both",
        )
    if savepath:
        plt.savefig(savepath, dpi=96, format="pdf")
    plt.show()


def plot_lorentzian_map(
    lparam_df, stations_df, veneto_map, veneto_map_inset, save_path=None
):
    """
    Args:
    res (dict): {station: {'A': value, 'B': value, 'c': value}}
    """

    lorentzian_df = lparam_df.merge(
        stations_df[["station_id"]], on="station_id", how="left"
    )

    # Parameter settings for Lorentzian
    lorentz_params_settings = {
        "A": {"cmap": "plasma", "title": "$A \ [h]$"},
        "B": {"cmap": "plasma", "title": "$f_0 \ [h^{-1}]$"},
        "c": {"cmap": "plasma", "title": "$c$"},
    }
    labels = ["(a)", "(b)", "(c)"]
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3, wspace=0)

    # Create maps for each parameter
    for i, (param, settings) in enumerate(lorentz_params_settings.items()):
        print(f"Generating plots for {param} ...")
        map_fpath = f"output/lorentzian_{param}.png"

        # Choose map type based on first or subsequent plots
        current_map = veneto_map_inset if i == 0 else veneto_map

        # Add scatter plot to map
        current_map.add_scatter(
            data_df=lorentzian_df,
            data_col=param,
            crs=4326,
            temporary=True,
            to_show=["raster", "base"],
            x_col="Lon",
            y_col="Lat",
            s=100,
            cmap=settings["cmap"],
            cb_label=settings["title"],
            save_filepath=map_fpath,
        )

        # Add subplot
        ax = fig.add_subplot(gs[0, i])
        img = mpimg.imread(map_fpath)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(labels[i], fontsize=12)
        os.remove(map_fpath)

    # Save the complete figure
    if save_path:
        plt.savefig(save_path, dpi=300, format="pdf")
        plt.close()


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
    integral_limit = (min(K_w), max(K_w))

    psd_w, K_w = psd_w[:-1], K_w[:-1]  # Ignore largest scale in the lorentzian fit

    C = 2 * np.pi if angular else 1

    # Initial parameter estimation
    c, _, breakpt = fit_psd_power_law(psd_w, K_w, integral_scale, angular=angular)
    params_init = [integral_scale / np.pi / C, 10**breakpt, -c]

    # Fit wavelet spectrum
    res_w = fit_lorentzian(
        psd_w,
        K_w,
        params_init,
        integral_limit=integral_limit,
        return_err=return_err,
        bootstrap_samples=1000,
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


def fit_lorentzian(
    psd, K, params_init, integral_limit, return_err=True, bootstrap_samples=1000
):
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
    # perr = np.sqrt(np.diag(pcov))

    ###### Add AUC = 1 constraint to refine B ######
    def residuals(p, K, psd):
        integral = quad(
            lambda f: lorentzian(f, popt[0], p, popt[2]), *integral_limit, epsrel=1e-9
        )[0]
        penalization = abs(1 - integral) * 10000
        return (
            np.log(psd)
            - log_lorentzian((popt[0], p, popt[2]), np.log(K))
            - penalization
        )

    out = least_squares(residuals, popt[1], args=(K, psd))

    params = {
        "A": popt[0],
        "B": out.x[0],
        "c": popt[2],
    }

    if not return_err:
        return params

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
                p0=(params["A"], params["B"], params["c"]),
            )
            # popt_resampled_refined = least_squares(residuals,
            #             params['B'],
            #             args=(K_w, psd_w))

            # boot_params.append((popt_resampled[0], popt_resampled_refined.x[0], popt_resampled[2]))
            boot_params.append(popt_resampled)
        except RuntimeError:
            continue

    boot_params = np.array(boot_params)
    boot_se = np.std(boot_params, axis=0)

    return params, boot_se


def compute_spectra(stations, delta_t, fit_lorentzian=False, return_wT=False):
    psi = pywt.Wavelet("haar")
    spectra = {}
    lparams = {}
    std_error = {}
    int_scale = {}
    wavelet_coeff = {}

    for station in stations:
        df = pd.read_csv(
            PRCP_FOLDER + station + ".csv",
            parse_dates=["datetime"],
            index_col="datetime",
        )
        data = df["PRCP"].values

        # Normalize
        x = data - np.nanmean(data)
        x = x / np.nanstd(x)

        # ACF Analysis
        _, _, integral_scale, _ = acf_analysis(x, delta_t)

        # Wavelet Transform
        wT, tau = wavelet_transform(x, psi, delta_t, mode="per")
        psd_w, K_w = wavelet_psd(wT, tau, delta_t, angular=False)
        psd_F, K_F = fourier_psd(
            x, delta_t, window="hann", nperseg=8 * 8192, angular=False
        )
        spectra.update({station: {"wavelet": (psd_w, K_w), "fourier": (psd_F, K_F)}})
        int_scale.update({station: integral_scale})
        if return_wT:
            wavelet_coeff.update({station: {"wT": wT, "scales": tau}})

        if fit_lorentzian:
            print(f"------------ Processing {station} ------------")
            try:
                p, perr = lorentzian_spectra(
                    (psd_w, K_w),
                    integral_scale=integral_scale,
                    return_err=True,
                    angular=False,
                )
                lparams.update({station: p})
                std_error.update({station: perr})

                ## Check
                area = quad(
                    lambda f: lorentzian(f, p["A"], p["B"], p["c"]),
                    min(K_w),
                    max(K_w),
                    epsrel=1e-9,
                )[0]
                print(
                    f"{station} & A = {p['A']:.2f} +/- {perr[0]:.2f} & f0 = {p['B']:.3f}  +/- {perr[1]:.3f} & c = {p['c']:.2f} +/- {perr[2]:.2f} & area = {area:.4f} \\\\"
                )

            except:
                print("!!!!!!!! OPTIMIZATION FAILED !!!!!!!!")
                continue

    if fit_lorentzian:
        lparams_df = pd.DataFrame.from_dict(lparams, orient="index").rename_axis(
            "station_id"
        )
        std_error_df = (
            pd.DataFrame.from_dict(std_error, orient="index")
            .rename_axis("station_id")
            .rename(columns={0: "A", 1: "B", 2: "c"})
        )
        lparams_df.to_csv("data/lparams.csv")
        std_error_df.to_csv("output/lorentzian_std_error.csv")

    if return_wT:
        return spectra, wavelet_coeff, int_scale
    return spectra, int_scale


###################################################################################
###################################################################################
# %%
#### Loading data #####
stations_df = pd.read_csv("data/gt_30y_lt_1perc_missing.csv")
stations = stations_df["station_id"].values
delta_t = 5 / 60

if RECOMPUTE_LPARAMS:
    spectra, int_scale = compute_spectra(
        stations, delta_t=delta_t, fit_lorentzian=True, return_wT=False
    )
    lparams = pd.read_csv("data/lparams.csv")
    lparams = lparams.merge(
        stations_df[["station_id", "Elv", "Lat", "Lon", "distance_to_coast"]],
        on="station_id",
    )
    lparams["int_scale"] = lparams["station_id"].map(int_scale)
    min_K = {stn: min(v["wavelet"][1]) for stn, v in spectra.items()}
    max_K = {stn: max(v["wavelet"][1]) for stn, v in spectra.items()}

    lparams = lparams.assign(
        min_K=lparams["station_id"].map(min_K),
        max_K=lparams["station_id"].map(max_K),
    )

    lparams = lparams.set_index("station_id")
    lparams.to_csv("data/lparams.csv")
else:
    lparams = pd.read_csv("data/lparams.csv").set_index("station_id")

std_err = pd.read_csv("output/lorentzian_std_error.csv", index_col="station_id")
veneto_map_inset = create_veneto_map(cb=1.5, show=False, add_inset=True)
veneto_map = create_veneto_map(cb=1.5, show=False, add_inset=False)
# %%
##### Fig 3: Plotting Parameter Map #####
print("------------ Plotting Map ------------")
plot_lorentzian_map(
    lparams,
    stations_df,
    veneto_map,
    veneto_map_inset,
    save_path="output/lorentzian_parameters_maps.pdf",
)
plt.close()
# %%
##### Fig 1: Conceptual Plot ######
conceptual_plot("output/conceptual_plot.pdf")


# %%
###### Fig 2: Spectra Plot #########
def add_extra_wT(ax, station):
    df = pd.read_csv(
        PRCP_FOLDER + station + ".csv",
        parse_dates=["datetime"],
        index_col="datetime",
    )
    data = df["PRCP"].values

    # Normalize
    x = data - np.nanmean(data)
    x = x / np.nanstd(x)

    colors = ["g", "y", "b"]
    markers = ["s", "v", "x"]
    for i, wav in enumerate(["sym3", "coif3", "db2"]):
        psi = pywt.Wavelet(wav)
        J = pywt.dwt_max_level(len(x), psi)
        wT_ = pywt.wavedec(x, psi, level=J, mode="per")[::-1][:-1]
        scales_ = 2 ** (np.arange(1, J + 1)) * delta_t
        psd_w_, K_w_ = wavelet_psd(wT_, scales_, delta_t, angular=False)
        ax.scatter(K_w_, psd_w_, s=20, c=colors[i], marker=markers[i], zorder=3)


stations = ["003_BL_Ar", "127_VR_Bu", "168_VE_Ch"]
spectra, wavelet_coeff, int_scale = compute_spectra(
    stations, delta_t=delta_t, fit_lorentzian=False, return_wT=True
)

labels = ["(a)", "(b)", "(c)"]
fig, axs = plt.subplots(1, 3, figsize=(9, 3), tight_layout=True)

for col, ax in enumerate(axs):
    station = stations[col]
    ax.plot(*spectra[station]["fourier"][::-1], alpha=0.8, linewidth=1, c="k", zorder=2)
    ax.scatter(*spectra[station]["wavelet"][::-1], s=30, fc="None", ec="r", zorder=4)
    add_extra_wT(ax, station)
    p = lparams.loc[station]
    KK = np.logspace(
        np.log10(min(spectra[station]["wavelet"][1])),
        np.log10(max(spectra[station]["wavelet"][1])),
        100,
    )
    ax.plot(KK, lorentzian(KK, p["A"], p["B"], p["c"]), "-r", linewidth=2, zorder=4)
    axis_settings = {
        "ylabel": "$\hat{E} \, (f \,)  \ [h]$",
        "xlabel": "$f \ [h^{-1}]$",
        "xscale": "log",
        "yscale": "log",
        "ylim": (1e-2, 5e1),
        "title": labels[col],
    }

    # Context
    map_ax = ax.inset_axes([0.05, 0.05, 0.4, 0.4])
    map_fpath = f"./data/maps/{station}.png"

    img = mpimg.imread(map_fpath)
    map_ax.imshow(img)
    map_ax.axis("off")

    ax.axhline(y=int_scale[station] / np.pi, linestyle="--", c="k", linewidth=0.5)
    ax.add_artist(
        plt.Rectangle(
            (1, ax.get_ylim()[0]),
            1e3,
            1e3,
            alpha=0.5,
            zorder=1,
            color="grey",
            ec="None",
        )
    )  # 1H region where artificial smoothing (viscosity) due to sensor sampling averaging takes effect (Paschalis paper)

    configure_axis(ax, axis_settings)
    add_time_axis(ax, wavelet_coeff[station]["scales"], wavenumber=False, loc="top")

plt.savefig("output/energy_spectra.pdf", dpi=600, format="pdf")
plt.show()
# plt.close()
# %%
############ Fig 4: Parameter Correlations ###################
fig, axs = plt.subplots(1, 3, figsize=(9, 3), tight_layout=True)
labels = ["(a)", "(b)", "(c)"]
axs[0].scatter(
    lparams["A"],
    lparams["B"],
    c=lparams["Elv"],
    cmap="copper",
    alpha=0.75,
    ec="k",
    linewidth=1,
)
axs[1].scatter(
    lparams["A"],
    lparams["c"],
    c=lparams["Elv"],
    cmap="copper",
    alpha=0.75,
    ec="k",
    linewidth=1,
)
axs[2].scatter(
    lparams["B"],
    lparams["c"],
    c=lparams["Elv"],
    cmap="copper",
    alpha=0.75,
    ec="k",
    linewidth=1,
)

axes_labels = [
    ("$A \ [h]$", "$f_0 \ [h^{-1}]$"),
    ("$A \ [h]$", "$c$"),
    ("$f_0 \ [h^{-1}]$", "$c$"),
]

for i, ax in enumerate(axs):
    ax.set_xlabel(axes_labels[i][0])
    ax.set_ylabel(axes_labels[i][1])
    ax.set_title(labels[i])
    # if i == 0:
    #     ax.set_yscale('log')
    #     ax.set_ylim((1e-3, 1e-1))
    # if i == 2:
    #     ax.set_xscale('log')
    #     ax.set_xlim((1e-3, 1e-1))

plt.savefig("output/parameter_correlations.pdf", dpi=96, format="pdf")
plt.show()
# plt.close()
# %%
######### Fig 5: Elevation - Distance to coast plot ##########
params = ["A", "B", "c"]
labels = ["(a)", "(b)", "(c)"]
cb_labels = ["$A \ [h]$", "$f_0 \ [h^{-1}]$", "$c$"]

fig = plt.figure(figsize=(9, 7), constrained_layout=True)
top, bottom = fig.subfigures(2, 1)

# top.suptitle('Top')
axs_top = top.subplots(1, 3)
for i, ax in enumerate(axs_top):
    sc = ax.scatter(
        lparams["distance_to_coast"],
        lparams["Elv"],
        c=lparams[params[i]],
        cmap="plasma",
        edgecolors="k",
        alpha=0.75,
        s=50,
    )
    ax.set_xlabel("Distance to Coast [km]")
    ax.set_ylabel("Elevation [m]")
    ax.set_title(labels[i])
    # ax.set_yscale('log')
    fig.colorbar(sc, ax=ax, label=cb_labels[i], orientation="horizontal")

# bottom.suptitle('Bottom')
axs_bottom = bottom.subplots(1, 3)
labels = ["(e)", "(f)", "(g)"]
plain = lparams.loc[lparams["Elv"] < 250]

axs_bottom[0].scatter(
    plain["distance_to_coast"],
    plain["A"],
    c=plain["Elv"],
    cmap="copper",
    edgecolors="k",
    alpha=0.75,
    s=50,
)
axs_bottom[1].scatter(
    plain["distance_to_coast"],
    plain["B"],
    c=plain["Elv"],
    cmap="copper",
    edgecolors="k",
    alpha=0.75,
    s=50,
)
sc = axs_bottom[2].scatter(
    plain["distance_to_coast"],
    plain["c"],
    c=plain["Elv"],
    cmap="copper",
    edgecolors="k",
    alpha=0.75,
    s=50,
)

for i, ax in enumerate(axs_bottom):
    ax.set_xlabel("Distance to Coast [km]")
    ax.set_ylabel(cb_labels[i])
    ax.set_title(labels[i])
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal")
    cb.set_label(label="Elevation [m]", weight=400)

plt.savefig("output/parameters_features.pdf", dpi=96, format="pdf")
plt.show()


# %%
###### Fig 7: Compute Source #######
def source_term(f, A, B, c, m):
    h = f / B
    return (
        -(
            A
            * f
            * u0
            * (
                c**2 * m * h**c * (-1 + h**c)
                - c * h**c * (1 + h**c) * (2 * m - 1)
                - 2 * (1 + h**c) ** 2
            )
        )
        / (1 + h**c) ** 3
    )


def normalized_source(f, A, B, c, m, int_scale):
    normalization = source_term(1 / int_scale, A, B, c, m)
    return source_term(f, A, B, c, m) / normalization


f = np.logspace(-6, 1, 100)
u0 = 1
m_values = [0, 1, 10]

params_df = lparams.copy()
for m in m_values:
    # params_df[f"source_m{m}"] = params_df.apply(lambda row: source_term(f, row["A"], row["B"], row["c"], m), axis=1)
    params_df[f"source_m{m}"] = params_df.apply(
        lambda row: normalized_source(
            f, row["A"], row["B"], row["c"], m, row["int_scale"]
        ),
        axis=1,
    )
    params_df["f_normalized"] = params_df.apply(
        lambda row: row["int_scale"] * f, axis=1
    )

orographic = params_df.loc[params_df["Elv"] > 250]
coast = params_df.loc[(params_df["Elv"] < 250) & (params_df["distance_to_coast"] < 30)]
mixed = params_df.loc[(params_df["Elv"] < 250) & (params_df["distance_to_coast"] >= 30)]

fig = plt.figure(figsize=(9, 9), constrained_layout=True)
fig1, fig2, fig3 = fig.subfigures(3, 1)
labels = [
    "(a) Elevation $\geq$ 250m",
    "(b)  Elevation $<$ 250m and Distance to Coast $\leq$ 30km",
    "(c) Elevation $<$ 250m and Distance to Coast $>$ 30km",
]
dfs = [orographic, coast, mixed]

for i, fig in enumerate((fig1, fig2, fig3)):
    axs = fig.subplots(1, 3)
    fig.suptitle(labels[i])

    for col, m in enumerate(m_values):
        y = np.vstack(dfs[i][f"source_m{m}"].values)
        # x = np.array([f]*len(y))
        x = np.vstack(dfs[i]["f_normalized"].values)
        axs[col].plot(x.T, y.T, "k-", linewidth=0.3)
        axs[col].plot(x.mean(axis=0), y.mean(axis=0), "r--", linewidth=3)
        axs[col].set_xscale("log")
        axs[col].set_xlabel("$\Gamma_0 f$")
        axs[col].set_ylabel("$S(f)~/~S(\Gamma_0 ^{-1})$")
        axs[col].set_title(f"m={m}")
        axs[col].axhline(y=1, linestyle="--", c="grey", linewidth=0.5)
        axs[col].axvline(x=1, linestyle="--", c="grey", linewidth=0.5)
        axs[col].set_xlim((1e-6, 1e2))
        axs[col].set_ylim((0, 3))

plt.savefig("output/source_term.pdf", dpi=96, format="pdf")
plt.show()


# %%
########## Fig 8: Shannon Entropy ################
def shannon_entropy(A, B, c, f_min, f_max):
    # Return normalized shannon entropy
    def entropy_integrand(f):
        p = lorentzian(f, A, B, c)
        return -p * np.log(p) if p > 0 else 0  # Avoid log(0) issues

    # Compute entropy using numerical integration
    entropy, _ = quad(entropy_integrand, f_min, f_max)
    return entropy / np.log(f_max - f_min)


lparams["Shannon_Entropy"] = lparams.apply(
    lambda row: shannon_entropy(
        row["A"], row["B"], row["c"], row["min_K"], row["max_K"]
    ),
    axis=1,
)
# Fit hyperplane
# Assuming lparams is your existing DataFrame with B, c, and Shannon_Entropy columns

X = lparams[["B", "c"]].values
X = sm.add_constant(X)  # constant intercept
y = lparams["Shannon_Entropy"].values

# OLS regression model
model = sm.OLS(y, X).fit()

# print(model.summary())

intercept = model.params[0]
slope_B = model.params[1]
slope_c = model.params[2]

print(
    f"Hyperplane equation: H(f) = {slope_B:.3f} × B + {slope_c:.3f} × c + {intercept:.3f}"
)


B_min, B_max = lparams["B"].min(), lparams["B"].max()
c_min, c_max = lparams["c"].min(), lparams["c"].max()

B_range = np.linspace(B_min, B_max, 10)
c_range = np.linspace(c_min, c_max, 10)
B_grid, c_grid = np.meshgrid(B_range, c_range)
z_grid = intercept + slope_B * B_grid + slope_c * c_grid


# %%
########## Fig 8a: Shannon Entropy ################
fig = go.Figure()

fig.add_trace(
    go.Surface(
        x=B_grid,
        y=c_grid,
        z=z_grid,
        colorscale="Reds",
        opacity=0.6,
        name="Regression Plane",
        showscale=False,
        contours={
            "x": {"show": True, "width": 2, "color": "darkred"},
            "y": {"show": True, "width": 2, "color": "darkred"},
            "z": {"show": False},
        },
        hidesurface=False,
    )
)

fig.add_trace(
    go.Scatter3d(
        x=lparams["B"],
        y=lparams["c"],
        z=lparams["Shannon_Entropy"],
        mode="markers",
        marker=dict(
            size=12,
            color=lparams["Shannon_Entropy"],
            colorscale="plasma",
            line=dict(color="black", width=5),
            opacity=0.9,
        ),
        name="Data Points",
    )
)

camera = dict(
    up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=-2, y=1.5, z=0.5)
)

fig.update_layout(
    scene_camera=camera,
    font=dict(size=15, family="Arial"),
    scene=dict(
        xaxis=dict(
            tickvals=[0.02, 0.04, 0.06],
            range=[0, 0.08],
            title=dict(
                text="f<sub>0</sub>",
                font=dict(size=24, weight=500),
            ),
            backgroundcolor="white",
            gridcolor="lightgrey",
            showbackground=True,
            zerolinecolor="white",
        ),
        yaxis=dict(
            title=dict(
                text="c",
                font=dict(size=24, weight=500),
            ),
            backgroundcolor="white",
            gridcolor="lightgrey",
            showbackground=True,
            zerolinecolor="white",
        ),
        zaxis=dict(
            title=dict(
                text=r"H(f)",
                font=dict(size=24, weight=500),
            ),
            backgroundcolor="white",
            gridcolor="lightgrey",
            showbackground=True,
            zerolinecolor="white",
        ),
    ),
    width=1000,
    height=800,
    margin=dict(r=5, l=5, b=5, t=5),
)

fig.add_annotation(
    x=0.1,
    y=0.9,
    xref="paper",
    yref="paper",
    text=f"H(f) = {slope_B:.3f} × f<sub>0</sub> - {np.abs(slope_c):.3f} × c + {intercept:.3f}<br>R² = {model.rsquared:.2f}",
    showarrow=False,
    font=dict(size=18),
    bgcolor="white",
    bordercolor="black",
    borderwidth=1,
    align="center",
)
fig.write_image(
    "output/shannon_entropy_regression.png",
    width=1000,
    height=800,
    scale=10,
)
fig.show()


# %%
########## Fig 8b: Shannon Entropy ################
veneto_map_inset.add_scatter(
    data_df=lparams,
    data_col="Shannon_Entropy",
    crs=4326,
    temporary=True,
    to_show=["raster", "base"],
    x_col="Lon",
    y_col="Lat",
    s=100,
    cmap="plasma",
    cb_label="Normalized Shannon Entropy",
    save_filepath="output/shannon_entropy.png",
)

# %%
############## Seasonal Analysis ##############
from scipy.stats import linregress


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


# frequency bounds in h^-1 -> period ranges of [1day-1hour] and [1hour-1min]
freq_ranges = {
    "1day-1hour": (1 / 24, 1),
    "1hour-1min": (1, 60),
}

psi = pywt.Wavelet("haar")
spectra = {}
lparams = {}
std_error = {}
int_scale = {}

# All stations (already includes the three example stations used for the spectra plot below)
stations = stations_df["station_id"].values
example_stations = ["003_BL_Ar", "127_VR_Bu", "168_VE_Ch"]
scaling_records = []

for station in stations:
    df = pd.read_csv(
        PRCP_FOLDER + station + ".csv",
        parse_dates=["datetime"],
        index_col="datetime",
    )

    summer = df.copy()
    summer.loc[~summer.index.month.isin([6, 7, 8]), "PRCP"] = np.nan  # JJA

    winter = df.copy()
    winter.loc[~winter.index.month.isin([12, 1, 2]), "PRCP"] = np.nan  # DJF

    result = {}
    ints = {}

    for season, data in zip(["summer", "winter"], [summer, winter]):
        data = data["PRCP"].values

        # Normalize
        x = data - np.nanmean(data)
        x = x / np.nanstd(x)

        # ACF Analysis
        _, _, integral_scale, _ = acf_analysis(x, delta_t)

        # Wavelet Transform
        wT, tau = wavelet_transform(x, psi, delta_t, mode="per")
        f = 1 / tau

        psd = np.array([np.nanmean(w**2) * delta_t / np.log(2) for w in wT])
        psd, f, tau = psd[psd > 0], f[psd > 0], tau[psd > 0]
        result.update({season: (psd, f)})
        ints.update({season: integral_scale})

        for range_name, rng in freq_ranges.items():
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
scaling_df.to_csv("output/seasonal_scaling_slopes.csv", index=False)
scaling_df = scaling_df.merge(
    stations_df[["station_id", "Elv", "Lat", "Lon"]], on="station_id"
)

# %%
############## Fig 6: Seasonal Analysis ##############
labels = ["(a)", "(b)", "(c)"]
fig, axs = plt.subplots(1, 3, figsize=(9, 3), tight_layout=True)
psd1_A = [3e-2, 5e-2, 5e-2]

for col, ax in enumerate(axs):
    station = example_stations[col]
    ax.plot(*spectra[station]["summer"][::-1], "r-", zorder=4, marker="o")
    ax.plot(*spectra[station]["winter"][::-1], "b-", zorder=4, marker="s")
    k = np.logspace(-0.9, -0.1, 3)
    psd1 = psd1_A[col] * k**-1
    psd2 = 7e-1 * k**-0.5
    ax.plot(k, psd1, "k--", linewidth=1)
    ax.plot(k, psd2, "k--", linewidth=1)

    ax.text(
        k[1] - 0.05,
        psd1[1],
        "$\sim k^{-1}$",
        fontsize=8,
        color="k",
        ha="right",
        va="top",
    )
    ax.text(
        k[0] + 0.1,
        psd2[0] - 0.5,
        "$\sim k^{-0.5}$",
        fontsize=8,
        color="k",
        ha="left",
        va="bottom",
    )

    axis_settings = {
        "ylabel": "$\hat{E} \, (f \,)  \ [h]$",
        "xlabel": "$f \ [h^{-1}]$",
        "xscale": "log",
        "yscale": "log",
        "ylim": (1e-2, 5e1),
        "title": labels[col],
    }

    # Context
    map_ax = ax.inset_axes([0.05, 0.05, 0.4, 0.4])
    map_fpath = f"./data/maps/{station}.png"

    img = mpimg.imread(map_fpath)
    map_ax.imshow(img)
    map_ax.axis("off")

    ax.add_artist(
        plt.Rectangle(
            (1, ax.get_ylim()[0]),
            1e3,
            1e3,
            alpha=0.5,
            zorder=1,
            color="grey",
            ec="None",
        )
    )  # 1H region where artificial smoothing (viscosity) due to sensor sampling averaging takes effect (Paschalis paper)

    configure_axis(ax, axis_settings)
    add_time_axis(ax, tau, wavenumber=False, loc="top", pad_frac=0.12)
plt.savefig("output/seasonal_analysis.pdf", dpi=600, format="pdf")
plt.show()

# %%
# Scaling-slope comparison across all stations
############## Fig S1: Seasonal Analysis ##############
freq_range_order = ["1day-1hour", "1hour-1min"]
panel_labels = ["(a)", "(b)"]
season_colors = {"summer": "#C65F5F", "winter": "#3B75AF"}
positions = {"summer": 1, "winter": 2}

rng = np.random.default_rng(0)  # reproducible jitter

fig, axs = plt.subplots(1, 2, figsize=(6, 3), tight_layout=True)

for ax, freq_range, label in zip(axs, freq_range_order, panel_labels):
    sub = scaling_df[scaling_df["freq_range"] == freq_range]

    for season, pos in positions.items():
        values = sub.loc[sub["season"] == season, "slope"].values
        print(f"{season} ({freq_range}) = {values.mean():.2f}")
        parts = ax.violinplot(
            values, positions=[pos], showmeans=True, showextrema=False
        )
        for body in parts["bodies"]:
            body.set_facecolor(season_colors[season])
            body.set_edgecolor("k")
            body.set_alpha(0.6)

        jitter = rng.uniform(-0.05, 0.05, size=len(values))
        ax.scatter(
            np.full(len(values), pos) + jitter,
            values,
            color="k",
            alpha=0.6,
            s=15,
            zorder=3,
        )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["JJA", "DJF"])
    ax.set_title(label)
    ax.set_ylabel("$c$" if ax is axs[0] else "")

plt.savefig("output/seasonal_slope_comparison.pdf", dpi=600, format="pdf")
plt.show()

# %%
############## Fig S2: Seasonal Analysis ##############
# seasons = ["summer", "winter"]
# freq_ranges = ["1day-1hour", "1hour-1min"]

# slope_dfs = {}
# for season in seasons:
#     for freq_range in freq_ranges:
#         key = f"{season}_{freq_range}"
#         slope_dfs[key] = scaling_df[
#             (scaling_df["season"] == season) & (scaling_df["freq_range"] == freq_range)
#         ].copy()

# freq_range_labels = {
#     "1day-1hour": "[1 day - 1h]",
#     "1hour-1min": "[1h - 5min]",
# }

# for key, df in slope_dfs.items():
#     season, freq_range = df["season"].iloc[0], df["freq_range"].iloc[0]
#     cb_label = f"c ({season.capitalize()} - {freq_range_labels[freq_range]})"
#     veneto_map_inset.add_scatter(
#         data_df=df,
#         data_col="slope",
#         crs=4326,
#         temporary=True,
#         to_show=["raster", "base"],
#         x_col="Lon",
#         y_col="Lat",
#         s=100,
#         cmap="plasma",
#         cb_label=cb_label,
#         save_filepath=f"output/slope_map_{key}.png",
#     )


# %%
def plot_slope_map(scaling_df, veneto_map, veneto_map_inset, save_path=None):
    """
    Args:
        scaling_df: DataFrame with columns [station_id, season, freq_range, slope, std_error, Lon, Lat, Elv]
    """
    seasons = ["summer", "winter"]
    freq_ranges = ["1day-1hour", "1hour-1min"]
    freq_range_labels = {
        "1day-1hour": "[1 day - 1h]",
        "1hour-1min": "[1h - 5min]",
    }
    clabel = {"summer": "JJA", "winter": "DJF"}
    labels = [["(a)", "(b)"], ["(c)", "(d)"]]  # row = freq_range, col = season

    # Split into 4 subsets, one per (season, freq_range) combo
    slope_dfs = {}
    for season in seasons:
        for freq_range in freq_ranges:
            key = f"{season}_{freq_range}"
            slope_dfs[key] = scaling_df[
                (scaling_df["season"] == season)
                & (scaling_df["freq_range"] == freq_range)
            ].copy()

    fig = plt.figure(figsize=(6, 7))
    subfigs = fig.subfigures(2, 1, hspace=0.01)

    first_plot = True
    for row, freq_range in enumerate(freq_ranges):
        subfig = subfigs[row]
        subfig.suptitle(freq_range_labels[freq_range], fontsize=13)
        axs = subfig.subplots(1, 2, gridspec_kw={"wspace": 0.01})

        for col, season in enumerate(seasons):
            key = f"{season}_{freq_range}"
            df_here = slope_dfs[key]
            print(f"Generating plot for {season} - {freq_range} ...")

            cb_label = f"c ({clabel[season]})"
            map_fpath = f"output/slope_map_{key}.png"

            current_map = veneto_map_inset if first_plot else veneto_map
            first_plot = False

            current_map.add_scatter(
                data_df=df_here,
                data_col="slope",
                crs=4326,
                temporary=True,
                to_show=["raster", "base"],
                x_col="Lon",
                y_col="Lat",
                s=100,
                cmap="plasma",
                cb_label=cb_label,
                save_filepath=map_fpath,
            )

            ax = axs[col]
            img = mpimg.imread(map_fpath)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(labels[row][col], fontsize=12)
            os.remove(map_fpath)

    if save_path:
        plt.savefig(save_path, dpi=300, format="pdf")
        plt.close()


plot_slope_map(
    scaling_df, veneto_map, veneto_map_inset, save_path="output/c_map_seasonal.pdf"
)
# %%
