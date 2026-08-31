# %%
"""
Spectral budget of precipitation time series - manuscript figure generation.

Reads the per-station Lorentzian parameters produced by the integral-scale
analysis and generates every figure and table for the manuscript, writing
them to ``output/``.

PREREQUISITE
------------
``integral_scale_analysis.py`` must be run **before** this script. It produces
the inputs consumed here:

    data/lparams.csv                  Lorentzian parameters per station
    output/lorentzian_std_error.csv   standard errors of those parameters

Running ``main.py`` against a stale or missing ``lparams.csv`` will either fail
on load or silently reproduce figures from outdated fits.

OTHER INPUTS
------------
    data/gt_30y_lt_1perc_missing.csv  station metadata (id, Lat, Lon, Elv)
    PRCP_FOLDER/<station_id>.csv      5-minute precipitation series

The script runs top to bottom as a sequence of ``# %%`` cells. Execution order
is significant: ``lparams``, ``spectra`` and ``stations`` are deliberately
rebound partway through (the seasonal analysis reuses those names for its own
per-station dictionaries), so sections must not be reordered.
"""

import os

import numpy as np
import pandas as pd
import pywt
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
from scipy.integrate import quad
from scipy.stats import linregress

from src.helpers import *
from src.helpers.maps_helper import *
from src.helpers.integral_scale_analysis import compute_spectra
from src.helpers.spectral_model import fit_entropy_hyperplane, normalized_source
from src.helpers.seasonal_analysis import (
    EXAMPLE_STATIONS,
    compute_seasonal_scaling,
)

PRCP_FOLDER = "/home/santa/Shared/data/filled_sampling/"
RECOMPUTE_LPARAMS = False


# %%
# ==========================================================================
# Plot helpers used by the figure sections below
# ==========================================================================
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


# ==========================================================================
# End plot helpers
# ==========================================================================
# %%
# --------------------------------------------------------------------------
# Load data: station metadata, Lorentzian parameters, base maps
# --------------------------------------------------------------------------
stations_df = pd.read_csv("data/gt_30y_lt_1perc_missing.csv")
stations = stations_df["station_id"].values
delta_t = 5 / 60

lparams = pd.read_csv("data/lparams.csv").set_index("station_id")
lparams["int_scale"] = lparams["gamma_spectrum"]

std_err = pd.read_csv("output/lorentzian_std_error.csv", index_col="station_id")
veneto_map_inset = create_veneto_map(cb=1.5, show=False, add_inset=True)
veneto_map = create_veneto_map(cb=1.5, show=False, add_inset=False)
# %%
# --------------------------------------------------------------------------
# Fig 3: Lorentzian parameter maps
# --------------------------------------------------------------------------
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
# --------------------------------------------------------------------------
# Fig 1: Conceptual spectrum
# --------------------------------------------------------------------------
conceptual_plot("output/conceptual_plot.pdf")


# %%
# --------------------------------------------------------------------------
# Fig 2: Energy spectra for three example stations
# --------------------------------------------------------------------------
def add_extra_wT(ax, station):
    df = load_station_series(PRCP_FOLDER, station)
    data = df["PRCP"].values

    # Normalize
    x = standardize(data)

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
spectra, wavelet_coeff = compute_spectra(
    stations, lparams["gamma_acf"].to_dict(), PRCP_FOLDER, delta_t=delta_t, fit_lorentzian_flag=False, return_wT=True
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

    ax.axhline(y=lparams.loc[station, "gamma_acf"]*4, linestyle="--", c="k", linewidth=0.5)
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
# --------------------------------------------------------------------------
# Fig 4: Parameter correlations
# --------------------------------------------------------------------------
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
# --------------------------------------------------------------------------
# Fig 5: Parameters vs. elevation and distance to coast
# --------------------------------------------------------------------------
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
# --------------------------------------------------------------------------
# Fig 7: Source term
# --------------------------------------------------------------------------
f = np.logspace(-6, 1, 100)
u0 = 1
m_values = [0, 1, 10]

params_df = lparams.copy()
for m in m_values:
    # params_df[f"source_m{m}"] = params_df.apply(lambda row: source_term(f, row["A"], row["B"], row["c"], m), axis=1)
    params_df[f"source_m{m}"] = params_df.apply(
        lambda row: normalized_source(
            f, row["A"], row["B"], row["c"], m, row["int_scale"], u0
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
# --------------------------------------------------------------------------
# Fig 8: Shannon entropy - compute and regress
# --------------------------------------------------------------------------
(
    model,
    intercept,
    slope_B,
    slope_c,
    B_min,
    B_max,
    c_min,
    c_max,
) = fit_entropy_hyperplane(lparams)

B_range = np.linspace(B_min, B_max, 1000)
c_range = np.linspace(c_min, c_max, 1000)
B_grid, c_grid = np.meshgrid(B_range, c_range)
z_grid = intercept + slope_B * B_grid + slope_c * c_grid

# %%
# --------------------------------------------------------------------------
# Fig 8a: Shannon entropy regression surface
# --------------------------------------------------------------------------
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection="3d")
 
B_range = np.linspace(B_min, B_max, 200)
c_range = np.linspace(c_min, c_max, 200)
B_grid, c_grid = np.meshgrid(B_range, c_range)
z_grid = intercept + slope_B * B_grid + slope_c * c_grid

n_lines = 5
rstride = max(1, z_grid.shape[0] // n_lines)
cstride = max(1, z_grid.shape[1] // n_lines)

ax.computed_zorder = False        

ax.plot_surface(
    B_grid, c_grid, z_grid,
    cmap="Reds",
    alpha=0.6,
    linewidth=0,
    antialiased=True,
    shade=False,
    rcount=200,
    ccount=200,
    zorder=1,              
)

ax.plot_wireframe(
    B_grid, c_grid, z_grid,
    rstride=rstride,
    cstride=cstride,
    color=mcolors.to_rgba("darkred", 0.1),   
    linewidth=0.5,
    zorder=2,
)

ax.scatter(
    lparams["B"], lparams["c"], lparams["Shannon_Entropy"],
    c=lparams["Shannon_Entropy"],
    cmap="plasma",
    s=120,
    edgecolors="black",
    linewidths=0.5,
    alpha=0.9,
    depthshade=False,
    zorder=3,                       
)

ax.set_xlim(0, 0.08)
ax.set_xticks([0.02, 0.04, 0.06])
ax.set_yticks([0.7, 0.8, 0.9])
ax.tick_params(axis="x", labelsize=12)
ax.tick_params(axis="y", labelsize=12)
ax.tick_params(axis="z", labelsize=12)  
ax.xaxis.minorticks_off()
ax.yaxis.minorticks_off()

ax.set_xlabel(r"$f_0$", fontsize=18, fontweight="medium", labelpad=18)
ax.set_ylabel(r"$c$", fontsize=18, fontweight="medium", labelpad=18)
ax.set_zlabel(r"$H(f)$", fontsize=18, fontweight="medium", labelpad=18)
 
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.set_pane_color((1.0, 1.0, 1.0, 1.0))     
    axis.pane.set_edgecolor("white")
    axis._axinfo["grid"].update(color="lightgrey", linewidth=0.3)
    axis.line.set_color("white")
 
ax.grid(True)
ax.set_box_aspect((1, 1, 1))


eye = np.array([-2.0, 1.5, 0.5])
r = np.linalg.norm(eye)
elev = np.degrees(np.arcsin(eye[2] / r))          
azim = np.degrees(np.arctan2(eye[1], eye[0]))     
ax.view_init(elev=elev, azim=azim)

ax.text2D(
    0.5,
    0.90,
    f"$H(f) = {slope_B:.3f} \\times f_0 - {np.abs(slope_c):.3f} \\times c "
    f"+ {intercept:.3f}$\n$R^2 = {model.rsquared:.2f}$",
    transform=ax.transAxes,
    fontsize=14,
    ha="center",
    va="center",
    bbox=dict(facecolor="white", edgecolor="black", linewidth=1, boxstyle="square,pad=0.5"),
)
 
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
fig.savefig(
    "output/shannon_entropy_regression.png",
    dpi=300,         
    bbox_inches="tight",
    facecolor="white",
)


# %%
veneto_map_inset = create_veneto_map(cb=1.5, add_inset=True)
# --------------------------------------------------------------------------
# Fig 8b: Shannon entropy map
# --------------------------------------------------------------------------
def plot_shannon_entropy_map(veneto_map_inset, lparams):
    print("Fig 8b")
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


plot_shannon_entropy_map(veneto_map_inset, lparams)

# %%
# --------------------------------------------------------------------------
# Seasonal analysis: per-station summer/winter scaling slopes
# --------------------------------------------------------------------------
print("Seasonal Analysis")

spectra, int_scale, scaling_df, tau = compute_seasonal_scaling(
    stations_df, PRCP_FOLDER, delta_t
)
example_stations = EXAMPLE_STATIONS

# %%
# --------------------------------------------------------------------------
# Fig 6: Seasonal spectra
# --------------------------------------------------------------------------
print("Fig 6")
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
# --------------------------------------------------------------------------
# Fig S1: Seasonal slope comparison
# --------------------------------------------------------------------------
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
# --------------------------------------------------------------------------
# Fig S2: Seasonal slope maps
# --------------------------------------------------------------------------

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

veneto_map_inset = create_veneto_map(cb=1.5, add_inset=True)
veneto_map = create_veneto_map(cb=1.5, add_inset=False)
plot_slope_map(
    scaling_df, veneto_map, veneto_map_inset, save_path="output/c_map_seasonal.pdf"
)
# %%
