"""
Spectrum figures for the manuscript.

The conceptual spectrum sketch (Fig 1), the measured energy spectra of the
example stations (Fig 2), the seasonal spectra (Fig 6), and the spectral
source term (Fig 7).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pywt

from .analysis import (
    add_time_axis,
    load_station_series,
    lorentzian,
    standardize,
    wavelet_psd,
)
from .integral_scale_analysis import compute_spectra
from .plotting_helper import configure_axis
from .spectral_model import normalized_source


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


def add_extra_wT(ax, station, delta_t, folder):
    df = load_station_series(folder, station)
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


def plot_energy_spectra(
    lparams, delta_t, folder, save_path="output/energy_spectra.pdf"
):
    """Fig 2: Fourier and wavelet spectra of the three example stations."""
    stations = ["003_BL_Ar", "127_VR_Bu", "168_VE_Ch"]
    spectra, wavelet_coeff = compute_spectra(
        stations, lparams["gamma_acf"].to_dict(), folder, delta_t=delta_t, fit_lorentzian_flag=False, return_wT=True
    )

    labels = ["(a)", "(b)", "(c)"]
    fig, axs = plt.subplots(1, 3, figsize=(9, 3), tight_layout=True)

    for col, ax in enumerate(axs):
        station = stations[col]
        ax.plot(*spectra[station]["fourier"][::-1], alpha=0.8, linewidth=1, c="k", zorder=2)
        ax.scatter(*spectra[station]["wavelet"][::-1], s=30, fc="None", ec="r", zorder=4)
        add_extra_wT(ax, station, delta_t, folder)
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

    plt.savefig(save_path, dpi=600, format="pdf")
    plt.show()
    # plt.close()


def plot_seasonal_spectra(
    spectra, example_stations, tau, save_path="output/seasonal_analysis.pdf"
):
    """
    Fig 6: summer/winter spectra of the example stations.

    ``tau`` is the wavelet scale grid used for the secondary time axis.
    """
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
    plt.savefig(save_path, dpi=600, format="pdf")
    plt.show()


def plot_source_term(lparams, save_path="output/source_term.pdf"):
    """Fig 7: normalized spectral source term by terrain class."""
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

    plt.savefig(save_path, dpi=96, format="pdf")
    plt.show()
