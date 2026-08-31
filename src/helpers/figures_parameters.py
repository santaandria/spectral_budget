"""
Parameter-space figures for the manuscript.

Correlations between the fitted Lorentzian parameters (Fig 4), their
dependence on elevation and distance to the coast (Fig 5), the Shannon
entropy regression surface (Fig 8a), and the seasonal slope comparison
(Fig S1).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_parameter_correlations(
    lparams, save_path="output/parameter_correlations.pdf"
):
    """Fig 4: pairwise scatter of A, f0 and c, coloured by elevation."""
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

    plt.savefig(save_path, dpi=96, format="pdf")
    plt.show()
    # plt.close()


def plot_parameters_vs_features(
    lparams, save_path="output/parameters_features.pdf"
):
    """Fig 5: parameters against elevation and distance to coast."""
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

    plt.savefig(save_path, dpi=96, format="pdf")
    plt.show()
    # plt.close()


def plot_shannon_entropy_surface(
    lparams,
    intercept,
    slope_B,
    slope_c,
    B_min,
    B_max,
    c_min,
    c_max,
    model,
    save_path="output/shannon_entropy_regression.png",
):
    """Fig 8a: entropy scatter with the fitted OLS regression plane."""
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
        save_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )


def plot_seasonal_slope_comparison(
    scaling_df, save_path="output/seasonal_slope_comparison.pdf"
):
    """Fig S1: summer vs winter scaling slopes across all stations."""
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

    plt.savefig(save_path, dpi=600, format="pdf")
    plt.show()
