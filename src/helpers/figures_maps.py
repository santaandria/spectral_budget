"""
Map figures for the manuscript.

Each of these renders one or more eomaps layers to a temporary PNG, reads it
back with matplotlib and composes the final panel, then deletes the temporary
file. They operate on the ``veneto_map`` / ``veneto_map_inset`` objects built
by :func:`maps_helper.create_veneto_map`.
"""

import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg



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
