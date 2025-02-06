import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def configure_axis(ax, axis_settings):
    """
    Configure a Matplotlib axis with provided settings.

    Parameters:
    - ax: Matplotlib axis object.
    - axis_settings: Dictionary containing axis configuration settings.
                     Example:
                     {
                         "title": "Main Title",
                         "xlabel": "X-Axis Label",
                         "ylabel": "Y-Axis Label",
                         "xscale": "log",
                         "yscale": "linear",
                         "xlim": (0, 10),
                         "ylim": (0, 100),
                         "xticks_rotation": 45,
                         "grid": True,
                         "xformatter": lambda x, _: f"{x:.1f} units",
                     }
    """
    # Set title and labels
    if "title" in axis_settings:
        ax.set_title(
            axis_settings["title"], fontsize=axis_settings.get("title_fontsize", 12)
        )
    if "xlabel" in axis_settings:
        ax.set_xlabel(
            axis_settings["xlabel"], fontsize=axis_settings.get("label_fontsize", 10)
        )
    if "ylabel" in axis_settings:
        ax.set_ylabel(
            axis_settings["ylabel"], fontsize=axis_settings.get("label_fontsize", 10)
        )

    # Set scales
    if "xscale" in axis_settings:
        ax.set_xscale(axis_settings["xscale"])
    if "yscale" in axis_settings:
        ax.set_yscale(axis_settings["yscale"])

    # Set limits
    if "xlim" in axis_settings:
        ax.set_xlim(axis_settings["xlim"])
    if "ylim" in axis_settings:
        ax.set_ylim(axis_settings["ylim"])

    # Rotate x-ticks
    if "xticks_rotation" in axis_settings:
        plt.setp(ax.get_xticklabels(), rotation=axis_settings["xticks_rotation"])

    # Add grid
    if axis_settings.get("grid", False):
        ax.grid(
            True,
            which=axis_settings.get("grid_which", "major"),
            linestyle="--",
            alpha=0.7,
        )

    # Add custom formatters
    if "xformatter" in axis_settings:
        ax.xaxis.set_major_formatter(FuncFormatter(axis_settings["xformatter"]))
    if "yformatter" in axis_settings:
        ax.yaxis.set_major_formatter(FuncFormatter(axis_settings["yformatter"]))

    # Return the configured axis for further chaining if needed
    return ax


def plot_timeseries(ax, t, x, axis_settings):
    markerline, stemlines, baseline = ax.stem(t, x, markerfmt='none')
    stemlines.set_linewidth(0.5)
    configure_axis(ax, axis_settings)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_tick_params(which="minor", bottom=False)
    ax.xaxis.set_tick_params(bottom=False)
    plt.show()
