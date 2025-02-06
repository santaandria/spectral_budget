# __init__.py
import matplotlib.pyplot as plt
from cycler import cycler


def set(family=None):
    # Figure size
    plt.rcParams["figure.figsize"] = (4.25279, 2.628)
    plt.rcParams["figure.dpi"] = 200

    # Color and line style cycle
    plt.rcParams["axes.prop_cycle"] = cycler(color=["k", "r", "b", "g"]) + cycler(
        ls=["-", "--", ":", "-."]
    )

    # Font sizes
    plt.rcParams["font.size"] = 8

    # Font family
    if family == "Fira Sans":
        plt.rcParams["font.family"] = "Fira Sans"
    else:
        plt.rcParams["font.family"] = family
        # plt.rcParams["font.sans-serif"] = font
    plt.rcParams["font.weight"] = "300"
    plt.rcParams["axes.labelweight"] = "300"
    plt.rcParams["axes.titleweight"] = "300"

    # X-axis settings
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["xtick.major.size"] = 3
    plt.rcParams["xtick.major.width"] = 0.5
    plt.rcParams["xtick.minor.size"] = 1.5
    plt.rcParams["xtick.minor.width"] = 0.5
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["xtick.top"] = True

    # Y-axis settings
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["ytick.major.size"] = 3
    plt.rcParams["ytick.major.width"] = 0.5
    plt.rcParams["ytick.minor.size"] = 1.5
    plt.rcParams["ytick.minor.width"] = 0.5
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["ytick.right"] = True

    # Line widths
    plt.rcParams["axes.linewidth"] = 0.5
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["lines.linewidth"] = 1.0

    # Legend settings
    plt.rcParams["legend.frameon"] = False

    # Savefig settings
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.05

    # LaTeX settings
    # plt.rcParams['text.usetex'] = True
    plt.rcParams["text.latex.preamble"] = (
        r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{sfmath} \usepackage{helvet} \usepackage{sansmath} \sansmath"
    )