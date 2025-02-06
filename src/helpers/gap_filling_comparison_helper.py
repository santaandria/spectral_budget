# Missing value helper/scheme comparison
import numpy as np
from scipy.stats import kurtosis, skew
from .analysis import acf_analysis


def compare_gfs_moments(gap_filling_schemes):
    def calculate_moments(array):
        mean = np.nanmean(array)
        variance = np.nanvar(array)
        cv = np.nanstd(array) / mean
        skewness = skew(array, nan_policy="omit")
        kurt = kurtosis(array, nan_policy="omit")
        return mean, variance, cv, skewness, kurt

    moments = ["Mean", "Variance", "CV", "Skewness", "Kurtosis"]
    results = {}
    for scheme, data in gap_filling_schemes.items():
        results[scheme] = calculate_moments(data)

    # Print results in a structured format
    print("Comparison of Moments:")
    for i, moment in enumerate(moments):
        comparisons = " | ".join(
            f"{scheme}: {values[i]:.4f}" for scheme, values in results.items()
        )
        print(f"{moment}: {comparisons}")


def analyze_gfs_autocorrelation(schemes, delta_t):
    autocorr_results = {}
    for scheme, data in schemes.items():
        autocorr, lags, integral_scale, first_zero_crossing = acf_analysis(
            data, delta_t
        )
        autocorr_results[scheme] = {
            "autocorr": autocorr,
            "lags": lags,
            "integral_scale": integral_scale,
            "first_zero_crossing": first_zero_crossing,
        }
    return autocorr_results


def plot_gfs_autocorrelations(autocorr_results, ax):
    colors = {"Original": "k-", "Interpolated": "b:", "Sampling": "r--"}

    for scheme, results in autocorr_results.items():
        lags = results["lags"]
        autocorr = results["autocorr"]
        integral_scale = results["integral_scale"]
        first_zero_crossing = results["first_zero_crossing"]

        ax.plot(
            lags[:first_zero_crossing] / integral_scale,
            autocorr[:first_zero_crossing],
            colors[scheme],
            label=f"{scheme} $\\tau_0$ = {integral_scale:.2f} h",
        )
    ax.set_xlabel("$\\tau$/$\\tau_0$")
    ax.set_ylabel("$\\rho(\\tau)$")
    ax.legend()
    ax.set_title("ACF Comparison")
    return ax


def plot_gfs_moments(gap_filling_schemes, ax):
    def calculate_moments(array):
        mean = np.nanmean(array)
        variance = np.nanvar(array)
        skewness = skew(array, nan_policy="omit")
        kurt = kurtosis(array, nan_policy="omit")
        return mean, variance, skewness, kurt

    # Define moments and calculate results
    moments = ["Mean", "Variance", "Skewness", "Kurtosis"]
    results = {
        scheme: calculate_moments(data) for scheme, data in gap_filling_schemes.items()
    }

    # Separate moments into axes
    moments_left = ["Mean", "Variance"]
    moments_right_1 = ["Skewness"]
    moments_right_2 = ["Kurtosis"]

    x_left = [moments.index(m) for m in moments_left]
    x_right_1 = [moments.index(m) for m in moments_right_1]
    x_right_2 = [moments.index(m) for m in moments_right_2]

    # Bar width and color settings
    width = 0.25
    colors = ["black", "grey", "white"]
    edge_color = "black"

    # Set up figure and axes
    ax2 = ax.twinx()  # Second y-axis for skewness
    ax3 = ax.twinx()  # Third y-axis for kurtosis
    ax3.spines["right"].set_position(
        ("outward", 60)
    )  # Offset third axis to avoid overlap

    # Plot "Mean" and "Variance" on the left y-axis
    for i, (scheme, values) in enumerate(results.items()):
        ax.bar(
            [x + (i - len(results) // 2) * width for x in x_left],
            [values[moments.index(m)] for m in moments_left],
            width,
            color=colors[i % len(colors)],
            edgecolor=edge_color,
            label=scheme,
        )

    # Plot "Skewness" on the second y-axis
    for i, (scheme, values) in enumerate(results.items()):
        ax2.bar(
            [x + (i - len(results) // 2) * width for x in x_right_1],
            [values[moments.index(m)] for m in moments_right_1],
            width,
            color=colors[i % len(colors)],
            edgecolor=edge_color,
        )

    # Plot "Kurtosis" on the third y-axis
    for i, (scheme, values) in enumerate(results.items()):
        ax3.bar(
            [x + (i - len(results) // 2) * width for x in x_right_2],
            [values[moments.index(m)] for m in moments_right_2],
            width,
            color=colors[i % len(colors)],
            edgecolor=edge_color,
        )

    # Configure axes labels and ticks
    ax.set_ylabel("Mean & Variance")
    ax2.set_ylabel("Skewness", color="blue")
    ax2.tick_params(axis="y", colors="blue", which="both")
    ax2.spines["right"].set_color("blue")

    ax3.set_ylabel("Kurtosis", color="red")
    ax3.tick_params(axis="y", colors="red", which="both")
    ax3.spines["right"].set_color("red")
    ax3.spines["right"].set_position(("axes", 1.2))

    ax.set_xticks(range(len(moments)))
    ax.set_xticklabels(moments)
    ax.get_xticklabels()[2].set_color("blue")
    ax.get_xticklabels()[3].set_color("red")

    # Add legends and titles
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3)
    ax.set_title("Gap-Filling Scheme Moment Comparison")
