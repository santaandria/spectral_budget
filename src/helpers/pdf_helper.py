import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.integrate import quad

from scipy.integrate import quad
from .Time_Unit_Converter import TimeUnitConverter
from .plotting_helper import configure_axis
from scipy.special import beta as B


def generate_exp_bins(x, initial_step=0.05, multiplier=2):
    """
    Generate exponentially growing bins for both positive and negative values.

    Parameters:
    x (array-like): Input array to determine the range of bins
    initial_step (float): Initial step size (default: 0.02)

    Returns:
    array: Sorted unique bins including 0 and covering the range of x
    """
    # Initialize bins with 0
    # bins = [0]
    bins = []

    # Generate positive bins
    step = initial_step
    current_bin = 0
    while current_bin < max(x):
        current_bin += step
        bins.append(current_bin)
        step *= multiplier

    # Generate negative bins
    step = initial_step
    current_bin = 0
    while current_bin > min(x):
        current_bin -= step
        bins.append(current_bin)
        step *= multiplier

    # Sort and remove duplicates
    return np.sort(np.unique(bins))


def empirical_pdf(
    data, bins=None, x_as_median=False, separate_pos_neg=False, offset=0.1
):
    """
    Compute the empirical probabilithy density.

    Parameters:
    data (array-like): Data
    bins (int or array-like): If bins is of type array-like, it represents the bin edges. If of type int, it is the number of equally spaced bins
    x_as_median (bool): Use the bin median instead of the midpoint if true
    separate_pos_neg (bool): Only valid if bins is an int. If True, the equally spaced bins are created from min(data):offset, offset:max(data) and [-offset, offset] to estimate the middle value

    Returns:
    numpy.ndarray: The median of each bin
    numpy.ndarray: Symmetric log-spaced array
    """
    if not isinstance(bins, (list, pd.core.series.Series, np.ndarray)):
        if separate_pos_neg:
            bins = np.unique(
                np.concat(
                    [
                        np.linspace(min(data), -offset, bins + 1),
                        np.linspace(offset, max(data), bins + 1),
                    ]
                )
            )
        else:
            bins = np.linspace(min(data), max(data), bins + 1)

    bin_xs = []
    bin_densities = []
    for i in range(len(bins) - 1):
        bin_data = data[(data >= bins[i]) & (data < bins[i + 1])]
        if len(bin_data) > 0:
            bin_x = np.median(bin_data) if x_as_median else (bins[i + 1] + bins[i]) / 2
            bin_density = len(bin_data) / (len(data) * (bins[i + 1] - bins[i]))
            bin_xs.append(bin_x)
            bin_densities.append(bin_density)

    bin_xs = np.array(bin_xs)
    bin_densities = np.array(bin_densities)
    return bin_xs, bin_densities


def calculate_bounds(q, beta, a, c1, x_r=[-1000, 1000], h=1):
    """
    Calculate the range where the unnormalized Beck-Tsallis distribution is defined based on a first order FD approximation

    Parameters:
    h (float): spacing between two consecutive points in the FD
    """

    def find_bounds(x_range, increasing=True):
        x = np.arange(x_range[0], x_range[1], h)
        y = modified_tsallis(x, q, beta, a, c1)

        # Calculate differences between consecutive points
        diff = np.diff(y)
        valid_indices = np.where(diff > 0 if increasing else diff < 0)[0]

        # Handle NaN values
        nan_indices = np.where(np.isnan(y))[0]
        if nan_indices.size > 0:
            if increasing:
                valid_indices = valid_indices[valid_indices > nan_indices[-1]]
            else:
                valid_indices = valid_indices[valid_indices < nan_indices[0]]

        if valid_indices.size > 0:
            return x[valid_indices[0]] if increasing else x[valid_indices[-1]]
        return x[0] if increasing else x[-1]

    C_left = find_bounds((x_r[0], 0), increasing=True)
    C_right = find_bounds((0, x_r[1]), increasing=False)
    return min(abs(C_left), abs(C_right))


def tsallis_normalization_constant(q=None, beta=None, a=None):
    """
    F.M. Ramos et al. (2004)
    """
    alpha = 1 - (q - 1) / a
    # alpha = a
    l = 1 / (q - 1)
    m_0 = (1 - alpha) / alpha
    phi_0 = (1 + m_0) / 2
    chi_0 = l - phi_0
    zeta = np.sqrt(l / beta)
    return (np.power(zeta, m_0 + 1) / alpha) * B(phi_0, chi_0)


def modified_tsallis(x, q=None, beta=None, a=None, c1=None):
    alpha = 1 - (q - 1) / a
    # alpha = a
    energy = np.power(np.abs(x), (2 * alpha)) - c1 * np.sign(x) * (
        np.power(np.abs(x), (alpha)) - (1 / 3) * np.power(np.abs(x), (3 * alpha))
    )
    Z = tsallis_normalization_constant(q, beta, a)
    return np.power(1 + beta * (q - 1) * energy, 1 / (1 - q)) / (Z + 1e-15)


def least_sq_obj(
    params, bin_x, empirical_pdf, free_param_names, fixed_params=None, free_beta=True
):
    """
    Computes the square of the log-residuals. Allow for flexible definition of free parameters.

    Parameters:
    fixed_params (dict): {'parameter name': value}
    free_beta (bool): If False, the Formula for beta in Beck et al. (2001) is used.

    """
    full_params = {
        name: fixed_params.get(name, None) for name in ["q", "a", "c1", "beta"]
    }

    # Assign free parameters dynamically
    for name, value in zip(free_param_names, params):
        full_params[name] = value

    # Compute beta if it's not a free parameter
    q = full_params["q"]
    if not free_beta:
        full_params["beta"] = 2 / (5 - 3 * q)

    q, a, c1, beta = full_params.values()
    # To ignore parameters values that do not provide a well-defined pdf over the range of data, np.nan_to_num(..., nan=9999) is used to return an arbitrary large value of the ojective function
    return np.sum(
        np.nan_to_num(
            (
                np.log10(empirical_pdf)
                - np.log10(modified_tsallis(bin_x, q, beta, a, c1))
            )
            ** 2,
            nan=999999,
        )
    )


def prepare_optimization(params, bounds, fixed_params, free_beta):
    """
    Prepares initial values and bounds for flexible optimization.

    Parameters:
    params (array-like): Initial values of [q, a, c1, beta], irrespective of whether some of them will be fixed or not. E.g. [1.4, 1, 0.124, 1]
    bounds: Plausible bounds for the [q, a, c1, beta]. E.g. [(1.01, 5 / 3), (0.1, 1.3), (-0.1, 0.1), (0.8, 1e3)]
    fixed_params (dict): {'parameter name': value}
    free_beta (bool): If True, beta will be added as a free parameter.

    Returns:
    Parameters needed for least_sq_obj()

    """
    free_params = []
    free_bounds = []
    free_param_names = []
    param_list = ["q", "a", "c1", "beta"] if free_beta else ["q", "a", "c1"]

    for i, name in enumerate(param_list):
        if name not in fixed_params:
            free_params.append(params[i])
            free_bounds.append(bounds[i])
            free_param_names.append(name)

    return free_param_names, free_params, free_bounds


def parse_result(result_x, free_param_names, fixed_params, free_beta):
    """
    Parses optimization results to return a complete parameter set [q, a, c1, beta].
    """

    full_params = {name: None for name in ["q", "a", "c1", "beta"]}
    full_params.update(fixed_params)

    for name, value in zip(free_param_names, result_x):
        full_params[name] = value

    if not free_beta:
        q = full_params["q"]
        full_params["beta"] = 2 / (5 - 3 * q)

    print(
        f'Estimated parameters: q = {full_params["q"]:.2f}, a = {full_params["a"]:.2f}, c1 = {full_params["c1"]:.4f}, beta = {full_params["beta"]:.2f}'
    )
    return (full_params["q"], full_params["beta"], full_params["a"], full_params["c1"])


def plot_tsallis(ax, params, bin_x, empirical_pdf, shift, label=False):
    x_vals = np.linspace(min(bin_x), max(bin_x), 1000)
    pdf_vals = modified_tsallis(x_vals, *params)
    ax.plot(
        bin_x,
        empirical_pdf * shift,
        "-ok",
        label="Empirical" if label else "",
        markersize=3,
        markerfacecolor="None",
    )
    ax.plot(x_vals, pdf_vals * shift, "-r", label="Modified Tsalis" if label else "")
    return np.nanmin(pdf_vals * shift)


def fit_scale_wise_pdf(wT, scales, breakpt):
    params = []
    bin_x_list, emp_pdf_list = [], []

    initial_params = [1.1, 1, 0.124, 1]  # [q, a, c1, beta]
    bounds = [(1.01, 5 / 3), (0.1, 10), (-0.5, 0.5), (0.8, 100)]
    fixed_params = {"a": 1}  # Change here
    free_beta = True  # False == beta = f(q)

    free_param_names, free_params, free_bounds = prepare_optimization(
        initial_params, bounds, fixed_params, free_beta
    )

    for i, w in enumerate(wT):
        if 2 * np.pi / scales[i] < breakpt:
            continue
        w = w[np.abs(w) > 1e-13]
        w = w / np.std(w)

        bin_x, emp_pdf = empirical_pdf(
            w, 30, separate_pos_neg=True, offset=0, x_as_median=True
        )

        bin_x_list.append(bin_x)
        emp_pdf_list.append(emp_pdf)

        result = minimize(
            least_sq_obj,
            free_params,
            args=(bin_x, emp_pdf, free_param_names, fixed_params, free_beta),
            options={"maxiter": 1e5},
            bounds=free_bounds,
            method="trust-constr",  # "L-BFGS-B", "Nelder-Mead"
        )

        if result.success:
            params.append(
                parse_result(result.x, free_param_names, fixed_params, free_beta)
            )
        else:
            print("i = {i} Optimization failed:", result.message)
    return params, bin_x_list, emp_pdf_list


def plot_scale_wise_pdf(params, bin_x_list, emp_pdf_list, scales, ax=None):
    if not ax:
        fig, ax = plt.subplots(figsize=(4.5, 4))
    N = len(params)
    vertical_scaling = 1e5  # Adjust this value to control the spacing between plots
    t_ax = ax.twinx()
    time_strings = [
        TimeUnitConverter.convert_time_units(t, from_unit="h") for t in scales[:N]
    ]

    tick_positions = []

    for i in range(N):
        tick_pos = plot_tsallis(
            ax,
            params[i],
            bin_x_list[i],
            emp_pdf_list[i],
            shift=vertical_scaling**i,
            label=not i,
        )
        tick_positions.append(np.log10(tick_pos))

    xlim = ax.get_xlim()
    xlim = (-max(np.abs(xlim)), max(np.abs(xlim)))
    axis_settings = {
        "ylabel": " $\\text{log-pdf }(W_{\psi})$",
        "xlabel": "$W_{\psi}$",
        "yscale": "log",
        "xlim": xlim,
    }

    configure_axis(ax, axis_settings)
    t_ax.set_ylim(np.log10(ax.get_ylim()))
    t_ax.set_yticks(tick_positions[::2])
    t_ax.minorticks_off()
    t_ax.yaxis.set_tick_params(which="both", labelrotation=90, length=0)
    t_ax.set_yticklabels(time_strings[::2], va="bottom")
    ax.get_yaxis().set_ticks([])
    return ax
