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

The script runs top to bottom as a sequence of ``# %%`` cells: load the data,
then call one function per figure, in order. The figures themselves live in
``src/helpers/figures_spectra``, ``figures_parameters`` and ``figures_maps``;
the computations behind them in ``spectral_model`` and ``seasonal_analysis``.

Execution order still matters -- fit_entropy_hyperplane() adds the
``Shannon_Entropy`` column that Figs 8a and 8b read, and the seasonal analysis
supplies the inputs for Figs 6, S1 and S2 -- so sections should not be
reordered.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.helpers.maps_helper import create_veneto_map
from src.helpers.spectral_model import fit_entropy_hyperplane
from src.helpers.seasonal_analysis import EXAMPLE_STATIONS, compute_seasonal_scaling
from src.helpers.figures_spectra import (
    conceptual_plot,
    plot_energy_spectra,
    plot_seasonal_spectra,
    plot_source_term,
)
from src.helpers.figures_parameters import (
    plot_parameter_correlations,
    plot_parameters_vs_features,
    plot_seasonal_slope_comparison,
    plot_shannon_entropy_surface,
)
from src.helpers.figures_maps import (
    plot_lorentzian_map,
    plot_shannon_entropy_map,
    plot_slope_map,
)

PRCP_FOLDER = "/home/santa/Shared/data/filled_sampling/"
RECOMPUTE_LPARAMS = False

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
plot_energy_spectra(lparams, delta_t, PRCP_FOLDER)

# %%
# --------------------------------------------------------------------------
# Fig 4: Parameter correlations
# --------------------------------------------------------------------------
plot_parameter_correlations(lparams)

# %%
# --------------------------------------------------------------------------
# Fig 5: Parameters vs. elevation and distance to coast
# --------------------------------------------------------------------------
plot_parameters_vs_features(lparams)

# %%
# --------------------------------------------------------------------------
# Fig 7: Source term
# --------------------------------------------------------------------------
plot_source_term(lparams)

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
plot_shannon_entropy_surface(
    lparams,
    intercept,
    slope_B,
    slope_c,
    B_min,
    B_max,
    c_min,
    c_max,
    model,
)

# %%
# --------------------------------------------------------------------------
# Fig 8b: Shannon entropy map
# --------------------------------------------------------------------------
veneto_map_inset = create_veneto_map(cb=1.5, add_inset=True)
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
plot_seasonal_spectra(spectra, example_stations, tau)

# %%
# --------------------------------------------------------------------------
# Fig S1: Seasonal slope comparison
# --------------------------------------------------------------------------
plot_seasonal_slope_comparison(scaling_df)

# %%
# --------------------------------------------------------------------------
# Fig S2: Seasonal slope maps
# --------------------------------------------------------------------------
veneto_map_inset = create_veneto_map(cb=1.5, add_inset=True)
veneto_map = create_veneto_map(cb=1.5, add_inset=False)
plot_slope_map(
    scaling_df, veneto_map, veneto_map_inset, save_path="output/c_map_seasonal.pdf"
)
