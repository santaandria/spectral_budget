import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from src.helpers.Time_Unit_Converter import TimeUnitConverter
from src.helpers.maps_helper import create_veneto_map

# Create output directory for parameter maps
os.makedirs("output/param_maps", exist_ok=True)
stations = pd.read_csv("data/gt_30y_lt_1perc_missing.csv")

# Time scales setup
delta_t = 5 / 60
scales = 2 ** (np.arange(2, 11)) * delta_t

# Create base maps
veneto_map_inset = create_veneto_map(cb=1, show=False, add_inset=True)
veneto_map = create_veneto_map(cb=1, show=False, add_inset=False)

############ Lorentzian Parameters ##########################
# Read Lorentzian parameters
print("---------- Processing Lorenzian Parameters -------------")
lorentzian_df = pd.read_csv("output/analysis_results_csv/lorentzian_parameters.csv")
lorentzian_df = lorentzian_df.merge(
    stations[["station_id", "Lat", "Lon"]], on="station_id", how="left"
)

# Parameter settings for Lorentzian
lorentz_params_settings = {
    "A": {"cmap": "plasma", "title": "A"},
    "B": {"cmap": "viridis", "title": "B"},
    "c": {"cmap": "magma", "title": "c"},
}

# Create figure
fig = plt.figure(figsize=(12, 5))
gs = fig.add_gridspec(1, 3, wspace=0)

# Create maps for each parameter
for i, (param, settings) in enumerate(lorentz_params_settings.items()):
    print(f"Generating plots for {param} ...")
    map_fpath = f"output/param_maps/lorentzian_{param}.png"

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
    ax.set_title(f'Parameter {settings["title"]}', fontsize=12)

# Add overall title
fig.suptitle(
    "Lorentzian Parameters $E(K) = \\frac{A}{1+\left(\\frac{K}{B}\\right)^{c}}$",
    fontsize=14,
    y=1.05,
)

# Save the complete figure
plt.savefig("output/param_maps/lorentzian_parameters_maps.png", dpi=300)
plt.close()


############ Tsallis Parameters ##########################
print("---------- Processing Distribution Parameters -------------")
# Parameter settings
params_settings = {
    "q": {"cmap": "plasma", "title": "q"},
    "beta": {"cmap": "plasma", "title": r"$\beta$"},
    "a": {"cmap": "plasma", "title": "a"},
    "c": {"cmap": "plasma", "title": "c"},
}

# Process each parameter
for param in params_settings.keys():
    print(f"Generating plots for {param} ...")
    # Read data
    param_df = pd.read_csv(f"output/analysis_results_csv/{param}_parameters.csv")
    param_df = param_df.merge(
        stations[["station_id", "Lat", "Lon"]], on="station_id", how="left"
    )

    # Create figure
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 3, wspace=0, hspace=0.1)

    # Create maps for each tau value
    for i in range(9):  # 9 tau values
        map_fpath = f"output/param_maps/{param}_tau_{i+1}.png"

        # Choose map type based on first or subsequent plots
        current_map = veneto_map_inset if i == 0 else veneto_map

        # Add scatter plot to map
        current_map.add_scatter(
            data_df=param_df,
            data_col=f"tau_{i+1}",
            crs=4326,
            temporary=True,
            to_show=["raster", "base"],
            x_col="Lon",
            y_col="Lat",
            s=100,
            cmap=params_settings[param]["cmap"],
            cb_label=params_settings[param]["title"],
            save_filepath=map_fpath,
        )

        # Add subplot
        ax = fig.add_subplot(gs[i // 3, i % 3])
        img = mpimg.imread(map_fpath)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f'$\\tau = ${TimeUnitConverter.convert_time_units(scales[i], from_unit="h")}',
            fontsize=10,
        )

    # Add overall title
    fig.suptitle(
        f'Modified Tsallis Distribution Parameter {params_settings[param]["title"]} for different τ values',
        fontsize=14,
        y=0.95,
    )

    # Save the complete figure
    plt.savefig(f"output/param_maps/{param}_all_tau_maps.png", dpi=300)
    plt.close()
