import warnings
import numpy as np
import pandas as pd
import pywt
from src.helpers.maps_helper import create_veneto_map
from src.helpers.analysis import create_combined_plot, load_analysis_results
import os
import logging
from datetime import datetime
from tqdm import tqdm

DATA_FOLDER = "/mnt/d/climate_data/ARPAV_5min/filled_sampling/"
DELTA_T = 5 / 60


def setup_logging(output_dir="output/logs"):
    """Set up logging configuration"""
    os.makedirs(output_dir, exist_ok=True)

    # Create a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"processing_{timestamp}.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def initialize_environment():
    """Initialize environment settings and create necessary directories"""
    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Set random seed for reproducibility
    np.random.seed(0)

    # Create output directories
    directories = ["output/maps", "output/fig", "output/logs"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def process_station(row, index, stations, veneto_map, psi, logger):
    """Process a single station's data"""
    try:
        station_id = row["station_id"]
        logger.info(f"Processing station {station_id}")

        # Create map
        map_fpath = f"./output/maps/{station_id}_map.png"
        veneto_map.add_scatter(
            data_df=stations.loc[[index]],
            data_col="Elv",
            crs=4326,
            temporary=True,
            to_show=["raster", "base"],
            x_col="Lon",
            y_col="Lat",
            s=100,
            cmap="r",
            cb_label="Elevation [m]",
            save_filepath=map_fpath,
        )

        # Create and save plots
        logger.info(f"Creating plots for station {station_id}")
        fig = create_combined_plot(row, map_fpath, DELTA_T, psi, DATA_FOLDER)
        fig_path = f"./output/fig/{station_id}_analysis.pdf"
        fig.savefig(fig_path, dpi=300, format="pdf")

        logger.info(f"Successfully completed analysis for station {station_id}")
        return True

    except Exception as e:
        logger.error(f"Error processing station {row['station_id']}: {str(e)}")
        return False


def results_to_csv(stations_df):
    station_data = {"q": {}, "beta": {}, "a": {}, "c": {}}

    lorentzian_param_row = []

    # First pass: collect all τ values to determine the complete set of columns
    all_tau_values = set()
    for index, row in stations_df.iterrows():
        try:
            lorentzian_param, params, *_ = load_analysis_results(
                station_id=row["station_id"]
            )
            all_tau_values.update(range(1, len(params) + 1))
        except:
            continue

    # Create column names for each τ value
    tau_columns = [f"tau_{i}" for i in sorted(all_tau_values)]

    # Second pass: collect data for each station
    for index, row in stations_df.iterrows():

        try:
            lorentzian_params, params, *_ = load_analysis_results(
                station_id=row["station_id"]
            )
            # Process Lorentzian Fit results
            lorentzian_params, params, _, _ = load_analysis_results(
                station_id=row["station_id"]
            )
            print(f'----- Processing {row["station_id"]} -----')
            lorentzian_param = lorentzian_params['wavelet'][0] ## Change here
            lorentzian_param["station_id"] = row["station_id"]
            del lorentzian_param["breakpoint"]
            lorentzian_param_row.append(lorentzian_param)

            # Create dictionaries for each parameter with NaN for missing values
            station_id = row["station_id"]
            # Initialize all columns with NaN for this station
            if station_id not in station_data["q"]:
                station_data["q"][station_id] = {col: np.nan for col in tau_columns}
                station_data["beta"][station_id] = {col: np.nan for col in tau_columns}
                station_data["a"][station_id] = {col: np.nan for col in tau_columns}
                station_data["c"][station_id] = {col: np.nan for col in tau_columns}

            # Fill in the available values
            for tau_idx, param in enumerate(params, start=1):
                col_name = f"tau_{tau_idx}"
                station_data["q"][station_id][col_name] = param[0]
                station_data["beta"][station_id][col_name] = param[1]
                station_data["a"][station_id][col_name] = param[2]
                station_data["c"][station_id][col_name] = param[3]
        except:
            continue
    if len(lorentzian_param_row) > 0:
        # Create DataFrames from collected data
        lorentzian_df = pd.DataFrame.from_dict(lorentzian_param_row, orient="columns")
        q_df = pd.DataFrame.from_dict(
            station_data["q"], orient="index", columns=tau_columns
        )
        beta_df = pd.DataFrame.from_dict(
            station_data["beta"], orient="index", columns=tau_columns
        )
        a_df = pd.DataFrame.from_dict(
            station_data["a"], orient="index", columns=tau_columns
        )
        c_df = pd.DataFrame.from_dict(
            station_data["c"], orient="index", columns=tau_columns
        )

        q_df.index.name = "station_id"
        beta_df.index.name = "station_id"
        a_df.index.name = "station_id"
        c_df.index.name = "station_id"

        # Create output directory if it doesn't exist
        os.makedirs("output/analysis_results_csv/", exist_ok=True)

        # Save each DataFrame to CSV
        lorentzian_df.to_csv("output/analysis_results_csv/lorentzian_parameters.csv")
        q_df.to_csv("output/analysis_results_csv/q_parameters.csv")
        beta_df.to_csv("output/analysis_results_csv/beta_parameters.csv")
        a_df.to_csv("output/analysis_results_csv/a_parameters.csv")
        c_df.to_csv("output/analysis_results_csv/c_parameters.csv")


def main():
    # Set up logging
    logger = setup_logging()
    logger.info("Starting analysis pipeline")

    try:
        # Initialize environment and parameters
        initialize_environment()

        # Load station data
        # Global
        psi = pywt.Wavelet("coif3")
        stations = pd.read_csv("data/gt_30y_lt_1perc_missing.csv")
        logger.info(f"Loaded {len(stations)} stations for processing")

        # Create base map
        veneto_map = create_veneto_map(cb=0, show=False)

        # Process each station with progress bar
        successful = 0
        failed = 0

        for index, row in tqdm(
            stations.iterrows(), total=len(stations), desc="Processing stations"
        ):
            if process_station(row, index, stations, veneto_map, psi, logger):
                successful += 1
            else:
                failed += 1

        # Log summary
        logger.info(
            f"Processing completed. "
            f"Successful: {successful}, Failed: {failed}, "
            f"Total: {len(stations)}"
        )
        results_to_csv(stations)
        logger.info("Results saved!")

    except Exception as e:
        logger.error(f"Critical error in main process: {str(e)}")
        raise
    # Log summary
    if successful > 0:
        logger.info(f"Saving the results into .csv files ")
        results_to_csv(stations)
        logger.info("Results saved!")


if __name__ == "__main__":
    main()
