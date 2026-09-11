"""Second-order stability diagnostic for all precipitation stations.

Produces:
  - per-station diagnostic figures  -> OUTPUT_DIR/
  - station-level summary CSV       -> OUTPUT_DIR/stationarity_summary.csv
"""
# %%
import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from src.helpers.integral_scale_analysis import (
    analyze_all_stations,
    build_summary_dataframe,
    compare_integral_scale_models,
    compute_spectra,
    plot_integral_scale_comparison,
    plot_station_diagnostics,
)

# ---- Configuration ---------------------------------------------------------
PRCP_FOLDER = "/home/santa/Shared/data/filled_sampling/"
STATIONS_CSV = "data/gt_30y_lt_1perc_missing.csv"
OUTPUT_DIR = "output/stationarity"
DELTA_T = 5 / 60  # hours

MAX_RELATIVE_DRIFT = 0.2
MIN_YEAR_COVERAGE = 0.80
MIN_VALID_YEARS = 10
MAX_ACF_LAG = None  
RECOMPUTE_LPARAMS = False
# ----------------------------------------------------------------------------

# %%
def main() -> None:
    # %%
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stations_df = pd.read_csv(STATIONS_CSV)
    stations = stations_df["station_id"].tolist()

    results = analyze_all_stations(
        stations,
        PRCP_FOLDER,
        DELTA_T,
        max_relative_drift=MAX_RELATIVE_DRIFT,
        min_year_coverage=MIN_YEAR_COVERAGE,
        min_valid_years=MIN_VALID_YEARS,
        max_acf_lag=MAX_ACF_LAG,
    )

    summary = build_summary_dataframe(results)
    csv_path = os.path.join(OUTPUT_DIR, "stationarity_summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\nSummary CSV written to {csv_path}")

    n_stable = (summary["status"] == "stable").sum()
    n_unstable = (summary["status"] == "unstable").sum()
    n_undet = summary["status"].str.startswith("undetermined").sum()
    print(f"Stations: {len(summary)}  |  stable: {n_stable}  |  "
          f"unstable: {n_unstable}  |  undetermined: {n_undet}")

    print("\nGenerating diagnostic plots ...")
    for station, res in results.items():
        plot_station_diagnostics(station, res, output_dir=OUTPUT_DIR)

    

    # %%
    analysis_df = pd.read_csv("output/stationarity/stationarity_summary.csv").set_index("station")
    stations_df = pd.read_csv("data/gt_30y_lt_1perc_missing.csv")
    stations = stations_df["station_id"].values
    delta_t = 5 / 60
    int_scale = analysis_df["gamma_acf"].to_dict()

    print("Getting Lorentzian parameters")

    if RECOMPUTE_LPARAMS:
        spectra = compute_spectra(
            stations, int_scale, PRCP_FOLDER, delta_t=delta_t,
            fit_lorentzian_flag=True, return_wT=False,
        )
        lparams = pd.read_csv("data/lparams.csv")
        lparams = lparams.merge(
            stations_df[["station_id", "Elv", "Lat", "Lon", "distance_to_coast"]],
            on="station_id",
        )
        min_K = {stn: min(v["wavelet"][1]) for stn, v in spectra.items()}
        max_K = {stn: max(v["wavelet"][1]) for stn, v in spectra.items()}

        lparams = lparams.assign(
            min_K=lparams["station_id"].map(min_K),
            max_K=lparams["station_id"].map(max_K),
        )

        lparams = lparams.set_index("station_id")
        lparams.to_csv("data/lparams.csv")
    else:
        lparams = pd.read_csv("data/lparams.csv").set_index("station_id")

    # %%
    lparams = lparams.drop(
        columns=[c for c in ("gamma_acf", "gamma_stable") if c in lparams.columns]
    )
    lparams = lparams.join(analysis_df[["gamma_acf", "gamma_stable"]])
    lparams["gamma_spectrum"] = lparams["A"] / 4

    # %%
    mask = (lparams["gamma_stable"] == True)
    gamma_spectrum = np.asarray(lparams.loc[mask, "gamma_spectrum"], dtype=float)
    gamma_acf = np.asarray(lparams.loc[mask, "gamma_acf"], dtype=float)

    comparison = compare_integral_scale_models(gamma_spectrum, gamma_acf)
    comparison.to_csv("output/integral_scale_model_comparison.csv", index=False)
    print(comparison)

    # %%
    plot_integral_scale_comparison(
        gamma_spectrum, gamma_acf, comparison,
        output_path="output/int_scale_models.pdf",
    )

    # %%
    lparams.to_csv("data/lparams.csv")
    print("Done.")
    
    # %%


if __name__ == "__main__":
    main()

