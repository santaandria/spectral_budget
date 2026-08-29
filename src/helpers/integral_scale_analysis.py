"""Second-order stability diagnostics for precipitation time series.

Provides functions to assess whether mean, variance, and ACF integral scale
show meaningful systematic drift over multi-year records, supporting the use
of a single spectrum estimated from the complete record.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from scipy.integrate import quad, trapezoid
from scipy.optimize import curve_fit, least_squares
from scipy.stats import linregress
from scipy.stats import t as t_dist
from statsmodels.tsa.stattools import acf

from .analysis import (
    fit_psd_power_law,
    fourier_psd,
    lorentzian,
    wavelet_psd,
    wavelet_transform,
)


# ---------------------------------------------------------------------------
# ACF utilities (refactored from analysis.acf_analysis)
# ---------------------------------------------------------------------------

def compute_acf(
    x: np.ndarray,
    delta_t: float,
    max_lag: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the autocorrelation function of *x*.

    Parameters
    ----------
    x : array-like
        Time series (may contain NaN; handled via ``missing='conservative'``).
    delta_t : float
        Time step in the same physical unit as the desired lag axis.
    max_lag : int, optional
        Maximum number of lags.  Defaults to the largest power of two
        not exceeding ``len(x)``.

    Returns
    -------
    autocorr : 1-D array
        ACF values from lag 0 to *max_lag*.
    lags : 1-D array
        Corresponding lag times (same unit as *delta_t*).
    """
    if max_lag is None:
        max_lag = 2 ** int(np.log2(len(x)))
    autocorr = acf(x, missing="conservative", nlags=max_lag)
    lags = np.arange(len(autocorr)) * delta_t
    return autocorr, lags


def find_first_zero_crossing(
    autocorr: np.ndarray,
    max_lag_index: Optional[int] = None,
) -> Optional[int]:
    """Return the index of the first negative ACF value.

    Parameters
    ----------
    autocorr : 1-D array
        ACF values starting at lag 0.
    max_lag_index : int, optional
        Search only up to this index.

    Returns
    -------
    int or None
        Index of first negative value, or *None* if no crossing is found
        within the search window.
    """
    search = autocorr[:max_lag_index] if max_lag_index is not None else autocorr
    neg = np.where(search < 0)[0]
    if len(neg) == 0:
        return None
    return int(neg[0])


def compute_truncated_integral_scale(
    autocorr: np.ndarray,
    lags: np.ndarray,
    tau0_index: int,
) -> float:
    """Integrate the ACF from lag 0 to *tau0_index* using the trapezoidal rule.

    Parameters
    ----------
    autocorr : 1-D array
    lags : 1-D array
    tau0_index : int
        Index corresponding to the upper integration limit (exclusive).

    Returns
    -------
    float
        Integral scale (same unit as *lags*).
    """
    return float(trapezoid(autocorr[:tau0_index], lags[:tau0_index]))


# ---------------------------------------------------------------------------
# Trend diagnostics
# ---------------------------------------------------------------------------

_MEAN_ABS_FLOOR = 1e-12


def compute_trend_diagnostics(
    years: np.ndarray,
    values: np.ndarray,
    max_relative_drift: float = 0.10,
    mean_abs_floor: float = _MEAN_ABS_FLOOR,
) -> dict:
    """Fit a linear trend to an annual statistic and quantify drift.

    Parameters
    ----------
    years : 1-D array of int/float
        Year labels (abscissa).
    values : 1-D array of float
        Annual statistic values (ordinate).  NaN entries are dropped.
    max_relative_drift : float
        Tolerance for the normalised drift *D_X*.
    mean_abs_floor : float
        If ``|mean(X)| < mean_abs_floor`` the normalised drift is undefined.

    Returns
    -------
    dict
        Keys: slope, pvalue, r2, mean, std, cv, relative_drift,
        relative_drift_pct, stable, drift_defined.
    """
    mask = np.isfinite(values)
    yy, vv = np.asarray(years[mask], dtype=float), values[mask]
    n = len(vv)

    result: dict = {}

    if n < 3:
        for k in ("slope", "pvalue", "r2", "mean", "std", "cv",
                   "relative_drift", "relative_drift_pct"):
            result[k] = np.nan
        result["stable"] = np.nan
        result["drift_defined"] = False
        return result

    slope, intercept, r, p, se = linregress(yy, vv)
    mean_val = float(np.mean(vv))
    std_val = float(np.std(vv, ddof=1))
    span = float(yy.max() - yy.min())

    abs_mean = abs(mean_val)
    drift_defined = abs_mean >= mean_abs_floor

    if drift_defined:
        relative_drift = abs(slope) * span / abs_mean
    else:
        relative_drift = np.nan

    result["slope"] = slope
    result["pvalue"] = p
    result["r2"] = r ** 2
    result["mean"] = mean_val
    result["std"] = std_val
    result["cv"] = std_val / abs_mean if drift_defined else np.nan
    result["relative_drift"] = relative_drift
    result["relative_drift_pct"] = relative_drift * 100 if drift_defined else np.nan
    result["stable"] = bool(relative_drift <= max_relative_drift) if drift_defined else np.nan
    result["drift_defined"] = drift_defined

    return result


# ---------------------------------------------------------------------------
# Per-station analysis
# ---------------------------------------------------------------------------

def analyze_station_stationarity(
    data: pd.Series,
    delta_t: float,
    *,
    max_relative_drift: float = 0.10,
    min_year_coverage: float = 0.80,
    min_valid_years: int = 10,
    max_acf_lag: Optional[int] = None,
) -> dict:
    """Run the second-order stability diagnostic for a single station.

    Parameters
    ----------
    data : pd.Series
        Precipitation time series with a ``DatetimeIndex``.
    delta_t : float
        Time step in hours.
    max_relative_drift : float
        Tolerance for normalised drift D_X.
    min_year_coverage : float
        Minimum fraction of non-NaN values in a calendar year for that year
        to be included.
    min_valid_years : int
        Minimum number of valid years required for the analysis.
    max_acf_lag : int, optional
        Maximum ACF lag (in number of time steps) to compute.  Defaults to
        the largest power-of-two not exceeding the record length.

    Returns
    -------
    dict
        Complete diagnostics (see module docstring for column catalogue).
    """
    # -- Partition into calendar-year blocks ------------------------------------
    x = data.copy()
    x.index = pd.to_datetime(x.index)
    years_all = x.index.year
    unique_years = np.sort(np.unique(years_all))

    n_steps_per_year = int(round(365.25 * 24 / delta_t))

    annual_means = []
    annual_vars = []
    valid_years = []

    for yr in unique_years:
        block = x[years_all == yr]
        frac_valid = block.notna().sum() / max(n_steps_per_year, len(block))
        if frac_valid < min_year_coverage:
            continue
        vals = block.dropna().values
        if len(vals) < 2:
            continue
        annual_means.append(np.mean(vals))
        pos = vals[vals > 0]
        annual_vars.append(np.var(pos, ddof=1) if len(pos) > 1 else np.nan)
        valid_years.append(yr)

    valid_years = np.array(valid_years)
    annual_means = np.array(annual_means)
    annual_vars = np.array(annual_vars)

    result: dict = {
        "n_valid_years": len(valid_years),
        "valid_years": valid_years,
    }

    if len(valid_years) < min_valid_years:
        result["status"] = "insufficient_years"
        return result

    # -- Full-record ACF and zero crossing -------------------------------------
    x_full = x.dropna().values
    autocorr_full, lags_full = compute_acf(x_full, delta_t, max_lag=max_acf_lag)
    effective_max_idx = len(autocorr_full)
    tau0_index = find_first_zero_crossing(autocorr_full, max_lag_index=effective_max_idx)

    zero_crossing_exists = tau0_index is not None
    if zero_crossing_exists and tau0_index <= 1:
        zero_crossing_exists = False
        tau0_index = None

    result["zero_crossing_exists"] = zero_crossing_exists
    result["tau0_index"] = tau0_index
    result["tau0"] = lags_full[tau0_index] if zero_crossing_exists else np.nan
    if zero_crossing_exists:
        result["autocorr_to_tau0"] = autocorr_full[:tau0_index]
        result["lags_to_tau0"] = lags_full[:tau0_index]
        result["gamma_acf"] = compute_truncated_integral_scale(
            autocorr_full, lags_full, tau0_index,
        )
    else:
        result["gamma_acf"] = np.nan

    # -- Annual Gamma ----------------------------------------------------------
    annual_gammas = []

    for yr in valid_years:
        block = x[years_all == yr].dropna().values
        if len(block) < 10:
            annual_gammas.append(np.nan)
            continue
        yr_max_lag = min(
            max_acf_lag if max_acf_lag is not None else 2 ** int(np.log2(len(block))),
            len(block) - 1,
        )
        acf_yr, lags_yr = compute_acf(block, delta_t, max_lag=yr_max_lag)

        if zero_crossing_exists and tau0_index <= len(acf_yr):
            gamma = compute_truncated_integral_scale(acf_yr, lags_yr, tau0_index)
        else:
            gamma = np.nan
        annual_gammas.append(gamma)

    annual_gammas = np.array(annual_gammas)
    result["annual_means"] = annual_means
    result["annual_vars"] = annual_vars
    result["annual_gammas"] = annual_gammas

    # -- Trend diagnostics for each quantity -----------------------------------
    for name, vals in [("mean", annual_means), ("variance", annual_vars),
                       ("gamma", annual_gammas)]:
        diag = compute_trend_diagnostics(valid_years, vals,
                                         max_relative_drift=max_relative_drift)
        for k, v in diag.items():
            result[f"{k}_{name}"] = v

    # -- Overall stability flag ------------------------------------------------
    mean_stable = result.get("stable_mean")
    var_stable = result.get("stable_variance")
    gam_stable = result.get("stable_gamma")

    result["mean_stable"] = mean_stable
    result["variance_stable"] = var_stable
    result["gamma_stable"] = gam_stable

    if not zero_crossing_exists or np.all(np.isnan(annual_gammas)):
        result["second_order_stable"] = np.nan
        result["status"] = "undetermined_acf"
    else:
        if isinstance(var_stable, float) and np.isnan(var_stable):
            result["second_order_stable"] = np.nan
            result["status"] = "undetermined_variance"
        elif isinstance(gam_stable, float) and np.isnan(gam_stable):
            result["second_order_stable"] = np.nan
            result["status"] = "undetermined_acf"
        else:
            result["second_order_stable"] = bool(var_stable and gam_stable)
            result["status"] = "stable" if result["second_order_stable"] else "unstable"

    return result


# ---------------------------------------------------------------------------
# Multi-station wrapper
# ---------------------------------------------------------------------------

def analyze_all_stations(
    stations: list[str],
    prcp_folder: str,
    delta_t: float,
    *,
    max_relative_drift: float = 0.10,
    min_year_coverage: float = 0.80,
    min_valid_years: int = 10,
    max_acf_lag: Optional[int] = None,
) -> dict[str, dict]:
    """Run the stationarity diagnostic for every station.

    Returns
    -------
    dict
        ``{station_id: result_dict}``
    """
    results = {}
    for i, station in enumerate(stations):
        print(f"[{i+1}/{len(stations)}] {station}")
        df = pd.read_csv(
            prcp_folder + station + ".csv",
            parse_dates=["datetime"],
            index_col="datetime",
        )
        data = df["PRCP"] - df["PRCP"].mean()
        res = analyze_station_stationarity(
            data, delta_t,
            max_relative_drift=max_relative_drift,
            min_year_coverage=min_year_coverage,
            min_valid_years=min_valid_years,
            max_acf_lag=max_acf_lag,
        )
        results[station] = res
    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_summary_dataframe(results: dict[str, dict]) -> pd.DataFrame:
    """Collapse per-station result dicts into a single summary DataFrame."""
    rows = []
    for station, r in results.items():
        if r.get("status") == "insufficient_years":
            rows.append({"station": station, "status": "insufficient_years"})
            continue

        row: dict = {
            "station": station,
            "n_valid_years": r["n_valid_years"],
            "zero_crossing_exists": r["zero_crossing_exists"],
            "tau0": r.get("tau0", np.nan),
            "gamma_acf": r.get("gamma_acf", np.nan),
        }

        for name in ("mean", "variance", "gamma"):
            row[f"mean_annual_{name}"] = r.get(f"mean_{name}", np.nan)
            row[f"slope_{name}"] = r.get(f"slope_{name}", np.nan)
            row[f"pvalue_{name}"] = r.get(f"pvalue_{name}", np.nan)
            row[f"r2_{name}"] = r.get(f"r2_{name}", np.nan)
            row[f"cv_{name}"] = r.get(f"cv_{name}", np.nan)
            row[f"relative_drift_{name}"] = r.get(f"relative_drift_{name}", np.nan)
            row[f"{name}_stable"] = r.get(f"{name}_stable", np.nan)

        row["second_order_stable"] = r.get("second_order_stable", np.nan)
        row["status"] = r.get("status", "unknown")
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_station_diagnostics(
    station: str,
    result: dict,
    output_dir: str = "output/stationarity",
    dpi: int = 150,
) -> None:
    """Save a diagnostic figure for a single station.

    Three panels: annual mean, variance, and Gamma vs year, each with a
    fitted linear trend and annotated normalised drift.
    """
    if result.get("status") == "insufficient_years":
        return

    years = result["valid_years"]
    quantities = [
        ("Mean", result["annual_means"], "mean"),
        ("Variance", result["annual_vars"], "variance"),
        (r"$\Gamma$", result["annual_gammas"], "gamma"),
    ]

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5), tight_layout=True)
    fig.suptitle(station, fontsize=11, y=1.02)

    for ax, (label, vals, name) in zip(axs, quantities):
        mask = np.isfinite(vals)
        ax.plot(years[mask], vals[mask], "ko", ms=4, alpha=0.7)

        slope = result.get(f"slope_{name}", np.nan)
        mean_val = result.get(f"mean_{name}", np.nan)
        r2 = result.get(f"r2_{name}", np.nan)
        p = result.get(f"pvalue_{name}", np.nan)
        rd = result.get(f"relative_drift_{name}", np.nan)
        stable = result.get(f"{name}_stable", np.nan)

        if np.isfinite(slope) and mask.sum() >= 2:
            yy = years[mask].astype(float)
            ax.plot(yy, slope * yy + (mean_val - slope * yy.mean()),
                    "r-", lw=1.5, alpha=0.8)

        drift_str = f"{rd*100:.1f}%" if np.isfinite(rd) else "n/a"
        stable_str = "stable" if stable is True else ("unstable" if stable is False else "n/a")
        info = f"D={drift_str}  ({stable_str})\n$R^2$={r2:.3f}  p={p:.2e}" if np.isfinite(r2) else ""
        ax.set_title(f"{label}", fontsize=10)
        ax.set_xlabel("Year")
        ax.annotate(info, xy=(0.03, 0.97), xycoords="axes fraction",
                    va="top", fontsize=7, family="monospace")

        if name == "gamma" and result.get("zero_crossing_exists"):
            tau0 = result.get("tau0", np.nan)
            if np.isfinite(tau0):
                ax.set_ylabel(rf"$\Gamma$ (h)   [$\tau_0$={tau0:.1f} h]", fontsize=8)
            else:
                ax.set_ylabel(r"$\Gamma$ (h)", fontsize=8)
        elif name == "mean":
            ax.set_ylabel("Mean precip.", fontsize=8)
        else:
            ax.set_ylabel("Variance", fontsize=8)

    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, f"{station}_stationarity_diagnostic.png")
    fig.savefig(fname, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Gaussian information criteria
# ---------------------------------------------------------------------------

def gaussian_information_criteria(
    y: np.ndarray,
    y_hat: np.ndarray,
    n_regression_params: int,
) -> dict:
    """Compute RSS, RMSE, AIC, and AICc assuming independent Gaussian residuals.

    Parameters
    ----------
    y : array-like
        Observed values.
    y_hat : array-like
        Fitted values.
    n_regression_params : int
        Number of fitted regression parameters (excluding residual variance).
    """
    y = np.asarray(y)
    y_hat = np.asarray(y_hat)

    n_obs = len(y)
    residuals = y - y_hat
    rss = np.sum(residuals**2)
    rmse = np.sqrt(rss / n_obs)

    sigma2_mle = rss / n_obs
    loglike = -0.5 * n_obs * (np.log(2.0 * np.pi) + 1.0 + np.log(sigma2_mle))

    K = n_regression_params + 1
    aic = -2.0 * loglike + 2.0 * K

    if n_obs > K + 1:
        aicc = aic + 2.0 * K * (K + 1) / (n_obs - K - 1)
    else:
        aicc = np.nan

    return {
        "rss": rss,
        "rmse": rmse,
        "loglike": loglike,
        "aic": aic,
        "aicc": aicc,
    }


# ---------------------------------------------------------------------------
# Integral scale model comparison  (gamma_ACF  vs  gamma_spectrum = A/4)
# ---------------------------------------------------------------------------

def compare_integral_scale_models(
    gamma_spectrum: np.ndarray,
    gamma_acf: np.ndarray,
) -> pd.DataFrame:
    """Compare four nested models relating gamma_spectrum (A/4) to gamma_acf.

    Models
    ------
    M0 : identity          gamma_acf = gamma_spectrum
    M1 : multiplicative    gamma_acf = kappa * gamma_spectrum
    M2 : additive          gamma_acf = alpha + gamma_spectrum
    M3 : unconstrained     gamma_acf = alpha + kappa * gamma_spectrum

    Returns a DataFrame sorted by AICc with delta-AIC and Akaike weights.
    """
    x = np.asarray(gamma_spectrum, dtype=float)
    y = np.asarray(gamma_acf, dtype=float)
    n = len(x)

    # M0: identity ---------------------------------------------------------
    yhat_m0 = x.copy()
    stats_m0 = gaussian_information_criteria(y, yhat_m0, 0)

    # M1: multiplicative (through origin) ----------------------------------
    kappa_m1 = np.sum(x * y) / np.sum(x**2)
    yhat_m1 = kappa_m1 * x
    resid_m1 = y - yhat_m1
    dof_m1 = n - 1
    sigma2_m1 = np.sum(resid_m1**2) / dof_m1
    kappa_se_m1 = np.sqrt(sigma2_m1 / np.sum(x**2))
    tcrit_m1 = t_dist.ppf(0.975, dof_m1)
    kappa_ci_m1 = (
        kappa_m1 - tcrit_m1 * kappa_se_m1,
        kappa_m1 + tcrit_m1 * kappa_se_m1,
    )
    stats_m1 = gaussian_information_criteria(y, yhat_m1, 1)

    # M2: additive (unit slope) --------------------------------------------
    difference = y - x
    alpha_m2 = np.mean(difference)
    alpha_se_m2 = np.std(difference, ddof=1) / np.sqrt(n)
    dof_m2 = n - 1
    tcrit_m2 = t_dist.ppf(0.975, dof_m2)
    alpha_ci_m2 = (
        alpha_m2 - tcrit_m2 * alpha_se_m2,
        alpha_m2 + tcrit_m2 * alpha_se_m2,
    )
    yhat_m2 = x + alpha_m2
    stats_m2 = gaussian_information_criteria(y, yhat_m2, 1)

    # M3: unconstrained linear ---------------------------------------------
    result = linregress(x, y)
    kappa_m3 = result.slope
    alpha_m3 = result.intercept
    kappa_se_m3 = result.stderr
    alpha_se_m3 = result.intercept_stderr
    dof_m3 = n - 2
    tcrit_m3 = t_dist.ppf(0.975, dof_m3)
    kappa_ci_m3 = (
        kappa_m3 - tcrit_m3 * kappa_se_m3,
        kappa_m3 + tcrit_m3 * kappa_se_m3,
    )
    alpha_ci_m3 = (
        alpha_m3 - tcrit_m3 * alpha_se_m3,
        alpha_m3 + tcrit_m3 * alpha_se_m3,
    )
    yhat_m3 = alpha_m3 + kappa_m3 * x
    stats_m3 = gaussian_information_criteria(y, yhat_m3, 2)

    # Build comparison table -----------------------------------------------
    comparison = pd.DataFrame(
        [
            {
                "model": "M0: identity",
                "equation": "gamma_acf = gamma_spectrum",
                "n_regression_params": 0,
                "intercept": 0.0,
                "slope": 1.0,
                "intercept_ci_low": np.nan,
                "intercept_ci_high": np.nan,
                "slope_ci_low": np.nan,
                "slope_ci_high": np.nan,
                **stats_m0,
            },
            {
                "model": "M1: multiplicative",
                "equation": "gamma_acf = kappa * gamma_spectrum",
                "n_regression_params": 1,
                "intercept": 0.0,
                "slope": kappa_m1,
                "intercept_ci_low": np.nan,
                "intercept_ci_high": np.nan,
                "slope_ci_low": kappa_ci_m1[0],
                "slope_ci_high": kappa_ci_m1[1],
                **stats_m1,
            },
            {
                "model": "M2: additive",
                "equation": "gamma_acf = alpha + gamma_spectrum",
                "n_regression_params": 1,
                "intercept": alpha_m2,
                "slope": 1.0,
                "intercept_ci_low": alpha_ci_m2[0],
                "intercept_ci_high": alpha_ci_m2[1],
                "slope_ci_low": np.nan,
                "slope_ci_high": np.nan,
                **stats_m2,
            },
            {
                "model": "M3: unconstrained",
                "equation": "gamma_acf = alpha + kappa * gamma_spectrum",
                "n_regression_params": 2,
                "intercept": alpha_m3,
                "slope": kappa_m3,
                "intercept_ci_low": alpha_ci_m3[0],
                "intercept_ci_high": alpha_ci_m3[1],
                "slope_ci_low": kappa_ci_m3[0],
                "slope_ci_high": kappa_ci_m3[1],
                **stats_m3,
            },
        ]
    )

    comparison["delta_aic"] = comparison["aic"] - comparison["aic"].min()
    comparison["delta_aicc"] = comparison["aicc"] - comparison["aicc"].min()
    aicc_weights = np.exp(-0.5 * comparison["delta_aicc"])
    comparison["akaike_weight_aicc"] = aicc_weights / aicc_weights.sum()
    comparison = comparison.sort_values("aicc").reset_index(drop=True)

    return comparison


# ---------------------------------------------------------------------------
# Integral scale comparison plot
# ---------------------------------------------------------------------------

def plot_integral_scale_comparison(
    gamma_spectrum: np.ndarray,
    gamma_acf: np.ndarray,
    comparison: pd.DataFrame,
    output_path: str = "output/int_scale_models.pdf",
) -> None:
    """Scatter plot of gamma_ACF vs A/4 with four fitted models."""
    models = comparison.set_index("model")
    kappa_m1 = models.loc["M1: multiplicative", "slope"]
    alpha_m2 = models.loc["M2: additive", "intercept"]
    alpha_m3 = models.loc["M3: unconstrained", "intercept"]
    kappa_m3 = models.loc["M3: unconstrained", "slope"]

    x = np.asarray(gamma_spectrum, dtype=float)
    y = np.asarray(gamma_acf, dtype=float)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(x, y, fc="none", ec="k", lw=1, s=30, label="Stations")

    x_fit = np.linspace(0, x.max() * 1.05, 200)

    ax.plot(x_fit, x_fit, ls="--", lw=1.5,
            label=r"$M_0:\ \Gamma_{\rm ACF}=A/4$")
    ax.plot(x_fit, kappa_m1 * x_fit, lw=1.5,
            label=rf"$M_1:\ \Gamma_{{\rm ACF}}={kappa_m1:.3f}(A/4)$")
    ax.plot(x_fit, alpha_m2 + x_fit, lw=1.5,
            label=rf"$M_2:\ \Gamma_{{\rm ACF}}={alpha_m2:.3f}+A/4$")
    ax.plot(x_fit, alpha_m3 + kappa_m3 * x_fit, lw=1.5,
            label=rf"$M_3:\ \Gamma_{{\rm ACF}}={alpha_m3:.3f}+{kappa_m3:.3f}(A/4)$")

    ax.set_xlabel(r"$A/4$")
    ax.set_ylabel(r"$\Gamma_{\mathrm{ACF}}$")
    ax.legend(fontsize=8, frameon=False)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, format="pdf")
    plt.tight_layout()
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Lorentzian fitting (used by compute_spectra)
# ---------------------------------------------------------------------------

def _fit_lorentzian(
    psd, K, params_init, integral_limit, return_err=True, bootstrap_samples=1000
):
    """Fit a Lorentzian in log-space with an AUC = 1 refinement step."""
    psd, K = np.asarray(psd), np.asarray(K)

    def log_lorentzian(params, u):
        A, B, c = params
        eps = 1e-15
        return np.log(max(A, eps)) - np.logaddexp(
            0, c * (u - np.log(max(B, eps)))
        )

    popt, pcov = curve_fit(
        lambda u, A, B, c: log_lorentzian((A, B, c), u),
        np.log(K),
        np.log(psd),
        p0=params_init,
    )

    def residuals(p, K, psd):
        p = p[0]
        integral = quad(
            lambda f: lorentzian(f, popt[0], p, popt[2]), *integral_limit, epsrel=1e-9
        )[0]
        penalization = abs(1 - integral) * 10000
        return (
            np.log(psd)
            - log_lorentzian((popt[0], p, popt[2]), np.log(K))
            - penalization
        )

    out = least_squares(residuals, [popt[1]], args=(K, psd))

    params = {"A": popt[0], "B": out.x[0], "c": popt[2]}

    if not return_err:
        return params

    boot_params = []
    rng = np.random.default_rng()
    for _ in range(bootstrap_samples):
        indices = rng.integers(0, len(psd), size=len(psd))
        try:
            popt_r, _ = curve_fit(
                lambda u, A, B, c: log_lorentzian((A, B, c), u),
                np.log(K[indices]),
                np.log(psd[indices]),
                p0=(params["A"], params["B"], params["c"]),
            )
            boot_params.append(popt_r)
        except RuntimeError:
            continue

    boot_params = np.array(boot_params)
    boot_se = np.std(boot_params, axis=0)
    return params, boot_se


def _lorentzian_spectra(
    wavelets, integral_scale, fourier=None, return_err=False, angular=True
):
    """Lorentzian fitting with AUC = 1 soft constraint."""
    psd_w, K_w = map(np.asarray, wavelets)
    integral_limit = (min(K_w), max(K_w))
    psd_w, K_w = psd_w[:-1], K_w[:-1]

    C = 2 * np.pi if angular else 1

    c, _, breakpt = fit_psd_power_law(psd_w, K_w, integral_scale, angular=angular)
    params_init = [integral_scale * 4 / C, 10**breakpt, -c]

    res_w = _fit_lorentzian(
        psd_w,
        K_w,
        params_init,
        integral_limit=integral_limit,
        return_err=return_err,
        bootstrap_samples=1000,
    )

    if return_err:
        res_w[0]["breakpoint"] = 10**breakpt
    else:
        res_w["breakpoint"] = 10**breakpt

    if fourier is None:
        return res_w

    psd_F, K_F = map(np.asarray, fourier)
    mask = K_F < max(K_w)
    psd_F, K_F = psd_F[mask], K_F[mask]
    K_F[K_F == 0] = 1e-15

    wavelet_params = res_w[0] if return_err else res_w
    params_init = [wavelet_params["A"], wavelet_params["B"], wavelet_params["c"]]

    res_F = _fit_lorentzian(
        psd_F, K_F, params_init, return_err=return_err, bootstrap_samples=1000
    )

    return {"wavelet": res_w, "fourier": res_F}


# ---------------------------------------------------------------------------
# Spectral computation (wavelet + optional Lorentzian fit
# ---------------------------------------------------------------------------

def compute_spectra(
    stations: list[str],
    int_scale: dict[str, float],
    prcp_folder: str,
    delta_t: float,
    *,
    fit_lorentzian_flag: bool = False,
    return_wT: bool = False,
) -> dict | tuple[dict, dict]:
    """Compute wavelet/Fourier spectra and optionally fit Lorentzians.

    Parameters
    ----------
    int_scale : dict
        ``{station_id: integral_scale}`` from the ACF analysis.
    prcp_folder : str
        Path to the directory containing per-station CSV files.
    """
    psi = pywt.Wavelet("haar")
    spectra: dict = {}
    lparams: dict = {}
    std_error: dict = {}
    wavelet_coeff: dict = {}

    for station in stations:
        df = pd.read_csv(
            prcp_folder + station + ".csv",
            parse_dates=["datetime"],
            index_col="datetime",
        )
        data = df["PRCP"].values

        x = data - np.nanmean(data)
        x = x / np.nanstd(x)

        wT, tau = wavelet_transform(x, psi, delta_t, mode="per")
        psd_w, K_w = wavelet_psd(wT, tau, delta_t, angular=False)
        psd_F, K_F = fourier_psd(
            x, delta_t, window="hann", nperseg=8 * 8192, angular=False
        )
        spectra[station] = {"wavelet": (psd_w, K_w), "fourier": (psd_F, K_F)}
        if return_wT:
            wavelet_coeff[station] = {"wT": wT, "scales": tau}

        if fit_lorentzian_flag:
            print(f"------------ Processing {station} ------------")
            try:
                p, perr = _lorentzian_spectra(
                    (psd_w, K_w),
                    integral_scale=int_scale[station],
                    return_err=True,
                    angular=False,
                )
                lparams[station] = p
                std_error[station] = perr

                area = quad(
                    lambda f: lorentzian(f, p["A"], p["B"], p["c"]),
                    min(K_w),
                    max(K_w),
                    epsrel=1e-9,
                )[0]
                print(
                    f"{station} & A = {p['A']:.2f} +/- {perr[0]:.2f} "
                    f"& f0 = {p['B']:.3f}  +/- {perr[1]:.3f} "
                    f"& c = {p['c']:.2f} +/- {perr[2]:.2f} "
                    f"& area = {area:.4f} \\\\"
                )
            except Exception as e:
                print("!!!!!!!! OPTIMIZATION FAILED !!!!!!!!")
                print(e)
                continue

    if fit_lorentzian_flag:
        lparams_df = pd.DataFrame.from_dict(
            lparams, orient="index"
        ).rename_axis("station_id")
        std_error_df = (
            pd.DataFrame.from_dict(std_error, orient="index")
            .rename_axis("station_id")
            .rename(columns={0: "A", 1: "B", 2: "c"})
        )
        lparams_df.to_csv("data/lparams.csv")
        std_error_df.to_csv("output/lorentzian_std_error.csv")

    if return_wT:
        return spectra, wavelet_coeff
    return spectra
