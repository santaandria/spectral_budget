"""
Analytical quantities derived from the fitted Lorentzian spectrum.

These are closed-form manuscript formulas evaluated on the per-station
Lorentzian parameters (A, B, c) rather than on the measured series: the
spectral source term and the normalized Shannon entropy of the spectrum,
plus the OLS hyperplane relating that entropy to B and c.
"""

import numpy as np
import statsmodels.api as sm
from scipy.integrate import quad

from .analysis import lorentzian


def source_term(f, A, B, c, m, u0):
    """
    Spectral source term for a Lorentzian spectrum.

    Parameters:
        f: frequency
        A, B, c: Lorentzian parameters (amplitude, scale, exponent)
        m: power-law exponent of the advection term
        u0: advection velocity scale
    """
    h = f / B
    return (
        -(
            A
            * f
            * u0
            * (
                c**2 * m * h**c * (-1 + h**c)
                - c * h**c * (1 + h**c) * (2 * m - 1)
                - 2 * (1 + h**c) ** 2
            )
        )
        / (1 + h**c) ** 3
    )


def normalized_source(f, A, B, c, m, int_scale, u0):
    """Source term normalized by its value at the integral scale."""
    normalization = source_term(1 / int_scale, A, B, c, m, u0)
    return source_term(f, A, B, c, m, u0) / normalization


def shannon_entropy(A, B, c, f_min, f_max):
    # Return normalized shannon entropy
    def entropy_integrand(f):
        p = lorentzian(f, A, B, c)
        return -p * np.log(p) if p > 0 else 0  # Avoid log(0) issues

    # Compute entropy using numerical integration
    entropy, _ = quad(entropy_integrand, f_min, f_max)
    return entropy / np.log(f_max - f_min)


def fit_entropy_hyperplane(lparams):
    """
    Add the ``Shannon_Entropy`` column to ``lparams`` and regress it on B and c.

    ``lparams`` is modified in place (the added column is read by the Fig 8a
    and Fig 8b sections). Returns the fitted OLS model together with the
    coefficients and the B/c ranges used to draw the regression surface.
    """
    lparams["Shannon_Entropy"] = lparams.apply(
        lambda row: shannon_entropy(
            row["A"], row["B"], row["c"], row["min_K"], row["max_K"]
        ),
        axis=1,
    )
    # Fit hyperplane
    # Assuming lparams is your existing DataFrame with B, c, and Shannon_Entropy columns

    X = lparams[["B", "c"]].values
    X = sm.add_constant(X)  # constant intercept
    y = lparams["Shannon_Entropy"].values

    # OLS regression model
    model = sm.OLS(y, X).fit()

    # print(model.summary())

    intercept = model.params[0]
    slope_B = model.params[1]
    slope_c = model.params[2]

    print(
        f"Hyperplane equation: H(f) = {slope_B:.3f} × B + {slope_c:.3f} × c + {intercept:.3f}"
    )

    B_min, B_max = lparams["B"].min(), lparams["B"].max()
    c_min, c_max = lparams["c"].min(), lparams["c"].max()

    return model, intercept, slope_B, slope_c, B_min, B_max, c_min, c_max
