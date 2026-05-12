"""A small DGP simulator for unit tests and quickstart demos.

The DGP:

    η_i  ~ N(0, σ_η^2)               (unit fixed effect)
    x_it ~ AR(1) with persistence ρ + cross-correlation with η_i if exogenous=False
    y_it = α y_{i,t-1} + β x_it + η_i + ε_it,    ε_it ~ N(0, σ_ε^2)

We initialize y_{i,0} at its stationary mean conditional on η_i and burn in
``burn`` periods to get rid of the initial-condition transient. Without
this burn-in, the dynamic-panel bias goes haywire and tests break in
the small-T regime.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def simulate_dynamic_panel(
    n: int = 200,
    t: int = 8,
    alpha: float = 0.5,
    beta: float = 1.0,
    rho_x: float = 0.5,
    sigma_eta: float = 1.0,
    sigma_eps: float = 1.0,
    sigma_x: float = 1.0,
    x_corr_eta: float = 0.25,
    burn: int = 50,
    seed: int | None = None,
) -> pl.DataFrame:
    """Simulate a dynamic panel y_it = alpha * y_{i,t-1} + beta * x_it + eta_i + eps_it.

    Parameters
    ----------
    n : int
        Number of cross-sectional units.
    t : int
        Number of *kept* time periods per unit (after burn-in).
    alpha, beta : float
        True dynamic and regression coefficients.
    rho_x : float
        AR(1) persistence in the x process. Setting this > 0 makes the
        problem realistic — fully iid x is too easy.
    sigma_eta, sigma_eps, sigma_x : float
        Standard deviations of the fixed effect, error, and x shock.
    x_corr_eta : float
        Correlation of x_it with η_i. Set > 0 to make x endogenous-ish
        (the classic "correlated regressor" case that motivates within
        estimators).
    burn : int
        Burn-in periods before we start recording.
    seed : int | None
        For reproducibility.

    Returns
    -------
    pl.DataFrame
        Columns: id, t, y, x, eta. (eta is observable here only because
        we're simulating — in real data you wouldn't see it.)
    """
    rng = np.random.default_rng(seed)
    eta = rng.normal(0, sigma_eta, size=n)

    total_t = burn + t
    # Pre-allocate
    y_full = np.zeros((n, total_t))
    x_full = np.zeros((n, total_t))

    # x starts at its conditional mean: x_corr_eta * eta_i (plus shock)
    x_full[:, 0] = x_corr_eta * eta + rng.normal(0, sigma_x, size=n)
    # y starts at its conditional stationary mean: η_i / (1 - α) + small noise
    if abs(alpha) < 1:
        y_full[:, 0] = eta / (1 - alpha) + rng.normal(0, sigma_eps / np.sqrt(1 - alpha**2), size=n)
    else:
        y_full[:, 0] = rng.normal(0, 1, size=n)

    for k in range(1, total_t):
        x_full[:, k] = (
            rho_x * x_full[:, k - 1]
            + x_corr_eta * eta
            + rng.normal(0, sigma_x, size=n)
        )
        y_full[:, k] = (
            alpha * y_full[:, k - 1]
            + beta * x_full[:, k]
            + eta
            + rng.normal(0, sigma_eps, size=n)
        )

    # Keep only the post-burnin window
    y = y_full[:, burn:]
    x = x_full[:, burn:]

    # Tidy long format
    ids = np.repeat(np.arange(n), t)
    ts = np.tile(np.arange(1, t + 1), n)
    etas = np.repeat(eta, t)
    return pl.DataFrame({
        "id": ids,
        "t": ts,
        "y": y.flatten(),
        "x": x.flatten(),
        "eta": etas,
    })
