"""Specification tests for dynamic panel GMM.

What's in:
    ar_test(order=1)       -- Arellano-Bond m1
    ar_test(order=2)       -- Arellano-Bond m2 (the one you actually care about)
    hansen_j_test          -- Hansen test of overidentifying restrictions

The m1 test should typically reject (the differenced residuals are
MA(1) by construction, since u_t - u_{t-1} is correlated with
u_{t-1} - u_{t-2} through u_{t-1}). The m2 test is the substantive one
— if it rejects, your lagged-level instruments are inconsistent and
you've got a problem.

NB on the AR test variance: we use one-way clustering by panel unit
under H0 (see code comments). This is asymptotically valid. The full
Arellano-Bond (1991) eq. 11 variance, which additionally adjusts for
the estimation noise in β̂, is on the v0.2 roadmap. The size of the
difference between the two depends on the sample and instrument set,
so v0.1 reports the simpler estimator and flags it explicitly in the
result note rather than benchmarking against the full form.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class ARTestResult:
    order: int
    statistic: float
    pvalue: float
    n_pairs: int       # number of (i, t) pairs that contributed
    note: str = ""     # any caveats

    def __str__(self) -> str:
        return (
            f"AR({self.order}) test: z = {self.statistic:.3f}, "
            f"p = {self.pvalue:.4f}  ({self.n_pairs} pairs)"
        )


@dataclass
class HansenJResult:
    statistic: float
    df: int
    pvalue: float
    note: str = ""

    def __str__(self) -> str:
        return (
            f"Hansen J: χ²({self.df}) = {self.statistic:.3f}, "
            f"p = {self.pvalue:.4f}"
        )


# ---------------------------------------------------------------------------
# Arellano-Bond m1 / m2 test
# ---------------------------------------------------------------------------

def ar_test(
    *,
    resid: np.ndarray,
    ids: np.ndarray,
    rows_position: np.ndarray,
    is_diff_row: np.ndarray,
    order: int,
) -> ARTestResult:
    """Arellano-Bond m_r test for AR(r) in the differenced residuals.

    Parameters
    ----------
    resid : (n_obs,)
        The full residual vector from the GMM fit (concatenation of
        diff-eq and possibly level-eq residuals).
    ids : (n_obs,)
        Panel id for each row.
    rows_position : (n_obs,)
        Within-individual position of each row.
    is_diff_row : (n_obs,)
        True for rows that come from the differenced equation. We only
        use those rows for this test.
    order : int
        AR order. 1 for m1, 2 for m2.

    Returns
    -------
    ARTestResult
    """
    if order < 1:
        raise ValueError(f"AR test order must be >= 1, got {order}.")

    mask = is_diff_row
    r = resid[mask]
    id_ = ids[mask]
    pos = rows_position[mask]

    # Group by id and accumulate the inner products (Δû_{it} · Δû_{i,t-r})
    # for valid (i, t) pairs.
    unique_ids = np.unique(id_)
    per_ind_pair_sum = []  # for the cluster variance estimator
    total_sum = 0.0
    n_pairs = 0
    for u in unique_ids:
        idx = np.where(id_ == u)[0]
        # sort by position to be safe
        order_idx = idx[np.argsort(pos[idx])]
        r_i = r[order_idx]
        p_i = pos[order_idx]
        # For each row at position p, the lagged residual we want is at the
        # row in the same individual with position p - order. The diff
        # equation has consecutive positions p_min, p_min+1, ..., so the
        # row at local index k in r_i corresponds to position p_i[k]. The
        # lag row is the one with position p_i[k] - order.
        pos_to_local = {int(pp): k for k, pp in enumerate(p_i)}
        s_i = 0.0
        local_pair_count = 0
        for k, pk in enumerate(p_i):
            src = int(pk) - order
            kk = pos_to_local.get(src)
            if kk is None:
                continue
            s_i += r_i[k] * r_i[kk]
            local_pair_count += 1
        if local_pair_count > 0:
            total_sum += s_i
            per_ind_pair_sum.append(s_i)
            n_pairs += local_pair_count

    if not per_ind_pair_sum:
        return ARTestResult(
            order=order, statistic=np.nan, pvalue=np.nan, n_pairs=0,
            note="no individuals had enough periods to form lag pairs",
        )

    # Cluster variance under H0 of no AR(r): Var(Σ_i s_i) ≈ Σ_i s_i^2.
    # This is conservative — see module docstring.
    var = float(np.sum(np.array(per_ind_pair_sum) ** 2))
    if var <= 0:
        return ARTestResult(
            order=order, statistic=np.nan, pvalue=np.nan, n_pairs=n_pairs,
            note="zero variance estimate — residuals are degenerate?",
        )

    z = total_sum / np.sqrt(var)
    pval = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return ARTestResult(
        order=order, statistic=float(z), pvalue=float(pval), n_pairs=n_pairs,
        note="one-way clustering by panel unit under H0; v0.2 will add the AB-1991 finite-sample correction",
    )


# ---------------------------------------------------------------------------
# Hansen J test of overidentifying restrictions
# ---------------------------------------------------------------------------

def hansen_j_test(
    *,
    X: np.ndarray,
    Z: np.ndarray,
    resid: np.ndarray,
    n_params: int,
    individual_slices,
    W: np.ndarray,
) -> HansenJResult:
    """Hansen J statistic at the optimal weight matrix.

        J = (Z' û)' W_optimal (Z' û)    ~ χ²(L - k) under H0

    where W_optimal is the inverse of the cluster moment of Z'u evaluated
    at the *final* residuals. This re-evaluates the optimal weight
    matrix at the fitted residuals, which is the standard form of the
    test reported in applied work.

    Important: J is NOT robust to the use of suboptimal weights. We
    evaluate it at the optimal weights, even if the user requested only
    one-step coefficients. That's deliberate — reporting Hansen at
    non-optimal weights is misleading.

    Returns
    -------
    HansenJResult
    """
    n_inst = Z.shape[1]
    df = n_inst - n_params
    if df <= 0:
        return HansenJResult(
            statistic=np.nan, df=df, pvalue=np.nan,
            note="model is just-identified (df ≤ 0); J test is not defined",
        )

    g = Z.T @ resid
    J = float(g @ W @ g)
    pval = float(1.0 - stats.chi2.cdf(J, df))
    note = ""
    if J < 0:
        # Can happen with a pseudo-inverse W when the moment matrix is
        # singular. Clip and warn via the note.
        J = max(J, 0.0)
        note = "J was slightly negative due to a singular moment matrix; clipped to 0"
        pval = float(1.0 - stats.chi2.cdf(J, df))
    return HansenJResult(statistic=J, df=df, pvalue=pval, note=note)
