"""Windmeijer (2005) finite-sample correction for two-step GMM standard errors.

.. warning::
   **Experimental in v0.1.** The closed form below is implemented from
   Windmeijer (2005) eq. (2.10) / Roodman (2009) eq. (25), but
   cross-validation against Stata's ``xtabond2`` on canonical datasets
   has not yet landed in CI. Treat the corrected SEs as preliminary
   until the v0.2 replication tests are merged.



Two-step GMM standard errors are notoriously downward biased in finite
samples — sometimes severely so. Windmeijer (2005) derives an analytical
correction that accounts for the dependence of the optimal weight matrix
W_2 on the first-step coefficient β̂_1.

The derivation (compactly):

    β̂_2 = A_2 X'Z W_2 Z'y    where  A_2 = (X'Z W_2 Z'X)^{-1}

    W_2 = Σ_2(β̂_1)^{-1}  where  Σ_2(β) = Σ_i Z_i' u_i(β) u_i(β)' Z_i

Treating W_2 as a function of β̂_1:

    ∂β̂_2/∂β̂_{1,c} = -A_2 (X'Z W_2) G_c (W_2 Z' û_2)

where

    G_c = ∂Σ_2/∂β̂_{1,c}
        = -Σ_i [ (Z_i'X_{i,c}) (Z_i'û_{1,i})' + (Z_i'û_{1,i}) (Z_i'X_{i,c})' ]

Note the two minus signs cancel — the implementation below uses the
positive form directly.

Letting D be the k×k matrix with D[:, c] = ∂β̂_2/∂β̂_{1,c}, the
Windmeijer-corrected variance (Roodman 2009 eq. 25) is:

    V_w = A_2  +  D A_2  +  A_2 D'  +  D V_1 D'

where V_1 is the cluster-robust one-step variance.

References
----------
Windmeijer, F. (2005). "A finite sample correction for the variance of
    linear efficient two-step GMM estimators." Journal of Econometrics,
    126(1):25–51.

Roodman, D. (2009). "How to do Xtabond2: An introduction to difference
    and system GMM in Stata." Stata Journal, 9(1):86–136.
"""

from __future__ import annotations

import numpy as np


def windmeijer_correction(
    *,
    X: np.ndarray,
    Z: np.ndarray,
    beta_2: np.ndarray,
    resid_1: np.ndarray,
    resid_2: np.ndarray,
    W2: np.ndarray,
    A2: np.ndarray,
    V1: np.ndarray,
    individual_slices,
) -> np.ndarray:
    """Compute the Windmeijer-corrected covariance matrix for β̂_2.

    Parameters
    ----------
    X, Z
        Stacked design and instrument matrices (n_obs × k, n_obs × L).
    beta_2
        Two-step coefficient vector (k,).
    resid_1, resid_2
        One-step and two-step residuals (n_obs,).
    W2
        Two-step weight matrix (L × L). Should equal
        (sum_i Z_i' û_1i û_1i' Z_i)^{-1}.
    A2
        (X' Z W_2 Z' X)^{-1}, the model-based two-step variance.
    V1
        Cluster-robust one-step variance (k × k).
    individual_slices
        List of slice objects, one per panel unit, defining its rows.

    Returns
    -------
    V_w : (k, k)
        Windmeijer-corrected variance.
    """
    n_obs, k = X.shape
    L = Z.shape[1]

    # Precompute reusable quantities.
    XtZ = X.T @ Z              # (k, L)
    XtZ_W2 = XtZ @ W2          # (k, L)
    A2_XtZ_W2 = A2 @ XtZ_W2    # (k, L)
    W2_Ztu2 = W2 @ (Z.T @ resid_2)  # (L,)

    # Build D column by column. D[:, c] = -A2 X'Z W2 G_c W2 Z'û_2.
    D = np.zeros((k, k))
    for c in range(k):
        # G_c = -sum_i [m_i^X(c) m_i^u(1) ' + m_i^u(1) m_i^X(c)']
        # The leading minus sign and the leading minus sign in D cancel,
        # so we compute G_c without the minus and use a + sign on D.
        G_c = np.zeros((L, L))
        for sl in individual_slices:
            Xi_c = X[sl, c]
            Zi = Z[sl]
            u1i = resid_1[sl]
            mX = Zi.T @ Xi_c          # (L,)
            mU = Zi.T @ u1i           # (L,)
            G_c += np.outer(mX, mU) + np.outer(mU, mX)
        D[:, c] = A2_XtZ_W2 @ G_c @ W2_Ztu2

    # Roodman (2009) eq. 25:
    #   V_w = A_2 + D A_2 + A_2 D' + D V_1 D'
    V_w = A2 + D @ A2 + A2 @ D.T + D @ V1 @ D.T

    # Guard against tiny negative diagonal entries from numerical noise.
    diag = np.diag(V_w)
    if np.any(diag < 0):
        # Symmetrize and clip diagonal. Keep off-diagonals intact.
        V_w = 0.5 * (V_w + V_w.T)
        d = np.diag(V_w)
        d_clipped = np.maximum(d, 0)
        np.fill_diagonal(V_w, d_clipped)
    return V_w
