"""System and Difference GMM estimators.

The implementation follows Arellano & Bond (1991), Arellano & Bover (1995),
Blundell & Bond (1998), and Roodman's (2009) *Stata Journal* article on
``xtabond2``.

A few choices worth flagging up front:

1. We build moment matrices in dense numpy. For the panel sizes that
   dynamic GMM actually works well on (a few thousand i × a dozen t),
   this is fine. If you have N > 50k, you probably have bigger problems
   than dense Z.

2. Two-step weight matrix:
       W_2 = (sum_i Z_i' u_1i u_1i' Z_i)^{-1}
   We fall back to a Moore-Penrose pseudo-inverse when the moment
   matrix is near-singular, which happens easily without
   ``collapse=True``. This mirrors the practical behavior of GMM
   software that has to handle singular moment covariances when L
   gets large relative to N. See Roodman (2009) section 4 for the
   conceptual discussion of this scenario.

3. Windmeijer (2005) correction: in ``_windmeijer.py``. The closed form
   in his eq. (2.10) / Roodman eq. (25). Default ON when steps=2
   because uncorrected two-step SEs are often substantially downward
   biased in small samples.

4. SEs default to one-way clustering on the panel unit. ``robust="twoway"``
   is not implemented in v0.1 and raises ``NotImplementedError`` rather
   than silently falling back — a user can't accidentally end up with a
   different variance than they asked for. True two-way clustering is on
   the v0.2 roadmap.

TODO: forward orthogonal deviations (Arellano-Bover 1995) is not wired
through. The flag is in the constructor signature so we don't break the
API later, but for now ``orthogonal_deviations=True`` raises.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from scipy import linalg

from ._data import Individual, split_into_individuals, to_polars, validate_panel
from ._formula import Formula, Term, parse_formula
from ._instruments import (
    AssembledMatrices,
    assemble,
    build_blocks_for_individual,
)
from ._results import GMMResults
from ._windmeijer import windmeijer_correction


# ---------------------------------------------------------------------------
# Public estimator classes
# ---------------------------------------------------------------------------


class GMMBase:
    """Shared scaffolding for Difference GMM and System GMM.

    Don't instantiate this directly. Use ``DifferenceGMM`` or ``SystemGMM``.
    """

    _equation: str  # "difference" or "system" — set in subclasses

    def __init__(
        self,
        formula: str,
        *,
        data,
        panel_var: str,
        time_var: str,
        gmm_instruments: list[str] | None = None,
        iv_instruments: list[str] | None = None,
        gmm_lags: dict[str, tuple[int, int | None]] | None = None,
        minlag: int = 2,
        maxlag: int | None = None,
        collapse: bool = False,
        orthogonal_deviations: bool = False,
        constant: bool = True,
    ):
        """
        Parameters
        ----------
        formula
            Lag-aware formula, e.g. ``"y ~ L(1).y + x"``.
        data
            pandas, polars, or dict-of-arrays.
        panel_var, time_var
            Column names of the panel id and (integer) time index.
        gmm_instruments
            Variables to instrument GMM-style (lagged levels in the
            differenced equation; lagged differences in the level
            equation under System GMM). Default: the dependent variable.
            These are the variables you suspect are *endogenous* or
            *predetermined*.
        iv_instruments
            Variables that are *strictly exogenous*. Used as themselves
            (or as their first differences in the differenced equation),
            i.e. one moment per observation rather than GMM-style.
        gmm_lags
            Optional per-variable lag specification. Map of
            ``{"var_name": (minlag, maxlag)}`` overriding the global
            defaults. Use this to distinguish:

            - **endogenous**:      ``{"y": (2, None)}``
            - **predetermined**:   ``{"w": (1, None)}``  (corr w/ past
              shocks only — lag 1 of the level is uncorrelated with
              Δu_it under standard assumptions)

            Variables in ``gmm_instruments`` that are not in
            ``gmm_lags`` use the global ``minlag`` / ``maxlag``.
        minlag, maxlag
            Default lag bounds for GMM instruments. ``minlag=2`` is the
            convention for endogenous variables (which is the right
            default to be conservative).
        collapse
            Collapse instrument matrix per Roodman (2009). *Strongly
            recommended.* Without it, instrument counts grow as O(T²)
            and the two-step weight matrix becomes ill-conditioned.
        orthogonal_deviations
            Forward orthogonal deviations (Arellano-Bover 1995). Not
            implemented in v0.1; raises ``NotImplementedError``.
        constant
            Include a constant in the level equation of System GMM.
            Ignored (no effect) for pure Difference GMM, since the
            constant is swept out by first-differencing.
        """
        # ---- validate orthogonal_deviations early so we fail loud --------
        if orthogonal_deviations:
            raise NotImplementedError(
                "orthogonal_deviations=True (forward orthogonal deviations, "
                "Arellano-Bover 1995) is on the v0.2 roadmap but not implemented "
                "yet. For now, use the default first-differences transform."
            )

        if minlag < 1:
            raise ValueError(f"minlag must be >= 1, got {minlag}.")
        if maxlag is not None and maxlag < minlag:
            raise ValueError(
                f"maxlag ({maxlag}) must be >= minlag ({minlag})."
            )

        self.formula_str = formula
        self.formula: Formula = parse_formula(formula)
        self.panel_var = panel_var
        self.time_var = time_var
        # Default GMM instrument list: the dependent variable itself,
        # which is what you almost always want.
        self.gmm_instruments = list(gmm_instruments) if gmm_instruments else [self.formula.dep_var]
        self.iv_instruments = list(iv_instruments) if iv_instruments else []

        # A variable can't be in both lists — it must be classified one
        # way or the other. Silent misclassification is exactly the kind
        # of bug that fits a model but produces wrong standard errors.
        overlap = set(self.gmm_instruments) & set(self.iv_instruments)
        if overlap:
            raise ValueError(
                f"the following variables appear in BOTH gmm_instruments and "
                f"iv_instruments: {sorted(overlap)}. "
                f"A variable is either endogenous/predetermined (gmm_instruments) "
                f"or strictly exogenous (iv_instruments) — it can't be both."
            )

        self.minlag = int(minlag)
        self.maxlag = None if maxlag is None else int(maxlag)
        self.collapse = bool(collapse)
        self.orthogonal_deviations = bool(orthogonal_deviations)
        self.constant = bool(constant)

        # Per-variable lag spec. Validate and resolve defaults.
        self.gmm_lags: dict[str, tuple[int, int | None]] = {}
        gmm_lags = gmm_lags or {}
        for v, spec in gmm_lags.items():
            if v not in self.gmm_instruments:
                raise ValueError(
                    f"gmm_lags has an entry for {v!r}, but {v!r} is not in "
                    f"gmm_instruments. Either add it to gmm_instruments or "
                    f"remove the override."
                )
            try:
                lo, hi = spec
            except (TypeError, ValueError):
                raise ValueError(
                    f"gmm_lags[{v!r}] must be a (minlag, maxlag) tuple, got {spec!r}."
                ) from None
            if lo is None or int(lo) < 1:
                raise ValueError(
                    f"gmm_lags[{v!r}] minlag must be >= 1, got {lo!r}."
                )
            if hi is not None and int(hi) < int(lo):
                raise ValueError(
                    f"gmm_lags[{v!r}] maxlag ({hi}) must be >= minlag ({lo})."
                )
            self.gmm_lags[v] = (int(lo), None if hi is None else int(hi))
        # Fill in defaults for any gmm_instrument without an explicit spec.
        for v in self.gmm_instruments:
            self.gmm_lags.setdefault(v, (self.minlag, self.maxlag))

        # Collect all variables we need to pull from the data
        needed = {self.formula.dep_var}
        for t in self.formula.rhs_terms:
            needed.add(t.var)
        for v in self.gmm_instruments:
            needed.add(v)
        for v in self.iv_instruments:
            needed.add(v)

        df = to_polars(data)
        df = validate_panel(df, panel_var, time_var, required_vars=sorted(needed))

        self._individuals: list[Individual] = split_into_individuals(
            df, panel_var, time_var, needed_vars=sorted(needed),
        )

        if len(self._individuals) < 2:
            raise ValueError(
                f"need at least 2 panel units (got {len(self._individuals)}). "
                f"You can't do panel GMM with one individual."
            )

        # Built lazily in fit()
        self._fitted: GMMResults | None = None

    # ---- public API ------------------------------------------------------

    def fit(
        self,
        *,
        steps: Literal[1, 2] = 2,
        windmeijer: bool = True,
        robust: str = "cluster",
    ) -> GMMResults:
        """Estimate the model.

        Parameters
        ----------
        steps
            1 for one-step GMM, 2 for two-step.
        windmeijer
            Apply Windmeijer (2005) finite-sample correction to two-step
            SEs. Ignored (with a warning) when ``steps=1``.
        robust
            Variance estimator for SEs. ``"cluster"`` (also accepted:
            ``"i"``, ``"id"``, ``"panel"``) clusters on the panel id —
            the standard choice for dynamic panels and the only
            supported option in v0.1. ``"twoway"`` raises
            ``NotImplementedError`` rather than silently falling back,
            so the user can't accidentally believe they got a twoway
            variance.

        Returns
        -------
        GMMResults
        """
        if steps not in (1, 2):
            raise ValueError(f"steps must be 1 or 2, got {steps!r}.")
        windmeijer_applied = bool(windmeijer and steps == 2)
        if windmeijer and steps == 1:
            warnings.warn(
                "windmeijer=True is meaningless with steps=1; ignoring.",
                stacklevel=2,
            )

        # Normalize robust= to a canonical value. We deliberately reject
        # "twoway" rather than silently falling back to one-way clustering
        # — a user who specified twoway and didn't notice the warning
        # would walk away believing they had different SEs than they do.
        if robust in ("i", "id", "cluster", "panel"):
            robust = "cluster"
        elif robust == "twoway":
            raise NotImplementedError(
                "robust='twoway' is not implemented in v0.1. Use "
                "robust='cluster' (one-way cluster on panel_var) for now."
            )
        else:
            raise ValueError(
                f"robust={robust!r} is not supported in v0.1. Use 'cluster' "
                f"(default, one-way cluster on panel_var). Model-based and "
                f"twoway variance estimators are on the v0.2 roadmap."
            )

        # ---- assemble matrices -------------------------------------------
        diff_blocks = []
        level_blocks = []
        for ind in self._individuals:
            db, lb = build_blocks_for_individual(
                ind=ind,
                dep_var=self.formula.dep_var,
                rhs_terms=self.formula.all_terms,
                gmm_instruments=self.gmm_instruments,
                iv_instruments=self.iv_instruments,
                gmm_lags=self.gmm_lags,
                equation=self._equation,
                collapse=self.collapse,
            )
            diff_blocks.append(db)
            level_blocks.append(lb)

        include_level = self._equation == "system"
        mats = assemble(
            individuals=self._individuals,
            blocks_diff=diff_blocks,
            blocks_level=level_blocks,
            include_level=include_level,
            add_const_to_X=self.constant and include_level,
        )

        # Column names for X (regressors)
        coef_names = self.formula.names
        if self.constant and include_level:
            coef_names = coef_names + ["const"]

        n_obs, n_reg = mats.X.shape
        n_inst = mats.Z.shape[1]
        if n_inst < n_reg:
            raise ValueError(
                f"underidentified: {n_inst} instruments < {n_reg} regressors. "
                f"Either add more lags / IVs, or lower the model order."
            )

        # Instrument proliferation warning. Hansen tests lose power and
        # the two-step weight matrix gets ill-conditioned when the
        # instrument count is large relative to the cross section.
        # Thresholds follow common practitioner advice (Roodman 2009 and
        # others suggest keeping L < N as a soft cap).
        N = len(self._individuals)
        if n_inst >= N:
            warnings.warn(
                f"instrument count ({n_inst}) is >= number of panel units "
                f"({N}). Hansen tests are likely to be unreliable; consider "
                f"setting collapse=True or reducing maxlag.",
                stacklevel=2,
            )
        elif n_inst > N / 2:
            warnings.warn(
                f"instrument count ({n_inst}) is large relative to the "
                f"number of panel units ({N}). Consider collapse=True or "
                f"reducing maxlag if Hansen / Windmeijer look off.",
                stacklevel=2,
            )

        # ---- compute the one-step weight matrix --------------------------
        W1 = self._one_step_weight(mats)

        # ---- step 1 ------------------------------------------------------
        b1, V1, resid1, A1, ZuuZ_1 = _gmm_step(
            mats.y, mats.X, mats.Z, W1, mats.ids, mats.individual_slices,
        )
        se1 = _safe_sqrt(np.diag(V1))

        result_dict = {
            "step": 1 if steps == 1 else 2,
            "coef_names": coef_names,
            "coef_step1": b1,
            "se_step1": se1,
            "V_step1": V1,
            "resid_step1": resid1,
            "A_step1": A1,
            "W_step1": W1,
            "ZuuZ_step1": ZuuZ_1,
            "windmeijer_applied": False,
            "robust": robust,
        }

        if steps == 1:
            beta = b1
            se = se1
            V = V1
            resid = resid1
            W_used = W1
        else:
            # ---- step 2 --------------------------------------------------
            W2 = _safe_inv(ZuuZ_1)
            # _gmm_step returns the cluster-robust sandwich variance.
            # This is the variance we report when the user asks for
            # two-step *without* the Windmeijer correction.
            b2, V2_uncorrected, resid2, A2, ZuuZ_2 = _gmm_step(
                mats.y, mats.X, mats.Z, W2, mats.ids, mats.individual_slices,
            )
            se2_uncorrected = _safe_sqrt(np.diag(V2_uncorrected))

            # Separately track the model-based variance, A_2 = (X'Z W_2 Z'X)^{-1}.
            # This is the quantity Windmeijer's (2005) correction is built on
            # top of — V_w = A_2 + D A_2 + A_2 D' + D V_1 D' — so we expose
            # it explicitly for diagnostics rather than reconstructing.
            V2_model_based = A2
            se2_model_based = _safe_sqrt(np.diag(V2_model_based))

            if windmeijer_applied:
                V_w = windmeijer_correction(
                    X=mats.X, Z=mats.Z, beta_2=b2,
                    resid_1=resid1, resid_2=resid2,
                    W2=W2, A2=A2, V1=V1,
                    individual_slices=mats.individual_slices,
                )
                V_used = V_w
                se = _safe_sqrt(np.diag(V_w))
            else:
                # Without Windmeijer, we report the cluster-robust sandwich.
                V_used = V2_uncorrected
                se = se2_uncorrected

            beta = b2
            V = V_used
            resid = resid2
            W_used = W2

            result_dict.update({
                "coef_step2": b2,
                "se_step2": se,
                "V_step2": V_used,
                # Both of these are stored so users can reconstruct any
                # comparison without re-running the fit. They are NOT the
                # same thing: V_step2_uncorrected is the cluster-robust
                # sandwich; V_step2_model_based is A_2 = (X'Z W_2 Z'X)^{-1}.
                "V_step2_uncorrected": V2_uncorrected,
                "se_step2_uncorrected": se2_uncorrected,
                "V_step2_model_based": V2_model_based,
                "se_step2_model_based": se2_model_based,
                "resid_step2": resid2,
                "A_step2": A2,
                "W_step2": W2,
                "ZuuZ_step2": ZuuZ_2,
                "windmeijer_applied": windmeijer_applied,
            })

        # ---- diagnostics -------------------------------------------------
        from ._diagnostics import ar_test, hansen_j_test

        m1 = ar_test(
            resid=resid, ids=mats.ids,
            rows_position=mats.rows_position,
            is_diff_row=mats.is_diff_row,
            order=1,
        )
        m2 = ar_test(
            resid=resid, ids=mats.ids,
            rows_position=mats.rows_position,
            is_diff_row=mats.is_diff_row,
            order=2,
        )
        hansen = hansen_j_test(
            X=mats.X, Z=mats.Z, resid=resid,
            n_params=n_reg,
            individual_slices=mats.individual_slices,
            # Always use the optimally-weighted form for Hansen, regardless
            # of whether the user asked for one-step coefficients.
            W=_safe_inv(_cluster_moment_from_slices(mats.Z, resid, mats.individual_slices)),
        )

        # ---- pack & return ----------------------------------------------
        results = GMMResults(
            model=self,
            mats=mats,
            beta=beta,
            se=se,
            V=V,
            resid=resid,
            W=W_used,
            ar1=m1,
            ar2=m2,
            hansen=hansen,
            n_obs=n_obs,
            n_individuals=len(self._individuals),
            n_inst=n_inst,
            n_reg=n_reg,
            extras=result_dict,
        )
        self._fitted = results
        return results

    # ---- subclass hooks --------------------------------------------------

    def _one_step_weight(self, mats: AssembledMatrices) -> np.ndarray:
        """Compute W_1 = (Z' Ω Z)^{-1}.

        For DifferenceGMM, Ω is the block-diagonal AR(1)-MA(1) matrix H
        per individual: 2 on the diagonal, -1 on adjacent off-diagonals.

        For SystemGMM, Ω is block-diagonal with H on the diff-eq rows and
        I on the level-eq rows.

        We construct Z' Ω Z directly without materializing Ω (which would
        be n_obs × n_obs and waste memory for any non-trivial panel).
        """
        Z = mats.Z
        is_diff = mats.is_diff_row
        L = Z.shape[1]
        ZOZ = np.zeros((L, L))

        for sl in mats.individual_slices:
            Zi = Z[sl]
            # Build per-individual Ω_i.
            n_i = Zi.shape[0]
            Omega_i = np.eye(n_i)
            is_diff_i = is_diff[sl]
            diff_rows_local = np.flatnonzero(is_diff_i)
            level_rows_local = np.flatnonzero(~is_diff_i)
            # For the diff-eq sub-block, set 2 on diagonal and -1 on
            # adjacent off-diagonals, but only when the adjacent diff rows
            # come from consecutive original positions (no calendar gap).
            pos_i = mats.rows_position[sl]
            for r in diff_rows_local:
                Omega_i[r, r] = 2.0
            for a, b in zip(diff_rows_local[:-1], diff_rows_local[1:]):
                if pos_i[b] - pos_i[a] == 1:
                    Omega_i[a, b] = -1.0
                    Omega_i[b, a] = -1.0
            # Level rows already have 1.0 on the diagonal (from np.eye).
            ZOZ += Zi.T @ Omega_i @ Zi
        return _safe_inv(ZOZ)


class DifferenceGMM(GMMBase):
    """Arellano-Bond (1991) difference GMM."""
    _equation = "difference"


class SystemGMM(GMMBase):
    """Blundell-Bond (1998) system GMM."""
    _equation = "system"


# ---------------------------------------------------------------------------
# GMM core algebra
# ---------------------------------------------------------------------------


def _gmm_step(y, X, Z, W, ids, individual_slices):
    """Single GMM step given a weight matrix W.

        β̂ = (X' Z W Z' X)^{-1} X' Z W Z' y

    Returns
    -------
    beta : (k,)
    V : (k, k)   cluster-robust sandwich variance
    resid : (n,)
    A : (k, k)   = (X' Z W Z' X)^{-1}, the model-based variance under
                   optimal weighting. Kept for Windmeijer / diagnostics.
    ZuuZ : (L, L)  = sum_i Z_i' u_i u_i' Z_i, the cluster moment of
                   residuals weighted by Z. Reused as the inverse of the
                   two-step weight matrix and for the Hansen statistic.
    """
    ZtX = Z.T @ X
    Zty = Z.T @ y

    XtZ_W = ZtX.T @ W
    bread = XtZ_W @ ZtX
    rhs = XtZ_W @ Zty
    A = _safe_inv(bread)
    beta = A @ rhs

    resid = y - X @ beta

    ZuuZ = _cluster_moment_from_slices(Z, resid, individual_slices)
    meat = XtZ_W @ ZuuZ @ XtZ_W.T
    V = A @ meat @ A

    return beta, V, resid, A, ZuuZ


def _cluster_moment_from_slices(Z: np.ndarray, u: np.ndarray, slices) -> np.ndarray:
    """Compute sum_i (Z_i' u_i)(Z_i' u_i)' across individual slices.

    This is the empirical cluster moment matrix Σ̂ that appears in both the
    cluster-robust sandwich and the two-step weight matrix.
    """
    L = Z.shape[1]
    S = np.zeros((L, L))
    for sl in slices:
        m = Z[sl].T @ u[sl]
        S += np.outer(m, m)
    return S


# ---------------------------------------------------------------------------
# Linear-algebra utilities
# ---------------------------------------------------------------------------


def _safe_inv(A: np.ndarray) -> np.ndarray:
    """Invert A, falling back to pseudo-inverse when near-singular.

    Near-singular moment matrices are *extremely* common when you forget
    ``collapse=True`` and have a tall instrument matrix. This mirrors
    the practical behavior of GMM software that has to gracefully
    handle singular moment covariances rather than crash.
    """
    try:
        return linalg.inv(A)
    except linalg.LinAlgError:
        return linalg.pinv(A)


def _safe_sqrt(v: np.ndarray) -> np.ndarray:
    """sqrt of array, clipping tiny negatives that come from numerical noise."""
    return np.sqrt(np.clip(v, 0.0, None))
