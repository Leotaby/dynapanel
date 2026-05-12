"""Instrument matrix construction for Difference and System GMM.

This is the most fiddly part of the library and the part that people get
wrong most often when they roll their own implementation. The conventions
we follow here are Roodman's (2009) `xtabond2` defaults:

  * Difference equation rows: positions p in each individual such that
    we can form ΔLHS at p AND ΔRHS for every regressor. With a maximum
    LHS lag of L on the RHS of the formula, the smallest diff-row
    position is p = L + 1 (we need y[p], y[p-1] for ΔLHS and y[p-L-1]
    for ΔL(L).y).

  * Diff-eq instruments (GMM-style) for variable w:
        E[ w_{i,t-s} Δu_{it} ] = 0    for s = minlag, minlag+1, ..., maxlag
        (defaults: minlag=2, maxlag=∞)

  * Diff-eq instruments (IV-style, exogenous) for variable z:
        E[ Δz_{it} Δu_{it} ] = 0
    These appear as Δz at each diff-eq row.

  * Level equation rows (System GMM only): same positions as diff eq.
    Instrument for w is Δw_{i,t-1}. Single moment per (i, t) when
    collapsed, one moment per t when not collapsed.

  * Constant in instruments: we add a column of 1's to the level-eq
    block in System GMM (this is what `xtabond2` does — it identifies
    the constant from the level equation, where the within-i mean is
    actual information).

  * Collapsing (Roodman 2009 section "Collapsing instruments"):
    instead of one column per (t, s) moment, sum across t for each s.
    This is *strongly* recommended in practice — without it, the
    instrument count grows as O(T^2) and the two-step weight matrix
    becomes a numerical nightmare.

Notation we use in the code:
    p   = position index inside an individual's sorted time series (0-based)
    t   = calendar time = individual.times[p]
    s   = lag (in positions), s >= 1
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._data import Individual


@dataclass
class EquationBlock:
    """Per-individual row block for one equation (difference or level).

    Attributes
    ----------
    y : (n_rows,) array
        LHS for these rows.
    X : (n_rows, k) array
        Regressors for these rows (same column order as the formula).
    Z_cols : dict[tuple, np.ndarray]
        Map from a moment-identifier (variable, kind, ...) to the column
        values for this individual at these rows. We let the outer
        assembler decide column ordering globally.
    rows_calendar_time : (n_rows,) array
        Calendar t at each row. Used for AR tests, level-row alignment,
        and (in non-collapsed mode) for indexing the (t, s) moments.
    rows_position : (n_rows,) array
        Position within the individual's sorted data. Used to align
        diff-eq and level-eq rows when stacking.
    """
    y: np.ndarray
    X: np.ndarray
    Z_cols: dict[tuple, np.ndarray]
    rows_calendar_time: np.ndarray
    rows_position: np.ndarray


def _xrow_levels(ind: Individual, term, p: int) -> float:
    """Regressor in level form for row at position p."""
    src = p - term.lag
    if src < 0:
        return np.nan
    return ind.values[term.var][src]


def _xrow_diff(ind: Individual, term, p: int) -> float:
    """Regressor for the differenced equation at position p.

    Δ(L(k).var)_t = var[p-k] - var[p-k-1].
    """
    a = p - term.lag
    b = p - term.lag - 1
    if a < 0 or b < 0:
        return np.nan
    return ind.values[term.var][a] - ind.values[term.var][b]


# ---------------------------------------------------------------------------
# Public construction
# ---------------------------------------------------------------------------

def build_blocks_for_individual(
    ind: Individual,
    dep_var: str,
    rhs_terms,
    gmm_instruments: list[str],
    iv_instruments: list[str],
    gmm_lags: dict[str, tuple[int, int | None]],
    equation: str,                  # "difference" or "system"
    collapse: bool,
) -> tuple[EquationBlock, EquationBlock | None]:
    """Construct the diff-eq block (and level-eq block if system) for one individual.

    Returns (diff_block, level_block_or_None). The caller stacks blocks
    across individuals and aligns columns globally.

    ``gmm_lags`` maps each variable in ``gmm_instruments`` to a
    ``(minlag, maxlag)`` tuple specifying the lag depth used to build
    GMM-style instruments for that variable. ``maxlag=None`` means
    "all available lags".
    """
    # Figure out the largest LHS lag that appears on the RHS. With
    # `y ~ L(1).y + ...`, that's 1; with `y ~ L(2).y`, that's 2; etc.
    # The first diff-eq row exists at position p_min = max_lhs_lag + 1.
    lhs_lags = [t.lag for t in rhs_terms if t.var == dep_var]
    max_lhs_lag = max(lhs_lags) if lhs_lags else 0
    max_rhs_lag = max((t.lag for t in rhs_terms if t.var != dep_var), default=0)
    # For ΔL(k).x to exist at position p, need p - k - 1 >= 0.
    p_min_for_rhs = max_rhs_lag + 1
    p_min_diff = max(max_lhs_lag + 1, p_min_for_rhs)

    T = ind.n_periods
    diff_positions = np.arange(p_min_diff, T)

    diff_block = _build_diff_block(
        ind, dep_var, rhs_terms, gmm_instruments, iv_instruments,
        gmm_lags, collapse, diff_positions,
    )

    level_block: EquationBlock | None = None
    if equation == "system":
        # Level equation rows are at the same positions as diff-eq rows.
        # We need Δw_{i,t-1} = w[p-1] - w[p-2] as instrument for L1.w, so
        # we need p - 2 >= 0. That's already guaranteed by p_min_diff >= 1.
        level_block = _build_level_block(
            ind, dep_var, rhs_terms, gmm_instruments, iv_instruments,
            collapse, diff_positions,
        )

    return diff_block, level_block


def _build_diff_block(
    ind: Individual,
    dep_var: str,
    rhs_terms,
    gmm_instruments: list[str],
    iv_instruments: list[str],
    gmm_lags: dict[str, tuple[int, int | None]],
    collapse: bool,
    positions: np.ndarray,
) -> EquationBlock:
    n_rows = len(positions)
    n_reg = len(rhs_terms)

    y = np.empty(n_rows, dtype=float)
    X = np.empty((n_rows, n_reg), dtype=float)

    y_var = ind.values[dep_var]
    for r, p in enumerate(positions):
        y[r] = y_var[p] - y_var[p - 1]
        for j, term in enumerate(rhs_terms):
            X[r, j] = _xrow_diff(ind, term, p)

    Z_cols: dict[tuple, np.ndarray] = {}

    # GMM-style instruments: w_{i,t-s} for s in [minlag_w, maxlag_w]
    # where the lag bounds are looked up per-variable in gmm_lags.
    for w in gmm_instruments:
        w_arr = ind.values[w]
        minlag, maxlag = gmm_lags[w]
        # Largest valid lag s for this individual: at the deepest row,
        # we can go down to s = positions[-1] (which gives p-s = 0).
        s_max_individual = int(positions[-1])
        s_cap = s_max_individual if maxlag is None else min(s_max_individual, maxlag)
        if s_cap < minlag:
            continue  # nothing to add

        if collapse:
            # One column per s, key = (block_id, "gmm", w, s).
            # We use block_id="diff" so diff and level cols never collide.
            for s in range(minlag, s_cap + 1):
                col = np.zeros(n_rows, dtype=float)
                for r, p in enumerate(positions):
                    src = p - s
                    if src >= 0:
                        col[r] = w_arr[src]
                Z_cols[("diff", "gmm", w, s)] = col
        else:
            # Standard "GMM-style" block-diagonal: one column per (calendar t, s).
            # The column has the lagged value at the row where the moment
            # applies, and zero everywhere else for this individual.
            for r, p in enumerate(positions):
                t = int(ind.times[p])
                for s in range(minlag, min(s_cap, p) + 1):
                    src = p - s
                    if src < 0:
                        continue
                    key = ("diff", "gmm", w, t, s)
                    col = Z_cols.get(key)
                    if col is None:
                        col = np.zeros(n_rows, dtype=float)
                        Z_cols[key] = col
                    col[r] = w_arr[src]

    # IV-style (exogenous) instruments enter as their own differences.
    # One column per IV variable; same column shape across individuals.
    for z in iv_instruments:
        z_arr = ind.values[z]
        col = np.empty(n_rows, dtype=float)
        for r, p in enumerate(positions):
            if p - 1 >= 0:
                col[r] = z_arr[p] - z_arr[p - 1]
            else:
                col[r] = np.nan
        Z_cols[("diff", "iv", z)] = col

    return EquationBlock(
        y=y, X=X, Z_cols=Z_cols,
        rows_calendar_time=ind.times[positions],
        rows_position=positions.copy(),
    )


def _build_level_block(
    ind: Individual,
    dep_var: str,
    rhs_terms,
    gmm_instruments: list[str],
    iv_instruments: list[str],
    collapse: bool,
    positions: np.ndarray,
) -> EquationBlock:
    """Level-equation block for System GMM.

    Convention: one moment per (i, t) of the form Δw_{i,t-1}, which is
    the standard Blundell-Bond single-lag-of-differences instrument set.
    Add more lags (Δw_{i,t-2}, etc.) later if we ever care.
    """
    n_rows = len(positions)
    n_reg = len(rhs_terms)

    y = np.empty(n_rows, dtype=float)
    X = np.empty((n_rows, n_reg), dtype=float)

    y_var = ind.values[dep_var]
    for r, p in enumerate(positions):
        y[r] = y_var[p]
        for j, term in enumerate(rhs_terms):
            X[r, j] = _xrow_levels(ind, term, p)

    Z_cols: dict[tuple, np.ndarray] = {}

    for w in gmm_instruments:
        w_arr = ind.values[w]
        if collapse:
            # Single column: Δw_{i,t-1} at each row.
            col = np.empty(n_rows, dtype=float)
            for r, p in enumerate(positions):
                # Δw_{i,t-1} = w[p-1] - w[p-2]
                if p - 2 >= 0:
                    col[r] = w_arr[p - 1] - w_arr[p - 2]
                else:
                    col[r] = np.nan
            Z_cols[("level", "gmm", w, 1)] = col
        else:
            # Standard: one column per (calendar t).
            for r, p in enumerate(positions):
                t = int(ind.times[p])
                if p - 2 < 0:
                    continue
                key = ("level", "gmm", w, t)
                col = Z_cols.get(key)
                if col is None:
                    col = np.zeros(n_rows, dtype=float)
                    Z_cols[key] = col
                col[r] = w_arr[p - 1] - w_arr[p - 2]

    for z in iv_instruments:
        z_arr = ind.values[z]
        col = z_arr[positions].astype(float)
        Z_cols[("level", "iv", z)] = col

    # Constant in the level-eq instrument matrix. This is the standard
    # System-GMM convention: the intercept gets identified from the level
    # equation, where the within-i mean carries actual information. The
    # diff equation sweeps the FE out, so no constant is needed there.
    Z_cols[("level", "const")] = np.ones(n_rows, dtype=float)

    return EquationBlock(
        y=y, X=X, Z_cols=Z_cols,
        rows_calendar_time=ind.times[positions],
        rows_position=positions.copy(),
    )


# ---------------------------------------------------------------------------
# Cross-individual assembly
# ---------------------------------------------------------------------------

@dataclass
class AssembledMatrices:
    y: np.ndarray              # (n_total_rows,)
    X: np.ndarray              # (n_total_rows, k)
    Z: np.ndarray              # (n_total_rows, L)
    ids: np.ndarray            # (n_total_rows,) cluster ids (panel ids)
    rows_calendar_time: np.ndarray
    rows_position: np.ndarray
    is_diff_row: np.ndarray    # bool mask: True for diff-eq rows, False for level
    instrument_names: list[str]
    # bookkeeping per individual: where each individual's rows live
    individual_slices: list[slice]


def assemble(
    individuals: list[Individual],
    blocks_diff: list[EquationBlock],
    blocks_level: list[EquationBlock | None],
    include_level: bool,
    add_const_to_X: bool,
) -> AssembledMatrices:
    """Stack per-individual blocks and align instrument columns globally.

    Drop rows with any NaN in (y, X). This implicitly handles the leading
    rows we couldn't construct (e.g., position p=0 in a diff equation).
    Also drop instrument columns that are entirely zero/NaN — they happen
    when no individual ever contributes to that moment.
    """
    # Build a global ordering of instrument keys. We sort for reproducibility.
    all_keys: set = set()
    for bd in blocks_diff:
        all_keys.update(bd.Z_cols.keys())
    if include_level:
        for bl in blocks_level:
            if bl is not None:
                all_keys.update(bl.Z_cols.keys())

    # Sort keys with a stable ordering: equation block first (diff before level),
    # then by kind (gmm before iv before const), then by everything else.
    def _key_sort(k):
        eq = 0 if k[0] == "diff" else 1
        kind = {"gmm": 0, "iv": 1, "const": 2}.get(k[1], 3)
        rest = tuple(str(x) for x in k[2:])
        return (eq, kind, rest)

    key_order = sorted(all_keys, key=_key_sort)
    col_index = {k: i for i, k in enumerate(key_order)}
    n_inst = len(key_order)

    # Stack rows
    y_chunks: list[np.ndarray] = []
    X_chunks: list[np.ndarray] = []
    Z_chunks: list[np.ndarray] = []
    id_chunks: list[np.ndarray] = []
    t_chunks: list[np.ndarray] = []
    p_chunks: list[np.ndarray] = []
    is_diff_chunks: list[np.ndarray] = []
    individual_slices: list[slice] = []
    row_cursor = 0

    for ind, bd, bl in zip(individuals, blocks_diff, blocks_level):
        # diff block
        rows_d = bd.y.shape[0]
        y_chunks.append(bd.y)
        X_chunks.append(bd.X)
        Zd = np.zeros((rows_d, n_inst))
        for k, col in bd.Z_cols.items():
            j = col_index[k]
            Zd[:, j] = col
        Z_chunks.append(Zd)
        id_chunks.append(np.full(rows_d, ind.id))
        t_chunks.append(bd.rows_calendar_time)
        p_chunks.append(bd.rows_position)
        is_diff_chunks.append(np.ones(rows_d, dtype=bool))

        rows_used = rows_d

        if include_level and bl is not None:
            rows_l = bl.y.shape[0]
            y_chunks.append(bl.y)
            X_chunks.append(bl.X)
            Zl = np.zeros((rows_l, n_inst))
            for k, col in bl.Z_cols.items():
                j = col_index[k]
                Zl[:, j] = col
            Z_chunks.append(Zl)
            id_chunks.append(np.full(rows_l, ind.id))
            t_chunks.append(bl.rows_calendar_time)
            p_chunks.append(bl.rows_position)
            is_diff_chunks.append(np.zeros(rows_l, dtype=bool))
            rows_used += rows_l

        individual_slices.append(slice(row_cursor, row_cursor + rows_used))
        row_cursor += rows_used

    y = np.concatenate(y_chunks)
    X = np.vstack(X_chunks)
    Z = np.vstack(Z_chunks)
    ids = np.concatenate(id_chunks)
    rows_calendar_time = np.concatenate(t_chunks)
    rows_position = np.concatenate(p_chunks)
    is_diff_row = np.concatenate(is_diff_chunks)

    # Drop rows that have NaN anywhere in y or X. Z NaNs are coerced to 0 —
    # a NaN in an instrument means "no information from this row for that
    # moment", which is exactly what zero means in the GMM moment sum.
    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    Z = np.where(np.isnan(Z), 0.0, Z)

    if not valid.all():
        # We need to recompute the individual slices after dropping rows.
        keep_idx = np.flatnonzero(valid)
        y = y[keep_idx]
        X = X[keep_idx]
        Z = Z[keep_idx]
        ids = ids[keep_idx]
        rows_calendar_time = rows_calendar_time[keep_idx]
        rows_position = rows_position[keep_idx]
        is_diff_row = is_diff_row[keep_idx]

        # Rebuild individual slices.
        new_slices = []
        cursor = 0
        unique_ids, first_idx = np.unique(ids, return_index=True)
        # iterate in encounter order
        order = np.argsort(first_idx)
        for u in unique_ids[order]:
            mask = ids == u
            count = int(mask.sum())
            new_slices.append(slice(cursor, cursor + count))
            cursor += count
        individual_slices = new_slices

    # Drop all-zero instrument columns (can happen if e.g. minlag is so high
    # that no individual has enough periods to contribute to that column).
    col_nonzero = np.any(Z != 0, axis=0)
    if not col_nonzero.all():
        Z = Z[:, col_nonzero]
        key_order = [k for k, keep in zip(key_order, col_nonzero) if keep]

    # Optional constant in X (intercept). We sweep it out in the diff eq,
    # so it's only meaningful if level rows exist OR the user is doing
    # something weird with `constant=True` on a pure-diff model.
    if add_const_to_X and include_level:
        const_col = np.where(is_diff_row, 0.0, 1.0).reshape(-1, 1)
        X = np.hstack([X, const_col])

    instrument_names = [_pretty_key(k) for k in key_order]

    return AssembledMatrices(
        y=y, X=X, Z=Z, ids=ids,
        rows_calendar_time=rows_calendar_time,
        rows_position=rows_position,
        is_diff_row=is_diff_row,
        instrument_names=instrument_names,
        individual_slices=individual_slices,
    )


def _pretty_key(k: tuple) -> str:
    eq, kind, *rest = k
    if kind == "gmm":
        if eq == "diff":
            if len(rest) == 2:
                w, s = rest
                return f"gmm({w}, L{s}, diff)"
            else:
                w, t, s = rest
                return f"gmm({w}, t={t}, L{s}, diff)"
        else:
            if len(rest) == 2:
                w, _ = rest
                return f"gmm(Δ{w}, level)"
            else:
                w, t = rest
                return f"gmm(Δ{w}, t={t}, level)"
    if kind == "iv":
        if eq == "diff":
            return f"iv(Δ{rest[0]})"
        return f"iv({rest[0]})"
    if kind == "const":
        return "const"
    return str(k)
