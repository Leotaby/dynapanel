"""Panel data wrangling: validation, sorting, per-individual slicing.

We prefer polars internally because lag and diff construction on a
panel is much faster with polars window functions than with
pandas groupby+shift, and the typical dynamic-panel dataset is
small enough that the conversion overhead is negligible.

We do NOT silently re-sort the user's data in-place. We sort a copy,
require a panel id and a time index, and we error out loudly if the
panel index doesn't uniquely identify rows. Better to fail here with
a clear message than to get a KeyError fifty lines deep in the GMM
moment-matrix builder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl


def to_polars(data) -> pl.DataFrame:
    """Accept pandas, polars, dict-of-arrays, list-of-dicts. Return polars."""
    if isinstance(data, pl.DataFrame):
        return data
    if isinstance(data, pd.DataFrame):
        # If the user left a MultiIndex on the frame, lift it back into columns.
        if isinstance(data.index, pd.MultiIndex) or (
            data.index.name is not None and data.index.name not in data.columns
        ):
            data = data.reset_index()
        return pl.from_pandas(data)
    if isinstance(data, dict):
        return pl.DataFrame(data)
    if isinstance(data, list):
        return pl.DataFrame(data)
    raise TypeError(
        f"data must be a polars or pandas DataFrame (or a dict / list of dicts), "
        f"got {type(data).__name__}."
    )


def validate_panel(
    df: pl.DataFrame,
    panel_var: str,
    time_var: str,
    required_vars: list[str] | None = None,
) -> pl.DataFrame:
    """Check the panel index and return a copy sorted by (i, t).

    What we check:
      * panel_var and time_var exist
      * all required_vars exist
      * (panel_var, time_var) uniquely identifies rows
      * time_var is integer-ish (we can coerce float-like to int safely)

    What we DON'T check:
      * gaps in t (dynapanel handles gaps fine; we just won't manufacture
        instruments out of nothing — a missing y_{i,t-2} means that
        moment condition simply doesn't contribute for that row)
      * NaNs in regressors (we drop them in the per-individual slice;
        if the user has NaN-mountains, that's an upstream cleaning issue)
    """
    missing = [v for v in [panel_var, time_var] + list(required_vars or [])
               if v not in df.columns]
    if missing:
        raise ValueError(
            f"the following columns are not in your data: {missing}.\n"
            f"Available columns: {df.columns}"
        )

    n_unique = df.select([panel_var, time_var]).unique().height
    if n_unique != df.height:
        n_dups = df.height - n_unique
        raise ValueError(
            f"({panel_var!r}, {time_var!r}) does not uniquely identify rows: "
            f"found {n_dups} duplicate (id, time) pair(s). Dynamic panel "
            f"estimators require a unique panel index."
        )

    t_dtype = df.schema[time_var]
    if not t_dtype.is_integer():
        try:
            df = df.with_columns(pl.col(time_var).cast(pl.Int64))
        except Exception as e:
            raise ValueError(
                f"time_var {time_var!r} must be an integer (year, quarter index, etc.). "
                f"Got dtype {t_dtype}. Cast it yourself before passing it in — we "
                f"won't guess your calendar convention."
            ) from e

    return df.sort([panel_var, time_var])


@dataclass
class Individual:
    """The slice of the panel for a single unit i, packed into numpy.

    times is the integer time index. values is a dict mapping variable
    name to a 1-D numpy array aligned with `times`. The dict is the
    superset of every variable referenced anywhere in the formula or
    the instrument lists.
    """
    id: object
    times: np.ndarray
    values: dict[str, np.ndarray]

    @property
    def n_periods(self) -> int:
        return len(self.times)

    def lag(self, var: str, k: int) -> np.ndarray:
        """Return the k-lagged series for `var`, with NaN where unavailable.

        Time-aware: we shift by k *positions in the sorted-times array*, NOT
        by k literal time units. If the data has gaps (e.g., a missing year),
        L(1) means "the previous observed period", not "exactly one year ago".

        TODO: add a strict mode that requires consecutive time, since for
        some moment conditions the difference between these is meaningful.
        """
        if k == 0:
            return self.values[var]
        v = self.values[var]
        out = np.full_like(v, np.nan, dtype=float)
        if k > 0:
            out[k:] = v[:-k]
        else:
            out[:k] = v[-k:]
        return out

    def diff(self, var: str) -> np.ndarray:
        """First difference of `var`: v_t - v_{t-1}, NaN at the start."""
        v = self.values[var]
        out = np.full_like(v, np.nan, dtype=float)
        out[1:] = v[1:] - v[:-1]
        return out


def split_into_individuals(
    df: pl.DataFrame,
    panel_var: str,
    time_var: str,
    needed_vars: list[str],
) -> list[Individual]:
    """Slice a validated panel df into a list of Individual objects.

    We materialize numpy arrays here, once, so the estimator can iterate
    over individuals without paying polars overhead per row.
    """
    out: list[Individual] = []
    # group_by + maintain_order to preserve the sort we already did
    for key, group in df.group_by(panel_var, maintain_order=True):
        sub = group.sort(time_var)
        times = sub[time_var].to_numpy()
        values = {v: sub[v].to_numpy().astype(float, copy=False) for v in needed_vars}
        id_val = key[0] if isinstance(key, tuple) else key
        out.append(Individual(id=id_val, times=times, values=values))
    return out
