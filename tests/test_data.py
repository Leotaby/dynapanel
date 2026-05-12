"""Tests for the data wrangling layer."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from dynapanel._data import (
    Individual,
    split_into_individuals,
    to_polars,
    validate_panel,
)


def _toy_df():
    return pl.DataFrame({
        "id": [1, 1, 1, 2, 2, 2],
        "year": [2010, 2011, 2012, 2010, 2011, 2012],
        "y": [1.0, 2.0, 3.0, 0.5, 1.5, 2.5],
        "x": [0.1, 0.2, 0.3, -0.1, -0.2, -0.3],
    })


def test_to_polars_accepts_polars():
    df = _toy_df()
    out = to_polars(df)
    assert out is df  # no copy


def test_to_polars_accepts_pandas():
    pdf = _toy_df().to_pandas()
    out = to_polars(pdf)
    assert isinstance(out, pl.DataFrame)
    assert out.shape == pdf.shape


def test_to_polars_rejects_garbage():
    with pytest.raises(TypeError):
        to_polars(42)


def test_validate_panel_passes():
    df = _toy_df()
    out = validate_panel(df, "id", "year", required_vars=["y", "x"])
    assert out.shape == df.shape


def test_validate_panel_missing_col():
    df = _toy_df()
    with pytest.raises(ValueError, match="not in your data"):
        validate_panel(df, "id", "year", required_vars=["z"])


def test_validate_panel_duplicate_idt():
    df = pl.DataFrame({
        "id": [1, 1, 1],
        "year": [2010, 2010, 2011],
        "y": [1.0, 2.0, 3.0],
    })
    with pytest.raises(ValueError, match="uniquely identify"):
        validate_panel(df, "id", "year")


def test_individual_lag_and_diff():
    ind = Individual(
        id=1,
        times=np.array([2010, 2011, 2012, 2013]),
        values={"y": np.array([1.0, 3.0, 6.0, 10.0])},
    )
    np.testing.assert_array_equal(
        ind.diff("y")[1:], np.array([2.0, 3.0, 4.0])
    )
    np.testing.assert_array_equal(
        ind.lag("y", 1)[1:], np.array([1.0, 3.0, 6.0])
    )
    # The leading entries should be NaN
    assert np.isnan(ind.diff("y")[0])
    assert np.isnan(ind.lag("y", 1)[0])


def test_split_into_individuals_preserves_order():
    df = _toy_df()
    inds = split_into_individuals(
        validate_panel(df, "id", "year"), "id", "year",
        needed_vars=["y", "x"],
    )
    assert len(inds) == 2
    np.testing.assert_array_equal(inds[0].times, [2010, 2011, 2012])
    np.testing.assert_array_equal(inds[1].times, [2010, 2011, 2012])
    assert inds[0].id == 1
    assert inds[1].id == 2
