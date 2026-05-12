"""Tests for the formula parser."""

import pytest

from dynapanel import parse_formula
from dynapanel._formula import Term


def test_basic_formula():
    f = parse_formula("y ~ L(1).y + x")
    assert f.dep_var == "y"
    assert f.lhs_lags == [1]
    assert f.rhs_terms == [Term("x", 0)]
    assert f.names == ["L1.y", "x"]


def test_lag_range():
    f = parse_formula("y ~ L(1:2).y + L(0:1).w + k")
    assert f.lhs_lags == [1, 2]
    # contemporaneous regressors plus the L(0).w / L(1).w pair
    rhs_strs = [str(t) for t in f.rhs_terms]
    assert "w" in rhs_strs
    assert "L1.w" in rhs_strs
    assert "k" in rhs_strs


def test_whitespace_tolerance():
    f1 = parse_formula("y ~ L(1).y + x")
    f2 = parse_formula("y~L(1).y+x")
    f3 = parse_formula("  y  ~  L( 1 ).y  +  x  ")
    assert f1.names == f2.names == f3.names


def test_missing_tilde():
    with pytest.raises(ValueError, match="separator"):
        parse_formula("y L(1).y + x")


def test_lhs_must_be_simple_name():
    with pytest.raises(ValueError, match="LHS"):
        parse_formula("L(1).y ~ x")


def test_dep_var_on_rhs_without_lag():
    with pytest.raises(ValueError, match="without a lag"):
        parse_formula("y ~ y + x")


def test_lag_zero_of_dep_var_errors():
    with pytest.raises(ValueError, match="lag 0"):
        parse_formula("y ~ L(0).y + x")


def test_duplicate_terms_error():
    with pytest.raises(ValueError, match="duplicate"):
        parse_formula("y ~ L(1).y + L(1).y + x")


def test_empty_rhs():
    with pytest.raises(ValueError, match="nothing on the RHS"):
        parse_formula("y ~")


def test_bogus_term():
    with pytest.raises(ValueError, match="can't parse"):
        parse_formula("y ~ x + 2*z")


def test_lag_range_inverted():
    with pytest.raises(ValueError, match="less than start"):
        parse_formula("y ~ L(3:1).y")


def test_unique_rhs_vars_order():
    f = parse_formula("y ~ L(1).y + L(2).y + L(0:1).w + k")
    assert f.unique_rhs_vars[0] == "y"
    assert "w" in f.unique_rhs_vars
    assert "k" in f.unique_rhs_vars
