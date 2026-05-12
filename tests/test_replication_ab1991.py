"""Replication test: dynapanel vs Stata's xtabond2 on the AB(1991) panel.

This test activates the moment the user provides two things:

  1. ``tests/fixtures/abdata.parquet``    (run ``scripts/fetch_abdata.py``)
  2. ``tests/fixtures/ab1991_reference.json``  (run
     ``scripts/xtabond2_reference.do`` in Stata, paste the numbers into
     the .template.json, drop the ``.template`` suffix)

Until both files exist the test pytest-skips with a clear message. Once
they exist the test asserts dynapanel's coefficients, Windmeijer SEs,
and Hansen / AR statistics all match Stata's within tolerances declared
in the fixture itself.

This is the v0.2 credibility unlock — when this test goes green in CI,
the experimental banner can come off the README.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from dynapanel import DifferenceGMM, SystemGMM

FIX_DIR = Path(__file__).parent / "fixtures"
ABDATA_PATH = FIX_DIR / "abdata.parquet"
REF_PATH = FIX_DIR / "ab1991_reference.json"


def _load_fixture():
    """Load the reference fixture or skip the test if not ready yet."""
    if not ABDATA_PATH.exists():
        pytest.skip(
            f"abdata not at {ABDATA_PATH}. "
            "Run `python scripts/fetch_abdata.py` to fetch it."
        )
    if not REF_PATH.exists():
        pytest.skip(
            f"Stata reference fixture not at {REF_PATH}. "
            "Run scripts/xtabond2_reference.do in Stata and save numbers "
            "into ab1991_reference.json (template in the same folder)."
        )
    ref = json.loads(REF_PATH.read_text())
    # Validate the fixture isn't still full of TBDs.
    if "TBD" in json.dumps(ref):
        pytest.skip(
            "ab1991_reference.json still has 'TBD' placeholders. "
            "Fill in the Stata-output numbers and re-run."
        )
    return ref


def _check_block(actual_model, ref_block, tols):
    """Assert each captured number in ref_block matches the fitted model."""
    coef_tol = tols["coef_abs"]
    se_tol = tols["se_abs"]
    j_tol = tols["hansen_J_abs"]
    z_tol = tols["ar_z_abs"]

    coef_map = dict(zip(actual_model.coef_names, actual_model.beta))
    se_map = dict(zip(actual_model.coef_names, actual_model.se))

    for stata_name, stata_val in ref_block["coefs"].items():
        py_name = _stata_to_dynapanel(stata_name)
        assert py_name in coef_map, (
            f"missing coef {py_name!r} in dynapanel output; have {list(coef_map)}"
        )
        assert abs(coef_map[py_name] - float(stata_val)) < coef_tol, (
            f"coef {stata_name}: stata={stata_val}, dynapanel={coef_map[py_name]}, "
            f"|diff|={abs(coef_map[py_name] - float(stata_val))}, tol={coef_tol}"
        )

    for stata_name, stata_val in ref_block["se_windmeijer"].items():
        py_name = _stata_to_dynapanel(stata_name)
        assert abs(se_map[py_name] - float(stata_val)) < se_tol, (
            f"SE {stata_name}: stata={stata_val}, dynapanel={se_map[py_name]}, "
            f"tol={se_tol}"
        )

    # Hansen
    assert abs(actual_model.hansen.statistic - float(ref_block["hansen_J"])) < j_tol
    if tols["df_exact"]:
        assert actual_model.hansen.df == int(ref_block["hansen_df"])

    # AR tests
    assert abs(actual_model.ar1.statistic - float(ref_block["ar1_z"])) < z_tol
    assert abs(actual_model.ar2.statistic - float(ref_block["ar2_z"])) < z_tol


def _stata_to_dynapanel(stata_name: str) -> str:
    """Translate Stata's variable naming into dynapanel's coef labels.

    Stata uses ``L.n``, ``L2.n`` (lags) and ``_cons`` (intercept).
    dynapanel uses ``L1.n``, ``L2.n``, ``const``.
    """
    if stata_name == "_cons":
        return "const"
    if stata_name.startswith("L."):
        return f"L1.{stata_name[2:]}"
    return stata_name


@pytest.fixture(scope="module")
def fixture():
    return _load_fixture()


@pytest.fixture(scope="module")
def abdata():
    # _load_fixture already skips if either file is missing, but the
    # `abdata` fixture can be evaluated independently — so re-check here.
    if not ABDATA_PATH.exists():
        pytest.skip(f"abdata not at {ABDATA_PATH}; run scripts/fetch_abdata.py")
    return pl.read_parquet(ABDATA_PATH)


def test_replication_difference_gmm(abdata, fixture):
    """dynapanel DifferenceGMM matches Stata xtabond2's two-step Windmeijer run."""
    model = DifferenceGMM(
        "n ~ L(1:2).n + L(0:1).w + L(0:1).k",
        gmm_instruments=["n", "w", "k"],
        data=abdata,
        panel_var="id",
        time_var="year",
        collapse=True,
    ).fit(steps=2, windmeijer=True)

    _check_block(
        model,
        fixture["difference_gmm_twostep_windmeijer"],
        fixture["tolerances"],
    )


def test_replication_system_gmm(abdata, fixture):
    """dynapanel SystemGMM matches Stata xtabond2's two-step Windmeijer run."""
    model = SystemGMM(
        "n ~ L(1:2).n + L(0:1).w + L(0:1).k",
        gmm_instruments=["n", "w", "k"],
        data=abdata,
        panel_var="id",
        time_var="year",
        collapse=True,
    ).fit(steps=2, windmeijer=True)

    _check_block(
        model,
        fixture["system_gmm_twostep_windmeijer"],
        fixture["tolerances"],
    )
