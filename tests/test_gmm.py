"""Integration tests: simulate, fit, sanity-check.

These tests aren't tight numerical checks against another implementation
(that's what the replication notebooks are for) — they're checks that
the estimator recovers the true coefficients to within a reasonable
tolerance for sample sizes where it should, and that the diagnostics
fire correctly.
"""

import numpy as np
import pytest

from dynapanel import DifferenceGMM, SystemGMM, simulate_dynamic_panel

# ---------------------------------------------------------------------------
# Sanity: basic fit doesn't crash
# ---------------------------------------------------------------------------

def test_system_gmm_runs_end_to_end():
    df = simulate_dynamic_panel(n=150, t=8, alpha=0.5, beta=1.0, seed=1)
    model = SystemGMM(
        "y ~ L(1).y + x",
        gmm_instruments=["y", "x"],
        data=df,
        panel_var="id",
        time_var="t",
        collapse=True,
    ).fit(steps=2, windmeijer=True)

    assert model.beta.shape == (3,)  # L1.y, x, const
    assert np.all(np.isfinite(model.beta))
    assert np.all(model.se >= 0)
    assert model.n_individuals == 150


def test_difference_gmm_runs_end_to_end():
    df = simulate_dynamic_panel(n=150, t=8, alpha=0.5, beta=1.0, seed=2)
    model = DifferenceGMM(
        "y ~ L(1).y + x",
        gmm_instruments=["y", "x"],
        data=df,
        panel_var="id",
        time_var="t",
        collapse=True,
    ).fit(steps=2, windmeijer=True)

    assert model.beta.shape == (2,)  # L1.y, x (no const in pure diff)
    assert np.all(np.isfinite(model.beta))


# ---------------------------------------------------------------------------
# Numerical: does System GMM recover the true coefficients reasonably?
# ---------------------------------------------------------------------------

def test_system_gmm_recovers_true_coefs():
    # System GMM should recover (α, β) within a few SDs in this sample.
    # We use a fairly persistent x and a generous N to keep the test stable.
    df = simulate_dynamic_panel(
        n=400, t=10, alpha=0.6, beta=0.8,
        rho_x=0.6, x_corr_eta=0.3, seed=123,
    )
    model = SystemGMM(
        "y ~ L(1).y + x",
        gmm_instruments=["y", "x"],
        data=df,
        panel_var="id",
        time_var="t",
        collapse=True,
    ).fit(steps=2, windmeijer=True)

    coef = dict(zip(model.coef_names, model.beta))
    # Loose tolerances — this is a Monte Carlo, not a regression test.
    assert abs(coef["L1.y"] - 0.6) < 0.12, f"L1.y off: {coef['L1.y']}"
    assert abs(coef["x"]    - 0.8) < 0.12, f"x off:    {coef['x']}"


def test_difference_gmm_recovers_true_coefs():
    # Difference GMM is biased downward for α in small T with persistent
    # data (the weak-instrument problem that motivated System GMM in the
    # first place). We pick a setting where DiffGMM is still defensible.
    df = simulate_dynamic_panel(
        n=400, t=10, alpha=0.4, beta=0.8,
        rho_x=0.4, x_corr_eta=0.2, seed=456,
    )
    model = DifferenceGMM(
        "y ~ L(1).y + x",
        gmm_instruments=["y", "x"],
        data=df,
        panel_var="id",
        time_var="t",
        collapse=True,
    ).fit(steps=2, windmeijer=True)

    coef = dict(zip(model.coef_names, model.beta))
    assert abs(coef["L1.y"] - 0.4) < 0.15
    assert abs(coef["x"]    - 0.8) < 0.15


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def test_ar2_should_not_reject_under_correct_dgp():
    df = simulate_dynamic_panel(n=300, t=8, alpha=0.5, beta=1.0, seed=7)
    model = SystemGMM(
        "y ~ L(1).y + x",
        gmm_instruments=["y", "x"],
        data=df, panel_var="id", time_var="t",
        collapse=True,
    ).fit()
    # Under the correctly-specified DGP, m2 should not reject (p > some
    # liberal threshold). One simulation; we just check it's not absurdly
    # rejecting.
    assert model.ar2.pvalue > 0.01, f"AR(2) p = {model.ar2.pvalue}"


def test_ar1_typically_rejects():
    df = simulate_dynamic_panel(n=300, t=8, alpha=0.5, beta=1.0, seed=8)
    model = SystemGMM(
        "y ~ L(1).y + x",
        gmm_instruments=["y", "x"],
        data=df, panel_var="id", time_var="t",
        collapse=True,
    ).fit()
    # m1 should reject under the model (Δu_t and Δu_{t-1} share u_{t-1}).
    # We test "small p-value" with a loose threshold for stability.
    assert model.ar1.pvalue < 0.20, f"AR(1) p = {model.ar1.pvalue}"


def test_hansen_does_not_reject_under_correct_dgp():
    # x_corr_eta=0 makes x strictly exogenous so all moment conditions hold
    # cleanly. With x_corr_eta>0, even though Δx_{t-1} is asymptotically
    # uncorrelated with η_i, the finite-sample correlation is enough to
    # make a single Hansen draw rejection-prone — the test would only
    # work in expectation across many simulations.
    df = simulate_dynamic_panel(n=300, t=8, alpha=0.5, beta=1.0,
                                x_corr_eta=0.0, seed=10)
    model = SystemGMM(
        "y ~ L(1).y + x",
        gmm_instruments=["y", "x"],
        data=df, panel_var="id", time_var="t",
        collapse=True,
    ).fit(steps=2)
    assert model.hansen.pvalue > 0.05, f"Hansen p = {model.hansen.pvalue}"


# ---------------------------------------------------------------------------
# Edge cases / API checks
# ---------------------------------------------------------------------------

def test_orthogonal_deviations_is_not_implemented_yet():
    df = simulate_dynamic_panel(n=50, t=6, seed=10)
    with pytest.raises(NotImplementedError):
        SystemGMM(
            "y ~ L(1).y + x",
            data=df, panel_var="id", time_var="t",
            orthogonal_deviations=True,
        )


def test_summary_text_fallback_runs(capsys):
    df = simulate_dynamic_panel(n=100, t=8, seed=11)
    model = SystemGMM(
        "y ~ L(1).y + x",
        data=df, panel_var="id", time_var="t",
        collapse=True,
    ).fit()
    # Force the text path by deliberately calling the private fallback.
    model._print_text_summary()
    captured = capsys.readouterr()
    assert "step" in captured.out.lower()
    assert "L1.y" in captured.out
    assert "Hansen" in captured.out


def test_pandas_input_works():
    df = simulate_dynamic_panel(n=80, t=6, seed=12).to_pandas()
    model = SystemGMM(
        "y ~ L(1).y + x",
        data=df, panel_var="id", time_var="t",
        collapse=True,
    ).fit()
    assert np.all(np.isfinite(model.beta))


def test_one_step_then_two_step_agree_qualitatively():
    df = simulate_dynamic_panel(n=200, t=8, alpha=0.5, beta=1.0, seed=13)
    m1 = SystemGMM("y ~ L(1).y + x", data=df, panel_var="id", time_var="t",
                   collapse=True).fit(steps=1)
    m2 = SystemGMM("y ~ L(1).y + x", data=df, panel_var="id", time_var="t",
                   collapse=True).fit(steps=2)
    # Coefs should be close-ish (one- and two-step have different finite-sample bias).
    diff_alpha = abs(m1.coef["L1.y"] - m2.coef["L1.y"])
    diff_beta  = abs(m1.coef["x"]    - m2.coef["x"])
    assert diff_alpha < 0.10
    assert diff_beta  < 0.10


def test_underidentified_raises():
    df = simulate_dynamic_panel(n=30, t=4, seed=14)
    # Pathologically high minlag should leave no instruments at all.
    with pytest.raises((ValueError, IndexError)):
        SystemGMM(
            "y ~ L(1).y + x",
            data=df, panel_var="id", time_var="t",
            gmm_instruments=["y"], minlag=10, collapse=True,
        ).fit()


# ---------------------------------------------------------------------------
# Per-variable lag specification (endogenous vs predetermined)
# ---------------------------------------------------------------------------

def test_gmm_lags_accepts_per_variable_spec():
    df = simulate_dynamic_panel(n=200, t=8, alpha=0.5, beta=1.0,
                                x_corr_eta=0.0, seed=20)
    # x is predetermined: minlag=1 should be valid.
    model = SystemGMM(
        "y ~ L(1).y + x",
        gmm_instruments=["y", "x"],
        gmm_lags={"y": (2, None), "x": (1, 3)},
        data=df, panel_var="id", time_var="t",
        collapse=True,
    ).fit(steps=2, windmeijer=True)
    assert np.all(np.isfinite(model.beta))
    # Sanity: instrument count is finite and reasonable
    assert 0 < model.n_inst < 50


def test_gmm_lags_rejects_unknown_variable():
    df = simulate_dynamic_panel(n=50, t=6, seed=21)
    with pytest.raises(ValueError, match="not in gmm_instruments"):
        SystemGMM(
            "y ~ L(1).y + x",
            gmm_instruments=["y"],
            gmm_lags={"q_nonexistent": (1, 4)},
            data=df, panel_var="id", time_var="t",
        )


def test_gmm_lags_rejects_bad_tuple():
    df = simulate_dynamic_panel(n=50, t=6, seed=22)
    with pytest.raises(ValueError):
        SystemGMM(
            "y ~ L(1).y + x",
            gmm_instruments=["y"],
            gmm_lags={"y": (0, 4)},  # minlag < 1
            data=df, panel_var="id", time_var="t",
        )
    with pytest.raises(ValueError):
        SystemGMM(
            "y ~ L(1).y + x",
            gmm_instruments=["y"],
            gmm_lags={"y": (5, 2)},  # maxlag < minlag
            data=df, panel_var="id", time_var="t",
        )


def test_predetermined_uses_more_instruments_than_endogenous():
    """If we set minlag=1 (predetermined) we get more instrument columns
    than minlag=2 (endogenous). Sanity check on the lag-handling path."""
    df = simulate_dynamic_panel(n=200, t=8, seed=23)
    m_endo = SystemGMM(
        "y ~ L(1).y + x",
        gmm_instruments=["y", "x"],
        gmm_lags={"y": (2, None), "x": (2, None)},
        data=df, panel_var="id", time_var="t",
        collapse=True,
    ).fit()
    m_pred = SystemGMM(
        "y ~ L(1).y + x",
        gmm_instruments=["y", "x"],
        gmm_lags={"y": (2, None), "x": (1, None)},
        data=df, panel_var="id", time_var="t",
        collapse=True,
    ).fit()
    assert m_pred.n_inst > m_endo.n_inst, (
        f"predetermined ({m_pred.n_inst}) should yield more instruments "
        f"than endogenous ({m_endo.n_inst})"
    )


# ---------------------------------------------------------------------------
# robust= handling
# ---------------------------------------------------------------------------

def test_robust_cluster_aliases_all_work():
    df = simulate_dynamic_panel(n=100, t=6, seed=30)
    runs = []
    for r in ("cluster", "i", "id", "panel"):
        m = SystemGMM(
            "y ~ L(1).y + x",
            data=df, panel_var="id", time_var="t",
            collapse=True,
        ).fit(steps=2, robust=r)
        runs.append(m.beta)
    # All aliases should produce identical estimates.
    for b in runs[1:]:
        np.testing.assert_array_almost_equal(b, runs[0])


def test_robust_twoway_is_rejected():
    """In v0.1 we raise instead of silently falling back, so a user who
    sets twoway can't walk away thinking they got a twoway variance."""
    df = simulate_dynamic_panel(n=100, t=6, seed=31)
    with pytest.raises(NotImplementedError, match="twoway"):
        SystemGMM(
            "y ~ L(1).y + x",
            data=df, panel_var="id", time_var="t",
            collapse=True,
        ).fit(steps=2, robust="twoway")


def test_robust_unknown_raises():
    df = simulate_dynamic_panel(n=50, t=6, seed=32)
    with pytest.raises(ValueError, match="not supported"):
        SystemGMM(
            "y ~ L(1).y + x",
            data=df, panel_var="id", time_var="t",
        ).fit(robust="bootstrap")


# ---------------------------------------------------------------------------
# Windmeijer sanity checks
# ---------------------------------------------------------------------------

def test_windmeijer_se_differs_from_model_based():
    """The Windmeijer correction is built on top of A_2 = (X'Z W_2 Z'X)^{-1},
    the model-based two-step variance. The corrected SE should be
    materially different from sqrt(diag(A_2)) — otherwise the correction
    isn't being applied. (Magnitude is a v0.2-cross-validation question.)"""
    df = simulate_dynamic_panel(n=150, t=6, alpha=0.5, beta=1.0,
                                x_corr_eta=0.0, seed=40)
    m = SystemGMM(
        "y ~ L(1).y + x",
        gmm_instruments=["y", "x"],
        data=df, panel_var="id", time_var="t",
        collapse=True,
    ).fit(steps=2, windmeijer=True)

    se_wm = m.se
    se_model = m.extras["se_step2_model_based"]
    diff = np.abs(se_wm - se_model)
    assert np.all(diff > 1e-6), (
        f"Windmeijer SEs ({se_wm}) are identical to model-based ({se_model}) — "
        f"correction may not be applied"
    )


def test_windmeijer_se_is_typically_larger_than_model_based():
    """In standard settings, the uncorrected model-based two-step SEs are
    downward biased in finite samples, so Windmeijer's correction usually
    inflates them. We test on average across a few seeds — single draws
    can go either way."""
    n_larger = 0
    n_total = 5
    for seed in range(50, 50 + n_total):
        df = simulate_dynamic_panel(n=120, t=6, alpha=0.5, beta=1.0,
                                    x_corr_eta=0.0, seed=seed)
        m = SystemGMM(
            "y ~ L(1).y + x",
            gmm_instruments=["y", "x"],
            data=df, panel_var="id", time_var="t",
            collapse=True,
        ).fit(steps=2, windmeijer=True)
        idx_lag = m.coef_names.index("L1.y")
        if m.se[idx_lag] > m.extras["se_step2_model_based"][idx_lag]:
            n_larger += 1
    assert n_larger >= 3, (
        f"Windmeijer SE was larger than model-based in {n_larger}/{n_total} "
        f"draws; expected at least 3/5. Possible bug in the correction."
    )


def test_uncorrected_se_storage_is_consistent():
    """A safety net for the bug we fixed in v0.1.1: V_step2_uncorrected
    and se_step2_uncorrected must come from the same matrix."""
    df = simulate_dynamic_panel(n=100, t=6, seed=61)
    m = SystemGMM(
        "y ~ L(1).y + x",
        data=df, panel_var="id", time_var="t",
        collapse=True,
    ).fit(steps=2, windmeijer=True)
    se_from_V = np.sqrt(np.clip(np.diag(m.extras["V_step2_uncorrected"]), 0, None))
    np.testing.assert_array_almost_equal(
        se_from_V, m.extras["se_step2_uncorrected"]
    )
    se_from_model = np.sqrt(np.clip(np.diag(m.extras["V_step2_model_based"]), 0, None))
    np.testing.assert_array_almost_equal(
        se_from_model, m.extras["se_step2_model_based"]
    )


# ---------------------------------------------------------------------------
# Defensive validation: catch silent misuse
# ---------------------------------------------------------------------------

def test_gmm_and_iv_instrument_overlap_is_rejected():
    """A variable can't be both endogenous/predetermined (gmm) and
    strictly exogenous (iv). Silent acceptance would be a footgun."""
    df = simulate_dynamic_panel(n=80, t=6, seed=70)
    with pytest.raises(ValueError, match="(?i)both gmm_instruments and iv_instruments"):
        SystemGMM(
            "y ~ L(1).y + x",
            gmm_instruments=["y", "x"],
            iv_instruments=["x"],   # x in both — should fail
            data=df, panel_var="id", time_var="t",
        )


def test_instrument_proliferation_warns():
    """When the instrument count exceeds N/2 (or N), warn the user.
    This is a quality-of-life feature — Hansen tests and Windmeijer
    correction both degrade with too many instruments."""
    df = simulate_dynamic_panel(n=30, t=12, seed=71)
    # collapse=False with T=12 and gmm_instruments=["y","x"] should
    # blow up the instrument count well past N/2 = 15.
    with pytest.warns(UserWarning, match="instrument count"):
        SystemGMM(
            "y ~ L(1).y + x",
            gmm_instruments=["y"],
            iv_instruments=["x"],
            data=df, panel_var="id", time_var="t",
            collapse=False,   # let instruments proliferate
        ).fit(steps=2, windmeijer=False)


def test_windmeijer_off_means_uncorrected_sandwich():
    """If Windmeijer is off, the reported SE should equal the uncorrected
    cluster-robust sandwich two-step SE."""
    df = simulate_dynamic_panel(n=120, t=6, seed=60)
    m = SystemGMM(
        "y ~ L(1).y + x",
        data=df, panel_var="id", time_var="t",
        collapse=True,
    ).fit(steps=2, windmeijer=False)
    np.testing.assert_array_almost_equal(m.se, m.extras["se_step2_uncorrected"])
    assert m.extras["windmeijer_applied"] is False
