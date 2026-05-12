# dynapanel

> dynapanel is to xtabond2 what pyfixest is to fixest.

A modern Python implementation of **Difference and System GMM** for
dynamic panel data — Arellano-Bond, Blundell-Bond, Windmeijer-corrected
standard errors, Hansen / AR(1) / AR(2) tests, collapsed instruments,
and publication-ready output tables. Inspired by Stata's `xtabond2`,
written from scratch in Python.

!!! warning "Experimental v0.1"
    Code paths run and the unit tests pass, but point estimates and
    standard errors have not yet been cross-validated against Stata's
    `xtabond2` on canonical datasets — that's the v0.2 milestone. Until
    then, treat numbers as preliminary and cross-check against Stata,
    R, or another trusted implementation for any published work.

## Install

Until v0.1 lands on PyPI, install directly from GitHub:

```bash
pip install "dynapanel @ git+https://github.com/Leotaby/dynapanel.git"
```

With the optional `[all]` extra (great_tables + matplotlib):

```bash
pip install "dynapanel[all] @ git+https://github.com/Leotaby/dynapanel.git"
```

## The shortest possible quickstart

```python
from dynapanel import SystemGMM, simulate_dynamic_panel

df = simulate_dynamic_panel(n=300, t=8, alpha=0.6, beta=0.5,
                            x_corr_eta=0.0, seed=42)

model = SystemGMM(
    "y ~ L(1).y + x",
    gmm_instruments=["y"],     # endogenous: instrumented by deeper lags
    iv_instruments=["x"],      # strictly exogenous: used as itself
    data=df,
    panel_var="id",
    time_var="t",
    collapse=True,
).fit(steps=2, windmeijer=True)

model.summary()
model.diagnostics()
```

If you've used `xtabond2` in Stata, the API should feel like a polite,
Pythonic translation. If you haven't, see the
[getting started guide](getting-started.md) for the full walkthrough
of endogenous / predetermined / exogenous classification and the
specification tests.

## What's in the experimental v0.1

- Difference GMM (Arellano-Bond 1991), one-step & two-step
- System GMM (Blundell-Bond 1998), one-step & two-step
- Windmeijer (2005) finite-sample SE correction
- AR(1) and AR(2) tests on differenced residuals (Arellano-Bond m1/m2)
- Hansen J test of overidentifying restrictions
- Collapsed instruments (Roodman 2009)
- Per-variable lag specification distinguishing endogenous,
  predetermined, and strictly exogenous regressors
- Publication-ready tables via `great_tables` (with text fallback)
- Native support for pandas and polars data frames

## What's not in v0.1 yet

See the [README](https://github.com/Leotaby/dynapanel#whats-not-in-v01-prs-welcome)
for the candid list. Highlights: forward orthogonal deviations,
Diff-in-Hansen, Ahn-Schmidt nonlinear moments, panel VAR, proper
two-way clustering, and the cross-validation against `xtabond2` on the
AB(1991) / BB(1998) canonical examples (v0.2 milestone — scaffolding
already in the repo under `scripts/` and `tests/test_replication_ab1991.py`).
