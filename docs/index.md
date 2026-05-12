# dynapanel

> dynapanel is to xtabond2 what pyfixest is to fixest.

A modern Python library for dynamic panel data models — Arellano-Bond
difference GMM, Blundell-Bond system GMM, Windmeijer-corrected
standard errors, AR(1)/AR(2)/Hansen specification tests. Built to feel
natural to Stata refugees and natively pandas+polars users alike.

## Install

```bash
pip install -e ".[all]"
```

## The shortest possible quickstart

```python
from dynapanel import SystemGMM, simulate_dynamic_panel

df = simulate_dynamic_panel(n=300, t=8, alpha=0.6, beta=0.5, seed=42)

model = SystemGMM(
    "y ~ L(1).y + x",
    gmm_instruments=["y", "x"],
    data=df,
    panel_var="id",
    time_var="t",
    collapse=True,
).fit(steps=2, windmeijer=True)

model.summary()
```

If you've used `xtabond2` in Stata, the API should feel like a polite,
Pythonic translation. If you haven't, see the
[getting started guide](getting-started.md).

## What the library does well

- Clean formula syntax: `y ~ L(1:2).y + L(0:1).w + k`.
- Two-step efficient GMM with the Windmeijer (2005) finite-sample SE
  correction.
- Diagnostics that match the conventions reported in published papers
  (m1, m2, Hansen J).
- Beautiful summary tables powered by `great_tables` with a clean
  text fallback when it's not installed.
- Polars-native panel data wrangling with full pandas interop.

## What the library doesn't do yet

See the [README](https://github.com/Leotaby/dynapanel#whats-not-in-v01-prs-welcome)
for a candid list. Highlights of what's missing: forward orthogonal
deviations, Diff-in-Hansen, Ahn-Schmidt nonlinear moments, panel VAR,
proper two-way clustering.
