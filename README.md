# dynapanel

dynapanel is to xtabond2 what pyfixest is to fixest.

A modern Python implementation of **Difference and System GMM** for
dynamic panel data — Arellano-Bond, Blundell-Bond, Windmeijer-corrected
standard errors, Hansen / AR(1) / AR(2) tests, collapsed instruments,
and publication-ready output tables. Inspired by Stata's `xtabond2`,
written from scratch in Python.

> **Status: experimental v0.1.** The code paths described below all
> run, the unit tests all pass, and the estimator recovers known
> coefficients on simulated data. What v0.1 does *not* yet have is a
> full cross-check of point estimates and standard errors against
> Stata's `xtabond2` on canonical datasets. Until those replication
> tests land in CI (v0.2 milestone), please treat results as
> preliminary and report any discrepancies with a minimal reproducible
> example.
>
> **For published empirical work, please cross-check v0.1 results
> against Stata, R, or another trusted implementation until the v0.2
> replication suite is complete.**

## Why does this exist

If you do corporate finance, banking, growth, or development research
and you've ever needed System GMM in Python, you've probably noticed:

- `linearmodels` has lovely pooled / FE / RE panel estimators, but its
  generic IV-GMM machinery is not set up the way Holtz-Eakin /
  Arellano-Bond / Blundell-Bond instruments naturally are.
- Stata's `xtabond2` (Roodman) is the canonical reference but lives in
  Stata.
- R's `plm::pgmm` works, but the API isn't designed around modern
  Python workflows.
- Hand-rolling System GMM in numpy means re-deriving the Windmeijer
  correction every two years and getting it slightly wrong.

`dynapanel` is the library I wanted to exist. Design goals:

1. **Stata users feel at home.** The formula syntax is a clean
   Pythonic take on the `xtabond2` style: `y ~ L(1:2).y + L(0:1).w + k`.
2. **Polars-native where it matters.** Lag/diff construction is fast.
3. **Beautiful output.** `great_tables` summary tables you can drop
   into a paper.
4. **Reproducible.** Replication notebooks for Arellano-Bond (1991)
   and Blundell-Bond (1998) are the v0.2 milestone.
5. **Honest documentation.** When something is not yet validated, the
   docs say so — we don't bury caveats in footnotes.

## Install

Until v0.1 is published to PyPI, install directly from GitHub:

```bash
pip install "dynapanel @ git+https://github.com/Leotaby/dynapanel.git"
```

The optional `[all]` extra pulls in `great_tables` for publication-ready
summary tables and `matplotlib` for coefficient plots:

```bash
pip install "dynapanel[all] @ git+https://github.com/Leotaby/dynapanel.git"
```

Once the PyPI release lands, this becomes:

```bash
pip install dynapanel
pip install "dynapanel[all]"
```

To work on the package itself:

```bash
git clone https://github.com/Leotaby/dynapanel.git
cd dynapanel
pip install -e ".[dev]"
pytest
```

## Quick start

```python
from dynapanel import SystemGMM, simulate_dynamic_panel

df = simulate_dynamic_panel(n=300, t=8, alpha=0.6, beta=0.5,
                            x_corr_eta=0.0, seed=42)

model = SystemGMM(
    "y ~ L(1).y + x",
    gmm_instruments=["y"],     # endogenous: y instrumented by deeper lags
    iv_instruments=["x"],      # treat x as strictly exogenous
    data=df,
    panel_var="id",
    time_var="t",
    collapse=True,
).fit(steps=2, windmeijer=True)

model.summary()
model.diagnostics()
```

For variables that are *predetermined* (correlated with past but not
contemporaneous shocks), pass a per-variable lag spec:

```python
SystemGMM(
    "y ~ L(1).y + x + k",
    gmm_instruments=["y", "x"],
    gmm_lags={
        "y": (2, None),    # endogenous: lag 2 and up
        "x": (1, 4),       # predetermined: lag 1 to 4
    },
    iv_instruments=["k"],
    ...
)
```

## What's in the experimental v0.1

- Difference GMM (Arellano-Bond 1991), one-step & two-step
- System GMM (Blundell-Bond 1998), one-step & two-step
- Windmeijer (2005) finite-sample SE correction *(implemented from
  the closed form; cross-validation against `xtabond2` is v0.2)*
- AR(1) and AR(2) tests on differenced residuals (Arellano-Bond m1/m2)
  with one-way clustering by panel unit under H0
- Hansen J test of overidentifying restrictions
- Collapsed instruments (Roodman 2009)
- Per-variable lag specification (`gmm_lags={"y": (2, None), ...}`)
- `great_tables` publication-ready summary tables (with text fallback)
- Native pandas and polars support
- Built-in simulator (`simulate_dynamic_panel`) for tests and demos

## What's NOT in v0.1 (PRs welcome)

- **Forward orthogonal deviations** (Arellano-Bover 1995). The flag
  exists on the constructor so the API doesn't break later, but it
  currently raises `NotImplementedError` with a clear message.
- **Diff-in-Hansen** tests for instrument subsets.
- **Ahn-Schmidt (1995)** nonlinear moment conditions.
- **Panel VAR.**
- **True two-way clustering.** `robust="twoway"` raises
  `NotImplementedError` rather than silently falling back, so the
  user can't accidentally believe they got a twoway variance.
- **`predict` / out-of-sample prediction.**
- **Replication tests against `xtabond2`** on the AB (1991) and BB
  (1998) datasets. Until these land, treat v0.1 numbers as
  preliminary.
- **Finite-sample correction for the m1/m2 variance estimator.** The
  current code uses one-way clustering by panel unit under H0 of no
  AR(r) in the differenced residuals — asymptotically valid, but
  doesn't adjust for the estimation noise in β̂ that the full
  Arellano-Bond (1991) eq. 11 estimator includes.

## Honest limitations

v0.1 is the "scaffolding plus a working numerical core" release. The
implementation follows Roodman's (2009) *Stata Journal* article and
Windmeijer (2005), but until cross-validation against `xtabond2` is
in CI you should consider the standard errors and Hansen statistics
preliminary. If you find a coefficient or test statistic that diverges
materially from a Stata or R reference, please open an issue with a
minimal reproducer.

## A note on style

The code in this repo is deliberately a bit chatty in the comments.
If you want a clean line of derivations without the editorial voice,
the docs are where to look. The source is where I argue with myself
about why two-step weights need a pseudo-inverse fallback at 3am.

## License

MIT.

## Citation

If you use `dynapanel` in a paper, please cite it using the metadata
in [`CITATION.cff`](CITATION.cff). GitHub renders this directly as a
"Cite this repository" widget.
