# Replication: Arellano & Bond (1991)

> **Status:** stub — the replication notebook itself lives in
> `examples/arellano_bond_1991.py`. Once the reference numbers are
> captured from a documented Stata/`xtabond2` specification, this page
> will walk through a table-by-table comparison.

The Arellano-Bond (1991) employment equation in the original notation:

$$
n_{it} = \alpha_1 n_{i,t-1} + \alpha_2 n_{i,t-2}
       + \beta_0 w_{it} + \beta_1 w_{i,t-1}
       + \gamma_0 k_{it} + \gamma_1 k_{i,t-1}
       + \mu_t + \eta_i + u_{it}
$$

In dynapanel that's:

```python
from dynapanel import DifferenceGMM

model = DifferenceGMM(
    "n ~ L(1:2).n + L(0:1).w + L(0:1).k",
    gmm_instruments=["n", "w", "k"],
    data=abdata,
    panel_var="id",
    time_var="year",
    collapse=True,
).fit(steps=2, windmeijer=True)

model.summary()
```

## A note on the right comparison target

The original Arellano-Bond (1991) paper predates the Windmeijer (2005)
finite-sample SE correction by 14 years. So a clean comparison cannot
simply read coefficients and standard errors off the AB(1991) published
table and assert that dynapanel reproduces them. Specifically:

- **Point estimates** from AB(1991) and dynapanel's two-step GMM should
  match within rounding (modulo any specification-level differences
  e.g. time dummies), since the underlying estimator is the same.
- **Standard errors** from AB(1991)'s published table use the
  uncorrected two-step SE formula. dynapanel's default
  Windmeijer-corrected two-step SEs will *not* match those — and
  shouldn't.

The right reference for v0.2 is therefore a *Stata `xtabond2` run* on
the same dataset, with documented options, comparing modern outputs
to modern outputs:

| Quantity                              | dynapanel    | xtabond2 (Stata) | Tolerance |
| ------------------------------------- | ------------ | ---------------- | --------- |
| L1.n coefficient                      | TBD          | TBD              | 1e-3      |
| L2.n coefficient                      | TBD          | TBD              | 1e-3      |
| w coefficient                         | TBD          | TBD              | 1e-3      |
| L1.w coefficient                      | TBD          | TBD              | 1e-3      |
| k coefficient                         | TBD          | TBD              | 1e-3      |
| L1.k coefficient                      | TBD          | TBD              | 1e-3      |
| Two-step naive SE (L1.n)              | TBD          | TBD              | 1e-3      |
| Windmeijer-corrected SE (L1.n)        | TBD          | TBD              | 1e-2      |
| Hansen J statistic                    | TBD          | TBD              | 1e-2      |
| Hansen J degrees of freedom           | TBD          | TBD              | exact     |
| AR(1) z-statistic                     | TBD          | TBD              | 1e-2      |
| AR(2) z-statistic                     | TBD          | TBD              | 1e-2      |

Filling this in is one of the v0.2 milestones. Once `tests/test_replication_ab1991.py`
is in CI with these numbers as fixtures, the experimental banner on
the README comes off.

## A separate comparison to the published table

Independently, dynapanel's *uncorrected* two-step SEs (`windmeijer=False`)
should be comparable to AB(1991)'s reported SEs, modulo specification
details around time dummies. That's the right reference for the
historical numbers, and we'll document it alongside the modern
comparison.
