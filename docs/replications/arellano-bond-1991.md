# Replication: Arellano & Bond (1991)

> **Status: scaffolded, not yet validated.** The dataset fetcher, the
> Stata reference script, and the pytest harness are all in place — the
> only missing piece is the Stata-side numbers themselves. Once those
> are captured into `tests/fixtures/ab1991_reference.json`, the test
> suite asserts dynapanel matches Stata within documented tolerances on
> every commit.

## The specification

The Arellano-Bond (1991) employment equation in the original notation:

$$
n_{it} = \alpha_1 n_{i,t-1} + \alpha_2 n_{i,t-2}
       + \beta_0 w_{it} + \beta_1 w_{i,t-1}
       + \gamma_0 k_{it} + \gamma_1 k_{i,t-1}
       + \mu_t + \eta_i + u_{it}
$$

In dynapanel:

```python
from dynapanel import DifferenceGMM
import polars as pl

abdata = pl.read_parquet("tests/fixtures/abdata.parquet")

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
table:

- **Point estimates** from AB(1991) and dynapanel's two-step GMM should
  match within rounding (modulo specification-level differences in the
  time dummy set), since the underlying estimator is the same.
- **Standard errors** in AB(1991)'s published table use the uncorrected
  two-step formula. dynapanel's default Windmeijer-corrected two-step
  SEs will *not* match those — and shouldn't.

The right reference for v0.2 is therefore a *Stata `xtabond2` run* on
the same dataset, with documented options, comparing modern outputs to
modern outputs. That's what the replication harness here checks.

## The replication harness (already in the repo)

Three artifacts are already shipped in v0.1, ready to activate:

1. **`scripts/fetch_abdata.py`** — downloads the dataset from a known
   stable mirror and writes it as `tests/fixtures/abdata.parquet`.

2. **`scripts/xtabond2_reference.do`** — Stata script with the exact
   `xtabond2` invocation. Run it once you have Stata + `xtabond2`
   installed (`ssc install xtabond2`), with `abdata.dta` loaded.

3. **`tests/fixtures/ab1991_reference.template.json`** — the schema
   you fill in with the Stata-output numbers. Drop the `.template`
   suffix once filled in.

4. **`tests/test_replication_ab1991.py`** — the pytest test that asserts
   dynapanel matches Stata within documented tolerances. Currently
   skips with a clear message until both the dataset and the filled
   fixture exist.

## The workflow

```bash
# 1) Fetch the dataset.
python scripts/fetch_abdata.py
# → writes tests/fixtures/abdata.parquet

# 2) In Stata:
#       use tests/fixtures/abdata.dta, clear     (convert from parquet first)
#       do scripts/xtabond2_reference.do

# 3) Paste the printed coefficients, SEs, and Hansen/AR numbers into
#    tests/fixtures/ab1991_reference.template.json. Rename it to drop
#    the .template suffix.

# 4) Run the replication tests.
pytest tests/test_replication_ab1991.py -v
```

When green, the README banner changes from "experimental v0.1" to
"validated against xtabond2 on AB(1991)", and the v0.2 release is
earned.

## Reference table (will be filled in once Stata-side numbers exist)

| Quantity                          | dynapanel | xtabond2 (Stata) | |Δ|     | Tolerance |
| --------------------------------- | --------- | ---------------- | ------- | --------- |
| L1.n coefficient                  | TBD       | TBD              | TBD     | 1e-3      |
| L2.n coefficient                  | TBD       | TBD              | TBD     | 1e-3      |
| w coefficient                     | TBD       | TBD              | TBD     | 1e-3      |
| L1.w coefficient                  | TBD       | TBD              | TBD     | 1e-3      |
| k coefficient                     | TBD       | TBD              | TBD     | 1e-3      |
| L1.k coefficient                  | TBD       | TBD              | TBD     | 1e-3      |
| Windmeijer SE (L1.n)              | TBD       | TBD              | TBD     | 1e-2      |
| Hansen J                          | TBD       | TBD              | TBD     | 1e-2      |
| Hansen df                         | TBD       | TBD              | —       | exact     |
| AR(1) z-statistic                 | TBD       | TBD              | TBD     | 1e-2      |
| AR(2) z-statistic                 | TBD       | TBD              | TBD     | 1e-2      |
