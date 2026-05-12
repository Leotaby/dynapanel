# Getting started

!!! warning "Experimental v0.1"
    Point estimates and standard errors have not yet been cross-validated
    against Stata's `xtabond2` on canonical datasets — that work is the
    v0.2 milestone. Treat the numbers in this guide as illustrative.

## The model

dynapanel estimates dynamic linear panel models of the form

$$
y_{it} = \alpha_1 y_{i,t-1} + \dots + \alpha_p y_{i,t-p}
       + \beta' x_{it} + \eta_i + u_{it}
$$

where the unit-specific effect $\eta_i$ is allowed to be arbitrarily
correlated with the regressors. This is the setting where OLS and the
within estimator are biased, and where Arellano-Bond / Blundell-Bond
GMM is a standard approach. (Other options for the same problem
include bias-corrected LSDV, Bayesian dynamic panels, and structural
models — dynapanel doesn't try to cover those.)

## The formula

The formula syntax is a Pythonic take on the `xtabond2` lag notation:

| Syntax           | Meaning                                |
| ---------------- | -------------------------------------- |
| `L(1).y`         | one-period lag of `y`                  |
| `L(2).y`         | two-period lag of `y`                  |
| `L(1:3).y`       | lags 1, 2, and 3 of `y`                |
| `x`              | contemporaneous `x` (= `L(0).x`)       |
| `L(0:1).x`       | contemporaneous and one lag of `x`     |

Examples:

```python
# AR(2) with one exogenous regressor
"y ~ L(1:2).y + x"

# AR(1) with contemporaneous and lagged exogenous regressors
"y ~ L(1).y + L(0:1).w + k"
```

## Classifying regressors: endogenous, predetermined, exogenous

This is the part that matters most. In dynamic panel GMM, every RHS
variable needs to be classified by how its errors interact with the
shock $u_{it}$:

| Classification         | Meaning                                                      | Moment condition                                                       | Instrument logic                |
| ---------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------------- | ------------------------------- |
| **Endogenous**         | May correlate with **current and past** shocks, not future   | $E[x_{it} u_{is}] = 0$ for $s > t$ only                                | Deeper lags only: `(2, None)`   |
| **Predetermined**      | May correlate with **past** shocks only, not current/future  | $E[x_{it} u_{is}] = 0$ for $s \ge t$                                   | Lag 1 onward: `(1, None)`       |
| **Strictly exogenous** | Uncorrelated with past, current, and future shocks           | $E[x_{it} u_{is}] = 0$ for all $s$                                     | Use itself, not lags            |

Why the lag depths differ: the differenced equation has
$\Delta u_{it} = u_{it} - u_{i,t-1}$, so a valid instrument at row $t$
must be uncorrelated with **both** $u_{it}$ and $u_{i,t-1}$. For an
endogenous variable, that means $x_{i,t-2}$ is the closest valid lag.
For a predetermined variable, $x_{i,t-1}$ already satisfies both
conditions (it can correlate with $u_{i,t-2}$ and earlier, but not with
$u_{i,t-1}$ or $u_{it}$).

In dynapanel:

- `gmm_instruments=[...]` → variables to instrument GMM-style. Use for
  **endogenous** and **predetermined**.
- `iv_instruments=[...]` → variables that are strictly exogenous.
- `gmm_lags={"var": (minlag, maxlag), ...}` → per-variable lag depth.
  Use `(2, None)` for endogenous, `(1, None)` for predetermined.
  Optional — defaults to the global `minlag=2`/`maxlag=None`.

Example: $y$ endogenous, $x$ predetermined, $k$ strictly exogenous.

```python
SystemGMM(
    "y ~ L(1).y + x + k",
    gmm_instruments=["y", "x"],
    gmm_lags={
        "y": (2, None),     # endogenous
        "x": (1, None),     # predetermined
    },
    iv_instruments=["k"],
    data=df, panel_var="id", time_var="t",
    collapse=True,
).fit(steps=2, windmeijer=True)
```

## The estimator

```python
from dynapanel import SystemGMM

model = SystemGMM(
    "y ~ L(1).y + x",
    gmm_instruments=["y", "x"],
    iv_instruments=[],
    data=df,
    panel_var="id",
    time_var="t",
    minlag=2,                     # default lag depth for GMM instruments
    maxlag=None,                  # None = use all available lags
    collapse=True,                # collapse instrument matrix (recommended)
).fit(
    steps=2,                      # 1 or 2
    windmeijer=True,              # finite-sample SE correction
    robust="cluster",             # one-way clustering by panel unit (default)
)

model.summary()
model.diagnostics()
```

## Reading the output

A typical summary table looks like this:

```
  dynapanel System GMM  —  step 2  (Windmeijer-corrected SE)
  ==========================================================
  variable              coef    std.err     z         p
  ----------------------------------------------------------
  L1.y                0.6275    0.0433  14.50    0.0000  ***
  x                   0.5764    0.0736   7.83    0.0000  ***
  const              -0.0312    0.0571  -0.55    0.5846
  ----------------------------------------------------------
  N (units) = 300   obs = 3600   k = 3   L = 15
  AR(1) test: z = -11.644, p = 0.0000
  AR(2) test: z =   0.965, p = 0.3346
  Hansen J:   χ²(12) = 10.935, p = 0.5345
```

The three diagnostics worth always glancing at, in order:

1. **AR(1)** typically rejects (the differenced residuals are MA(1) by
   construction).
2. **AR(2)** should *not* reject — failure to reject is consistent with
   the lagged-level instruments being valid.
3. **Hansen J** should *not* reject — failure to reject is consistent
   with the overidentifying restrictions.

These are necessary, not sufficient: passing all three doesn't prove
the model is right, just that the moments aren't visibly inconsistent
with the data. A rejection — especially in m2 or Hansen — is more
informative.

## On instrument proliferation

The biggest practical pitfall in dynamic-panel GMM is the
"too many instruments" problem (Roodman 2009): if you let the
instrument matrix grow unchecked, the two-step weight matrix becomes
ill-conditioned, Windmeijer's correction loses precision, and
Hansen's test loses power.

**Recommendation**: always set `collapse=True`. If the instrument count
is still large relative to the cross-section (a common rule of thumb
is $L < N/2$), also set a reasonable `maxlag`.

## When to use Difference vs System

- **Difference GMM** (Arellano-Bond 1991): the original. Works well
  when the dependent variable is not too persistent (|α| small).
- **System GMM** (Blundell-Bond 1998): adds level-equation moments
  using lagged differences as instruments. Helps when α is close to 1
  (highly persistent series) — that's where Difference GMM's lagged
  levels become weak instruments for differenced lags.

A common diagnostic in applied work is to check whether the dynamic
coefficient lies in a plausible range — for example, between the OLS
and within estimates, which tend to bracket the truth from above and
below. This is a sanity check, not a proof of validity; the formal
specification tests above remain the primary tools.
