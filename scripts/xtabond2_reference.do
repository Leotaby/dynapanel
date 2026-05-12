/*
 * dynapanel — Stata reference run for the AB(1991) replication.
 *
 * Run this in Stata once `abdata.dta` is loaded. Copy the printed
 * coefficients, standard errors, and diagnostic statistics into
 * tests/fixtures/ab1991_reference.json. The test
 * `tests/test_replication_ab1991.py` will then activate and assert
 * dynapanel matches these numbers within documented tolerances.
 *
 * Prerequisites:
 *   ssc install xtabond2
 *   use abdata.dta, clear
 *
 * The specification below matches the AB(1991) employment equation
 * (Table 4 col b in the original paper, give or take a time-dummy set):
 *
 *   n_{it} = α₁ n_{i,t-1} + α₂ n_{i,t-2}
 *          + β₀ w_{it} + β₁ w_{i,t-1}
 *          + γ₀ k_{it} + γ₁ k_{i,t-1}
 *          + (year FEs) + η_i + u_{it}
 */

* Two-step DIFFERENCE GMM with Windmeijer-corrected SEs, collapsed instruments.
xtabond2 n L.n L2.n w L.w k L.k yr1979-yr1984, ///
    gmm(L.(n w k), lag(2 .) collapse) ///
    iv(yr1979-yr1984, eq(diff))      ///
    twostep robust                   ///
    nolevel                          ///
    artests(2)

* Print just the numbers we need to capture:
display _newline "=== AB(1991) DIFFERENCE GMM, two-step, Windmeijer ==="
display "Coefficients and SEs come from the table above."
display "AR(1) z   = " e(ar1)        "  p = " e(ar1p)
display "AR(2) z   = " e(ar2)        "  p = " e(ar2p)
display "Hansen J  = " e(sargan)     "  df = " e(sargandf) "  p = " e(sarganp)

* Now the SYSTEM GMM (Blundell-Bond) version on the same equation.
xtabond2 n L.n L2.n w L.w k L.k yr1979-yr1984, ///
    gmm(L.(n w k), lag(2 .) collapse) ///
    iv(yr1979-yr1984, eq(level))     ///
    twostep robust                   ///
    artests(2)

display _newline "=== AB(1991) SYSTEM GMM, two-step, Windmeijer ==="
display "Coefficients and SEs come from the table above."
display "AR(1) z   = " e(ar1)        "  p = " e(ar1p)
display "AR(2) z   = " e(ar2)        "  p = " e(ar2p)
display "Hansen J  = " e(sargan)     "  df = " e(sargandf) "  p = " e(sarganp)
