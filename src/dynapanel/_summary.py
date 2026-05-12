"""great_tables summary formatter.

Optional dependency. If `great_tables` isn't installed we let the
ImportError bubble up to the caller, which falls back to a text summary.

The table layout:

    --------------------------------------------------------------
    dynapanel System GMM  —  step 2  (Windmeijer-corrected SE)
    --------------------------------------------------------------
    variable        coef     se      z     p       95% CI
    L1.y           0.612  0.041  14.92  <.001   [0.531, 0.693]
    x              0.501  0.028  17.89  <.001   [0.446, 0.555]
    --------------------------------------------------------------
    N units = 300    obs = 2100    k = 2    L = 23
    AR(1)        z = -3.91   p < .001
    AR(2)        z = -1.04   p =  .298
    Hansen J     χ²(21) =  19.7   p =  .547
    --------------------------------------------------------------
"""

from __future__ import annotations

import great_tables as gt  # raise ImportError early if not installed
import polars as pl


def great_tables_summary(results) -> gt.GT:
    coef_names = results.coef_names
    beta = results.beta
    se = results.se
    z = results.t_stats
    p = results.pvalues
    ci = results.confint()

    df = pl.DataFrame({
        "term": coef_names,
        "coef": beta,
        "se": se,
        "z": z,
        "p": p,
        "ci_lo": ci[:, 0],
        "ci_hi": ci[:, 1],
    })

    eq = results.model._equation
    step = results.extras.get("step", "?")
    wm = results.extras.get("windmeijer_applied", False)
    title = f"dynapanel {eq.title()} GMM — step {step}"
    subtitle_bits = [
        f"N = {results.n_individuals} units, {results.n_obs} obs",
        f"k = {results.n_reg}, L = {results.n_inst} instruments",
    ]
    if wm:
        subtitle_bits.append("Windmeijer (2005) SE correction")
    subtitle = "  ·  ".join(subtitle_bits)

    table = (
        gt.GT(df)
        .tab_header(title=title, subtitle=subtitle)
        .cols_label(
            term="",
            coef="Coef.",
            se="Std. Err.",
            z="z",
            p="p",
            ci_lo="95% CI lo",
            ci_hi="95% CI hi",
        )
        .fmt_number(columns=["coef", "se", "ci_lo", "ci_hi"], decimals=4)
        .fmt_number(columns=["z"], decimals=2)
        .fmt_number(columns=["p"], decimals=4)
    )

    # Diagnostic footer
    foot_lines = [
        f"AR(1):    {results.ar1.statistic:7.3f}     p = {results.ar1.pvalue:.4f}",
        f"AR(2):    {results.ar2.statistic:7.3f}     p = {results.ar2.pvalue:.4f}",
        f"Hansen J: χ²({results.hansen.df}) = {results.hansen.statistic:.3f}  "
        f"p = {results.hansen.pvalue:.4f}",
    ]
    footer = "\n".join(foot_lines)
    return table.tab_source_note(source_note=footer)
