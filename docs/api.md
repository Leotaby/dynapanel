# API reference

## Estimators

::: dynapanel.SystemGMM
    options:
      members:
        - __init__
        - fit
      show_source: false

::: dynapanel.DifferenceGMM
    options:
      members:
        - __init__
        - fit
      show_source: false

## Results

::: dynapanel.GMMResults
    options:
      members:
        - coef
        - se
        - t_stats
        - pvalues
        - confint
        - summary
        - diagnostics
        - coef_plot

## Formula parser

::: dynapanel.parse_formula

::: dynapanel.Formula

::: dynapanel.Term

## Simulator

::: dynapanel.simulate_dynamic_panel
