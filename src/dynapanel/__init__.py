"""dynapanel — modern dynamic panel data models for Python.

Public API:
    DifferenceGMM    -- Arellano-Bond (1991) difference GMM estimator
    SystemGMM        -- Blundell-Bond (1998) system GMM estimator
    simulate_dynamic_panel -- toy DGP for tests & demos
    parse_formula    -- formula parser (mostly internal, exported for power users)

Typical use:

    from dynapanel import SystemGMM

    model = SystemGMM(
        "y ~ L(1).y + x",
        gmm_instruments=["y", "x"],
        data=df,
        panel_var="id",
        time_var="t",
        collapse=True,
    ).fit(steps=2, windmeijer=True)

    model.summary()
    model.diagnostics()
"""

from ._formula import Formula, Term, parse_formula
from ._gmm import DifferenceGMM, SystemGMM
from ._results import GMMResults
from ._simulate import simulate_dynamic_panel

__version__ = "0.1.1"

__all__ = [
    "DifferenceGMM",
    "SystemGMM",
    "GMMResults",
    "Formula",
    "Term",
    "parse_formula",
    "simulate_dynamic_panel",
    "__version__",
]
