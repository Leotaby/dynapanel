"""Results object returned by ``.fit()``.

The Results object's job is to *present* the fitted model to the user.
It holds the coefficients, standard errors, diagnostics, and provides:

    .summary()       -- great_tables (or plain text) table
    .diagnostics()   -- text summary of m1/m2/Hansen
    .coef_plot()     -- matplotlib coefficient plot

We keep the computational guts (residuals, moment matrices, etc.) as
attributes so power users can post-process whatever they need.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class GMMResults:
    """Output of a GMM fit. See ``SystemGMM.fit`` / ``DifferenceGMM.fit``."""

    # The model that produced these results (kept for introspection)
    model: Any

    # Assembled matrices (kept so the user can do extra moment computations)
    mats: Any

    # Estimation output
    beta: np.ndarray
    se: np.ndarray
    V: np.ndarray
    resid: np.ndarray
    W: np.ndarray

    # Diagnostics
    ar1: Any
    ar2: Any
    hansen: Any

    # Counts
    n_obs: int
    n_individuals: int
    n_inst: int
    n_reg: int

    # Per-step bookkeeping and extras (one-step coefs, naive SEs, etc.)
    extras: dict = field(default_factory=dict)

    # ---- convenience -----------------------------------------------------

    @property
    def coef_names(self) -> list[str]:
        return list(self.extras.get("coef_names", []))

    @property
    def coef(self) -> dict[str, float]:
        return {n: float(v) for n, v in zip(self.coef_names, self.beta)}

    @property
    def t_stats(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.se > 0, self.beta / self.se, np.nan)

    @property
    def pvalues(self) -> np.ndarray:
        z = np.abs(self.t_stats)
        return 2.0 * (1.0 - stats.norm.cdf(z))

    def confint(self, alpha: float = 0.05) -> np.ndarray:
        """Two-sided (1-alpha) confidence intervals using the normal approx."""
        crit = stats.norm.ppf(1 - alpha / 2)
        lo = self.beta - crit * self.se
        hi = self.beta + crit * self.se
        return np.column_stack([lo, hi])

    # ---- repr / summary --------------------------------------------------

    def __repr__(self) -> str:
        step = self.extras.get("step", "?")
        eq = self.model._equation
        return (
            f"<GMMResults equation={eq} step={step} "
            f"N={self.n_individuals} obs={self.n_obs} "
            f"k={self.n_reg} L={self.n_inst}>"
        )

    def summary(self):
        """Return a great_tables Table (if installed), else print a text table.

        We try ``great_tables`` first because it produces actually-good
        publication tables. If it's not installed (or if the user is in a
        non-HTML environment) we fall back to a clean text table.
        """
        # We try the great_tables path first.
        try:
            from ._summary import great_tables_summary
        except ImportError:  # great_tables not available — fall back
            self._print_text_summary()
            return None
        else:
            try:
                return great_tables_summary(self)
            except Exception as e:
                # Don't let a presentation bug kill the user's analysis.
                # Fall back to text and surface the error softly.
                print(f"(great_tables summary unavailable: {e})\n")
                self._print_text_summary()
                return None

    def _print_text_summary(self) -> None:
        """Plain-text summary table. Prints to stdout."""
        eq = self.model._equation
        step = self.extras.get("step", "?")
        wm = self.extras.get("windmeijer_applied", False)
        title = f"  dynapanel {eq.title()} GMM  —  step {step}"
        if wm:
            title += "  (Windmeijer-corrected SE)"
        print()
        print(title)
        print("  " + "=" * (len(title) - 2))

        # Header for coefficient table
        head = f"  {'variable':<18} {'coef':>12} {'std.err':>12} {'z':>8} {'p':>8}"
        sep = "  " + "-" * (len(head) - 2)
        print(sep)
        print(head)
        print(sep)
        z = self.t_stats
        p = self.pvalues
        for name, b, s, zi, pi in zip(self.coef_names, self.beta, self.se, z, p):
            stars = _pvalue_stars(pi)
            line = f"  {name:<18} {b:>12.4f} {s:>12.4f} {zi:>8.2f} {pi:>8.4f} {stars}"
            print(line)
        print(sep)

        # Diagnostic footer
        print(f"  N (units) = {self.n_individuals:<6} obs = {self.n_obs:<6} "
              f"k = {self.n_reg:<3} L = {self.n_inst}")
        print(f"  {self.ar1}")
        print(f"  {self.ar2}")
        print(f"  {self.hansen}")
        if wm:
            print("  Standard errors use Windmeijer (2005) finite-sample correction.")
        print()

    def diagnostics(self) -> None:
        """Print the specification-test footer alone."""
        print(self.ar1)
        print(self.ar2)
        print(self.hansen)

    # ---- coefficient plot ------------------------------------------------

    def coef_plot(self, ax=None, alpha: float = 0.05, drop_const: bool = True):
        """Quick matplotlib coefficient plot with confidence intervals.

        Returns the axes object so you can style it further.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "coef_plot requires matplotlib. Install with `pip install matplotlib`."
            ) from e

        names = list(self.coef_names)
        b = self.beta.copy()
        ci = self.confint(alpha=alpha)
        if drop_const and "const" in names:
            keep = [i for i, n in enumerate(names) if n != "const"]
            names = [names[i] for i in keep]
            b = b[keep]
            ci = ci[keep]

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 0.5 + 0.4 * len(names)))
        ypos = np.arange(len(names))[::-1]
        ax.errorbar(
            b, ypos,
            xerr=[b - ci[:, 0], ci[:, 1] - b],
            fmt="o", capsize=3,
        )
        ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_yticks(ypos)
        ax.set_yticklabels(names)
        ax.set_xlabel("coefficient (95% CI)" if alpha == 0.05 else f"coefficient ({int((1-alpha)*100)}% CI)")
        ax.set_title(f"{self.model.__class__.__name__} — step {self.extras.get('step', '?')}")
        return ax


def _pvalue_stars(p: float) -> str:
    """Old-school p-value stars. Yes, I know. Yes, I'm doing it anyway."""
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "."
    return ""
