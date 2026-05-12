"""Formula parsing.

We support a clean subset of the Stata/xtabond2 syntax that covers what 95%
of dynamic panel papers actually do:

    y ~ L(1).y + L(0:2).x + w

Concretely:
- LHS is a single dependent variable name. No lag operator on the LHS;
  it just doesn't make sense in this setting.
- RHS terms are either bare names (= contemporaneous regressor, equivalent
  to L(0).var) or lag operators L(k).var / L(a:b).var.
- Lags of the dependent variable end up on the RHS as regressors. We track
  them separately because the instrument-building code treats them a bit
  differently from "other" regressors.

This is intentionally hand-rolled. We could lean on `formulaic` here but
for the small surface we actually need it's not worth the dependency, and
the error messages end up much better when we control the parser ourselves.

TODO: add interaction operator (`*` / `:`), categorical factors, support
      `-1` to drop the intercept. For System GMM the intercept handling is
      different in level vs diff eq, so we manage it inside the estimator
      rather than in the formula for now.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Term:
    """A single regressor: a variable name plus a lag (0 = contemporaneous)."""

    var: str
    lag: int

    def __str__(self) -> str:
        if self.lag == 0:
            return self.var
        return f"L{self.lag}.{self.var}"


@dataclass
class Formula:
    """Parsed representation of a dynamic-panel formula."""

    dep_var: str
    lhs_lags: list[int] = field(default_factory=list)   # lags of dep_var that appear on RHS
    rhs_terms: list[Term] = field(default_factory=list)  # everything else on RHS
    raw: str = ""

    @property
    def all_terms(self) -> list[Term]:
        """All RHS regressors in canonical order: lhs-lags first, then others.

        Order matters: it's the column order of the design matrix X and
        thereby the coefficient vector. Keep it stable across the estimator.
        """
        out = [Term(self.dep_var, k) for k in sorted(self.lhs_lags)]
        out.extend(self.rhs_terms)
        return out

    @property
    def names(self) -> list[str]:
        return [str(t) for t in self.all_terms]

    @property
    def unique_rhs_vars(self) -> list[str]:
        """Unique variable names referenced anywhere on the RHS (incl. dep_var if lagged)."""
        seen: list[str] = []
        for t in self.all_terms:
            if t.var not in seen:
                seen.append(t.var)
        return seen


_NAME = r"[A-Za-z_][A-Za-z0-9_]*"
_LAG_RE = re.compile(rf"^L\((\d+)(?::(\d+))?\)\.({_NAME})$")
_NAME_RE = re.compile(rf"^{_NAME}$")


def parse_formula(formula: str) -> Formula:
    """Parse a dynamic-panel formula like ``y ~ L(1:2).y + x + L(0:1).w``.

    Lag operator syntax:

    - ``L(k).var``     -> lag k of var
    - ``L(a:b).var``   -> lags a, a+1, ..., b of var (inclusive)
    - ``var``          -> contemporaneous (equivalent to ``L(0).var``)

    The LHS is a single variable; the RHS is one or more terms joined with
    ``+``. The estimator handles the constant.

    Raises
    ------
    ValueError
        With a description of what's wrong and a suggestion if obvious.
    TypeError
        If `formula` is not a string.
    """
    if not isinstance(formula, str):
        raise TypeError(
            f"formula must be a string, got {type(formula).__name__}. "
            f"(Did you accidentally pass the parsed Formula back in?)"
        )

    raw = formula
    s = re.sub(r"\s+", "", formula)

    if "~" not in s:
        raise ValueError(
            f"formula {raw!r} is missing the '~' separator.\n"
            f"Expected something like:  'y ~ L(1).y + x1 + x2'."
        )

    lhs, _, rhs = s.partition("~")

    if not _NAME_RE.match(lhs):
        raise ValueError(
            f"the LHS of the formula must be a single variable name, got {lhs!r}.\n"
            f"(No lags, no transformations — dynamic panel models have a single dep var.)"
        )
    dep_var = lhs

    if not rhs:
        raise ValueError(
            f"formula {raw!r} has nothing on the RHS. You need at least one regressor."
        )

    terms_str = rhs.split("+")

    lhs_lags: list[int] = []
    rhs_terms: list[Term] = []
    seen: set[tuple[str, int]] = set()

    for raw_term in terms_str:
        term = raw_term.strip()
        if not term:
            raise ValueError(
                f"empty term in formula {raw!r}. (Stray '+' somewhere?)"
            )

        m = _LAG_RE.match(term)
        if m:
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) is not None else start
            var = m.group(3)
            if end < start:
                raise ValueError(
                    f"in {term!r}: lag range end ({end}) is less than start ({start})."
                )
            for lag in range(start, end + 1):
                key = (var, lag)
                if key in seen:
                    raise ValueError(
                        f"duplicate term {Term(var, lag)} in formula {raw!r}."
                    )
                seen.add(key)
                if var == dep_var:
                    if lag == 0:
                        # That's just the LHS again. Catch the typo loudly.
                        raise ValueError(
                            f"in {term!r}: lag 0 of the dependent variable {dep_var!r} "
                            f"makes no sense — that's just the LHS."
                        )
                    lhs_lags.append(lag)
                else:
                    rhs_terms.append(Term(var, lag))
        elif _NAME_RE.match(term):
            var = term
            key = (var, 0)
            if key in seen:
                raise ValueError(
                    f"duplicate term {var!r} in formula {raw!r}."
                )
            seen.add(key)
            if var == dep_var:
                raise ValueError(
                    f"in formula {raw!r}: you put {dep_var!r} on the RHS without a lag. "
                    f"Did you mean L(1).{dep_var}?"
                )
            rhs_terms.append(Term(var, 0))
        else:
            raise ValueError(
                f"can't parse term {term!r} in formula {raw!r}.\n"
                f"Expected a variable name or L(k).var / L(a:b).var lag operator."
            )

    if not lhs_lags and not rhs_terms:
        raise ValueError(f"formula {raw!r}: no regressors found.")

    return Formula(
        dep_var=dep_var,
        lhs_lags=sorted(lhs_lags),
        rhs_terms=rhs_terms,
        raw=raw,
    )
