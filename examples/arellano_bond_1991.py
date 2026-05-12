"""
Replication: Arellano & Bond (1991), employment equation.

STATUS: scaffold. The dataset isn't bundled here — its redistribution
terms haven't been verified for inclusion in this package. The intent
for v0.2 is to either:

    (a) bundle it as a parquet under `data/` if the license clearly
        permits redistribution, or
    (b) ship a small fixture-generation script that downloads it from
        a known stable mirror (e.g. the `plm` R package's data, or the
        `pydynpd` repository's copy) at test time.

Once the dataset is wired in, this script will:

    1. Load the dataset (140 UK firms × 7-9 years, manufacturing).
    2. Fit the AR(2) employment equation with wages and capital.
    3. Compare point estimates to a documented Stata/xtabond2 run,
       and compare uncorrected two-step SEs to AB(1991)'s published
       table (Windmeijer-corrected SEs are compared to xtabond2 only,
       since Windmeijer's correction post-dates AB(1991) by 14 years).

To-do (v0.2):
    [ ] Resolve dataset licensing / write the download fixture
    [ ] Document the Stata `xtabond2` reference invocation
    [ ] Capture reference numbers as JSON fixtures
    [ ] Wire into `tests/test_replication_ab1991.py` with CI
"""

from pathlib import Path

import polars as pl

from dynapanel import DifferenceGMM


def load_abdata() -> pl.DataFrame:
    """Try a few likely paths for abdata. Replace with packaged data later."""
    here = Path(__file__).parent
    candidates = [
        here / "abdata.parquet",
        here / "abdata.csv",
        here.parent / "data" / "abdata.parquet",
    ]
    for p in candidates:
        if p.exists():
            return pl.read_parquet(p) if p.suffix == ".parquet" else pl.read_csv(p)
    raise FileNotFoundError(
        "abdata not found. For now, download it manually from a known "
        "Arellano-Bond example source (the dataset is used in common "
        "Arellano-Bond examples across Stata/R workflows; check the source "
        "you obtain it from for license and citation requirements) and "
        "save it as examples/abdata.parquet."
    )


def main() -> None:
    df = load_abdata()
    model = DifferenceGMM(
        "n ~ L(1:2).n + L(0:1).w + L(0:1).k",
        gmm_instruments=["n", "w", "k"],
        data=df,
        panel_var="id",
        time_var="year",
        collapse=True,
    ).fit(steps=2, windmeijer=True)

    model.summary()


if __name__ == "__main__":
    main()
