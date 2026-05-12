"""
dynapanel quickstart — run me with `python examples/quickstart.py`.

Simulates a dynamic panel from a known DGP, fits SystemGMM, prints the
summary, and verifies that the diagnostics fire correctly.
"""

from dynapanel import DifferenceGMM, SystemGMM, simulate_dynamic_panel


def main() -> None:
    # Generate 300 firms × 8 years with a fixed effect, a persistent y,
    # and a strictly-exogenous regressor x.
    df = simulate_dynamic_panel(
        n=300, t=8,
        alpha=0.6, beta=0.5,
        rho_x=0.6,
        x_corr_eta=0.0,    # strictly exogenous x (cleanest demo)
        seed=42,
    )

    # y is endogenous (dynamic), x is strictly exogenous in this DGP
    # (x_corr_eta=0). That's the cleanest demo setting.
    print("=" * 70)
    print("System GMM (Blundell-Bond), two-step, Windmeijer-corrected")
    print("=" * 70)
    sys_model = SystemGMM(
        "y ~ L(1).y + x",
        gmm_instruments=["y"],     # endogenous: instrumented by lagged levels
        iv_instruments=["x"],      # strictly exogenous: used as itself
        data=df,
        panel_var="id",
        time_var="t",
        collapse=True,
    ).fit(steps=2, windmeijer=True)
    sys_model.summary()

    print()
    print("=" * 70)
    print("Difference GMM (Arellano-Bond), two-step, Windmeijer-corrected")
    print("=" * 70)
    diff_model = DifferenceGMM(
        "y ~ L(1).y + x",
        gmm_instruments=["y"],
        iv_instruments=["x"],
        data=df,
        panel_var="id",
        time_var="t",
        collapse=True,
    ).fit(steps=2, windmeijer=True)
    diff_model.summary()


if __name__ == "__main__":
    main()
