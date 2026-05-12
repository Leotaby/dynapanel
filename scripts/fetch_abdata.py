"""Fetch the Arellano-Bond (1991) UK employment dataset for replication tests.

The dataset (commonly known as ``abdata``) is the canonical example for
both Difference and System GMM. It contains 140 UK manufacturing firms
observed over 1976–1984. Variables include log employment (``n``), log
wages (``w``), log capital (``k``), log industry output (``ys``), and a
year column. It's the same dataset shipped with Stata's ``xtabond2``
helpers and R's ``plm::EmplUK``.

This script downloads the dataset from a stable mirror (the ``pydynpd``
repository, which redistributes it with an explicit MIT license) and
saves it as a parquet under ``tests/fixtures/abdata.parquet``.

Usage:
    python scripts/fetch_abdata.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import polars as pl

# The pydynpd repo (MIT-licensed) ships abdata as a CSV. This is the most
# stable redistributable copy I know of as of 2026. If this URL ever rots,
# the AER data archive or the `plm` R package's data folder are both
# viable alternates.
SOURCE_URL = (
    "https://raw.githubusercontent.com/dazhwu/pydynpd/main/test/data/abdata.csv"
)

DEST = Path(__file__).parent.parent / "tests" / "fixtures" / "abdata.parquet"


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)

    if DEST.exists():
        print(f"Already cached at {DEST}; skipping download.")
        return 0

    print(f"Downloading abdata from {SOURCE_URL} ...")
    try:
        with urlopen(SOURCE_URL, timeout=30) as resp:
            csv_bytes = resp.read()
    except URLError as e:
        print(
            f"ERROR: couldn't download from {SOURCE_URL!r}.\n"
            f"  Reason: {e}\n"
            f"\n"
            f"Fallbacks:\n"
            f"  - Install R + plm and export EmplUK to CSV / parquet.\n"
            f"  - Download from an AER data archive mirror.\n"
            f"  - Use any clean copy of Arellano-Bond's UK employment panel.\n"
            f"\n"
            f"Save the resulting file as {DEST}.",
            file=sys.stderr,
        )
        return 1

    # Save as parquet for speed and stable schema across pandas/polars versions.
    df = pl.read_csv(csv_bytes)
    df.write_parquet(DEST)
    print(f"Saved {df.height} rows × {df.width} cols to {DEST}.")
    print(f"Columns: {df.columns}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
