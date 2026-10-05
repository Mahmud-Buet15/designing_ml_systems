"""Tests run against a throwaway data dir filled with synthetic, TLC-shaped fixture data."""

import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="taxieta-test-"))
os.environ["TAXIETA_DATA"] = str(_TMP / "data")
os.environ["TAXIETA_REPORTS"] = str(_TMP / "reports")

import pytest  # noqa: E402

from taxieta.features import build  # noqa: E402
from taxieta.ingest.fixture import write_fixture  # noqa: E402
from taxieta.ingest.lake import ingest_month  # noqa: E402

MONTHS = [(2019, 1), (2019, 2), (2019, 3), (2020, 4)]


@pytest.fixture(scope="session")
def lake():
    write_fixture(MONTHS, n=4000)
    build._borough_map.cache_clear()
    reports = {f"{y}-{m:02d}": ingest_month(y, m) for y, m in MONTHS}
    return reports
